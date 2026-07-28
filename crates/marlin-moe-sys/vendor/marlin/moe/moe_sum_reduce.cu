// moe_sum_reduce.cu — fast top-k sum reduction for Marlin MoE outputs.
//
// Replaces Candle's generic `fast_sum_bf16` (the kernel called by `.sum(1)`
// on a `[M, top_k, K]` tensor) which dominates K2 GPU time at non-trivial M:
//
//     nsys (8× H200, Kimi K2.6 INT4, --cuda-graph-trace=node)
//     ──────────────────────────────────────────────────────────
//     bs=16  in=512  out=128 :  fast_sum_bf16  =  34.9 % of GPU time
//     bs=1   in=4096 out=16  :  fast_sum_bf16  =  53.8 % of GPU time
//
// Ported (with adaptations) from SGLang's
// `sgl-kernel/csrc/moe/moe_sum_reduce.cu` (Apache-2.0). Key changes:
//
// * Plain `extern "C"` entry points so we don't pull in torch headers.
// * Bare-pointer interface — strides in elements, callers own buffers.
// * `routed_scaling_factor` is *not* applied here — for Kimi K2 the scale
//   already lives on the per-row top-k weights that Marlin applies inside
//   its FC2 epilogue, so we pass `scale=1.0f` and skip the FMA. (The arg
//   stays in the ABI so we can fuse it in later if a model needs it.)
//
// Three kernel variants behind one dispatcher (`marlin_moe_impl_sum_reduce_bf16`):
//
//   1. **vectorized warp-per-token** (BF16 only, hidden % 8 == 0, M > 256)
//      Each warp owns one output token. Threads issue 16-byte (8×BF16)
//      `__ldg` loads in a `uint4`, accumulate in FP32, and store 16 B back.
//      Hits ~95 % of HBM peak on H200.
//
//   2. **warp-per-token scalar fallback** (M > 128)
//      Same warp ownership but one BF16 element per thread per top-k step.
//      Used when hidden is not 8-aligned or for FP16/F32 dtypes.
//
//   3. **block-per-token** (M ≤ 128)
//      Avoids the warp-occupancy cliff when there are very few tokens
//      (decode at bs=1 hits this path with M=1).
//
// Variants 2 and 3 specialise on `TOPK ∈ {2, 4, 8, 9}` for `#pragma unroll`
// and fall back to a runtime-`topk_num` general kernel otherwise. Kimi K2
// uses TOPK=8.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace modeld_moe_sum_reduce {

// -------- type helpers ------------------------------------------------------
template <typename T>
__device__ __forceinline__ float to_f32(T x) {
  return static_cast<float>(x);
}
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <>
__device__ __forceinline__ float to_f32<__half>(__half x) {
  return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T from_f32(float x);
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}
template <>
__device__ __forceinline__ __half from_f32<__half>(float x) {
  return __float2half_rn(x);
}

union Pack16B {
  uint4 v;
  __nv_bfloat16 b[8];
};

// -------- (1) vectorized BF16 warp-per-token --------------------------------
//
// Grid:  (cols_per_warp, tokens_per_block)
// Block: (THREADS = WARPS_PER_BLOCK * 32)
//
// Each warp processes one output token; threads within the warp stride
// through `hidden_dim / 8` 16-byte chunks. Inside each chunk we accumulate
// `topk` values in FP32 and store the result back as BF16.
template <int WARPS_PER_BLOCK>
__global__ void warp_per_token_vec_bf16(
    const __nv_bfloat16* __restrict__ x,  // [M, topk, hidden]
    __nv_bfloat16* __restrict__ y,        // [M, hidden]
    const int64_t num_tokens,
    const int64_t hidden_dim,
    const int64_t topk_num,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const float scale) {
  constexpr int VEC = 8;  // 8 BF16 elements per 16-byte load
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= num_tokens) return;

  const int64_t n_chunks = hidden_dim / VEC;

  for (int64_t chunk = (int64_t)blockIdx.x * 32 + lane; chunk < n_chunks;
       chunk += (int64_t)gridDim.x * 32) {
    const int64_t d = chunk * VEC;
    const int64_t base = t * stride_token + d;

    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

    for (int64_t k = 0; k < topk_num; ++k) {
      Pack16B pack;
      pack.v = __ldg(reinterpret_cast<const uint4*>(x + base + k * stride_topk));
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        acc[i] += __bfloat162float(pack.b[i]);
      }
    }

    Pack16B outp;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      outp.b[i] = __float2bfloat16_rn(acc[i] * scale);
    }
    *reinterpret_cast<uint4*>(y + t * out_stride_token + d) = outp.v;
  }
}

// -------- (2) warp-per-token scalar (TOPK template) -------------------------
template <typename T, int TOPK, int WARPS_PER_BLOCK>
__global__ void warp_per_token_scalar(
    const T* __restrict__ x,
    T* __restrict__ y,
    const int64_t num_tokens,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const float scale) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= num_tokens) return;

  for (int64_t d = (int64_t)blockIdx.x * 32 + lane; d < hidden_dim;
       d += (int64_t)gridDim.x * 32) {
    float acc = 0.0f;
    const int64_t base = t * stride_token + d;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      acc += to_f32<T>(__ldg(x + base + (int64_t)k * stride_topk));
    }
    y[t * out_stride_token + d] = from_f32<T>(acc * scale);
  }
}

// -------- (3) block-per-token (TOPK template) -------------------------------
template <typename T, int TOPK>
__global__ void block_per_token(
    const T* __restrict__ x,
    T* __restrict__ y,
    const int64_t num_tokens,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const float scale) {
  for (int64_t t = blockIdx.y; t < num_tokens; t += gridDim.y) {
    for (int64_t d = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         d < hidden_dim;
         d += (int64_t)blockDim.x * gridDim.x) {
      float acc = 0.0f;
      const int64_t base = t * stride_token + d;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        acc += to_f32<T>(__ldg(x + base + (int64_t)k * stride_topk));
      }
      y[t * out_stride_token + d] = from_f32<T>(acc * scale);
    }
  }
}

// -------- general-topk fallbacks --------------------------------------------
template <typename T, int WARPS_PER_BLOCK>
__global__ void warp_per_token_scalar_general(
    const T* __restrict__ x,
    T* __restrict__ y,
    const int64_t num_tokens,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const int topk_num,
    const float scale) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= num_tokens) return;

  for (int64_t d = (int64_t)blockIdx.x * 32 + lane; d < hidden_dim;
       d += (int64_t)gridDim.x * 32) {
    float acc = 0.0f;
    const int64_t base = t * stride_token + d;
    for (int k = 0; k < topk_num; ++k) {
      acc += to_f32<T>(__ldg(x + base + (int64_t)k * stride_topk));
    }
    y[t * out_stride_token + d] = from_f32<T>(acc * scale);
  }
}

template <typename T>
__global__ void block_per_token_general(
    const T* __restrict__ x,
    T* __restrict__ y,
    const int64_t num_tokens,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const int topk_num,
    const float scale) {
  for (int64_t t = blockIdx.y; t < num_tokens; t += gridDim.y) {
    for (int64_t d = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         d < hidden_dim;
         d += (int64_t)blockDim.x * gridDim.x) {
      float acc = 0.0f;
      const int64_t base = t * stride_token + d;
      for (int k = 0; k < topk_num; ++k) {
        acc += to_f32<T>(__ldg(x + base + (int64_t)k * stride_topk));
      }
      y[t * out_stride_token + d] = from_f32<T>(acc * scale);
    }
  }
}

// -------- dispatcher --------------------------------------------------------
//
// Picks one of the three families above based on `(dtype, M, hidden, topk)`.
// Returns a non-zero CUDA error code on failure so the caller can surface
// the right error message at the right call site.
template <typename T>
static cudaError_t dispatch(const T* x,
                            T* y,
                            int64_t num_tokens,
                            int64_t topk_num,
                            int64_t hidden_dim,
                            int64_t stride_token,
                            int64_t stride_topk,
                            int64_t out_stride_token,
                            float scale,
                            cudaStream_t stream) {
  if (num_tokens == 0 || hidden_dim == 0) return cudaSuccess;

  constexpr int MAX_GRID = 65535;
  const bool large_tokens = num_tokens > 128;

  // (1) vectorized BF16 fast path
  if (std::is_same<T, __nv_bfloat16>::value && hidden_dim % 8 == 0 &&
      num_tokens > 256) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    const int64_t n_chunks = hidden_dim / 8;
    int64_t gx = (n_chunks + 32 - 1) / 32;
    if (gx > MAX_GRID) gx = MAX_GRID;
    int64_t gy = (num_tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (gy > MAX_GRID) gy = MAX_GRID;
    dim3 grid((unsigned)gx, (unsigned)gy);
    warp_per_token_vec_bf16<WARPS_PER_BLOCK>
        <<<grid, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(y),
            num_tokens, hidden_dim, topk_num,
            stride_token, stride_topk, out_stride_token, scale);
    return cudaGetLastError();
  }

  // (2)/(3) topk-specialised templates for {2, 4, 8, 9}
  if (large_tokens) {
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int64_t gx = (hidden_dim + 32 - 1) / 32;
    if (gx > MAX_GRID) gx = MAX_GRID;
    int64_t gy = (num_tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (gy > MAX_GRID) gy = MAX_GRID;
    dim3 grid((unsigned)gx, (unsigned)gy);
    dim3 block(THREADS);
    switch (topk_num) {
      case 2:
        warp_per_token_scalar<T, 2, WARPS_PER_BLOCK>
            <<<grid, block, 0, stream>>>(x, y, num_tokens, hidden_dim,
                                         stride_token, stride_topk,
                                         out_stride_token, scale);
        break;
      case 4:
        warp_per_token_scalar<T, 4, WARPS_PER_BLOCK>
            <<<grid, block, 0, stream>>>(x, y, num_tokens, hidden_dim,
                                         stride_token, stride_topk,
                                         out_stride_token, scale);
        break;
      case 8:
        warp_per_token_scalar<T, 8, WARPS_PER_BLOCK>
            <<<grid, block, 0, stream>>>(x, y, num_tokens, hidden_dim,
                                         stride_token, stride_topk,
                                         out_stride_token, scale);
        break;
      case 9:
        warp_per_token_scalar<T, 9, WARPS_PER_BLOCK>
            <<<grid, block, 0, stream>>>(x, y, num_tokens, hidden_dim,
                                         stride_token, stride_topk,
                                         out_stride_token, scale);
        break;
      default:
        warp_per_token_scalar_general<T, WARPS_PER_BLOCK>
            <<<grid, block, 0, stream>>>(x, y, num_tokens, hidden_dim,
                                         stride_token, stride_topk,
                                         out_stride_token,
                                         static_cast<int>(topk_num), scale);
        break;
    }
    return cudaGetLastError();
  }

  // small-token block-per-token
  const int block_size = 256;
  int64_t gx = (hidden_dim + block_size - 1) / block_size;
  if (gx > MAX_GRID) gx = MAX_GRID;
  int64_t gy = num_tokens < MAX_GRID ? num_tokens : MAX_GRID;
  dim3 grid((unsigned)gx, (unsigned)gy);
  dim3 block(block_size);
  switch (topk_num) {
    case 2:
      block_per_token<T, 2><<<grid, block, 0, stream>>>(
          x, y, num_tokens, hidden_dim, stride_token, stride_topk,
          out_stride_token, scale);
      break;
    case 4:
      block_per_token<T, 4><<<grid, block, 0, stream>>>(
          x, y, num_tokens, hidden_dim, stride_token, stride_topk,
          out_stride_token, scale);
      break;
    case 8:
      block_per_token<T, 8><<<grid, block, 0, stream>>>(
          x, y, num_tokens, hidden_dim, stride_token, stride_topk,
          out_stride_token, scale);
      break;
    case 9:
      block_per_token<T, 9><<<grid, block, 0, stream>>>(
          x, y, num_tokens, hidden_dim, stride_token, stride_topk,
          out_stride_token, scale);
      break;
    default:
      block_per_token_general<T><<<grid, block, 0, stream>>>(
          x, y, num_tokens, hidden_dim, stride_token, stride_topk,
          out_stride_token, static_cast<int>(topk_num), scale);
      break;
  }
  return cudaGetLastError();
}

}  // namespace modeld_moe_sum_reduce

// ----------------------------------------------------------------------------
// C ABI
// ----------------------------------------------------------------------------
//
// `input`  : `[num_tokens, topk, hidden_dim]` (row-major in elements)
// `output` : `[num_tokens, hidden_dim]` (row-major in elements)
// Strides are in **elements**, not bytes. Caller is responsible for ensuring
// the buffers are valid for the (M, topk, hidden) extents — there is no
// bounds checking inside the kernel.
//
// Returns 0 on success or a CUDA error code on a launch failure.
extern "C" {

int marlin_moe_impl_sum_reduce_bf16(const void* input,
                               void* output,
                               long num_tokens,
                               long topk_num,
                               long hidden_dim,
                               long stride_token,
                               long stride_topk,
                               long out_stride_token,
                               float scale,
                               cudaStream_t stream) {
  return (int)modeld_moe_sum_reduce::dispatch<__nv_bfloat16>(
      reinterpret_cast<const __nv_bfloat16*>(input),
      reinterpret_cast<__nv_bfloat16*>(output),
      (int64_t)num_tokens, (int64_t)topk_num, (int64_t)hidden_dim,
      (int64_t)stride_token, (int64_t)stride_topk, (int64_t)out_stride_token,
      scale, stream);
}

int marlin_moe_impl_sum_reduce_fp16(const void* input,
                               void* output,
                               long num_tokens,
                               long topk_num,
                               long hidden_dim,
                               long stride_token,
                               long stride_topk,
                               long out_stride_token,
                               float scale,
                               cudaStream_t stream) {
  return (int)modeld_moe_sum_reduce::dispatch<__half>(
      reinterpret_cast<const __half*>(input),
      reinterpret_cast<__half*>(output),
      (int64_t)num_tokens, (int64_t)topk_num, (int64_t)hidden_dim,
      (int64_t)stride_token, (int64_t)stride_topk, (int64_t)out_stride_token,
      scale, stream);
}

}  // extern "C"

// moe_support.cu — support kernels needed to drive Marlin MoE.
//
// 1) `modeld_moe_align_block_size`
//    Single-block version of SGLang's `moe_align_kernel.cu` (Apache-2.0).
//    Sorts `[numel] topk_ids` into a `[max_num_tokens_padded] sorted_token_ids`
//    array grouped by expert, padded so each expert's run length is a
//    multiple of `block_size`. Also emits the per-block expert-id array
//    consumed by Marlin's MoE GEMM, the total padded count, and the
//    per-expert exclusive cumulative sum used by the scatter pass.
//
//    Kimi-K2.5 has 384 experts and per-rank batches well below 1k tokens,
//    so a single-block, serial-prefix-sum design is plenty: histogramming
//    is parallel, prefix sum is O(num_experts) on thread 0, and emitting
//    `expert_ids` is parallel.
//
// 2) `marlin_moe_impl_silu_and_mul_bf16` / `_fp16`
//    Fused gated-SiLU activation: out[m, n] = silu(gate[m, n]) * up[m, n],
//    where `in` is the [M, 2N] concatenation [gate | up].

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace marlin_moe_impl_support {

// gridDim=2:
//   block 0: histogram + per-expert padded prefix sum, emits `cumsum`,
//            `expert_ids`, and `total_tokens_post_pad`.
//   block 1: fills `sorted_token_ids` with the sentinel `numel` so any
//            padded slots that the scatter pass doesn't touch are inert
//            for Marlin's GEMM.
__global__ void moe_align_block_size_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t expert_id_offset,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    int32_t max_num_tokens_padded) {
  if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      const int32_t sentinel = (int32_t)numel;
      for (int32_t i = (int32_t)threadIdx.x; i < max_num_tokens_padded;
           i += (int32_t)blockDim.x) {
        sorted_token_ids[i] = sentinel;
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  // Layout:
  //   shared_counts[num_experts]  — raw counts per expert
  //   shared_cumsum[num_experts + 1] — exclusive prefix sum of padded counts
  int32_t* shared_counts = smem;
  int32_t* shared_cumsum = shared_counts + num_experts;

  const int32_t tid = (int32_t)threadIdx.x;
  const int32_t stride = (int32_t)blockDim.x;

  // Zero the counts.
  for (int32_t i = tid; i < num_experts; i += stride) shared_counts[i] = 0;
  __syncthreads();

  // Histogram. Rebase global expert ids onto this rank's local expert range
  // by subtracting `expert_id_offset`; values that fall outside
  // `[0, num_experts)` are silently dropped (they belong to another EP rank
  // and will be handled by the post-MoE all-reduce).
  for (size_t i = tid; i < numel; i += (size_t)stride) {
    int32_t e = topk_ids[i] - expert_id_offset;
    if (e >= 0 && e < num_experts) {
      atomicAdd(&shared_counts[e], 1);
    }
  }
  __syncthreads();

  // Serial exclusive prefix sum of padded counts (thread 0).
  if (tid == 0) {
    int32_t running = 0;
    shared_cumsum[0] = 0;
    for (int32_t e = 0; e < num_experts; ++e) {
      int32_t cnt = shared_counts[e];
      int32_t padded = (cnt + block_size - 1) / block_size * block_size;
      running += padded;
      shared_cumsum[e + 1] = running;
    }
    *total_tokens_post_pad = running;
  }
  __syncthreads();

  // Mirror the cumsum to global so the scatter pass can do atomicAdd onto it.
  for (int32_t i = tid; i <= num_experts; i += stride) {
    cumsum[i] = shared_cumsum[i];
  }

  // Emit one expert id per `block_size`-sized output tile, by binary search
  // over the exclusive padded prefix sum.
  const int32_t total = shared_cumsum[num_experts];
  const int32_t num_blocks = total / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    const int32_t block_start = i * block_size;
    // Find largest e in [0, num_experts) with shared_cumsum[e] <= block_start.
    int32_t lo = 0, hi = num_experts;
    while (lo < hi) {
      int32_t mid = (lo + hi + 1) >> 1;
      if (shared_cumsum[mid] <= block_start) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    expert_ids[i] = lo;
  }
}

// Scatter pass: for each entry `i` in topk_ids, place its index at the next
// available slot within its expert's run. Uses global `cumsum_buffer` (which
// already holds the exclusive padded prefix sum) as a per-expert atomic
// counter — the running value is overwritten with the post-scatter end of
// each expert's real-token region.
__global__ void count_and_sort_expert_tokens_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    int32_t num_experts,
    int32_t expert_id_offset,
    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < numel; i += stride) {
    int32_t e = topk_ids[i] - expert_id_offset;
    if (e < 0 || e >= num_experts) continue;
    int32_t slot = atomicAdd(&cumsum_buffer[e], 1);
    sorted_token_ids[slot] = (int32_t)i;
  }
}

// SiLU(gate) * up. `in` is `[M, 2N]` laid out as `[up | gate]`, matching the
// `flashinfer_moe.rs` loader's `Tensor::cat([&up, &gate], 0)` convention.
// `out` is `[M, N]`.
template <typename T>
__global__ void silu_and_mul_kernel(T* __restrict__ out,
                                    const T* __restrict__ in,
                                    int M, int N) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const T* up = in + m * (2 * N);
  const T* gate = up + N;
  T* o = out + m * N;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    float u = (float)up[n];
    float g = (float)gate[n];
    float s = g / (1.0f + expf(-g));  // SiLU(g) = g * sigmoid(g)
    o[n] = (T)(s * u);
  }
}

}  // namespace marlin_moe_impl_support

// ----------------------------------------------------------------------------
// C ABI
// ----------------------------------------------------------------------------
extern "C" {

void marlin_moe_impl_align_block_size_i32(
    const int32_t* topk_ids,
    int64_t numel,
    int32_t num_experts,
    int32_t expert_id_offset,
    int32_t block_size,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t* num_tokens_post_pad,
    int32_t* cumsum_buffer,
    int32_t max_num_tokens_padded,
    bool pad_sorted_token_ids,
    cudaStream_t stream) {
  using namespace marlin_moe_impl_support;

  // Single-block design: thread count is whatever's most useful for the
  // serial prefix sum + parallel histogram. 1024 is the maximum legal
  // block size and lets us issue up to 1024 atomic adds in parallel during
  // the histogram phase, which dominates this kernel.
  const int threads = 1024;
  const size_t shared_mem_size =
      ((size_t)num_experts + (size_t)num_experts + 1) * sizeof(int32_t);

  moe_align_block_size_kernel<<<2, threads, shared_mem_size, stream>>>(
      topk_ids,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_pad,
      num_experts,
      expert_id_offset,
      block_size,
      (size_t)numel,
      cumsum_buffer,
      pad_sorted_token_ids,
      max_num_tokens_padded);

  const int block_threads = 256;
  const int num_blocks_x =
      (int)((numel + block_threads - 1) / block_threads);
  const int max_blocks = 65535;
  const int actual_blocks = num_blocks_x < 1
                                ? 1
                                : (num_blocks_x < max_blocks ? num_blocks_x
                                                             : max_blocks);

  count_and_sort_expert_tokens_kernel<<<actual_blocks, block_threads, 0, stream>>>(
      topk_ids,
      sorted_token_ids,
      cumsum_buffer,
      num_experts,
      expert_id_offset,
      (size_t)numel);
}

void marlin_moe_impl_silu_and_mul_bf16(void* out, const void* in,
                              int M, int N, cudaStream_t stream) {
  const int threads = N < 1024 ? ((N + 31) / 32) * 32 : 1024;
  marlin_moe_impl_support::silu_and_mul_kernel<__nv_bfloat16>
      <<<M, threads, 0, stream>>>(static_cast<__nv_bfloat16*>(out),
                                  static_cast<const __nv_bfloat16*>(in),
                                  M, N);
}

void marlin_moe_impl_silu_and_mul_fp16(void* out, const void* in,
                              int M, int N, cudaStream_t stream) {
  const int threads = N < 1024 ? ((N + 31) / 32) * 32 : 1024;
  marlin_moe_impl_support::silu_and_mul_kernel<__half>
      <<<M, threads, 0, stream>>>(static_cast<__half*>(out),
                                  static_cast<const __half*>(in),
                                  M, N);
}

int marlin_moe_impl_sync_check(cudaStream_t stream) {
  cudaError_t err;
  if (stream != nullptr) {
    err = cudaStreamSynchronize(stream);
  } else {
    err = cudaDeviceSynchronize();
  }
  if (err != cudaSuccess) return (int)err;
  err = cudaGetLastError();
  return (int)err;
}

const char* marlin_moe_impl_cuda_error_name(int err) {
  return cudaGetErrorName((cudaError_t)err);
}

}  // extern "C"

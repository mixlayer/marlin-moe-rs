// Public C ABI for the Marlin port. All entry points are stream-asynchronous
// and assume the caller has already selected the target CUDA device.
//
// Quantization type ids are the same as vLLM's ScalarType::id() values, so
// the Rust binding can construct them via constexpr expressions:
//
//   kU4   (asymmetric INT4 with zero points) — same as AWQ
//   kU4B8 (symmetric INT4 with implicit bias-8) — same as GPTQ / compressed-tensors `symmetric: true`
//
// Activation / output / scale types follow the same scheme (kBFloat16, kFloat16).
//
// See modeld-kernels/kernels/marlin/core/scalar_type.hpp for the ids.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ScalarType ids exposed from C side so callers don't need to recompute them.
// Computed from scalar_type.hpp at build time.
int64_t marlin_moe_impl_scalar_type_id_bfloat16(void);
int64_t marlin_moe_impl_scalar_type_id_float16(void);
int64_t marlin_moe_impl_scalar_type_id_u4b8(void);
int64_t marlin_moe_impl_scalar_type_id_u4(void);
int64_t marlin_moe_impl_scalar_type_id_fe4m3fn(void);

// Packs row-major FP8 `[size_n, size_k]` bytes into I32 and transposes them
// to `[size_k / 4, size_n]` for the eight-bit Marlin repack kernel.
void marlin_moe_impl_fp8_pack_transpose(
    const uint8_t* weight_ptr,
    uint32_t* packed_transposed_ptr,
    int64_t size_k,
    int64_t size_n,
    cudaStream_t stream);

//
// GPTQ-style INT4 weight repack: takes packed `int32` weights laid out as
// `[size_k / pack_factor, size_n]` (row-major, pack_factor = 32 / num_bits)
// and emits the Marlin interleaved layout
// `[size_k / tile_k_size, size_n * tile_k_size / pack_factor]`.
//
// `perm_ptr` may be `nullptr` (or `perm_size == 0`) when act_order isn't in
// use — that is the compressed-tensors symmetric INT4 case.
//
// All tensors live on `device`. Errors abort the process.
//
void marlin_moe_impl_gptq_repack(
    const uint32_t* b_q_weight_ptr,   // [size_k / pack_factor, size_n] int32
    const uint32_t* perm_ptr,         // optional [size_k] int32, or nullptr
    int64_t perm_size,                // 0 if perm_ptr is unused
    uint32_t* out_ptr,                // [size_k / 16, size_n * 16 / pack_factor]
    int64_t size_k,
    int64_t size_n,
    int64_t num_bits,                 // 4 or 8
    bool is_a_8bit,                   // false for W4A16
    int device,
    cudaStream_t stream);

//
// MoE Marlin fused weight-only GEMM (BF16/FP16 × INT4/FP8).
//
// Weight layout follows `marlin_moe_impl_gptq_repack` output:
//   b_q_weight: [num_experts, size_k / 16, size_n * 16 / pack_factor] uint32_t
//   b_scales:   [num_experts, num_groups, size_n] BF16/FP16
// where num_groups = size_k / group_size.
//
// Token-to-expert routing buffers come from a Marlin-compatible
// `moe_align_block_size` pass (see modeld-kernels' own moe_align helper).
//   sorted_token_ids:        [padded_total] int32 — sorted by expert
//   expert_ids:              [padded_total / moe_block_size] int32
//   num_tokens_post_padded:  [1] int32 — total padded count
//   topk_weights:            [size_m * top_k] float — router weights
//
// `b_zeros_ptr` is non-null only for asymmetric INT4 (b_type == kU4).
// For Kimi K2.5 (symmetric, kU4B8) it must be nullptr.
//
// `workspace` is a transient int32 scratch buffer: at least
// `size_n / 64 * (padded_total / moe_block_size)` ints (or `sms * 4`,
// whichever is smaller — see Marlin's get_kernel_cache_size).
//
// `c_ptr` must be sized [size_m * top_k, size_n] and is written by the kernel.
//
void marlin_moe_impl_moe_wna16_gemm(
    const void* a_ptr,                  // [size_m, size_k] BF16/FP16
    void* c_ptr,                        // [size_m * top_k, size_n] same dtype as a
    float* c_tmp_ptr,                   // FP32 reduction scratch; sized by
                                        // marlin_moe_impl_moe_wna16_c_tmp_elems(...)
    const uint32_t* b_q_weight_ptr,     // [num_experts, K/16, N*16/pack_factor]
    const void* b_scales_ptr,           // [num_experts, num_groups, size_n] BF16/FP16
    const uint32_t* b_zeros_ptr,        // optional [E, num_groups, N/pack_factor], or nullptr
    const float* topk_weights_ptr,      // [size_m * top_k]
    const int32_t* sorted_token_ids_ptr,
    const int32_t* expert_ids_ptr,
    const int32_t* num_tokens_post_padded_ptr,
    int32_t* workspace_ptr,             // int32 scratch
    int64_t moe_block_size,             // 16, 32, 48, or 64
    int64_t num_experts,
    int64_t top_k,
    bool mul_topk_weights,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    int64_t a_type_id,                  // ScalarType::id() — kBFloat16 or kFloat16
    int64_t b_type_id,                  // kU4B8, kU4, or kFE4M3fn
    int64_t c_type_id,                  // matches a_type_id for 16-bit activations
    int64_t s_type_id,                  // matches c_type_id
    bool is_k_full,                     // true when there's no act_order
    int64_t group_size,                 // e.g. 32 for Kimi K2.5
    int device,
    cudaStream_t stream);

// Required size of the FP32 `c_tmp` reduce scratch buffer (in float elements)
// to feed `marlin_moe_impl_moe_wna16_gemm`. Mirrors the bound vLLM allocates
// inside its torch wrapper:
//   min(size_n * padded_total, sms * 4 * moe_block_size * 256) * (block_size==8 ? 2 : 1)
// `padded_total` is the size of `sorted_token_ids` (= `num_tokens_post_pad`
// upper bound). `device` is the CUDA ordinal used to query the SM count.
int64_t marlin_moe_impl_moe_wna16_c_tmp_elems(
    int64_t size_n, int64_t padded_total, int64_t moe_block_size, int device);

// ----------------------------------------------------------------------------
// MoE alignment + activation support kernels
// ----------------------------------------------------------------------------

// Single-block `moe_align_block_size` (adapted from SGLang). Computes the
// `sorted_token_ids` / `expert_ids` / `num_tokens_post_padded` triple Marlin
// expects, from a `[numel] int32 topk_ids` buffer.
//
// `expert_id_offset` is subtracted from each `topk_ids[i]` before clamping
// to `[0, num_experts)`. This lets callers pass a routing tensor that uses
// the *global* expert numbering while running expert-parallel inference: an
// EP rank with `num_local = num_global / ep_size` experts passes
// `expert_id_offset = ep_rank * num_local` and `num_experts = num_local`,
// and topk entries that aren't on this rank fall outside `[0, num_local)`
// and are silently dropped.
//
// `cumsum_buffer` is a transient scratch of `num_experts + 1` int32s.
// `max_num_tokens_padded` is `(numel + num_experts * (block_size - 1))`
// rounded up to whatever upper bound the caller pre-allocated for
// `sorted_token_ids`.
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
    cudaStream_t stream);

// Fused gated-SiLU activation: out[m, n] = silu(in[m, n]) * in[m, N + n],
// where `in` is `[M, 2N]` (gate | up concatenation) and `out` is `[M, N]`.
// Buffers must be the matching CUDA dtype (BF16 or FP16); typed as `void*`
// here to keep this header CUDA-free.
void marlin_moe_impl_silu_and_mul_bf16(void* out, const void* in,
                              int M, int N, cudaStream_t stream);
void marlin_moe_impl_silu_and_mul_fp16(void* out, const void* in,
                              int M, int N, cudaStream_t stream);

int marlin_moe_impl_sum_reduce_bf16(
    const void* input, void* output, int64_t num_tokens, int64_t topk_num,
    int64_t hidden_dim, int64_t stride_token, int64_t stride_topk,
    int64_t out_stride_token, float scale, cudaStream_t stream);
int marlin_moe_impl_sum_reduce_fp16(
    const void* input, void* output, int64_t num_tokens, int64_t topk_num,
    int64_t hidden_dim, int64_t stride_token, int64_t stride_topk,
    int64_t out_stride_token, float scale, cudaStream_t stream);

// ----------------------------------------------------------------------------
// Diagnostics
// ----------------------------------------------------------------------------

// Synchronize the given stream and return any sticky CUDA error code. Useful
// for pinpointing which kernel failed under `CUDA_LAUNCH_BLOCKING=1`. Returns
// 0 on success, otherwise the CUDA error code. Pass `stream = nullptr` to
// synchronize the device instead of a specific stream.
int marlin_moe_impl_sync_check(cudaStream_t stream);

// Human-readable name for a CUDA error code (returns "" if unknown). Buffer
// is shared, copy out if you need to hold it. Thread-safe (returns a pointer
// into the CUDA runtime's static table).
const char* marlin_moe_impl_cuda_error_name(int err);

#ifdef __cplusplus
}  // extern "C"
#endif

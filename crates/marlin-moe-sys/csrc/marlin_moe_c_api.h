#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum marlin_moe_status_t {
  MARLIN_MOE_STATUS_SUCCESS = 0,
  MARLIN_MOE_STATUS_INVALID_ARGUMENT = 1,
  MARLIN_MOE_STATUS_UNSUPPORTED_ARCH = 2,
  MARLIN_MOE_STATUS_CUDA_ERROR = 3,
  MARLIN_MOE_STATUS_INTERNAL_ERROR = 4,
} marlin_moe_status_t;

typedef enum marlin_moe_scalar_type_t {
  MARLIN_MOE_SCALAR_BFLOAT16 = 0,
  MARLIN_MOE_SCALAR_FLOAT16 = 1,
  MARLIN_MOE_SCALAR_U4B8 = 2,
  MARLIN_MOE_SCALAR_U4 = 3,
  MARLIN_MOE_SCALAR_FE4M3FN = 4,
} marlin_moe_scalar_type_t;

const char* marlin_moe_last_error(void);

marlin_moe_status_t marlin_moe_scalar_type_id(
    marlin_moe_scalar_type_t scalar_type, int64_t* id);

marlin_moe_status_t marlin_moe_gptq_repack(
    const uint32_t* b_q_weight_ptr, const uint32_t* perm_ptr,
    int64_t perm_size, uint32_t* out_ptr, int64_t size_k, int64_t size_n,
    int64_t num_bits, bool is_a_8bit, int device, cudaStream_t stream);

marlin_moe_status_t marlin_moe_fp8_pack_transpose(
    const uint8_t* weight_ptr, uint32_t* packed_transposed_ptr,
    int64_t size_k, int64_t size_n, cudaStream_t stream);

marlin_moe_status_t marlin_moe_wna16_c_tmp_elems(
    int64_t size_n, int64_t padded_total, int64_t moe_block_size, int device,
    int64_t* elements);

marlin_moe_status_t marlin_moe_wna16_gemm(
    const void* a_ptr, void* c_ptr, float* c_tmp_ptr,
    const uint32_t* b_q_weight_ptr, const void* b_scales_ptr,
    const uint32_t* b_zeros_ptr, const float* topk_weights_ptr,
    const int32_t* sorted_token_ids_ptr, const int32_t* expert_ids_ptr,
    const int32_t* num_tokens_post_padded_ptr, int32_t* workspace_ptr,
    int64_t moe_block_size, int64_t num_experts, int64_t top_k,
    bool mul_topk_weights, int64_t size_m, int64_t size_n, int64_t size_k,
    int64_t a_type_id, int64_t b_type_id, int64_t c_type_id,
    int64_t s_type_id, bool is_k_full, int64_t group_size, int device,
    cudaStream_t stream);

marlin_moe_status_t marlin_moe_align_block_size_i32(
    const int32_t* topk_ids, int64_t numel, int32_t num_experts,
    int32_t expert_id_offset, int32_t block_size, int32_t* sorted_token_ids,
    int32_t* expert_ids, int32_t* num_tokens_post_pad,
    int32_t* cumsum_buffer, int32_t max_num_tokens_padded,
    bool pad_sorted_token_ids, cudaStream_t stream);

marlin_moe_status_t marlin_moe_silu_and_mul_bf16(
    void* out, const void* input, int m, int n, cudaStream_t stream);
marlin_moe_status_t marlin_moe_silu_and_mul_fp16(
    void* out, const void* input, int m, int n, cudaStream_t stream);

marlin_moe_status_t marlin_moe_sum_reduce_bf16(
    const void* input, void* output, int64_t num_tokens, int64_t topk_num,
    int64_t hidden_dim, int64_t stride_token, int64_t stride_topk,
    int64_t out_stride_token, float scale, cudaStream_t stream);
marlin_moe_status_t marlin_moe_sum_reduce_fp16(
    const void* input, void* output, int64_t num_tokens, int64_t topk_num,
    int64_t hidden_dim, int64_t stride_token, int64_t stride_topk,
    int64_t out_stride_token, float scale, cudaStream_t stream);

marlin_moe_status_t marlin_moe_sync_check(cudaStream_t stream);

#ifdef __cplusplus
}
#endif

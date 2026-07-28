#include "marlin_moe_c_api.h"

#include <exception>
#include <stdexcept>
#include <string>

#include "marlin_moe_impl.h"

namespace {
thread_local std::string last_error;

marlin_moe_status_t invalid(const char* message) {
  last_error = message;
  return MARLIN_MOE_STATUS_INVALID_ARGUMENT;
}

marlin_moe_status_t cuda_status(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return MARLIN_MOE_STATUS_SUCCESS;
  last_error = std::string(context) + ": " + cudaGetErrorName(error) + ": " +
               cudaGetErrorString(error);
  return MARLIN_MOE_STATUS_CUDA_ERROR;
}

template <typename F>
marlin_moe_status_t guarded(F&& function) {
  last_error.clear();
  try {
    function();
    return cuda_status(cudaPeekAtLastError(), "CUDA launch failed");
  } catch (const std::invalid_argument& error) {
    last_error = error.what();
    return MARLIN_MOE_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    last_error = error.what();
    return MARLIN_MOE_STATUS_INTERNAL_ERROR;
  } catch (...) {
    last_error = "unknown native exception";
    return MARLIN_MOE_STATUS_INTERNAL_ERROR;
  }
}
}  // namespace

extern "C" {

const char* marlin_moe_last_error(void) { return last_error.c_str(); }

marlin_moe_status_t marlin_moe_scalar_type_id(
    marlin_moe_scalar_type_t scalar_type, int64_t* id) {
  if (id == nullptr) return invalid("scalar type output pointer is null");
  return guarded([&] {
    switch (scalar_type) {
      case MARLIN_MOE_SCALAR_BFLOAT16:
        *id = marlin_moe_impl_scalar_type_id_bfloat16();
        break;
      case MARLIN_MOE_SCALAR_FLOAT16:
        *id = marlin_moe_impl_scalar_type_id_float16();
        break;
      case MARLIN_MOE_SCALAR_U4B8:
        *id = marlin_moe_impl_scalar_type_id_u4b8();
        break;
      case MARLIN_MOE_SCALAR_U4:
        *id = marlin_moe_impl_scalar_type_id_u4();
        break;
      case MARLIN_MOE_SCALAR_FE4M3FN:
        *id = marlin_moe_impl_scalar_type_id_fe4m3fn();
        break;
      default:
        throw std::invalid_argument("unknown scalar type");
    }
  });
}

marlin_moe_status_t marlin_moe_gptq_repack(
    const uint32_t* b_q_weight_ptr, const uint32_t* perm_ptr,
    int64_t perm_size, uint32_t* out_ptr, int64_t size_k, int64_t size_n,
    int64_t num_bits, bool is_a_8bit, int device, cudaStream_t stream) {
  if (b_q_weight_ptr == nullptr || out_ptr == nullptr)
    return invalid("repack input and output pointers must be non-null");
  return guarded([&] {
    marlin_moe_impl_gptq_repack(b_q_weight_ptr, perm_ptr, perm_size, out_ptr,
                                size_k, size_n, num_bits, is_a_8bit, device,
                                stream);
  });
}

marlin_moe_status_t marlin_moe_fp8_pack_transpose(
    const uint8_t* weight_ptr, uint32_t* packed_transposed_ptr,
    int64_t size_k, int64_t size_n, cudaStream_t stream) {
  if (weight_ptr == nullptr || packed_transposed_ptr == nullptr)
    return invalid("FP8 pack input and output pointers must be non-null");
  if (size_k <= 0 || size_n <= 0 || size_k % 4 != 0)
    return invalid("FP8 pack dimensions must be positive and K divisible by four");
  return guarded([&] {
    marlin_moe_impl_fp8_pack_transpose(
        weight_ptr, packed_transposed_ptr, size_k, size_n, stream);
  });
}

marlin_moe_status_t marlin_moe_wna16_c_tmp_elems(
    int64_t size_n, int64_t padded_total, int64_t moe_block_size, int device,
    int64_t* elements) {
  if (elements == nullptr) return invalid("workspace size output is null");
  if (size_n <= 0 || padded_total <= 0 || moe_block_size <= 0)
    return invalid("workspace dimensions must be positive");
  return guarded([&] {
    *elements = marlin_moe_impl_moe_wna16_c_tmp_elems(
        size_n, padded_total, moe_block_size, device);
  });
}

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
    cudaStream_t stream) {
  if (a_ptr == nullptr || c_ptr == nullptr || c_tmp_ptr == nullptr ||
      b_q_weight_ptr == nullptr || b_scales_ptr == nullptr ||
      topk_weights_ptr == nullptr || sorted_token_ids_ptr == nullptr ||
      expert_ids_ptr == nullptr || num_tokens_post_padded_ptr == nullptr ||
      workspace_ptr == nullptr)
    return invalid("required GEMM pointer is null");
  return guarded([&] {
    marlin_moe_impl_moe_wna16_gemm(
        a_ptr, c_ptr, c_tmp_ptr, b_q_weight_ptr, b_scales_ptr, b_zeros_ptr,
        topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_post_padded_ptr, workspace_ptr, moe_block_size, num_experts,
        top_k, mul_topk_weights, size_m, size_n, size_k, a_type_id, b_type_id,
        c_type_id, s_type_id, is_k_full, group_size, device, stream);
  });
}

marlin_moe_status_t marlin_moe_align_block_size_i32(
    const int32_t* topk_ids, int64_t numel, int32_t num_experts,
    int32_t expert_id_offset, int32_t block_size, int32_t* sorted_token_ids,
    int32_t* expert_ids, int32_t* num_tokens_post_pad,
    int32_t* cumsum_buffer, int32_t max_num_tokens_padded,
    bool pad_sorted_token_ids, cudaStream_t stream) {
  if (topk_ids == nullptr || sorted_token_ids == nullptr ||
      expert_ids == nullptr || num_tokens_post_pad == nullptr ||
      cumsum_buffer == nullptr)
    return invalid("required alignment pointer is null");
  return guarded([&] {
    marlin_moe_impl_align_block_size_i32(
        topk_ids, numel, num_experts, expert_id_offset, block_size,
        sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum_buffer,
        max_num_tokens_padded, pad_sorted_token_ids, stream);
  });
}

marlin_moe_status_t marlin_moe_silu_and_mul_bf16(
    void* out, const void* input, int m, int n, cudaStream_t stream) {
  if (out == nullptr || input == nullptr) return invalid("activation pointer is null");
  return guarded(
      [&] { marlin_moe_impl_silu_and_mul_bf16(out, input, m, n, stream); });
}

marlin_moe_status_t marlin_moe_silu_and_mul_fp16(
    void* out, const void* input, int m, int n, cudaStream_t stream) {
  if (out == nullptr || input == nullptr) return invalid("activation pointer is null");
  return guarded(
      [&] { marlin_moe_impl_silu_and_mul_fp16(out, input, m, n, stream); });
}

#define WRAP_SUM(NAME, IMPL)                                                \
  marlin_moe_status_t NAME(                                                \
      const void* input, void* output, int64_t num_tokens, int64_t topk_num, \
      int64_t hidden_dim, int64_t stride_token, int64_t stride_topk,        \
      int64_t out_stride_token, float scale, cudaStream_t stream) {         \
    if (input == nullptr || output == nullptr)                              \
      return invalid("sum-reduction pointer is null");                     \
    return guarded([&] {                                                   \
      int error = IMPL(input, output, num_tokens, topk_num, hidden_dim,     \
                       stride_token, stride_topk, out_stride_token, scale,  \
                       stream);                                            \
      if (error != 0)                                                      \
        throw std::runtime_error(cudaGetErrorString((cudaError_t)error));   \
    });                                                                    \
  }

WRAP_SUM(marlin_moe_sum_reduce_bf16, marlin_moe_impl_sum_reduce_bf16)
WRAP_SUM(marlin_moe_sum_reduce_fp16, marlin_moe_impl_sum_reduce_fp16)

marlin_moe_status_t marlin_moe_sync_check(cudaStream_t stream) {
  last_error.clear();
  int error = marlin_moe_impl_sync_check(stream);
  return cuda_status((cudaError_t)error, "CUDA synchronization failed");
}

}  // extern "C"

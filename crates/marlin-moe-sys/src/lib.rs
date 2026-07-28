#![allow(non_camel_case_types)]
//! Raw status-returning Marlin MoE C ABI bindings.

use core::ffi::{c_char, c_int, c_void};

/// Opaque CUDA stream handle.
pub type cudaStream_t = *mut c_void;

/// Vendored Marlin source root selected by the build script.
pub const MARLIN_MOE_SOURCE_ROOT: &str = env!("MARLIN_MOE_SOURCE_ROOT");

/// Status returned by every operational Marlin MoE C ABI entry point.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum marlin_moe_status_t {
    /// The operation was validated and enqueued successfully.
    MARLIN_MOE_STATUS_SUCCESS = 0,
    /// A caller-provided pointer, dimension, dtype identifier, or layout was invalid.
    MARLIN_MOE_STATUS_INVALID_ARGUMENT = 1,
    /// The selected build or runtime GPU architecture is unsupported.
    MARLIN_MOE_STATUS_UNSUPPORTED_ARCH = 2,
    /// CUDA reported an API or asynchronous launch error.
    MARLIN_MOE_STATUS_CUDA_ERROR = 3,
    /// Native code threw or encountered an unexpected internal failure.
    MARLIN_MOE_STATUS_INTERNAL_ERROR = 4,
}

/// Scalar type selector used to retrieve stable native Marlin type identifiers.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum marlin_moe_scalar_type_t {
    /// BF16 activation, output, or scale type.
    MARLIN_MOE_SCALAR_BFLOAT16 = 0,
    /// FP16 activation, output, or scale type.
    MARLIN_MOE_SCALAR_FLOAT16 = 1,
    /// Symmetric INT4 weights with implicit bias eight.
    MARLIN_MOE_SCALAR_U4B8 = 2,
    /// Asymmetric unsigned INT4 weights with explicit zero points.
    MARLIN_MOE_SCALAR_U4 = 3,
}

#[cfg(not(feature = "unsupported_arch"))]
unsafe extern "C" {
    /// Returns a pointer to a thread-local diagnostic valid until the next ABI call on this thread.
    pub fn marlin_moe_last_error() -> *const c_char;

    /// Writes the native Marlin scalar identifier selected by `scalar_type` to `id`.
    pub fn marlin_moe_scalar_type_id(
        scalar_type: marlin_moe_scalar_type_t,
        id: *mut i64,
    ) -> marlin_moe_status_t;

    /// Enqueues GPTQ repacking from `[K / pack_factor, N]` I32 into Marlin tile layout.
    ///
    /// Device pointers must remain valid through stream completion. `out_ptr` is writable and
    /// non-overlapping; `perm_ptr` may be null exactly when `perm_size` is zero.
    pub fn marlin_moe_gptq_repack(
        b_q_weight_ptr: *const u32,
        perm_ptr: *const u32,
        perm_size: i64,
        out_ptr: *mut u32,
        size_k: i64,
        size_n: i64,
        num_bits: i64,
        is_a_8bit: bool,
        device: c_int,
        stream: cudaStream_t,
    ) -> marlin_moe_status_t;

    /// Calculates the number of F32 elements required by Marlin's reduction workspace.
    pub fn marlin_moe_wna16_c_tmp_elems(
        size_n: i64,
        padded_total: i64,
        moe_block_size: i64,
        device: c_int,
        elements: *mut i64,
    ) -> marlin_moe_status_t;

    /// Enqueues routed W4A16 GEMM on caller-owned CUDA buffers and stream.
    ///
    /// Pointer shapes, dtypes, layouts, device ownership, workspace lengths, and lifetimes are
    /// described and validated by `marlin-moe::GemmLaunchParams`.
    #[allow(clippy::too_many_arguments)]
    pub fn marlin_moe_wna16_gemm(
        a_ptr: *const c_void,
        c_ptr: *mut c_void,
        c_tmp_ptr: *mut f32,
        b_q_weight_ptr: *const u32,
        b_scales_ptr: *const c_void,
        b_zeros_ptr: *const u32,
        topk_weights_ptr: *const f32,
        sorted_token_ids_ptr: *const i32,
        expert_ids_ptr: *const i32,
        num_tokens_post_padded_ptr: *const i32,
        workspace_ptr: *mut i32,
        moe_block_size: i64,
        num_experts: i64,
        top_k: i64,
        mul_topk_weights: bool,
        size_m: i64,
        size_n: i64,
        size_k: i64,
        a_type_id: i64,
        b_type_id: i64,
        c_type_id: i64,
        s_type_id: i64,
        is_k_full: bool,
        group_size: i64,
        device: c_int,
        stream: cudaStream_t,
    ) -> marlin_moe_status_t;

    /// Enqueues expert alignment for contiguous I32 top-k ids and caller-owned I32 workspaces.
    #[allow(clippy::too_many_arguments)]
    pub fn marlin_moe_align_block_size_i32(
        topk_ids: *const i32,
        numel: i64,
        num_experts: i32,
        expert_id_offset: i32,
        block_size: i32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        cumsum_buffer: *mut i32,
        max_num_tokens_padded: i32,
        pad_sorted_token_ids: bool,
        stream: cudaStream_t,
    ) -> marlin_moe_status_t;

    /// Enqueues BF16 gated-SiLU from `[M, 2N]` into `[M, N]`.
    pub fn marlin_moe_silu_and_mul_bf16(
        out: *mut c_void,
        input: *const c_void,
        m: c_int,
        n: c_int,
        stream: cudaStream_t,
    ) -> marlin_moe_status_t;

    /// Enqueues FP16 gated-SiLU from `[M, 2N]` into `[M, N]`.
    pub fn marlin_moe_silu_and_mul_fp16(
        out: *mut c_void,
        input: *const c_void,
        m: c_int,
        n: c_int,
        stream: cudaStream_t,
    ) -> marlin_moe_status_t;

    /// Enqueues BF16 top-k reduction from `[M, top_k, hidden]` into `[M, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn marlin_moe_sum_reduce_bf16(
        input: *const c_void,
        output: *mut c_void,
        num_tokens: i64,
        topk_num: i64,
        hidden_dim: i64,
        stride_token: i64,
        stride_topk: i64,
        out_stride_token: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> marlin_moe_status_t;

    /// Enqueues FP16 top-k reduction from `[M, top_k, hidden]` into `[M, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn marlin_moe_sum_reduce_fp16(
        input: *const c_void,
        output: *mut c_void,
        num_tokens: i64,
        topk_num: i64,
        hidden_dim: i64,
        stride_token: i64,
        stride_topk: i64,
        out_stride_token: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> marlin_moe_status_t;

    /// Synchronizes the supplied CUDA stream and converts any CUDA error into ABI status.
    pub fn marlin_moe_sync_check(stream: cudaStream_t) -> marlin_moe_status_t;
}

#[cfg(feature = "unsupported_arch")]
mod unsupported {
    use super::*;

    static ERROR: &[u8] = b"Marlin MoE was built for an unsupported architecture\0";

    macro_rules! unsupported {
        ($name:ident($($arg:ident: $ty:ty),* $(,)?) -> $return:ty) => {
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn $name($($arg: $ty),*) -> $return {
                $(let _ = $arg;)*
                marlin_moe_status_t::MARLIN_MOE_STATUS_UNSUPPORTED_ARCH
            }
        };
    }

    /// Returns the diagnostic used by unsupported-architecture stubs.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn marlin_moe_last_error() -> *const c_char {
        ERROR.as_ptr().cast()
    }

    unsupported!(marlin_moe_scalar_type_id(
        scalar_type: marlin_moe_scalar_type_t,
        id: *mut i64
    ) -> marlin_moe_status_t);
    unsupported!(marlin_moe_gptq_repack(
        b_q_weight_ptr: *const u32, perm_ptr: *const u32, perm_size: i64,
        out_ptr: *mut u32, size_k: i64, size_n: i64, num_bits: i64,
        is_a_8bit: bool, device: c_int, stream: cudaStream_t
    ) -> marlin_moe_status_t);
    unsupported!(marlin_moe_wna16_c_tmp_elems(
        size_n: i64, padded_total: i64, moe_block_size: i64, device: c_int,
        elements: *mut i64
    ) -> marlin_moe_status_t);
    unsupported!(marlin_moe_wna16_gemm(
        a_ptr: *const c_void, c_ptr: *mut c_void, c_tmp_ptr: *mut f32,
        b_q_weight_ptr: *const u32, b_scales_ptr: *const c_void,
        b_zeros_ptr: *const u32, topk_weights_ptr: *const f32,
        sorted_token_ids_ptr: *const i32, expert_ids_ptr: *const i32,
        num_tokens_post_padded_ptr: *const i32, workspace_ptr: *mut i32,
        moe_block_size: i64, num_experts: i64, top_k: i64, mul_topk_weights: bool,
        size_m: i64, size_n: i64, size_k: i64, a_type_id: i64, b_type_id: i64,
        c_type_id: i64, s_type_id: i64, is_k_full: bool, group_size: i64,
        device: c_int, stream: cudaStream_t
    ) -> marlin_moe_status_t);
    unsupported!(marlin_moe_align_block_size_i32(
        topk_ids: *const i32, numel: i64, num_experts: i32, expert_id_offset: i32,
        block_size: i32, sorted_token_ids: *mut i32, expert_ids: *mut i32,
        num_tokens_post_pad: *mut i32, cumsum_buffer: *mut i32,
        max_num_tokens_padded: i32, pad_sorted_token_ids: bool, stream: cudaStream_t
    ) -> marlin_moe_status_t);
    unsupported!(marlin_moe_silu_and_mul_bf16(
        out: *mut c_void, input: *const c_void, m: c_int, n: c_int,
        stream: cudaStream_t
    ) -> marlin_moe_status_t);
    unsupported!(marlin_moe_silu_and_mul_fp16(
        out: *mut c_void, input: *const c_void, m: c_int, n: c_int,
        stream: cudaStream_t
    ) -> marlin_moe_status_t);
    unsupported!(marlin_moe_sum_reduce_bf16(
        input: *const c_void, output: *mut c_void, num_tokens: i64, topk_num: i64,
        hidden_dim: i64, stride_token: i64, stride_topk: i64,
        out_stride_token: i64, scale: f32, stream: cudaStream_t
    ) -> marlin_moe_status_t);
    unsupported!(marlin_moe_sum_reduce_fp16(
        input: *const c_void, output: *mut c_void, num_tokens: i64, topk_num: i64,
        hidden_dim: i64, stride_token: i64, stride_topk: i64,
        out_stride_token: i64, scale: f32, stream: cudaStream_t
    ) -> marlin_moe_status_t);
    unsupported!(marlin_moe_sync_check(
        stream: cudaStream_t
    ) -> marlin_moe_status_t);
}

#[cfg(feature = "unsupported_arch")]
pub use unsupported::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_values_are_stable() {
        assert_eq!(marlin_moe_status_t::MARLIN_MOE_STATUS_SUCCESS as i32, 0);
        assert_eq!(
            marlin_moe_status_t::MARLIN_MOE_STATUS_UNSUPPORTED_ARCH as i32,
            2
        );
    }

    #[cfg(feature = "unsupported_arch")]
    #[test]
    fn unsupported_stub_returns_status_and_diagnostic() {
        let mut id = 0;
        let status = unsafe {
            marlin_moe_scalar_type_id(
                marlin_moe_scalar_type_t::MARLIN_MOE_SCALAR_BFLOAT16,
                &mut id,
            )
        };
        assert_eq!(
            status,
            marlin_moe_status_t::MARLIN_MOE_STATUS_UNSUPPORTED_ARCH
        );
        let message = unsafe { std::ffi::CStr::from_ptr(marlin_moe_last_error()) };
        assert!(message.to_string_lossy().contains("unsupported"));
    }
}

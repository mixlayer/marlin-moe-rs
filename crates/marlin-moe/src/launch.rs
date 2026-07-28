use core::ffi::c_void;

use marlin_moe_sys::{
    marlin_moe_align_block_size_i32 as sys_align, marlin_moe_gptq_repack as sys_repack,
    marlin_moe_scalar_type_id as sys_scalar_type_id, marlin_moe_scalar_type_t,
    marlin_moe_silu_and_mul_bf16, marlin_moe_silu_and_mul_fp16, marlin_moe_status_t,
    marlin_moe_sum_reduce_bf16, marlin_moe_sum_reduce_fp16, marlin_moe_sync_check,
    marlin_moe_wna16_gemm as sys_gemm,
};

use crate::{ActivationDtype, Error, GemmDims, Quantization, RepackDims, Result, WeightDtype};

/// Native scalar identifier requested from the vendored Marlin type system.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ScalarType {
    /// BF16 activation, output, or scale.
    Bf16,
    /// FP16 activation, output, or scale.
    F16,
    /// Symmetric INT4 weight.
    U4B8,
    /// Asymmetric INT4 weight.
    U4,
}

/// Returns the native scalar type identifier used in raw GEMM parameters.
pub fn scalar_type_id(scalar_type: ScalarType) -> Result<i64> {
    let selector = match scalar_type {
        ScalarType::Bf16 => marlin_moe_scalar_type_t::MARLIN_MOE_SCALAR_BFLOAT16,
        ScalarType::F16 => marlin_moe_scalar_type_t::MARLIN_MOE_SCALAR_FLOAT16,
        ScalarType::U4B8 => marlin_moe_scalar_type_t::MARLIN_MOE_SCALAR_U4B8,
        ScalarType::U4 => marlin_moe_scalar_type_t::MARLIN_MOE_SCALAR_U4,
    };
    let mut id = 0;
    let status = unsafe { sys_scalar_type_id(selector, &mut id) };
    check(status, "Marlin scalar type lookup failed")?;
    Ok(id)
}

/// Caller-owned raw GPTQ repack buffers and CUDA execution context.
///
/// `input` points to contiguous I32 `[K / pack_factor, N]`; `output` points to writable I32
/// `[K / 16, N * 16 / pack_factor]`. `perm` is null when `perm_len` is zero or points to
/// contiguous U32 `[K]`. All pointers belong to `device`, remain live until `stream` completes,
/// and do not overlap incompatibly.
#[derive(Debug, Copy, Clone)]
pub struct RepackLaunchParams {
    /// Validated logical dimensions and bit width.
    pub dims: RepackDims,
    /// Packed input device pointer.
    pub input: *const u32,
    /// Optional permutation device pointer.
    pub perm: *const u32,
    /// Permutation element count.
    pub perm_len: usize,
    /// Writable packed output device pointer.
    pub output: *mut u32,
    /// Whether activations are eight bit.
    pub is_a_8bit: bool,
    /// CUDA device ordinal owning every pointer.
    pub device: i32,
    /// CUDA stream receiving the asynchronous launch.
    pub stream: *mut c_void,
}

/// Validates raw repack parameters and enqueues the asynchronous CUDA kernel.
///
/// # Safety
///
/// The caller must uphold every pointer, shape, dtype, device, lifetime, aliasing, and stream
/// requirement documented on [`RepackLaunchParams`].
pub unsafe fn gptq_repack(params: &RepackLaunchParams) -> Result<()> {
    params.dims.validate()?;
    if params.input.is_null() || params.output.is_null() {
        return Err(Error::InvalidArgument(
            "repack input and output pointers must be non-null".into(),
        ));
    }
    if params.perm_len != 0 && params.perm.is_null() {
        return Err(Error::InvalidArgument(
            "repack permutation is null with non-zero length".into(),
        ));
    }
    let status = unsafe {
        sys_repack(
            params.input,
            params.perm,
            i64::try_from(params.perm_len)
                .map_err(|_| Error::InvalidArgument("permutation length exceeds i64".into()))?,
            params.output,
            params.dims.k as i64,
            params.dims.n as i64,
            params.dims.bits as i64,
            params.is_a_8bit,
            params.device,
            params.stream,
        )
    };
    check(status, "Marlin GPTQ repack failed")
}

/// Caller-owned raw routed GEMM buffers and CUDA execution context.
///
/// `a` is contiguous `[M, K]` BF16/FP16. `c` is writable
/// `[M * top_k, N]` of the same dtype. `b_q_weight` is contiguous
/// `[num_experts, K / 16, N * 2]` I32 for INT4, scales are contiguous
/// `[num_experts, K / group_size, N]` with activation dtype, and optional zeros match the
/// asymmetric packed layout. Routing weights are F32 `[M, top_k]`; token ids, expert ids,
/// padded count, and workspace are contiguous I32 buffers sized by the Candle integration.
/// Every pointer belongs to `device`, remains live through `stream` completion, and is accessed
/// only in the direction indicated by its mutability.
#[derive(Debug, Copy, Clone)]
pub struct GemmLaunchParams {
    /// Logical GEMM and routing dimensions.
    pub dims: GemmDims,
    /// Quantization and dtype configuration.
    pub quantization: Quantization,
    /// Activation input pointer.
    pub a: *const c_void,
    /// Writable output pointer.
    pub c: *mut c_void,
    /// Writable F32 reduction workspace.
    pub c_tmp: *mut f32,
    /// Packed INT4 weight pointer.
    pub b_q_weight: *const u32,
    /// BF16 or FP16 scale pointer.
    pub b_scales: *const c_void,
    /// Optional asymmetric packed zero-point pointer.
    pub b_zeros: *const u32,
    /// F32 routing weights.
    pub topk_weights: *const f32,
    /// Sorted I32 token ids.
    pub sorted_token_ids: *const i32,
    /// I32 expert id per aligned block.
    pub expert_ids: *const i32,
    /// I32 scalar padded-token count.
    pub num_tokens_post_padded: *const i32,
    /// Writable I32 Marlin lock workspace.
    pub workspace: *mut i32,
    /// Whether this GEMM epilogue multiplies routed weights.
    pub mul_topk_weights: bool,
    /// Whether K includes all groups without activation-order indirection.
    pub is_k_full: bool,
    /// CUDA device ordinal owning all buffers.
    pub device: i32,
    /// CUDA stream receiving the asynchronous launch.
    pub stream: *mut c_void,
}

/// Validates raw GEMM parameters and enqueues Marlin W4A16 GEMM.
///
/// # Safety
///
/// The caller must uphold every pointer, shape, dtype, device, workspace, lifetime, aliasing, and
/// stream requirement documented on [`GemmLaunchParams`].
pub unsafe fn wna16_gemm(params: &GemmLaunchParams) -> Result<()> {
    params.dims.validate(params.quantization)?;
    for (name, pointer) in [
        ("a", params.a),
        ("c", params.c.cast_const()),
        ("c_tmp", params.c_tmp.cast()),
        ("b_q_weight", params.b_q_weight.cast()),
        ("b_scales", params.b_scales),
        ("topk_weights", params.topk_weights.cast()),
        ("sorted_token_ids", params.sorted_token_ids.cast()),
        ("expert_ids", params.expert_ids.cast()),
        (
            "num_tokens_post_padded",
            params.num_tokens_post_padded.cast(),
        ),
        ("workspace", params.workspace.cast()),
    ] {
        if pointer.is_null() {
            return Err(Error::InvalidArgument(format!(
                "required GEMM pointer {name} is null"
            )));
        }
    }
    if params.quantization.weight == WeightDtype::U4 && params.b_zeros.is_null() {
        return Err(Error::InvalidArgument(
            "asymmetric U4 weights require zero points".into(),
        ));
    }
    if params.quantization.weight == WeightDtype::U4B8 && !params.b_zeros.is_null() {
        return Err(Error::InvalidArgument(
            "symmetric U4B8 weights must not provide zero points".into(),
        ));
    }
    let activation = match params.quantization.activation {
        ActivationDtype::Bf16 => scalar_type_id(ScalarType::Bf16)?,
        ActivationDtype::F16 => scalar_type_id(ScalarType::F16)?,
    };
    let weight = match params.quantization.weight {
        WeightDtype::U4B8 => scalar_type_id(ScalarType::U4B8)?,
        WeightDtype::U4 => scalar_type_id(ScalarType::U4)?,
    };
    let status = unsafe {
        sys_gemm(
            params.a,
            params.c,
            params.c_tmp,
            params.b_q_weight,
            params.b_scales,
            params.b_zeros,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_padded,
            params.workspace,
            params.dims.block_size as i64,
            params.dims.num_experts as i64,
            params.dims.top_k as i64,
            params.mul_topk_weights,
            params.dims.m as i64,
            params.dims.n as i64,
            params.dims.k as i64,
            activation,
            weight,
            activation,
            activation,
            params.is_k_full,
            params.quantization.group_size as i64,
            params.device,
            params.stream,
        )
    };
    check(status, "Marlin W4A16 GEMM failed")
}

/// Caller-owned expert alignment buffers.
///
/// `topk_ids` is contiguous I32 `[numel]`; outputs are writable contiguous I32 buffers sized for
/// the documented maximum padded rows, blocks, scalar count, and `num_experts + 1` cumsum entries.
#[derive(Debug, Copy, Clone)]
pub struct AlignLaunchParams {
    /// Input top-k id pointer.
    pub topk_ids: *const i32,
    /// Number of input expert ids.
    pub numel: usize,
    /// Number of experts local to this rank.
    pub num_experts: usize,
    /// Global expert id belonging to local expert zero.
    pub expert_id_offset: usize,
    /// Alignment tile size.
    pub block_size: usize,
    /// Writable sorted token id pointer.
    pub sorted_token_ids: *mut i32,
    /// Writable expert id pointer.
    pub expert_ids: *mut i32,
    /// Writable scalar padded-token count.
    pub num_tokens_post_pad: *mut i32,
    /// Writable cumsum scratch pointer.
    pub cumsum_buffer: *mut i32,
    /// Capacity of `sorted_token_ids`.
    pub max_num_tokens_padded: usize,
    /// Whether unused sorted-token slots are padded.
    pub pad_sorted_token_ids: bool,
    /// CUDA stream receiving the asynchronous launch.
    pub stream: *mut c_void,
}

/// Validates alignment geometry and enqueues expert alignment.
///
/// # Safety
///
/// All pointers must satisfy the shapes, I32 dtypes, CUDA device ownership, lifetimes, and
/// writable capacities documented on [`AlignLaunchParams`].
pub unsafe fn align_block_size_i32(params: &AlignLaunchParams) -> Result<()> {
    if params.numel == 0 || params.num_experts == 0 || params.block_size == 0 {
        return Err(Error::InvalidArgument(
            "alignment dimensions must be positive".into(),
        ));
    }
    let status = unsafe {
        sys_align(
            params.topk_ids,
            checked_i64(params.numel, "numel")?,
            checked_i32(params.num_experts, "num_experts")?,
            checked_i32(params.expert_id_offset, "expert_id_offset")?,
            checked_i32(params.block_size, "block_size")?,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad,
            params.cumsum_buffer,
            checked_i32(params.max_num_tokens_padded, "max_num_tokens_padded")?,
            params.pad_sorted_token_ids,
            params.stream,
        )
    };
    check(status, "Marlin expert alignment failed")
}

/// Caller-owned gated-SiLU input and output buffers.
///
/// `input` is contiguous `[M, 2N]` and `output` is writable contiguous `[M, N]`; both use `dtype`
/// and remain live on the stream's CUDA device through launch completion.
#[derive(Debug, Copy, Clone)]
pub struct SiluLaunchParams {
    /// Activation/output dtype.
    pub dtype: ActivationDtype,
    /// Writable output pointer.
    pub output: *mut c_void,
    /// Input pointer.
    pub input: *const c_void,
    /// Row count.
    pub m: usize,
    /// Gated output feature count.
    pub n: usize,
    /// CUDA stream receiving the asynchronous launch.
    pub stream: *mut c_void,
}

/// Validates and enqueues gated-SiLU.
///
/// # Safety
///
/// The raw pointers must satisfy [`SiluLaunchParams`] shape, dtype, ownership, aliasing, and
/// stream-lifetime requirements.
pub unsafe fn silu_and_mul(params: &SiluLaunchParams) -> Result<()> {
    if params.output.is_null() || params.input.is_null() || params.m == 0 || params.n == 0 {
        return Err(Error::InvalidArgument(
            "activation pointers and dimensions must be non-zero".into(),
        ));
    }
    let status = unsafe {
        match params.dtype {
            ActivationDtype::Bf16 => marlin_moe_silu_and_mul_bf16(
                params.output,
                params.input,
                checked_i32(params.m, "activation M")?,
                checked_i32(params.n, "activation N")?,
                params.stream,
            ),
            ActivationDtype::F16 => marlin_moe_silu_and_mul_fp16(
                params.output,
                params.input,
                checked_i32(params.m, "activation M")?,
                checked_i32(params.n, "activation N")?,
                params.stream,
            ),
        }
    };
    check(status, "Marlin gated-SiLU failed")
}

/// Caller-owned top-k reduction buffers with element strides.
///
/// `input` is BF16/FP16 `[M, top_k, hidden]` and `output` is writable
/// `[M, hidden]`, potentially as offset-contiguous views. Strides are in dtype elements.
#[derive(Debug, Copy, Clone)]
pub struct SumReduceLaunchParams {
    /// Input/output dtype.
    pub dtype: ActivationDtype,
    /// Input pointer.
    pub input: *const c_void,
    /// Writable output pointer.
    pub output: *mut c_void,
    /// Token count.
    pub num_tokens: usize,
    /// Experts selected per token.
    pub top_k: usize,
    /// Hidden feature count.
    pub hidden: usize,
    /// Input stride between tokens, in elements.
    pub stride_token: usize,
    /// Input stride between top-k rows, in elements.
    pub stride_topk: usize,
    /// Output stride between tokens, in elements.
    pub out_stride_token: usize,
    /// Scaling factor applied after FP32 accumulation.
    pub scale: f32,
    /// CUDA stream receiving the asynchronous launch.
    pub stream: *mut c_void,
}

/// Validates and enqueues the Marlin MoE top-k sum reduction.
///
/// # Safety
///
/// The raw pointers and strides must satisfy every shape, dtype, device, lifetime, and aliasing
/// requirement documented on [`SumReduceLaunchParams`].
pub unsafe fn sum_reduce(params: &SumReduceLaunchParams) -> Result<()> {
    if params.input.is_null() || params.output.is_null() {
        return Err(Error::InvalidArgument(
            "sum-reduction pointers must be non-null".into(),
        ));
    }
    let arguments = (
        params.input,
        params.output,
        checked_i64(params.num_tokens, "num_tokens")?,
        checked_i64(params.top_k, "top_k")?,
        checked_i64(params.hidden, "hidden")?,
        checked_i64(params.stride_token, "stride_token")?,
        checked_i64(params.stride_topk, "stride_topk")?,
        checked_i64(params.out_stride_token, "out_stride_token")?,
        params.scale,
        params.stream,
    );
    let status = unsafe {
        match params.dtype {
            ActivationDtype::Bf16 => marlin_moe_sum_reduce_bf16(
                arguments.0,
                arguments.1,
                arguments.2,
                arguments.3,
                arguments.4,
                arguments.5,
                arguments.6,
                arguments.7,
                arguments.8,
                arguments.9,
            ),
            ActivationDtype::F16 => marlin_moe_sum_reduce_fp16(
                arguments.0,
                arguments.1,
                arguments.2,
                arguments.3,
                arguments.4,
                arguments.5,
                arguments.6,
                arguments.7,
                arguments.8,
                arguments.9,
            ),
        }
    };
    check(status, "Marlin top-k sum reduction failed")
}

/// Synchronizes a CUDA stream for diagnostics and converts sticky CUDA errors to Rust errors.
///
/// This function is intentionally not used on hot inference paths or during CUDA graph capture.
///
/// # Safety
///
/// `stream` is null for the default device synchronization or a valid live CUDA stream handle.
pub unsafe fn sync_check(stream: *mut c_void) -> Result<()> {
    let status = unsafe { marlin_moe_sync_check(stream) };
    check(status, "Marlin CUDA synchronization failed")
}

fn check(status: marlin_moe_status_t, context: &str) -> Result<()> {
    if status == marlin_moe_status_t::MARLIN_MOE_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::from_status(status, context))
    }
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidArgument(format!("{name} exceeds i32: {value}")))
}

fn checked_i64(value: usize, name: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::InvalidArgument(format!("{name} exceeds i64: {value}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_null_repack_pointers_before_ffi() {
        let result = unsafe {
            gptq_repack(&RepackLaunchParams {
                dims: RepackDims {
                    k: 128,
                    n: 64,
                    bits: 4,
                },
                input: core::ptr::null(),
                perm: core::ptr::null(),
                perm_len: 0,
                output: core::ptr::null_mut(),
                is_a_8bit: false,
                device: 0,
                stream: core::ptr::null_mut(),
            })
        };
        assert!(matches!(result, Err(Error::InvalidArgument(_))));
    }
}

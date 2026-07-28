//! Marlin MoE W4A16 INT4 kernel bindings (port of vLLM's Marlin).
//!
//! Used on Hopper (SM90) for Kimi-K2.5-style checkpoints
//! (compressed-tensors `symmetric: true`, `group_size: 32`, no act_order)
//! where FlashInfer's fused-MoE has no working W4A16 path that respects the
//! native group size.
//!
//! The Marlin kernel itself is an Ampere-MMA mainloop that runs unchanged
//! on SM80–SM90 and natively supports runtime `group_size ∈ {32, 64, 128}`
//! plus optional zero points. Asymmetric INT4 (kU4, AWQ-style) and
//! symmetric INT4 (kU4B8, GPTQ / compressed-tensors style) are both wired.
//!
//! Weight preprocessing
//! --------------------
//! Compressed-tensors `pack-quantized` ships INT4 weights as
//! `[out_features, in_features / 8]` INT32 (8 nibbles per int32) per
//! Linear, packed in GPTQ nibble order. After per-expert stacking that's
//!
//!     fc1_packed: [E, 2 * inter, K / 8]    INT32
//!     fc1_scale : [E, 2 * inter, K / group_size] BF16/FP16
//!     fc2_packed: [E, K, inter / 8]        INT32
//!     fc2_scale : [E, K, inter / group_size]     BF16/FP16
//!
//! Marlin's `gptq_marlin_repack` expects `[K / 8, N]` INT32 per expert
//! (transposed) and produces `[K / 16, N * 2]` INT32 in its tile-interleaved
//! layout. Scales additionally need a fixed 8×8-block permutation
//! (`get_scale_perms`) applied along the last axis.
//!
//! Public surface:
//! - [`MarlinInt4Moe::from_packed_weights`] — one-shot preprocessing
//!   wrapper that takes the same buffers `flashinfer_moe.rs` already
//!   loads (Flashinfer-style layout) and returns a Marlin-ready handle.
//! - [`MarlinInt4Moe::forward`] — full fc1 → SiLU·mul → fc2 → topk-sum
//!   pipeline; matches `_fused_marlin_moe` from vLLM.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::tensor::{cuda_any, cuda_f32, cuda_i32};
use candle::{
    DType, Device, IndexOp, Result, Tensor, bail,
    cuda::cudarc::driver::sys::{CUresult, cuMemsetD8Async},
};
use marlin_moe::{
    ActivationDtype, AlignLaunchParams, GemmDims, GemmLaunchParams, Quantization, RepackDims,
    RepackLaunchParams, SiluLaunchParams, SumReduceLaunchParams, WeightDtype,
};

fn marlin<T>(result: marlin_moe::Result<T>) -> Result<T> {
    result.map_err(|error| candle::Error::Msg(error.to_string()))
}

fn activation_dtype(dtype: DType) -> Result<ActivationDtype> {
    match dtype {
        DType::BF16 => Ok(ActivationDtype::Bf16),
        DType::F16 => Ok(ActivationDtype::F16),
        _ => bail!("Marlin: unsupported activation dtype {:?}", dtype),
    }
}

/// Synchronize `stream` and return the next CUDA error encountered, if any.
/// Used as a checkpoint after C ABI launches to surface OOB kernels at the
/// right call site when `CUDA_LAUNCH_BLOCKING=1` is in effect.
///
/// NOT used on the hot decode/prefill path — `cudaStreamSynchronize` is
/// expensive (host roundtrip every layer) and is illegal during CUDA graph
/// capture. Keep this function for ad-hoc debugging only.
#[allow(dead_code)]
fn cuda_check(stream: &candle::cuda::cudarc::driver::CudaStream, where_: &str) -> Result<()> {
    marlin(unsafe { marlin_moe::sync_check(stream.cu_stream().cast()) })
        .map_err(|error| candle::Error::Msg(format!("{where_}: {error}")))
}

// --------------------------------------------------------------------------
// Scale permutation (CPU host-side, runs once at load).
// --------------------------------------------------------------------------

fn marlin_scale_perm() -> [usize; 64] {
    let mut perm = [0usize; 64];
    let mut idx = 0;
    for i in 0..8 {
        for j in 0..8 {
            perm[idx] = i + 8 * j;
            idx += 1;
        }
    }
    perm
}

fn permute_marlin_scales<T: WithPrim + Copy>(
    scales: &[T],
    num_experts: usize,
    num_groups: usize,
    size_n: usize,
) -> Vec<T> {
    let perm = marlin_scale_perm();
    let perm_len = perm.len(); // 64
    assert!(
        (num_groups * size_n) % perm_len == 0,
        "Marlin scale layout requires num_groups * N to be a multiple of 64; got {} * {} = {}",
        num_groups,
        size_n,
        num_groups * size_n
    );
    let total_per_expert = num_groups * size_n;
    let blocks_per_expert = total_per_expert / perm_len;
    let mut out = vec![T::zero(); scales.len()];
    for e in 0..num_experts {
        let base = e * total_per_expert;
        for b in 0..blocks_per_expert {
            let block_base = base + b * perm_len;
            for (dst_idx, &src_idx) in perm.iter().enumerate() {
                out[block_base + dst_idx] = scales[block_base + src_idx];
            }
        }
    }
    out
}

trait WithPrim {
    fn zero() -> Self;
}
impl WithPrim for half::bf16 {
    fn zero() -> Self {
        half::bf16::ZERO
    }
}
impl WithPrim for half::f16 {
    fn zero() -> Self {
        half::f16::ZERO
    }
}

// --------------------------------------------------------------------------
// Public: GPU repack + Marlin MoE handle
// --------------------------------------------------------------------------

/// Marlin MoE block size — the M-tile size the kernel batches per expert.
pub const MARLIN_MOE_BLOCK_SIZE: i32 = 32;

/// Run the Marlin GPTQ repack op on a single packed weight tensor.
///
/// Input is `[size_k / 8, size_n]` INT32 (transposed compressed-tensors
/// layout). Output is `[size_k / 16, size_n * 2]` INT32 in Marlin's
/// tile-interleaved layout.
pub fn marlin_gptq_repack(b_q_weight: &Tensor) -> Result<Tensor> {
    let (rows, cols) = b_q_weight.dims2()?;
    let size_k = rows * 8;
    let size_n = cols;
    if size_k % 16 != 0 {
        bail!(
            "marlin_gptq_repack: size_k = {} must be divisible by 16",
            size_k
        );
    }
    if size_n % 64 != 0 {
        bail!(
            "marlin_gptq_repack: size_n = {} must be divisible by 64 (Marlin tile_n_size)",
            size_n
        );
    }
    if b_q_weight.dtype() != DType::I32 {
        bail!(
            "marlin_gptq_repack: input must be I32, got {:?}",
            b_q_weight.dtype()
        );
    }

    let cuda_dev = b_q_weight.device().as_cuda_device()?;
    let stream = cuda_dev.cuda_stream();
    let dev_ordinal = stream.context().ordinal() as i32;

    let b_q_weight = b_q_weight.contiguous()?;
    let out_shape = (size_k / 16, size_n * 2);
    let out = Tensor::zeros(out_shape, DType::I32, b_q_weight.device())?;

    // Scoped storage borrows so `out` can be moved into the return.
    {
        let (b_s, b_l) = b_q_weight.storage_and_layout();
        let (out_s, out_l) = out.storage_and_layout();
        let b_ptr = cuda_i32(&b_s, b_l.start_offset(), &stream)?;
        let out_ptr = cuda_i32(&out_s, out_l.start_offset(), &stream)?;
        marlin(unsafe {
            marlin_moe::gptq_repack(&RepackLaunchParams {
                dims: RepackDims {
                    k: size_k,
                    n: size_n,
                    bits: 4,
                },
                input: b_ptr.as_const::<u32>(),
                perm: core::ptr::null(),
                perm_len: 0,
                output: out_ptr.as_mut::<u32>(),
                is_a_8bit: false,
                device: dev_ordinal,
                stream: stream.cu_stream().cast(),
            })
        })?;
    }
    Ok(out)
}

/// Per-`M` scratch buffer set for `MarlinInt4Moe::forward`.
///
/// Each Marlin MoE forward needs ~10 transient tensors whose shapes depend on
/// the runtime batch dim `M`. The naive implementation calls `Tensor::zeros`
/// for all of them per forward, which translates to one `cuMemAllocAsync`
/// + `cuMemsetD8Async` + `cuMemFreeAsync` per buffer. Across 60 MoE layers
/// × 64 tokens × 8 ranks that's ~340 K stream-ordered alloc/free pairs
/// (≈ 16 % of CPU API time at bs=1 decode, from the nsys baseline).
///
/// We cache one `MarlinScratch` per `(top_k, inter_size, hidden_size,
/// num_local_experts, scale_dtype)` in a process-wide thread-local pool that
/// all MoE layers in the same model share (they all run sequentially on the
/// same rank and have identical dimensions). Buffers are sized for the
/// worst-case `max_m` configured at model load (see
/// `MarlinInt4Moe::prealloc_scratch`); each forward narrows the M-dependent
/// fields (`fc1_out`, `act_out`, `fc2_out`, `out`) to the active
/// `m * top_k` / `m` rows before passing them to the kernels.
///
/// This is intentional: previously the pool was keyed by exact `M` and had
/// no eviction, so a long-running process with many distinct prefill /
/// decode batch sizes would accumulate one ~1 GiB scratch entry per unique
/// `M` and eventually OOM on rank 0 (which already carries the largest
/// memory budget for logits + sampler state). Pinning the scratch to one
/// worst-case entry per geometry caps total scratch memory at a single
/// known-good value validated up front at load.
///
/// Only the four lock / counter buffers (`workspace`, `workspace2`,
/// `num_tokens_post_pad`, `cumsum_buffer`) plus the active region of
/// `fc2_out` are zeroed each forward; the rest are fully overwritten by
/// kernels and can be reused unchanged.
///
/// `Clone` is cheap — every field is an `Arc`-backed `Tensor`, so cloning
/// just bumps refcounts. The pool keeps the canonical copy alive; callers
/// take a clone so they can drop the `RefCell` borrow before launching
/// kernels.
#[derive(Debug, Clone)]
struct MarlinScratch {
    /// Worst-case `M` this scratch was sized for. Forwards with `m`
    /// exceeding this trigger a one-shot grow + log; they no longer add a
    /// second pool entry.
    max_m: usize,
    /// `[max_padded(max_m)]` I32 — token slot index per padded row, written
    /// by `moe_align_block_size_kernel`.
    sorted_token_ids: Tensor,
    /// `[max_blocks(max_m)]` I32 — expert id per output tile, written by
    /// `moe_align_block_size_kernel`.
    expert_ids: Tensor,
    /// `[1]` I32 — running count, atomically incremented by
    /// `moe_align_block_size_kernel`. Needs zeroing each call.
    num_tokens_post_pad: Tensor,
    /// `[max(num_local_experts+1, 64)]` I32 — exclusive cumulative sum
    /// scratch for `moe_align_block_size_kernel`. Needs zeroing each call.
    cumsum_buffer: Tensor,
    /// `[workspace_size(max_m)]` I32 — Marlin GEMM lock array (fc1 path).
    /// Needs zeroing each call.
    workspace: Tensor,
    /// `[workspace_size(max_m)]` I32 — Marlin GEMM lock array (fc2 path).
    /// Needs zeroing each call.
    workspace2: Tensor,
    /// `[c_tmp_elems(max_m)]` F32 — FP32 reduce scratch shared between fc1
    /// and fc2. Each Marlin CTA writes its own slot, so no pre-zero needed.
    c_tmp: Tensor,
    /// `[max_m*top_k, 2*inter]` BF16/FP16 — fc1 output. Narrowed to
    /// `[m*top_k, ...]` per call before being passed to kernels.
    fc1_out: Tensor,
    /// `[max_m*top_k, inter]` BF16/FP16 — `silu_and_mul` output. Narrowed
    /// per call.
    act_out: Tensor,
    /// `[max_m*top_k, K]` BF16/FP16 — fc2 output. Narrowed per call. Only
    /// the active region is zeroed in `reset_locks` (off-rank EP rows
    /// outside `[0..m*top_k]` are never read).
    fc2_out: Tensor,
    /// `[max_m, K]` BF16/FP16 — post `top_k` sum output. Narrowed to
    /// `[m, K]` per call so the value returned to the caller has the
    /// expected shape.
    out: Tensor,
}

impl MarlinScratch {
    /// Allocate a fresh worst-case scratch set sized for `max_m`. Called
    /// once per `(geometry)` at load time via
    /// `MarlinInt4Moe::prealloc_scratch`; the resulting buffers stay alive
    /// for the lifetime of the process. Forwards with `m <= max_m`
    /// narrow into these buffers in-place (no further allocation).
    #[allow(clippy::too_many_arguments)]
    fn alloc(
        max_m: usize,
        top_k: usize,
        block_size: i32,
        num_local_experts: i32,
        inter_size: usize,
        hidden_size: usize,
        scale_dtype: DType,
        dev_ordinal: i32,
        device: &Device,
    ) -> Result<Self> {
        // Worst-case padded total tokens after expert sort. Matches the
        // formula used in `MarlinInt4Moe::forward`.
        let numel = (max_m * top_k) as i32;
        let max_padded = numel + num_local_experts * (block_size - 1);
        let max_padded = ((max_padded + block_size - 1) / block_size) * block_size;
        let max_blocks = max_padded / block_size;

        // Marlin GEMM lock array (one int32 per output tile × `N / 64` tiles
        // along N). Sized for the larger of the two GEMMs and floored to the
        // Marlin minimum of `132 * 4` ints (SM-major scratch).
        let workspace_size = (max_blocks as usize) * ((2 * inter_size).max(hidden_size) / 64);
        let workspace_size = workspace_size.max(132 * 4);

        let c_tmp_fc1_elems = marlin(marlin_moe::c_tmp_elements(
            2 * inter_size,
            max_padded as usize,
            block_size as usize,
            dev_ordinal,
        ))?;
        let c_tmp_fc2_elems = marlin(marlin_moe::c_tmp_elements(
            hidden_size,
            max_padded as usize,
            block_size as usize,
            dev_ordinal,
        ))?;
        let c_tmp_elems = c_tmp_fc1_elems.max(c_tmp_fc2_elems).max(1);

        Ok(Self {
            max_m,
            sorted_token_ids: Tensor::zeros(max_padded as usize, DType::I32, device)?
                .contiguous()?,
            expert_ids: Tensor::zeros(max_blocks as usize, DType::I32, device)?.contiguous()?,
            num_tokens_post_pad: Tensor::zeros(1, DType::I32, device)?.contiguous()?,
            cumsum_buffer: Tensor::zeros(
                ((num_local_experts + 1) as usize).max(64),
                DType::I32,
                device,
            )?
            .contiguous()?,
            workspace: Tensor::zeros(workspace_size, DType::I32, device)?.contiguous()?,
            workspace2: Tensor::zeros(workspace_size, DType::I32, device)?.contiguous()?,
            c_tmp: Tensor::zeros(c_tmp_elems, DType::F32, device)?.contiguous()?,
            fc1_out: Tensor::zeros((max_m * top_k, 2 * inter_size), scale_dtype, device)?
                .contiguous()?,
            act_out: Tensor::zeros((max_m * top_k, inter_size), scale_dtype, device)?
                .contiguous()?,
            fc2_out: Tensor::zeros((max_m * top_k, hidden_size), scale_dtype, device)?
                .contiguous()?,
            out: Tensor::zeros((max_m, hidden_size), scale_dtype, device)?.contiguous()?,
        })
    }

    /// In-place zero of the buffers that must start each forward at zero:
    ///
    /// - `workspace` / `workspace2`: Marlin GEMM lock arrays. The kernel
    ///   atomically increments slots; stale non-zero values would prevent
    ///   tiles from being claimed.
    /// - `num_tokens_post_pad`: atomically incremented counter for
    ///   `moe_align_block_size_kernel`.
    /// - `cumsum_buffer`: scratch for `moe_align_block_size_kernel`'s
    ///   per-expert exclusive prefix sum.
    /// - `fc2_out`: under EP, ~`(world_size-1)/world_size` of the
    ///   `M*top_k` routing slots route to experts on other ranks. The Marlin
    ///   GEMM (correctly) skips those tiles, leaving their rows untouched.
    ///   The final `sum(top_k)` reads every row, so off-rank rows MUST be
    ///   zero or they'd contribute stale data from the previous call. The
    ///   original (per-call `Tensor::zeros`) implementation was implicitly
    ///   zero on every call; the hoisted-scratch path has to re-zero here.
    ///
    /// Combined size for Kimi K2.5 bs=1 decode is ≈ 16 KiB; for bs=128
    /// prefill ≈ 14 MiB. The latter is dominated by `fc2_out` and is on
    /// the order of 10 µs per layer per token on H200.
    ///
    /// `m` / `top_k` describe the *active* region of the forward call —
    /// because the scratch is sized for `max_m`, zeroing the full
    /// `fc2_out` on every decode step would scale memset bandwidth with
    /// `max_m` rather than the actual batch (60 layers × 469 MiB of D2D
    /// memset per step at `max_m=4096`). We instead narrow to the rows
    /// the GEMM actually writes; rows in `[m*top_k..max_m*top_k]` are
    /// never read by the downstream `moe_sum_reduce` (which only walks
    /// `m * top_k`) so their stale contents are irrelevant.
    fn reset_locks(
        &self,
        m: usize,
        top_k: usize,
        scale_dtype: DType,
        stream: &candle::cuda::cudarc::driver::CudaStream,
    ) -> Result<()> {
        zero_buffer(&self.workspace, DType::I32, stream)?;
        zero_buffer(&self.workspace2, DType::I32, stream)?;
        zero_buffer(&self.num_tokens_post_pad, DType::I32, stream)?;
        zero_buffer(&self.cumsum_buffer, DType::I32, stream)?;
        let active_rows = m * top_k;
        if active_rows > 0 {
            let fc2_active = self.fc2_out.narrow(0, 0, active_rows)?;
            zero_buffer(&fc2_active, scale_dtype, stream)?;
        }
        Ok(())
    }
}

/// In-place stream-ordered zero of `t`'s entire backing storage via
/// `cuMemsetD8Async`. Cheaper than `Tensor::zeros` (no alloc) and graph-safe
/// (memset is a recordable graph node).
fn zero_buffer(
    t: &Tensor,
    dtype: DType,
    stream: &candle::cuda::cudarc::driver::CudaStream,
) -> Result<()> {
    let (s, l) = t.storage_and_layout();
    let ptr = cuda_any(&s, l.start_offset(), dtype, stream)?;
    let num_bytes = t.elem_count() * dtype.size_in_bytes();
    let res = unsafe {
        cuMemsetD8Async(
            ptr.address() as candle::cuda::cudarc::driver::sys::CUdeviceptr,
            0,
            num_bytes,
            stream.cu_stream() as candle::cuda::cudarc::driver::sys::CUstream,
        )
    };
    if res != CUresult::CUDA_SUCCESS {
        bail!("Marlin: cuMemsetD8Async failed: {:?}", res);
    }
    Ok(())
}

// Process-wide Marlin scratch pool. Every MoE layer in a model has identical
// `inter_size`, `hidden_size`, `num_local_experts`, `top_k`, and `group_size`,
// and the 60+ MoE layers in Kimi K2.5 execute strictly sequentially per rank.
// One pool can therefore be shared across all layers and we only need to
// keep one entry per `(top_k, inter_size, hidden_size, num_local_experts,
// scale_dtype)`. Different geometries (e.g. two models in one process) get
// separate cache lines.
//
// The pool no longer keys on `M`. An earlier revision did, with no eviction:
// long-running prefill workloads accumulated one ~1 GiB scratch entry per
// unique `M` and rank 0 eventually OOM'd on
// `MarlinScratch::alloc` mid-forward — which silently abandoned the
// model's NCCL collectives and deadlocked the cluster. Now `max_m` is
// configured once at load time via `MarlinInt4Moe::prealloc_scratch` and
// each forward narrows into the pinned worst-case buffers.
//
// Memory: one scratch is ~`max_m * (top_k*hidden + top_k*2*inter +
// top_k*inter + hidden) + workspace + c_tmp`. For Kimi K2.5 at
// `max_m = 8 * 512 = 4096`, that's roughly 1 GiB BF/FP, shared across all
// 60 layers.
thread_local! {
    static MARLIN_SCRATCH_POOL: RefCell<HashMap<MarlinScratchKey, MarlinScratch>>
        = RefCell::new(HashMap::new());
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct MarlinScratchKey {
    top_k: usize,
    num_local_experts: usize,
    inter_size: usize,
    hidden_size: usize,
    scale_dtype_id: u8,
}

fn dtype_id(d: DType) -> u8 {
    match d {
        DType::BF16 => 1,
        DType::F16 => 2,
        DType::F32 => 3,
        DType::I32 => 4,
        DType::U32 => 5,
        _ => 0,
    }
}

/// Preprocessed Marlin MoE handle. Holds the tile-interleaved weights,
/// permuted scales, persistent workspace buffer, and shape metadata.
#[derive(Debug)]
pub struct MarlinInt4Moe {
    /// `[E, K / 16, 4 * inter]` INT32, Marlin tile-interleaved fc1 weights.
    fc1_qweight: Tensor,
    /// `[E, K / group_size, 2 * inter]` BF16/FP16, permuted scales for fc1.
    fc1_scales: Tensor,
    /// `[E, inter / 16, 2 * K]` INT32, Marlin tile-interleaved fc2 weights.
    fc2_qweight: Tensor,
    /// `[E, inter / group_size, K]` BF16/FP16, permuted scales for fc2.
    fc2_scales: Tensor,
    num_experts: usize,
    hidden_size: usize,
    inter_size: usize,
    group_size: usize,
    scale_dtype: DType,
}

impl MarlinInt4Moe {
    /// Returns the BF16 or FP16 dtype required for `xs` and used by the output.
    pub fn activation_dtype(&self) -> DType {
        self.scale_dtype
    }

    /// Build a Marlin handle from Flashinfer-style packed weights and scales:
    ///
    /// * `fc1_packed`: `[E, 2 * inter, K / 8]` INT32   (`[up | gate]` cat along axis 1)
    /// * `fc1_scales`: `[E, 2 * inter, K / group_size]` BF16/FP16
    /// * `fc2_packed`: `[E, K, inter / 8]` INT32
    /// * `fc2_scales`: `[E, K, inter / group_size]` BF16/FP16
    pub fn from_packed_weights(
        fc1_packed: &Tensor,
        fc1_scales: &Tensor,
        fc2_packed: &Tensor,
        fc2_scales: &Tensor,
        group_size: usize,
    ) -> Result<Self> {
        let (e1, two_inter, k_div_8) = fc1_packed.dims3()?;
        let (e2, k, inter_div_8) = fc2_packed.dims3()?;
        if e1 != e2 {
            bail!("Marlin: fc1/fc2 expert count mismatch ({} vs {})", e1, e2);
        }
        if two_inter % 2 != 0 {
            bail!("Marlin: fc1 out_features {} is not even", two_inter);
        }
        let inter = two_inter / 2;
        if inter_div_8 * 8 != inter {
            bail!(
                "Marlin: fc2 in_features ({} * 8 = {}) != inter ({})",
                inter_div_8,
                inter_div_8 * 8,
                inter
            );
        }
        if k_div_8 * 8 != k {
            bail!(
                "Marlin: fc1 in_features ({} * 8 = {}) != K ({})",
                k_div_8,
                k_div_8 * 8,
                k
            );
        }
        if k % group_size != 0 || inter % group_size != 0 {
            bail!(
                "Marlin: group_size {} must divide both K ({}) and inter ({})",
                group_size,
                k,
                inter
            );
        }

        let scale_dtype = fc1_scales.dtype();
        if scale_dtype != fc2_scales.dtype() {
            bail!(
                "Marlin: fc1_scales dtype ({:?}) != fc2_scales dtype ({:?})",
                scale_dtype,
                fc2_scales.dtype()
            );
        }
        if !matches!(scale_dtype, DType::BF16 | DType::F16) {
            bail!("Marlin: scales must be BF16 or F16, got {:?}", scale_dtype);
        }

        // ---- repack each expert's fc1 ----
        // [2*inter, K/8] -> transpose -> [K/8, 2*inter] -> repack -> [K/16, 4*inter]
        let _ = fc1_packed.device().as_cuda_device()?; // dtype/device check
        let mut fc1_marlin = Vec::with_capacity(e1);
        for ex in 0..e1 {
            let w = fc1_packed.i(ex)?;
            let w_t = w.t()?.contiguous()?;
            fc1_marlin.push(marlin_gptq_repack(&w_t)?);
        }
        let fc1_qweight = Tensor::stack(&fc1_marlin, 0)?.contiguous()?;

        // ---- repack each expert's fc2 ----
        // [K, inter/8] -> transpose -> [inter/8, K] -> repack -> [inter/16, 2*K]
        let mut fc2_marlin = Vec::with_capacity(e1);
        for ex in 0..e1 {
            let w = fc2_packed.i(ex)?;
            let w_t = w.t()?.contiguous()?;
            fc2_marlin.push(marlin_gptq_repack(&w_t)?);
        }
        let fc2_qweight = Tensor::stack(&fc2_marlin, 0)?.contiguous()?;

        // ---- transpose + permute scales ----
        // fc1_scales: [E, 2*inter, K/group_size]  -> [E, K/group_size, 2*inter] permuted
        // fc2_scales: [E, K, inter/group_size]    -> [E, inter/group_size, K]   permuted
        let fc1_scales = transpose_and_permute_scales(fc1_scales, e1, two_inter, k / group_size)?;
        let fc2_scales = transpose_and_permute_scales(fc2_scales, e1, k, inter / group_size)?;

        Ok(Self {
            fc1_qweight,
            fc1_scales,
            fc2_qweight,
            fc2_scales,
            num_experts: e1,
            hidden_size: k,
            inter_size: inter,
            group_size,
            scale_dtype,
        })
    }

    /// Pre-populate the process-wide scratch pool with one worst-case
    /// entry for this MoE's geometry. Call once per geometry at model
    /// load (e.g. from `FlashInferMoeBlock::load_int4`) with `max_m`
    /// equal to the largest `M` the dispatcher can ever submit
    /// (decode batch + prefill chunk, including any MTP draft tokens).
    ///
    /// Two reasons to do this:
    /// 1. **Fail-fast OOM**: rank 0 already carries the largest memory
    ///    footprint (sampler state, logits, xgrammar). A ~1 GiB Marlin
    ///    scratch on top can OOM. Allocating up front turns a silent
    ///    mid-run NCCL deadlock (rank 0 errors out, ranks 1..N spin in
    ///    `AllReduce` forever) into a clean startup failure that
    ///    surfaces in launcher logs.
    /// 2. **Bounded steady-state memory**: the runtime path will refuse
    ///    to grow the cached `max_m` after preallocation in practice
    ///    (any `m <= max_m` reuses the pinned entry; `m > max_m` logs
    ///    a warning and grows in-place rather than accumulating
    ///    additional pool entries per unique M).
    ///
    /// Calling this more than once for the same geometry with a smaller
    /// `max_m` is a no-op; calling it with a larger `max_m` grows the
    /// entry.
    pub fn prealloc_scratch(&self, max_m: usize, top_k: usize, device: &Device) -> Result<()> {
        let cuda_dev = match device {
            Device::Cuda(_) => device.as_cuda_device()?,
            _ => bail!("MarlinInt4Moe::prealloc_scratch: device must be CUDA"),
        };
        let dev_ordinal = cuda_dev.cuda_stream().context().ordinal() as i32;
        let key = MarlinScratchKey {
            top_k,
            num_local_experts: self.num_experts,
            inter_size: self.inter_size,
            hidden_size: self.hidden_size,
            scale_dtype_id: dtype_id(self.scale_dtype),
        };
        MARLIN_SCRATCH_POOL.with(|pool| -> Result<()> {
            let mut pool = pool.borrow_mut();
            let needs_alloc = match pool.get(&key) {
                Some(existing) => max_m > existing.max_m,
                None => true,
            };
            if needs_alloc {
                // Drop old before alloc to avoid transient 2× peak.
                pool.remove(&key);
                let s = MarlinScratch::alloc(
                    max_m,
                    top_k,
                    MARLIN_MOE_BLOCK_SIZE,
                    self.num_experts as i32,
                    self.inter_size,
                    self.hidden_size,
                    self.scale_dtype,
                    dev_ordinal,
                    device,
                )?;
                pool.insert(key, s);
                tracing::info!(
                    "MarlinInt4Moe: preallocated scratch max_m={} \
                     (top_k={}, num_local_experts={}, inter={}, hidden={}, \
                      dtype={:?})",
                    max_m,
                    top_k,
                    self.num_experts,
                    self.inter_size,
                    self.hidden_size,
                    self.scale_dtype,
                );
            }
            Ok(())
        })
    }

    /// Forward pass: hidden states + top-k routing → MoE output.
    ///
    /// `xs`: `[M, K]` BF16/FP16
    /// `topk_ids`: `[M, top_k]` I32 or U32. Values in `[0, expert_id_offset)`
    /// or `[expert_id_offset + num_local_experts, ∞)` are silently treated
    /// as "expert lives on a different EP rank" and produce zero contribution
    /// from this rank — the caller is expected to all-reduce across ranks
    /// after the MoE block.
    /// `topk_weights`: `[M, top_k]` F32 routing weights
    /// `expert_id_offset`: subtracted from each `topk_ids[i]` to rebase global
    /// expert numbering to this rank's local range
    /// `[0, num_local_experts)`. For a single-rank model pass `0`; for EP
    /// pass `ep_rank * (num_global_experts / ep_size)`.
    ///
    /// Returns `[M, K]` summed over top_k (this rank's partial contribution).
    pub fn forward(
        &self,
        xs: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        expert_id_offset: i32,
    ) -> Result<Tensor> {
        let (m, k) = xs.dims2()?;
        if k != self.hidden_size {
            bail!(
                "MarlinInt4Moe.forward: xs K dim {} != hidden_size {}",
                k,
                self.hidden_size
            );
        }
        let (m_tk, top_k) = topk_ids.dims2()?;
        if m_tk != m {
            bail!("MarlinInt4Moe.forward: topk_ids rows {} != M {}", m_tk, m);
        }
        if xs.dtype() != self.scale_dtype {
            bail!(
                "MarlinInt4Moe.forward: xs dtype {:?} must match scale dtype {:?}",
                xs.dtype(),
                self.scale_dtype
            );
        }

        let cuda_dev = match xs.device() {
            Device::Cuda(_) => xs.device().as_cuda_device()?,
            _ => bail!("MarlinInt4Moe.forward: xs must be on CUDA"),
        };
        let stream = cuda_dev.cuda_stream();
        let dev_ordinal = stream.context().ordinal() as i32;

        // ---- 1. moe_align_block_size ----
        let block_size = MARLIN_MOE_BLOCK_SIZE;
        // Marlin reads expert IDs as int32. Expert IDs are non-negative so a
        // U32 tensor has identical bit layout — accept either dtype and pass
        // the raw pointer through. Avoids depending on candle having a
        // CUDA `cast_u32_i32` kernel.
        let topk_ids_in = topk_ids.contiguous()?;
        let topk_ids_dtype = topk_ids_in.dtype();
        if !matches!(topk_ids_dtype, DType::I32 | DType::U32) {
            bail!(
                "MarlinInt4Moe.forward: topk_ids dtype must be I32 or U32, got {:?}",
                topk_ids_dtype
            );
        }
        let numel = (m * top_k) as i64;
        let num_local_experts = self.num_experts as i32;
        // Worst-case padded total. Must match `MarlinScratch::alloc`.
        let max_padded = (numel as i32) + num_local_experts * (block_size - 1);
        let max_padded = ((max_padded + block_size - 1) / block_size) * block_size;

        // ---- 2. Get-or-allocate scratch buffers from the shared pool ----
        // Lazily grows the pool the first time we see a given key. Decode at
        // fixed batch hits a single entry permanently. Cloning the
        // `MarlinScratch` is cheap (Arc bumps) and lets us drop the `RefCell`
        // borrow before doing any of the kernel launches below.
        let scratch_key = MarlinScratchKey {
            top_k,
            num_local_experts: self.num_experts,
            inter_size: self.inter_size,
            hidden_size: self.hidden_size,
            scale_dtype_id: dtype_id(self.scale_dtype),
        };
        // The scratch is pinned to a single worst-case `max_m` entry per
        // geometry (see `MarlinInt4Moe::prealloc_scratch`, called once at
        // model load). On a cold pool (e.g. unit test that bypasses
        // prealloc) we fall back to allocating for the current `m`; if a
        // forward later arrives with a larger `m` we grow the entry
        // in-place and log a warning, so memory ceiling moves to the new
        // worst case rather than accumulating siblings.
        let scratch: MarlinScratch = MARLIN_SCRATCH_POOL.with(|pool| -> Result<_> {
            let mut pool = pool.borrow_mut();
            let needs_alloc = match pool.get(&scratch_key) {
                Some(existing) => m > existing.max_m,
                None => true,
            };
            if needs_alloc {
                let prev = pool.get(&scratch_key).map(|s| s.max_m).unwrap_or(0);
                if prev > 0 {
                    tracing::warn!(
                        "MarlinInt4Moe: growing scratch from max_m={} to {} \
                         (top_k={}, num_local_experts={}, inter={}, hidden={}) — \
                         call MarlinInt4Moe::prealloc_scratch at load time with \
                         the largest expected M to avoid mid-run reallocation",
                        prev,
                        m,
                        top_k,
                        self.num_experts,
                        self.inter_size,
                        self.hidden_size,
                    );
                    // Drop the old entry first so its ~1 GiB freed before
                    // we try to allocate the replacement (avoids
                    // transient peak that would OOM rank 0).
                    pool.remove(&scratch_key);
                }
                let s = MarlinScratch::alloc(
                    m,
                    top_k,
                    block_size,
                    num_local_experts,
                    self.inter_size,
                    self.hidden_size,
                    self.scale_dtype,
                    dev_ordinal,
                    xs.device(),
                )?;
                pool.insert(scratch_key, s);
            }
            Ok(pool
                .get(&scratch_key)
                .expect("scratch just inserted")
                .clone())
        })?;

        // Re-zero the lock / counter buffers and the fc2 output (whose
        // off-rank rows under EP are not written by the GEMM and would
        // otherwise contribute stale values to the `sum(top_k)`). See
        // `MarlinScratch::reset_locks` for the full justification.
        scratch.reset_locks(m, top_k, self.scale_dtype, &stream)?;

        // The scratch is sized for `max_m >= m`. Narrow the M-dependent
        // fields to the active region so kernel pointers + element counts
        // computed from `.storage_and_layout()` match `m` exactly.
        let fc1_out_view = scratch.fc1_out.narrow(0, 0, m * top_k)?;
        let act_out_view = scratch.act_out.narrow(0, 0, m * top_k)?;
        let fc2_out_view = scratch.fc2_out.narrow(0, 0, m * top_k)?;
        let out_view = scratch.out.narrow(0, 0, m)?;

        // Cast top-k weights to F32 for Marlin (no-op if already F32). This
        // is the only per-call allocation left in the hot path; payload is
        // `M * top_k * 4` bytes (32 B at bs=1).
        let topk_weights_f32 = if topk_weights.dtype() == DType::F32 {
            topk_weights.contiguous()?
        } else {
            topk_weights.to_dtype(DType::F32)?.contiguous()?
        };

        // --- moe_align launch ---
        {
            let (topk_s, topk_l) = topk_ids_in.storage_and_layout();
            let (sorted_s, sorted_l) = scratch.sorted_token_ids.storage_and_layout();
            let (eids_s, eids_l) = scratch.expert_ids.storage_and_layout();
            let (count_s, count_l) = scratch.num_tokens_post_pad.storage_and_layout();
            let (cumsum_s, cumsum_l) = scratch.cumsum_buffer.storage_and_layout();
            let topk_ptr = cuda_any(&topk_s, topk_l.start_offset(), topk_ids_dtype, &stream)?;
            let sorted_ptr = cuda_i32(&sorted_s, sorted_l.start_offset(), &stream)?;
            let eids_ptr = cuda_i32(&eids_s, eids_l.start_offset(), &stream)?;
            let count_ptr = cuda_i32(&count_s, count_l.start_offset(), &stream)?;
            let cumsum_ptr = cuda_i32(&cumsum_s, cumsum_l.start_offset(), &stream)?;
            let expert_id_offset = usize::try_from(expert_id_offset)
                .map_err(|_| candle::Error::Msg("expert_id_offset must be non-negative".into()))?;
            marlin(unsafe {
                marlin_moe::align_block_size_i32(&AlignLaunchParams {
                    topk_ids: topk_ptr.as_const::<i32>(),
                    numel: numel as usize,
                    num_experts: num_local_experts as usize,
                    expert_id_offset,
                    block_size: block_size as usize,
                    sorted_token_ids: sorted_ptr.as_mut::<i32>(),
                    expert_ids: eids_ptr.as_mut::<i32>(),
                    num_tokens_post_pad: count_ptr.as_mut::<i32>(),
                    cumsum_buffer: cumsum_ptr.as_mut::<i32>(),
                    max_num_tokens_padded: max_padded as usize,
                    pad_sorted_token_ids: true,
                    stream: stream.cu_stream().cast(),
                })
            })?;
        }

        // --- fc1: xs * fc1 -> [M*top_k, 2*inter] ---
        self.run_marlin_gemm(
            xs,
            &fc1_out_view,
            &scratch.c_tmp,
            &self.fc1_qweight,
            &self.fc1_scales,
            &topk_weights_f32,
            &scratch.sorted_token_ids,
            &scratch.expert_ids,
            &scratch.num_tokens_post_pad,
            &scratch.workspace,
            m as i64,
            (2 * self.inter_size) as i64,
            self.hidden_size as i64,
            top_k as i64,
            /*mul_topk_weights=*/ false,
            block_size as i64,
            dev_ordinal,
            &stream,
        )?;

        // --- silu_and_mul: [M*top_k, 2*inter] -> [M*top_k, inter] ---
        {
            let (in_s, in_l) = fc1_out_view.storage_and_layout();
            let (out_s, out_l) = act_out_view.storage_and_layout();
            let in_ptr = cuda_any(&in_s, in_l.start_offset(), self.scale_dtype, &stream)?;
            let out_ptr = cuda_any(&out_s, out_l.start_offset(), self.scale_dtype, &stream)?;
            marlin(unsafe {
                marlin_moe::silu_and_mul(&SiluLaunchParams {
                    dtype: activation_dtype(self.scale_dtype)?,
                    output: out_ptr.as_mut_void(),
                    input: in_ptr.as_const_void(),
                    m: m * top_k,
                    n: self.inter_size,
                    stream: stream.cu_stream().cast(),
                })
            })?;
        }

        // --- fc2: act * fc2 -> [M*top_k, K] with mul_topk_weights ---
        self.run_marlin_gemm(
            &act_out_view,
            &fc2_out_view,
            &scratch.c_tmp,
            &self.fc2_qweight,
            &self.fc2_scales,
            &topk_weights_f32,
            &scratch.sorted_token_ids,
            &scratch.expert_ids,
            &scratch.num_tokens_post_pad,
            &scratch.workspace2,
            (m * top_k) as i64,
            self.hidden_size as i64,
            self.inter_size as i64,
            /*top_k=*/ 1,
            /*mul_topk_weights=*/ true,
            block_size as i64,
            dev_ordinal,
            &stream,
        )?;

        // --- sum across top_k ---
        // Custom warp-per-token reduction (see `kernels/moe_sum_reduce.cu`).
        // Replaces Candle's generic `fast_sum_bf16`, which used to consume
        // 35–54 % of GPU time in K2 prefill (`nsys` traces in
        // `benchmarks/kimi-k25-int4-marlin/nsys-post-rope-fix/`).
        //
        // The input tensor is `scratch.fc2_out` viewed as
        // `[M, top_k, K]` contiguous, so element strides collapse cleanly
        // to `top_k*K`, `K`, `1` — no `reshape().contiguous()` needed.
        {
            let (in_s, in_l) = fc2_out_view.storage_and_layout();
            let (out_s, out_l) = out_view.storage_and_layout();
            let in_ptr = cuda_any(&in_s, in_l.start_offset(), self.scale_dtype, &stream)?;
            let out_ptr = cuda_any(&out_s, out_l.start_offset(), self.scale_dtype, &stream)?;
            marlin(unsafe {
                marlin_moe::sum_reduce(&SumReduceLaunchParams {
                    dtype: activation_dtype(self.scale_dtype)?,
                    input: in_ptr.as_const_void(),
                    output: out_ptr.as_mut_void(),
                    num_tokens: m,
                    top_k,
                    hidden: self.hidden_size,
                    stride_token: top_k * self.hidden_size,
                    stride_topk: self.hidden_size,
                    out_stride_token: self.hidden_size,
                    scale: 1.0,
                    stream: stream.cu_stream().cast(),
                })
            })?;
        }
        // `out_view` is a contiguous view (narrow along dim 0 of a
        // contiguous tensor) into the pinned `scratch.out`. Returning it
        // is shape-correct (`[m, hidden_size]`) and storage-aliased with
        // the next MoE layer's scratch — safe because the caller's
        // residual add for layer N runs before layer N+1's MoE forward on
        // the same CUDA stream.
        Ok(out_view)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_marlin_gemm(
        &self,
        a: &Tensor,
        c: &Tensor,
        c_tmp: &Tensor,
        b_qweight: &Tensor,
        b_scales: &Tensor,
        topk_weights: &Tensor,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        num_tokens_post_pad: &Tensor,
        workspace: &Tensor,
        size_m: i64,
        size_n: i64,
        size_k: i64,
        top_k: i64,
        mul_topk_weights: bool,
        block_size: i64,
        dev_ordinal: i32,
        stream: &candle::cuda::cudarc::driver::CudaStream,
    ) -> Result<()> {
        let (a_s, a_l) = a.storage_and_layout();
        let (c_s, c_l) = c.storage_and_layout();
        let (ctmp_s, ctmp_l) = c_tmp.storage_and_layout();
        let (bq_s, bq_l) = b_qweight.storage_and_layout();
        let (bs_s, bs_l) = b_scales.storage_and_layout();
        let (tk_s, tk_l) = topk_weights.storage_and_layout();
        let (sids_s, sids_l) = sorted_token_ids.storage_and_layout();
        let (eids_s, eids_l) = expert_ids.storage_and_layout();
        let (cnt_s, cnt_l) = num_tokens_post_pad.storage_and_layout();
        let (ws_s, ws_l) = workspace.storage_and_layout();

        let a_ptr = cuda_any(&a_s, a_l.start_offset(), a.dtype(), stream)?;
        let c_ptr = cuda_any(&c_s, c_l.start_offset(), c.dtype(), stream)?;
        let ctmp_ptr = cuda_f32(&ctmp_s, ctmp_l.start_offset(), stream)?;
        let bq_ptr = cuda_i32(&bq_s, bq_l.start_offset(), stream)?;
        let bs_ptr = cuda_any(&bs_s, bs_l.start_offset(), b_scales.dtype(), stream)?;
        let tk_ptr = cuda_f32(&tk_s, tk_l.start_offset(), stream)?;
        let sids_ptr = cuda_i32(&sids_s, sids_l.start_offset(), stream)?;
        let eids_ptr = cuda_i32(&eids_s, eids_l.start_offset(), stream)?;
        let cnt_ptr = cuda_i32(&cnt_s, cnt_l.start_offset(), stream)?;
        let ws_ptr = cuda_i32(&ws_s, ws_l.start_offset(), stream)?;

        marlin(unsafe {
            marlin_moe::wna16_gemm(&GemmLaunchParams {
                dims: GemmDims {
                    m: usize::try_from(size_m).map_err(|_| {
                        candle::Error::Msg("Marlin GEMM M must be non-negative".into())
                    })?,
                    n: usize::try_from(size_n).map_err(|_| {
                        candle::Error::Msg("Marlin GEMM N must be non-negative".into())
                    })?,
                    k: usize::try_from(size_k).map_err(|_| {
                        candle::Error::Msg("Marlin GEMM K must be non-negative".into())
                    })?,
                    num_experts: self.num_experts,
                    top_k: usize::try_from(top_k).map_err(|_| {
                        candle::Error::Msg("Marlin GEMM top_k must be non-negative".into())
                    })?,
                    block_size: usize::try_from(block_size).map_err(|_| {
                        candle::Error::Msg("Marlin GEMM block size must be non-negative".into())
                    })?,
                },
                quantization: Quantization {
                    activation: activation_dtype(self.scale_dtype)?,
                    weight: WeightDtype::U4B8,
                    group_size: self.group_size,
                },
                a: a_ptr.as_const_void(),
                c: c_ptr.as_mut_void(),
                c_tmp: ctmp_ptr.as_mut::<f32>(),
                b_q_weight: bq_ptr.as_const::<u32>(),
                b_scales: bs_ptr.as_const_void(),
                b_zeros: core::ptr::null(),
                topk_weights: tk_ptr.as_const::<f32>(),
                sorted_token_ids: sids_ptr.as_const::<i32>(),
                expert_ids: eids_ptr.as_const::<i32>(),
                num_tokens_post_padded: cnt_ptr.as_const::<i32>(),
                workspace: ws_ptr.as_mut::<i32>(),
                mul_topk_weights,
                is_k_full: true,
                device: dev_ordinal,
                stream: stream.cu_stream().cast(),
            })
        })
    }
}

/// Transpose a `[E, N, num_groups]` scale tensor to `[E, num_groups, N]` and
/// apply Marlin's `get_scale_perms` 8×8-block permutation along the last axis.
fn transpose_and_permute_scales(
    scales: &Tensor,
    num_experts: usize,
    n_dim: usize,
    num_groups: usize,
) -> Result<Tensor> {
    let scales_t = scales.transpose(1, 2)?.contiguous()?;
    let scale_dtype = scales.dtype();
    let permuted = match scale_dtype {
        DType::BF16 => {
            let raw: Vec<half::bf16> = scales_t.flatten_all()?.to_vec1()?;
            let out = permute_marlin_scales(&raw, num_experts, num_groups, n_dim);
            Tensor::from_vec(out, (num_experts, num_groups, n_dim), scales.device())?
        }
        DType::F16 => {
            let raw: Vec<half::f16> = scales_t.flatten_all()?.to_vec1()?;
            let out = permute_marlin_scales(&raw, num_experts, num_groups, n_dim);
            Tensor::from_vec(out, (num_experts, num_groups, n_dim), scales.device())?
        }
        _ => bail!(
            "transpose_and_permute_scales: unsupported dtype {:?}",
            scale_dtype
        ),
    };
    permuted.contiguous()
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn scale_permutation_transposes_each_eight_by_eight_block() {
        let input: Vec<half::bf16> = (0..64)
            .map(|value| half::bf16::from_f32(value as f32))
            .collect();
        let output = permute_marlin_scales(&input, 1, 1, 64);
        for row in 0..8 {
            for column in 0..8 {
                assert_eq!(
                    output[row * 8 + column],
                    input[column * 8 + row],
                    "permutation mismatch at ({row}, {column})"
                );
            }
        }
    }

    #[test]
    fn packed_weight_validation_rejects_mismatched_experts_before_cuda() -> Result<()> {
        let fc1 = Tensor::zeros((2, 128, 16), DType::I32, &Device::Cpu)?;
        let fc2 = Tensor::zeros((1, 128, 8), DType::I32, &Device::Cpu)?;
        let fc1_scales = Tensor::zeros((2, 128, 4), DType::BF16, &Device::Cpu)?;
        let fc2_scales = Tensor::zeros((1, 128, 2), DType::BF16, &Device::Cpu)?;
        assert!(
            MarlinInt4Moe::from_packed_weights(&fc1, &fc1_scales, &fc2, &fc2_scales, 32,).is_err()
        );
        Ok(())
    }
}

#[cfg(test)]
mod sum_reduce_tests {
    //! Numerical correctness of the Marlin BF16 sum reduction vs. Candle's
    //! generic `.sum(1)` on a `[M, top_k, K]` BF16 tensor. Covers all three
    //! dispatcher branches:
    //!
    //!   * `M = 512, top_k = 8, K = 2048` → vectorized BF16 fast path
    //!   * `M = 200, top_k = 8, K = 2049` → scalar warp-per-token (unaligned K)
    //!   * `M = 8,   top_k = 8, K = 2048` → block-per-token (small M)
    //!
    //! Tolerance is set to `2 ** -7 * topk` absolute = ~0.0625 for top_k=8 to
    //! account for BF16's 7-bit mantissa and the different summation orders
    //! between the kernel (FP32 accumulator, per-element) and Candle
    //! (whatever reduction tree `fast_sum_bf16` uses).
    use super::*;
    use candle::Device;
    use core::ffi::c_void;

    fn run_one(m: usize, top_k: usize, hidden: usize, label: &str) -> Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                println!("[{label}] no CUDA device, skipping");
                return Ok(());
            }
        };

        // Deterministic-but-non-trivial input: small magnitudes so the
        // FP32 sum doesn't itself round when cast back to BF16.
        let x = Tensor::randn(0f32, 0.25f32, (m, top_k, hidden), &dev)?
            .to_dtype(DType::BF16)?
            .contiguous()?;

        let reference = x.sum(1)?.contiguous()?.to_dtype(DType::F32)?;
        let out = Tensor::zeros((m, hidden), DType::BF16, &dev)?.contiguous()?;

        let cuda_dev = x.device().as_cuda_device()?;
        let stream = cuda_dev.cuda_stream();
        let (in_s, in_l) = x.storage_and_layout();
        let (out_s, out_l) = out.storage_and_layout();
        let in_ptr = cuda_any(&in_s, in_l.start_offset(), DType::BF16, &stream)?;
        let out_ptr = cuda_any(&out_s, out_l.start_offset(), DType::BF16, &stream)?;

        marlin(unsafe {
            marlin_moe::sum_reduce(&SumReduceLaunchParams {
                dtype: ActivationDtype::Bf16,
                input: in_ptr.as_const_void(),
                output: out_ptr.as_mut_void(),
                num_tokens: m,
                top_k,
                hidden,
                stride_token: top_k * hidden,
                stride_topk: hidden,
                out_stride_token: hidden,
                scale: 1.0,
                stream: stream.cu_stream().cast(),
            })
        })?;

        // Force the launch to complete before we read back.
        marlin(unsafe { marlin_moe::sync_check(stream.cu_stream().cast()) })?;

        let got = out.to_dtype(DType::F32)?;
        let diff = (&got - &reference)?.abs()?;
        let max_abs: f32 = diff.flatten_all()?.max(0)?.to_scalar()?;
        // top_k summands × BF16 unit-roundoff (~1/128) × |x|_max (~0.25 * a few sigma)
        // gives a comfortable upper bound of 0.0625. We see ~0.01 in practice.
        let tol = 0.0625f32;
        assert!(
            max_abs < tol,
            "[{label}] max abs diff {max_abs} >= tol {tol}"
        );
        println!("[{label}] max abs diff = {max_abs}");
        Ok(())
    }

    #[test]
    fn moe_sum_reduce_bf16_vectorized_path() -> Result<()> {
        run_one(512, 8, 2048, "vectorized")
    }

    #[test]
    fn moe_sum_reduce_bf16_scalar_warp_path() -> Result<()> {
        // hidden = 2049 is intentionally not a multiple of 8 — forces the
        // vectorized fast-path off and exercises `warp_per_token_scalar`.
        run_one(200, 8, 2049, "scalar_warp")
    }

    #[test]
    fn moe_sum_reduce_bf16_block_per_token_path() -> Result<()> {
        // Small M kicks the dispatcher down to `block_per_token`.
        run_one(8, 8, 2048, "block_per_token")
    }

    #[test]
    #[ignore = "requires a visible SM90 CUDA GPU"]
    fn sm90_gptq_repack_smoke() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let packed = Tensor::zeros((8, 64), DType::I32, &device)?;
        let repacked = marlin_gptq_repack(&packed)?;
        assert_eq!(repacked.dims2()?, (4, 128));
        device.synchronize()?;
        Ok(())
    }

    fn synthetic_forward(dtype: DType) -> Result<()> {
        let device = Device::new_cuda(0)?;
        let fc1 = Tensor::zeros((1, 128, 8), DType::I32, &device)?;
        let fc2 = Tensor::zeros((1, 64, 8), DType::I32, &device)?;
        let fc1_scales = Tensor::ones((1, 128, 2), dtype, &device)?;
        let fc2_scales = Tensor::ones((1, 64, 2), dtype, &device)?;
        let moe = MarlinInt4Moe::from_packed_weights(&fc1, &fc1_scales, &fc2, &fc2_scales, 32)?;
        moe.prealloc_scratch(1, 1, &device)?;
        let xs = Tensor::zeros((1, 64), dtype, &device)?;
        let ids = Tensor::zeros((1, 1), DType::I32, &device)?;
        let weights = Tensor::ones((1, 1), DType::F32, &device)?;
        let output = moe.forward(&xs, &ids, &weights, 0)?;
        assert_eq!(output.dims2()?, (1, 64));
        assert_eq!(output.dtype(), dtype);
        device.synchronize()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a visible SM90 CUDA GPU"]
    fn sm90_bf16_forward_smoke() -> Result<()> {
        synthetic_forward(DType::BF16)
    }

    #[test]
    #[ignore = "requires a visible SM90 CUDA GPU"]
    fn sm90_fp16_forward_smoke() -> Result<()> {
        synthetic_forward(DType::F16)
    }
}

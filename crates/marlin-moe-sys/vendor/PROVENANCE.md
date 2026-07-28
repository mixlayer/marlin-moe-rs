# Vendored source provenance

This directory is a source snapshot, not a vLLM checkout or submodule.

## Sources

- Marlin core and GPTQ repack implementation: IST-DASLab/Marlin, subsequently adapted by vLLM.
- MoE WNA16 dispatch and generated SM80 instantiations: vLLM's
  `csrc/moe/marlin_moe_wna16` implementation.
- Expert alignment and gated-SiLU helpers: adapted from SGLang/sgl-kernel.
- MoE top-k sum reduction: adapted from SGLang's `csrc/moe/moe_sum_reduce.cu`.

The snapshot was extracted from the Mixlayer `modeld-kernels` tree on branch
`moe_refactor_2`. Existing source comments retain recoverable upstream URLs and revisions,
including vLLM revision `6dffa4b0a6120159ef2fe44d695a46817aff65bc` where recorded.

The SM120/SM121 FP8 MoE dispatch, generated instantiations, and FP8 pack-transpose operation were
ported from Mixlayer branch `ahadnagy/qwen-sm120` at revision
`95fcc5e906e10361f4b156daf175cddbb1173e26`.

## Mixlayer modifications

- Removed PyTorch, libtorch, registration, and tensor-wrapper dependencies.
- Added bare-pointer CUDA stream APIs.
- Added expert-parallel global-to-local expert offsets.
- Added reusable FP32 reduction scratch sizing and W4A16 BF16/FP16 entry points.
- Added SM120/SM121 BF16-activation, E4M3-weight Marlin dispatch plus guarded FP8 packing and
  transposition for 128-by-128 block-scaled checkpoints.
- Added the optimized top-k sum reduction.
- Wrapped every exported operation in `marlin_moe_status_t`, translated C++ exceptions and CUDA
  launch errors, and exposed a thread-local diagnostic through `marlin_moe_last_error`.

All upstream components listed above are distributed under Apache-2.0. See the repository
`LICENSE` and `NOTICE`.

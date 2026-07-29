# marlin-moe-rs

Rust bindings for the Marlin W4/W8A16 mixture-of-experts kernels used by Mixlayer.

- `marlin-moe-sys` builds a documented snapshot of adapted vLLM, Marlin, and SGLang CUDA sources
  and exposes a guarded raw C ABI.
- `marlin-moe` validates architecture, dimensions, quantization, workspaces, and raw launch
  parameters without depending on Candle.
- `candle-marlin-moe` integrates the safe launcher with CUDA Candle tensors and exposes
  `MarlinInt4Moe` and `MarlinFp8Moe`.

## Kernel support

| Compute capability | INT4 weights, BF16/FP16 activations | E4M3 FP8 weights, BF16 activations |
| --- | --- | --- |
| SM80, SM86, SM89, SM90 | Yes | No |
| SM120, SM121 | Yes | Yes |

The INT4 path supports symmetric and asymmetric weights with group sizes 32, 64, or 128. The FP8
path supports 128-by-128 block-scaled checkpoints and requires BF16 activations and outputs.
`MarlinFp8Moe` accepts E4M3 expert weights plus F32 block scales and performs the required Marlin
repacking and scale expansion.

Unsupported selected architectures compile ABI-compatible Rust stubs that return
`UnsupportedArch`, allowing dependency graphs to build without native kernels for that target.

## Building

SM90 is the default build target. Select one target with `CUDA_COMPUTE_CAP`:

```sh
CUDA_COMPUTE_CAP=120 cargo build --workspace
```

Select multiple targets with `MARLIN_MOE_ARCHS`:

```sh
MARLIN_MOE_ARCHS=80,86,89,90 cargo build --workspace
MARLIN_MOE_ARCHS=120,121 cargo build --workspace
```

SM120/SM121 cannot currently be combined with SM80-SM90 in one archive because FP8 kernel
selection is architecture-gated. Build separate archives for legacy and SM12x targets.

CUDA discovery uses `CUDA_HOME`, `CUDA_ROOT`, or `CUDA_PATH`, falling back to `nvcc` on `PATH`.
See [`crates/marlin-moe-sys/build.rs`](crates/marlin-moe-sys/build.rs) for the remaining supported
build environment variables.

## Validation

```sh
cargo fmt --all
cargo test -p marlin-moe
CUDA_COMPUTE_CAP=90 cargo test -p marlin-moe-sys
```

# Repository Guidelines

## Scope

These instructions apply to the entire `marlin-moe-rs` workspace.

## Project Shape

- Keep `marlin-moe-sys` limited to the raw C ABI and native CUDA build.
- Keep `marlin-moe` Candle-independent and responsible for validation.
- Keep Candle tensor ownership and pointer extraction in `candle-marlin-moe`.
- Do not add Python, PyTorch, libtorch, pybind11, Candle, or cudarc to `marlin-moe-sys`.

## API and CUDA Safety

- Document every public Rust item, including tensor shapes, dtypes, layouts, devices, streams,
  synchronization, pointer lifetime, and ownership where applicable.
- C ABI entry points return `marlin_moe_status_t`; exceptions and CUDA errors must never cross it.
- Preserve SM80 through SM90 kernel behavior and compile ABI-compatible unsupported-architecture
  stubs elsewhere.

## Validation

- Format with `cargo fmt --all`.
- Run CPU-only safe-layer tests with `cargo test -p marlin-moe`.
- Run native checks with `CUDA_COMPUTE_CAP=90 cargo test -p marlin-moe-sys`.


# marlin-moe-rs

Rust bindings for the Marlin W4A16 mixture-of-experts path used by Mixlayer.

The workspace follows the same separation as `flashmla-rs`:

- `marlin-moe-sys` builds a documented snapshot of adapted vLLM, Marlin, and SGLang CUDA sources
  and exposes a guarded raw C ABI.
- `marlin-moe` validates architecture, dimensions, quantization, workspaces, and raw launch
  parameters without depending on Candle.
- `candle-marlin-moe` integrates the safe launcher with CUDA Candle tensors and exposes
  `MarlinInt4Moe`.

Native kernels support SM80, SM86, SM89, and SM90. Other selected architectures compile Rust
stubs which return `UnsupportedArch`, allowing dependency graphs to build on unsupported hosts.

Set `CUDA_COMPUTE_CAP=90` for a Hopper build. Multi-architecture builds may be selected with
`MARLIN_MOE_ARCHS=80,86,89,90`. See `crates/marlin-moe-sys/build.rs` for supported build
environment variables.


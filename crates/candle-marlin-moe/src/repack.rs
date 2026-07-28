//! Candle GPTQ repack integration.

/// Repack contiguous I32 weights from `[K / 8, N]` into Marlin's tile-interleaved layout.
pub use crate::moe::marlin_gptq_repack;

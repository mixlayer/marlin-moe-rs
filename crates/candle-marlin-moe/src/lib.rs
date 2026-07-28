//! Candle CUDA tensor integration for Marlin W4A16 mixture-of-experts inference.

/// Candle error/result aliases.
pub mod error;
/// Model-facing Marlin MoE handle and implementation.
pub mod moe;
/// Packed-weight repack integration.
pub mod repack;
/// CUDA tensor pointer validation and guard retention.
mod tensor;
/// Reusable scratch workspace integration.
pub mod workspace;

pub use error::Result;
pub use moe::{MARLIN_MOE_BLOCK_SIZE, MarlinInt4Moe, marlin_gptq_repack};

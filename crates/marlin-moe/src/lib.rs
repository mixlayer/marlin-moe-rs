//! Candle-independent validation and safe status conversion for Marlin MoE.

/// Supported GPU architectures.
pub mod arch;
/// Validated dimensions and quantization configuration.
pub mod config;
/// Error and result types.
pub mod error;
/// Raw asynchronous launch interfaces.
pub mod launch;
/// Reusable workspace sizing.
pub mod workspace;

pub use arch::Arch;
pub use config::{ActivationDtype, GemmDims, Quantization, RepackDims, WeightDtype};
pub use error::{Error, Result};
pub use launch::{
    AlignLaunchParams, GemmLaunchParams, RepackLaunchParams, ScalarType, SiluLaunchParams,
    SumReduceLaunchParams, align_block_size_i32, gptq_repack, scalar_type_id, silu_and_mul,
    sum_reduce, sync_check, wna16_gemm,
};
pub use workspace::{ScratchLayout, c_tmp_elements};

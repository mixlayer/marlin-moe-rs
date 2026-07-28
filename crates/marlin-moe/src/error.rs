use std::fmt;

use marlin_moe_sys::{marlin_moe_last_error, marlin_moe_status_t};

/// Result type returned by safe Marlin MoE operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error returned by Marlin MoE validation or native launch operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A dimension, pointer, dtype, layout, or workspace was invalid.
    InvalidArgument(String),
    /// The build or runtime GPU architecture is unsupported.
    UnsupportedArch(String),
    /// CUDA reported an API or kernel launch failure.
    Cuda(String),
    /// Native Marlin code reported an unexpected internal failure.
    Internal(String),
}

impl Error {
    /// Converts a non-success C ABI status and thread-local diagnostic into a Rust error.
    pub fn from_status(status: marlin_moe_status_t, context: impl Into<String>) -> Self {
        let context = append_last_error(context.into());
        match status {
            marlin_moe_status_t::MARLIN_MOE_STATUS_SUCCESS => {
                Self::Internal(format!("{context}: unexpected success status"))
            }
            marlin_moe_status_t::MARLIN_MOE_STATUS_INVALID_ARGUMENT => {
                Self::InvalidArgument(context)
            }
            marlin_moe_status_t::MARLIN_MOE_STATUS_UNSUPPORTED_ARCH => {
                Self::UnsupportedArch(context)
            }
            marlin_moe_status_t::MARLIN_MOE_STATUS_CUDA_ERROR => Self::Cuda(context),
            marlin_moe_status_t::MARLIN_MOE_STATUS_INTERNAL_ERROR => Self::Internal(context),
        }
    }
}

fn append_last_error(context: String) -> String {
    let message = unsafe {
        let pointer = marlin_moe_last_error();
        (!pointer.is_null())
            .then(|| {
                std::ffi::CStr::from_ptr(pointer)
                    .to_string_lossy()
                    .into_owned()
            })
            .filter(|message| !message.is_empty())
    };
    message.map_or(context.clone(), |message| format!("{context}: {message}"))
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(formatter, "invalid argument: {message}"),
            Self::UnsupportedArch(message) => {
                write!(formatter, "unsupported architecture: {message}")
            }
            Self::Cuda(message) => write!(formatter, "CUDA error: {message}"),
            Self::Internal(message) => write!(formatter, "internal error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

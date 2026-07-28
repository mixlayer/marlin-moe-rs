use crate::{Error, Result};

/// NVIDIA architecture supported by the vendored Marlin W4A16 kernels.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Arch {
    /// Ampere SM80.
    Sm80,
    /// Ampere SM86.
    Sm86,
    /// Ada SM89.
    Sm89,
    /// Hopper SM90.
    Sm90,
}

impl Arch {
    /// Maps a CUDA compute capability to a supported Marlin architecture.
    pub fn from_compute_capability(major: u32, minor: u32) -> Result<Self> {
        match (major, minor) {
            (8, 0) => Ok(Self::Sm80),
            (8, 6) => Ok(Self::Sm86),
            (8, 9) => Ok(Self::Sm89),
            (9, 0) => Ok(Self::Sm90),
            _ => Err(Error::UnsupportedArch(format!(
                "expected SM80, SM86, SM89, or SM90; got SM{major}{minor}"
            ))),
        }
    }

    /// Returns the NVCC `sm_XX` architecture suffix.
    pub fn nvcc_arch(self) -> &'static str {
        match self {
            Self::Sm80 => "sm_80",
            Self::Sm86 => "sm_86",
            Self::Sm89 => "sm_89",
            Self::Sm90 => "sm_90",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_architectures() {
        assert_eq!(Arch::from_compute_capability(8, 0).unwrap(), Arch::Sm80);
        assert_eq!(Arch::from_compute_capability(9, 0).unwrap(), Arch::Sm90);
    }

    #[test]
    fn rejects_unsupported_architectures() {
        assert!(matches!(
            Arch::from_compute_capability(12, 1),
            Err(Error::UnsupportedArch(_))
        ));
    }
}

use crate::{Error, Result};

/// NVIDIA architecture supported by the vendored Marlin kernels.
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
    /// Blackwell SM120.
    Sm120,
    /// Blackwell SM121.
    Sm121,
}

impl Arch {
    /// Maps a CUDA compute capability to a supported Marlin architecture.
    pub fn from_compute_capability(major: u32, minor: u32) -> Result<Self> {
        match (major, minor) {
            (8, 0) => Ok(Self::Sm80),
            (8, 6) => Ok(Self::Sm86),
            (8, 9) => Ok(Self::Sm89),
            (9, 0) => Ok(Self::Sm90),
            (12, 0) => Ok(Self::Sm120),
            (12, 1) => Ok(Self::Sm121),
            _ => Err(Error::UnsupportedArch(format!(
                "expected SM80, SM86, SM89, SM90, SM120, or SM121; got SM{major}{minor}"
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
            Self::Sm120 => "sm_120",
            Self::Sm121 => "sm_121",
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
        assert_eq!(Arch::from_compute_capability(12, 1).unwrap(), Arch::Sm121);
    }

    #[test]
    fn rejects_unsupported_architectures() {
        assert!(matches!(
            Arch::from_compute_capability(10, 0),
            Err(Error::UnsupportedArch(_))
        ));
    }
}

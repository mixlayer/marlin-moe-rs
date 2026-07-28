use crate::{Error, Result};

/// Activation, output, and scale dtype accepted by W4A16 kernels.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ActivationDtype {
    /// IEEE FP16.
    F16,
    /// BFloat16.
    Bf16,
}

/// Packed four-bit weight interpretation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum WeightDtype {
    /// Symmetric INT4 with implicit bias eight.
    U4B8,
    /// Asymmetric unsigned INT4 with explicit zero points.
    U4,
}

/// Validated quantization configuration for a Marlin MoE GEMM.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Quantization {
    /// Activation, output, and scale dtype.
    pub activation: ActivationDtype,
    /// Packed weight interpretation.
    pub weight: WeightDtype,
    /// Number of K elements sharing one scale; must be 32, 64, or 128.
    pub group_size: usize,
}

/// Logical dimensions for one routed Marlin W4A16 GEMM.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GemmDims {
    /// Input rows before top-k expansion.
    pub m: usize,
    /// Output features; must be divisible by 64.
    pub n: usize,
    /// Input features; must be divisible by 16 and by the quantization group size.
    pub k: usize,
    /// Number of local experts.
    pub num_experts: usize,
    /// Experts selected per input row.
    pub top_k: usize,
    /// Expert-alignment tile size; 8 or a multiple of 16 through 64.
    pub block_size: usize,
}

impl GemmDims {
    /// Validates Marlin tile divisibility and checked integer conversions.
    pub fn validate(self, quantization: Quantization) -> Result<()> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(Error::InvalidArgument(
                "M, N, and K must be positive".into(),
            ));
        }
        if self.n % 64 != 0 {
            return Err(Error::InvalidArgument(format!(
                "N={} must be divisible by 64",
                self.n
            )));
        }
        if self.k % 16 != 0 {
            return Err(Error::InvalidArgument(format!(
                "K={} must be divisible by 16",
                self.k
            )));
        }
        if !matches!(quantization.group_size, 32 | 64 | 128)
            || self.k % quantization.group_size != 0
        {
            return Err(Error::InvalidArgument(format!(
                "group size {} must be 32, 64, or 128 and divide K={}",
                quantization.group_size, self.k
            )));
        }
        if self.num_experts == 0 || self.top_k == 0 {
            return Err(Error::InvalidArgument(
                "num_experts and top_k must be positive".into(),
            ));
        }
        if self.block_size != 8
            && (self.block_size < 16 || self.block_size > 64 || self.block_size % 16 != 0)
        {
            return Err(Error::InvalidArgument(format!(
                "block size {} must be 8 or a multiple of 16 through 64",
                self.block_size
            )));
        }
        for (name, value) in [
            ("m", self.m),
            ("n", self.n),
            ("k", self.k),
            ("num_experts", self.num_experts),
            ("top_k", self.top_k),
            ("block_size", self.block_size),
        ] {
            i32::try_from(value)
                .map_err(|_| Error::InvalidArgument(format!("{name} exceeds i32")))?;
        }
        Ok(())
    }
}

/// Dimensions for GPTQ-to-Marlin packed-weight repacking.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RepackDims {
    /// Logical K dimension; divisible by 16.
    pub k: usize,
    /// Logical N dimension; divisible by 64.
    pub n: usize,
    /// Packed bits per weight; 4 or 8.
    pub bits: usize,
}

impl RepackDims {
    /// Validates input/output tile geometry and checked integer conversions.
    pub fn validate(self) -> Result<()> {
        if self.k == 0 || self.k % 16 != 0 {
            return Err(Error::InvalidArgument(format!(
                "repack K={} must be non-zero and divisible by 16",
                self.k
            )));
        }
        if self.n == 0 || self.n % 64 != 0 {
            return Err(Error::InvalidArgument(format!(
                "repack N={} must be non-zero and divisible by 64",
                self.n
            )));
        }
        if !matches!(self.bits, 4 | 8) {
            return Err(Error::InvalidArgument(format!(
                "repack bits={} must be 4 or 8",
                self.bits
            )));
        }
        i64::try_from(self.k)
            .and_then(|_| i64::try_from(self.n))
            .map_err(|_| Error::InvalidArgument("repack dimensions exceed i64".into()))?;
        Ok(())
    }

    /// Returns the packed input I32 element count for row-major `[K / pack_factor, N]`.
    pub fn input_elements(self) -> Result<usize> {
        self.validate()?;
        self.k
            .checked_mul(self.n)
            .and_then(|elements| elements.checked_div(32 / self.bits))
            .ok_or_else(|| Error::InvalidArgument("repack input element count overflow".into()))
    }

    /// Returns the Marlin output I32 element count for `[K / 16, N * 16 / pack_factor]`.
    pub fn output_elements(self) -> Result<usize> {
        self.input_elements()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantization() -> Quantization {
        Quantization {
            activation: ActivationDtype::Bf16,
            weight: WeightDtype::U4B8,
            group_size: 32,
        }
    }

    #[test]
    fn validates_expected_w4a16_geometry() {
        GemmDims {
            m: 4,
            n: 2048,
            k: 4096,
            num_experts: 16,
            top_k: 8,
            block_size: 32,
        }
        .validate(quantization())
        .unwrap();
    }

    #[test]
    fn rejects_invalid_group_size_and_tiles() {
        let mut quantization = quantization();
        quantization.group_size = 48;
        assert!(
            GemmDims {
                m: 1,
                n: 65,
                k: 4096,
                num_experts: 1,
                top_k: 1,
                block_size: 32,
            }
            .validate(quantization)
            .is_err()
        );
    }

    #[test]
    fn detects_repack_overflow() {
        assert!(
            RepackDims {
                k: usize::MAX - 15,
                n: 64,
                bits: 4
            }
            .input_elements()
            .is_err()
        );
    }
}

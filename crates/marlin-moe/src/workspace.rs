use crate::{Error, Result};

/// Element counts for a reusable Marlin MoE scratch allocation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ScratchLayout {
    /// Upper bound on routed rows after per-expert block padding.
    pub max_padded: usize,
    /// Upper bound on expert blocks.
    pub max_blocks: usize,
    /// I32 Marlin lock elements needed independently for FC1 and FC2.
    pub lock_elements: usize,
}

impl ScratchLayout {
    /// Calculates checked scratch geometry for `max_tokens * top_k` routed rows.
    pub fn calculate(
        max_tokens: usize,
        top_k: usize,
        block_size: usize,
        num_local_experts: usize,
        max_output_features: usize,
    ) -> Result<Self> {
        if max_tokens == 0
            || top_k == 0
            || block_size == 0
            || num_local_experts == 0
            || max_output_features == 0
        {
            return Err(Error::InvalidArgument(
                "scratch dimensions must be positive".into(),
            ));
        }
        let routed = max_tokens
            .checked_mul(top_k)
            .ok_or_else(|| Error::InvalidArgument("routed row count overflow".into()))?;
        let padding = num_local_experts
            .checked_mul(block_size - 1)
            .ok_or_else(|| Error::InvalidArgument("expert padding overflow".into()))?;
        let unrounded = routed
            .checked_add(padding)
            .ok_or_else(|| Error::InvalidArgument("padded row count overflow".into()))?;
        let max_padded = unrounded
            .checked_add(block_size - 1)
            .and_then(|value| value.checked_div(block_size))
            .and_then(|value| value.checked_mul(block_size))
            .ok_or_else(|| Error::InvalidArgument("padded row rounding overflow".into()))?;
        let max_blocks = max_padded / block_size;
        let lock_elements = max_blocks
            .checked_mul(max_output_features / 64)
            .ok_or_else(|| Error::InvalidArgument("lock workspace overflow".into()))?
            .max(132 * 4);
        Ok(Self {
            max_padded,
            max_blocks,
            lock_elements,
        })
    }
}

/// Queries the device-dependent number of F32 reduction elements needed by a Marlin GEMM.
pub fn c_tmp_elements(
    output_features: usize,
    padded_rows: usize,
    block_size: usize,
    device: i32,
) -> Result<usize> {
    let mut elements = 0_i64;
    let status = unsafe {
        marlin_moe_sys::marlin_moe_wna16_c_tmp_elems(
            i64::try_from(output_features)
                .map_err(|_| Error::InvalidArgument("output features exceed i64".into()))?,
            i64::try_from(padded_rows)
                .map_err(|_| Error::InvalidArgument("padded rows exceed i64".into()))?,
            i64::try_from(block_size)
                .map_err(|_| Error::InvalidArgument("block size exceeds i64".into()))?,
            device,
            &mut elements,
        )
    };
    if status != marlin_moe_sys::marlin_moe_status_t::MARLIN_MOE_STATUS_SUCCESS {
        return Err(Error::from_status(status, "Marlin c_tmp sizing failed"));
    }
    usize::try_from(elements)
        .map_err(|_| Error::Internal(format!("native c_tmp size is invalid: {elements}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_layout_matches_padding_formula() {
        let layout = ScratchLayout::calculate(1, 8, 32, 16, 4096).unwrap();
        assert_eq!(layout.max_padded, 512);
        assert_eq!(layout.max_blocks, 16);
        assert!(layout.lock_elements >= 132 * 4);
    }

    #[test]
    fn scratch_layout_detects_overflow() {
        assert!(ScratchLayout::calculate(usize::MAX, 8, 32, 16, 4096).is_err());
    }
}

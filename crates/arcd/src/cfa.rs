//! Numeric conversion traits for CFA sample types.

/// A single CFA sample value that can be converted to the internal `f32`
/// working precision.
///
/// Implemented for `u8` and `u16` (the two input formats described in the
/// paper's specification: "8- or 16-bit Bayer RGGB input").
pub trait CfaSample: Copy {
    /// Sample value as `f32`, in the raw scale (e.g. `0..=255` or `0..=65535`).
    fn value(self) -> f32;
}

impl CfaSample for u8 {
    #[inline]
    fn value(self) -> f32 {
        f32::from(self)
    }
}

impl CfaSample for u16 {
    #[inline]
    fn value(self) -> f32 {
        f32::from(self)
    }
}

impl CfaSample for i16 {
    #[inline]
    fn value(self) -> f32 {
        f32::from(self)
    }
}

impl CfaSample for f32 {
    #[inline]
    fn value(self) -> f32 {
        self
    }
}

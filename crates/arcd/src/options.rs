//! Configuration for the ARCD algorithm (paper section 4.4).

use crate::error::{ArcdError, Result};
use crate::pattern::BayerPattern;

/// Pre-interpolation method used for flat regions (Stage 1) and as the fallback
/// green estimate used by the ratio statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreInterpolation {
    /// Plain bilinear interpolation. This is the paper's fast fallback and the
    /// default; flat regions receive it directly.
    #[default]
    Bilinear,
    /// Malvar-He-Cutler gradient-corrected 5x5 filters. Slightly more
    /// expensive, usually a better "difference-based" baseline for flat areas.
    Mhc,
}

/// Tunable parameters of ARCD (paper section 4.4, "Adjustable Parameters").
///
/// Defaults follow the paper's table:
///
/// | Parameter       | Field            | Default          | Range (paper)      |
/// |-----------------|------------------|------------------|--------------------|
/// | `tau_flat`      | [`Self::tau_flat`]     | `650.25` (= 0.01 * 255^2) | `[0, 0.1 * 255^2]` |
/// | `tau_anis`      | [`Self::tau_anis`]     | `2.0`            | `[1.5, 5.0]`       |
/// | `lambda_ratio`  | [`Self::lambda_ratio`] | `0.5`            | `[0, 1]`           |
/// | `eta_homog`     | [`Self::eta_l`], [`Self::eta_c`] | `(5.0, 3.0)` | -             |
/// | `rho_median`    | [`Self::rho_median`]   | `0.7`            | `[0, 1]`           |
///
/// `tau_flat` is expressed in *8-bit variance units* (i.e. squared 0..255
/// values). It is rescaled automatically to the configured [`Self::bit_depth`],
/// so the same value is meaningful for both 8-bit and 16-bit input.
#[derive(Debug, Clone, PartialEq)]
pub struct ArcdOptions {
    /// Bayer CFA phase of the input. Default: [`BayerPattern::Rggb`].
    pub pattern: BayerPattern,
    /// Effective bit depth of the samples (8 for 0..=255, 16 for 0..=65535, etc.).
    /// Used to derive `px_max` and to rescale `tau_flat`. Default: `8`.
    pub bit_depth: u8,
    /// Flat-region variance threshold, in 8-bit squared units. Default: `650.25`.
    pub tau_flat: f32,
    /// Edge anisotropy threshold: `max(Gh, Gv) / (min(Gh, Gv) + eps)`. Default: `2.0`.
    pub tau_anis: f32,
    /// Ratio/difference blend for Stage 2 (texture) green: `lambda * ratio-based
    /// estimate + (1 - lambda) * difference-based estimate`. Default: `0.5`.
    pub lambda_ratio: f32,
    /// AHD homogeneity luminance threshold in CIELAB L* units. Default: `5.0`.
    pub eta_l: f32,
    /// AHD homogeneity chrominance threshold in CIELAB (a*, b*) units. Default: `3.0`.
    pub eta_c: f32,
    /// Median-filter strength for the Stage 4 color-difference artifact
    /// suppression pass: `rho * median + (1 - rho) * original`. `0.0` disables
    /// filtering, `1.0` applies the full 3x3 median. Default: `0.7`.
    pub rho_median: f32,
    /// Numerical guard used as `epsilon` in anisotropy ratio and weight
    /// denominators. Default: `1e-4`.
    pub eps: f32,
    /// Pre-interpolation method for Stage 1 / flat regions. Default: [`PreInterpolation::Bilinear`].
    pub pre_interpolation: PreInterpolation,
}

impl ArcdOptions {
    /// Paper defaults for RGGB input.
    pub fn new(pattern: BayerPattern) -> Self {
        Self {
            pattern,
            bit_depth: 8,
            tau_flat: 0.01 * 255.0 * 255.0,
            tau_anis: 2.0,
            lambda_ratio: 0.5,
            eta_l: 5.0,
            eta_c: 3.0,
            rho_median: 0.7,
            eps: 1.0e-4,
            pre_interpolation: PreInterpolation::Bilinear,
        }
    }

    /// Paper defaults for a specific sample bit depth. `tau_flat` is kept in
    /// 8-bit squared units on purpose: the demosaicer rescales it internally
    /// based on `bit_depth`, so the paper's default is always meaningful.
    pub fn for_bit_depth(pattern: BayerPattern, bit_depth: u8) -> Self {
        let mut o = Self::new(pattern);
        o.bit_depth = bit_depth;
        o
    }

    /// Maximum representable sample value: `(1 << bit_depth) - 1`.
    #[inline]
    pub fn px_max(&self) -> f32 {
        ((1u32 << self.bit_depth) - 1) as f32
    }

    /// Ratio between this depth's full scale and the 8-bit full scale.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.px_max() / 255.0
    }

    /// Validate all fields, returning [`ArcdError::InvalidOptions`] when a value
    /// is out of range. Validation is intentionally looser than the paper's
    /// suggested ranges so that experimentation is not blocked.
    pub fn validate(&self) -> Result<()> {
        let fail = |msg: String| Err(ArcdError::InvalidOptions(msg));
        if self.bit_depth < 1 || self.bit_depth > 16 {
            return fail(format!(
                "bit_depth must be in 1..=16, got {}",
                self.bit_depth
            ));
        }
        if !(self.tau_flat.is_finite() && self.tau_flat >= 0.0) {
            return fail(format!("tau_flat must be >= 0, got {}", self.tau_flat));
        }
        if !(self.tau_anis.is_finite() && self.tau_anis >= 1.0) {
            return fail(format!("tau_anis must be >= 1.0, got {}", self.tau_anis));
        }
        if !(0.0..=1.0).contains(&self.lambda_ratio) {
            return fail(format!(
                "lambda_ratio must be in [0, 1], got {}",
                self.lambda_ratio
            ));
        }
        if !(self.eta_l.is_finite() && self.eta_l >= 0.0) {
            return fail(format!("eta_l must be >= 0, got {}", self.eta_l));
        }
        if !(self.eta_c.is_finite() && self.eta_c >= 0.0) {
            return fail(format!("eta_c must be >= 0, got {}", self.eta_c));
        }
        if !(0.0..=1.0).contains(&self.rho_median) {
            return fail(format!(
                "rho_median must be in [0, 1], got {}",
                self.rho_median
            ));
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            return fail(format!("eps must be > 0, got {}", self.eps));
        }
        Ok(())
    }
}

impl Default for ArcdOptions {
    fn default() -> Self {
        Self::new(BayerPattern::default())
    }
}

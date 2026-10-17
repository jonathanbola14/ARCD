//! Bayer CFA pattern definitions.

use crate::error::{ArcdError, Result};

/// The color sampled by a photosite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CfaColor {
    /// Red photosite (25% of sites).
    Red,
    /// Green photosite (50% of sites).
    Green,
    /// Blue photosite (25% of sites).
    Blue,
}

/// Bayer CFA phase (the color sampled at pixel `(0, 0)`), named by reading the
/// first two columns of the first two rows.
///
/// ```text
/// Rggb: R G   Grbg: G R   Gbrg: G B   Bggr: B G
///       G B         B G         R G         G R
/// ```
///
/// ARCD was designed for RGGB (see the paper), but all four phases are supported
/// because every stage of the algorithm only relies on two geometry invariants
/// that hold for *any* Bayer phase:
///
/// * the four axial (distance-1, axis-aligned) neighbors of a red or blue site
///   are always green sites, and
/// * moving a distance of 2 along an axis preserves the site color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BayerPattern {
    /// `R G / G B` — the canonical Bayer pattern assumed by the ARCD paper.
    #[default]
    Rggb,
    /// `G R / B G`
    Grbg,
    /// `G B / R G`
    Gbrg,
    /// `B G / G R`
    Bggr,
}

impl BayerPattern {
    /// Parse a pattern from a string such as `"rggb"` (case-insensitive).
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rggb" => Ok(BayerPattern::Rggb),
            "grbg" => Ok(BayerPattern::Grbg),
            "gbrg" => Ok(BayerPattern::Gbrg),
            "bggr" => Ok(BayerPattern::Bggr),
            other => Err(ArcdError::InvalidOptions(format!(
                "unknown Bayer pattern '{other}' (expected one of: rggb, grbg, gbrg, bggr)"
            ))),
        }
    }

    /// Color sampled at pixel `(x, y)` (x = column, y = row).
    ///
    /// Uses only the parity of the coordinates, so it is well defined for any
    /// integer (negative coordinates wrap correctly, e.g. `(-1) & 1 == 1`).
    #[inline]
    pub fn color_at(self, x: usize, y: usize) -> CfaColor {
        let cx = x & 1;
        let cy = y & 1;
        match self {
            BayerPattern::Rggb => match (cx, cy) {
                (0, 0) => CfaColor::Red,
                (1, 0) => CfaColor::Green,
                (0, 1) => CfaColor::Green,
                _ => CfaColor::Blue,
            },
            BayerPattern::Grbg => match (cx, cy) {
                (0, 0) => CfaColor::Green,
                (1, 0) => CfaColor::Red,
                (0, 1) => CfaColor::Blue,
                _ => CfaColor::Green,
            },
            BayerPattern::Gbrg => match (cx, cy) {
                (0, 0) => CfaColor::Green,
                (1, 0) => CfaColor::Blue,
                (0, 1) => CfaColor::Red,
                _ => CfaColor::Green,
            },
            BayerPattern::Bggr => match (cx, cy) {
                (0, 0) => CfaColor::Blue,
                (1, 0) => CfaColor::Green,
                (0, 1) => CfaColor::Green,
                _ => CfaColor::Red,
            },
        }
    }

    /// All four axial offsets (left, right, up, down) as `(dx, dy)`.
    pub(crate) const AXIAL: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    /// All four diagonal offsets as `(dx, dy)`.
    pub(crate) const DIAGONAL: [(isize, isize); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];
}

//! Stage 1: pre-interpolation (bilinear / MHC) and region classification.
//!
//! Paper section 4.1 (Stage 1) and 4.3 ("Local variance", "Directional
//! gradients").

use crate::buffer::{clamp_i, clamp_px, get};
use crate::options::ArcdOptions;
use crate::pattern::{BayerPattern, CfaColor};

/// Three full-resolution color planes (one `f32` per pixel per channel).
#[derive(Debug, Clone)]
pub struct Planes {
    /// Red plane.
    pub r: Vec<f32>,
    /// Green plane.
    pub g: Vec<f32>,
    /// Blue plane.
    pub b: Vec<f32>,
}

/// Mean of the axial neighbors (distance 1, axis-aligned) whose CFA color
/// matches `want`.
///
/// Neighbor coordinates are clamped *before* the color test so that the color
/// of the sample actually fetched is what decides inclusion; on image borders
/// this means some neighbors alias onto the center site and are skipped.
pub(crate) fn axial_mean(
    raw: &[f32],
    w: usize,
    h: usize,
    pat: BayerPattern,
    x: usize,
    y: usize,
    want: CfaColor,
) -> f32 {
    let mut sum = 0.0;
    let mut cnt = 0.0;
    for (dx, dy) in BayerPattern::AXIAL {
        let jx = clamp_i(x as isize + dx, w);
        let jy = clamp_i(y as isize + dy, h);
        if pat.color_at(jx, jy) == want {
            sum += raw[jy * w + jx];
            cnt += 1.0;
        }
    }
    if cnt > 0.0 {
        sum / cnt
    } else {
        0.0
    }
}

/// Mean of the diagonal neighbors whose CFA color matches `want` (clamped).
pub(crate) fn diagonal_mean(
    raw: &[f32],
    w: usize,
    h: usize,
    pat: BayerPattern,
    x: usize,
    y: usize,
    want: CfaColor,
) -> f32 {
    let mut sum = 0.0;
    let mut cnt = 0.0;
    for (dx, dy) in BayerPattern::DIAGONAL {
        let jx = clamp_i(x as isize + dx, w);
        let jy = clamp_i(y as isize + dy, h);
        if pat.color_at(jx, jy) == want {
            sum += raw[jy * w + jx];
            cnt += 1.0;
        }
    }
    if cnt > 0.0 {
        sum / cnt
    } else {
        0.0
    }
}

/// Mean of the two axial (distance-1) neighbors along one axis whose clamped
/// CFA color is green. Used by Stage 2/3 directional green candidates, where
/// the naive assumption "axials of a quarter-color site are green" breaks at
/// image borders.
pub(crate) fn axial_pair_green(
    raw: &[f32],
    w: usize,
    h: usize,
    pat: BayerPattern,
    x: usize,
    y: usize,
    horizontal: bool,
) -> f32 {
    let dirs: [(isize, isize); 2] = if horizontal { [(-1, 0), (1, 0)] } else { [(0, -1), (0, 1)] };
    let mut sum = 0.0;
    let mut cnt = 0.0;
    for (dx, dy) in dirs {
        let jx = clamp_i(x as isize + dx, w);
        let jy = clamp_i(y as isize + dy, h);
        if pat.color_at(jx, jy) == CfaColor::Green {
            sum += raw[jy * w + jx];
            cnt += 1.0;
        }
    }
    if cnt > 0.0 {
        sum / cnt
    } else {
        raw[y * w + x] // degenerate tiny images: fall back to the raw sample
    }
}

/// Plain bilinear interpolation (paper section 2.1).
///
/// Bayer invariants used here:
/// * green at G sites is the raw sample; at R/B sites it is the mean of the
///   four axial (always green) neighbors;
/// * red at R sites is raw; at G sites the two axial R neighbors; at B sites
///   the four diagonal (always R) neighbors;
/// * blue symmetric to red.
pub fn bilinear(raw: &[f32], w: usize, h: usize, pat: BayerPattern) -> Planes {
    let n = w * h;
    let mut out = Planes {
        r: vec![0.0; n],
        g: vec![0.0; n],
        b: vec![0.0; n],
    };
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            match pat.color_at(x, y) {
                CfaColor::Green => {
                    out.g[i] = raw[i];
                    out.r[i] = axial_mean(raw, w, h, pat, x, y, CfaColor::Red);
                    out.b[i] = axial_mean(raw, w, h, pat, x, y, CfaColor::Blue);
                }
                CfaColor::Red => {
                    out.r[i] = raw[i];
                    out.g[i] = axial_mean(raw, w, h, pat, x, y, CfaColor::Green);
                    out.b[i] = diagonal_mean(raw, w, h, pat, x, y, CfaColor::Blue);
                }
                CfaColor::Blue => {
                    out.b[i] = raw[i];
                    out.g[i] = axial_mean(raw, w, h, pat, x, y, CfaColor::Green);
                    out.r[i] = diagonal_mean(raw, w, h, pat, x, y, CfaColor::Red);
                }
            }
        }
    }
    out
}

/// Malvar-He-Cutler 5x5 kernels (Malvar, He & Cutler, ICASSP 2004, Table 1),
/// expressed directly on the CFA values and pre-normalized (each kernel sums
/// to 1, so constant images are reproduced exactly; this is verified by the
/// `mhc_kernels_sum_to_one` test). The integer forms below appear verbatim in
/// reference implementations (`GR_GB`, `Rg_RB_Bg_BR`, `Rg_BR_Bg_RB`,
/// `Rb_BB_Br_RR`, each divided by 8).
///
/// * `K_G_AT_RB` — green at a red or blue site (alpha = 1/2): axial green
///   bilinear weights (1/4 each) plus a Laplacian correction from the sampled
///   channel (1/2 at the center, -1/8 at distance 2 along both axes).
/// * `K_AT_G_H` / `K_AT_G_V` — red/blue at a green site whose matching-channel
///   axial pair lies on the horizontal (resp. vertical) axis; `K_AT_G_V` is the
///   transpose of `K_AT_G_H`.
/// * `K_DIAG` — red at a blue site or blue at a red site: diagonal bilinear
///   weights (1/4 each) plus a same-channel Laplacian correction (3/4 at the
///   center, -3/16 at axial distance 2).
const K_G_AT_RB: [[f32; 5]; 5] = [
    [0.0, 0.0, -0.125, 0.0, 0.0],
    [0.0, 0.0, 0.25, 0.0, 0.0],
    [-0.125, 0.25, 0.5, 0.25, -0.125],
    [0.0, 0.0, 0.25, 0.0, 0.0],
    [0.0, 0.0, -0.125, 0.0, 0.0],
];

const K_AT_G_H: [[f32; 5]; 5] = [
    [0.0, 0.0, 0.0625, 0.0, 0.0],
    [0.0, -0.125, 0.0, -0.125, 0.0],
    [-0.125, 0.5, 0.625, 0.5, -0.125],
    [0.0, -0.125, 0.0, -0.125, 0.0],
    [0.0, 0.0, 0.0625, 0.0, 0.0],
];

const K_AT_G_V: [[f32; 5]; 5] = [
    [0.0, 0.0, -0.125, 0.0, 0.0],
    [0.0, -0.125, 0.5, -0.125, 0.0],
    [0.0625, 0.0, 0.625, 0.0, 0.0625],
    [0.0, -0.125, 0.5, -0.125, 0.0],
    [0.0, 0.0, -0.125, 0.0, 0.0],
];

const K_DIAG: [[f32; 5]; 5] = [
    [0.0, 0.0, -0.1875, 0.0, 0.0],
    [0.0, 0.25, 0.0, 0.25, 0.0],
    [-0.1875, 0.0, 0.75, 0.0, -0.1875],
    [0.0, 0.25, 0.0, 0.25, 0.0],
    [0.0, 0.0, -0.1875, 0.0, 0.0],
];

/// 5x5 kernel applied to the raw CFA plane with clamped borders.
fn apply5(k: &[[f32; 5]; 5], raw: &[f32], w: usize, h: usize, x: usize, y: usize) -> f32 {
    let xi = x as isize;
    let yi = y as isize;
    let mut acc = 0.0;
    for (ky, row) in k.iter().enumerate() {
        let yy = yi + ky as isize - 2;
        for (kx, &c) in row.iter().enumerate() {
            if c != 0.0 {
                acc += c * get(raw, w, h, xi + kx as isize - 2, yy);
            }
        }
    }
    acc
}

/// Malvar-He-Cutler gradient-corrected interpolation (paper section 2.2).
///
/// The 5x5 kernels are applied only to the interior (`2..w-2`, `2..h-2`);
/// border pixels — where the kernel would have to fetch through clamped
/// borders and would alias neighbor colors — keep the bilinear estimate,
/// which is color-filtered and exact on constant images.
pub fn mhc(raw: &[f32], w: usize, h: usize, pat: BayerPattern, px_max: f32) -> Planes {
    let mut out = bilinear(raw, w, h, pat);
    if w < 5 || h < 5 {
        return out;
    }
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let i = y * w + x;
            let c = pat.color_at(x, y);
            // (target channel, kernel) pairs; the sampled channel itself keeps
            // the raw value.
            let (t0, k0, t1, k1) = match c {
                CfaColor::Green => {
                    // At a green site, the channel carried by the horizontal
                    // axial pair vs the vertical axial pair depends on the
                    // pattern and on the site parity; detect it explicitly.
                    let h_axis = pat.color_at(x + 1, y);
                    (h_axis, &K_AT_G_H, other_quarter(h_axis), &K_AT_G_V)
                }
                CfaColor::Red => (CfaColor::Green, &K_G_AT_RB, CfaColor::Blue, &K_DIAG),
                CfaColor::Blue => (CfaColor::Green, &K_G_AT_RB, CfaColor::Red, &K_DIAG),
            };
            out.r[i] = raw[i];
            out.g[i] = raw[i];
            out.b[i] = raw[i];
            for (target, k) in [(t0, k0), (t1, k1)] {
                let v = clamp_px(apply5(k, raw, w, h, x, y), px_max);
                match target {
                    CfaColor::Red => out.r[i] = v,
                    CfaColor::Green => out.g[i] = v,
                    CfaColor::Blue => out.b[i] = v,
                }
            }
        }
    }
    out
}

/// The other non-green CFA color.
#[inline]
fn other_quarter(c: CfaColor) -> CfaColor {
    match c {
        CfaColor::Red => CfaColor::Blue,
        CfaColor::Blue => CfaColor::Red,
        CfaColor::Green => CfaColor::Green,
    }
}

/// Region classes assigned in Stage 1 (paper section 4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RegionClass {
    /// Smooth region: served directly by the Stage 1 pre-interpolation.
    Flat = 0,
    /// Textured region: Stage 2 (ratio-corrected green).
    Texture = 1,
    /// Edge region: Stage 3 (homogeneity-directed refinement).
    Edge = 2,
}

/// Classification map as raw bytes ([`RegionClass`] values).
pub type ClassMap = Vec<u8>;

/// Stage 1 classification.
///
/// * `sigma2 < tau_flat` -> [`RegionClass::Flat`]
/// * else `max(Gh, Gv) / (min(Gh, Gv) + eps) > tau_anis` -> [`RegionClass::Edge`]
/// * else [`RegionClass::Texture`]
///
/// Returns the classification map and the number of EDGE pixels.
pub fn classify(raw: &[f32], w: usize, h: usize, opts: &ArcdOptions) -> (ClassMap, usize) {
    let n = w * h;
    let mut classes = vec![RegionClass::Flat as u8; n];
    let tau_flat = opts.tau_flat * opts.scale() * opts.scale();
    let mut edge_count = 0usize;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            // Local variance over a 5x5 window of the raw CFA.
            let mut sum = 0.0f64;
            let mut sumsq = 0.0f64;
            for dy in -2..=2isize {
                for dx in -2..=2isize {
                    let v = get(raw, w, h, x as isize + dx, y as isize + dy) as f64;
                    sum += v;
                    sumsq += v * v;
                }
            }
            let mean = sum / 25.0;
            let var = (sumsq / 25.0 - mean * mean).max(0.0);
            if (var as f32) < tau_flat {
                classes[i] = RegionClass::Flat as u8;
                continue;
            }
            // Directional gradients over distance-1 neighbors (paper 4.3).
            let gh = (get(raw, w, h, x as isize - 1, y as isize)
                - get(raw, w, h, x as isize + 1, y as isize))
            .abs();
            let gv = (get(raw, w, h, x as isize, y as isize - 1)
                - get(raw, w, h, x as isize, y as isize + 1))
            .abs();
            let hi = gh.max(gv);
            let lo = gh.min(gv);
            let anis = hi / (lo + opts.eps);
            if anis > opts.tau_anis {
                classes[i] = RegionClass::Edge as u8;
                edge_count += 1;
            } else {
                classes[i] = RegionClass::Texture as u8;
            }
        }
    }
    (classes, edge_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn const_raw(w: usize, h: usize, pat: BayerPattern, rg: f32, gg: f32, b: f32) -> Vec<f32> {
        let mut raw = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = match pat.color_at(x, y) {
                    CfaColor::Red => rg,
                    CfaColor::Green => gg,
                    CfaColor::Blue => b,
                };
                raw[y * w + x] = v;
            }
        }
        raw
    }

    #[test]
    fn mhc_kernels_sum_to_one() {
        // Guarantees constant images are reproduced exactly by MHC.
        for k in [&K_G_AT_RB, &K_AT_G_H, &K_AT_G_V, &K_DIAG] {
            let sum: f32 = k.iter().flat_map(|r| r.iter()).sum();
            assert!((sum - 1.0).abs() < 1e-6, "kernel sums to {sum}");
        }
    }

    #[test]
    fn bilinear_and_mhc_exact_on_constant_image() {
        for pat in [
            BayerPattern::Rggb,
            BayerPattern::Grbg,
            BayerPattern::Gbrg,
            BayerPattern::Bggr,
        ] {
            let (w, h) = (12usize, 9usize);
            let raw = const_raw(w, h, pat, 200.0, 150.0, 100.0);
            for name in ["bilinear", "mhc"] {
                let planes = if name == "bilinear" {
                    bilinear(&raw, w, h, pat)
                } else {
                    mhc(&raw, w, h, pat, 255.0)
                };
                for i in 0..w * h {
                    assert!(
                        (planes.r[i] - 200.0).abs() < 1e-4
                            && (planes.g[i] - 150.0).abs() < 1e-4
                            && (planes.b[i] - 100.0).abs() < 1e-4,
                        "{name} {pat:?} pixel {i}: {:?}",
                        (&planes.r[i], &planes.g[i], &planes.b[i])
                    );
                }
            }
        }
    }

    #[test]
    fn classify_flat_edge_texture() {
        // Left half constant, right half constant, hard vertical edge.
        let (w, h) = (24usize, 16usize);
        let mut raw = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                raw[y * w + x] = if x < 12 { 40.0 } else { 200.0 };
            }
        }
        let opts = ArcdOptions::new(BayerPattern::Rggb);
        let (classes, edges) = classify(&raw, w, h, &opts);
        assert!(edges > 0);
        // Far from the edge, pixels must be flat.
        for y in 2..h - 2 {
            for x in 2..10 {
                assert_eq!(classes[y * w + x], RegionClass::Flat as u8);
            }
        }
        // Near the edge (columns 11/12), pixels must be classified edge or
        // texture, never flat.
        for y in 3..h - 3 {
            for x in 10..14 {
                assert_ne!(classes[y * w + x], RegionClass::Flat as u8);
            }
        }
    }
}

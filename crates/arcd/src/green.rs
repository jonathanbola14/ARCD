//! Stage 2 (ratio-corrected green for texture regions, RCD-inspired) and
//! Stage 3 (homogeneity-directed refinement for edge regions, AHD-inspired).
//!
//! Paper sections 4.1 (Stages 2-3), 4.2 (pseudocode) and 4.3 ("Ratio-corrected
//! green", "Discrimination weights", "Homogeneity map").
//!
//! ## Interpretation notes
//!
//! The paper marks several Stage 2/3 details as "see the RCD source code for
//! exact coefficients". This implementation disambiguates them as follows:
//!
//! * **Measured ratio map.** For every quarter-color site `p` (red or blue),
//!   the measured channel-to-green ratio is `rho(p) = S(p) / G_bil(p)`, where
//!   `G_bil` is the Stage 1 pre-interpolated green. Bayer invariants guarantee
//!   that sites two steps away along either axis have the same color, so the
//!   ratio map is sampled at distance 2.
//! * **Directional ratio-corrected candidates.** `G_h = S / rho_h` and
//!   `G_v = S / rho_v`, with `rho_h = (2*rho_center + rho_left2 + rho_right2) / 4`
//!   (resp. vertical). Note that `S / rho_center` alone would collapse to
//!   `G_bil`; blending the neighboring ratios is exactly what makes the
//!   estimate directional, following the paper's "interpolate ratios, multiply
//!   by the interpolated green" idea.
//! * **Difference-based counterparts.** MHC-style axial Laplacian correction:
//!   `C_h = (G_left + G_right)/2 + (S - (S_left2 + S_right2)/2)/2`, and the
//!   vertical analogue. These are blended with weight `1 - lambda_ratio`,
//!   implementing the paper's "Ratio/difference blend" parameter.
//! * **Discrimination weights.** Following the paper (`w_h = 1/(1 + V_stat)`,
//!   `w_v = 1/(1 + H_stat)`), the statistics are defined so the formulae are
//!   self-consistent: `Dh`/`Dv` are the mean absolute difference of
//!   same-color CFA sample pairs along the horizontal/vertical axis over a
//!   5x5 window; `V_stat = Dh^2 / (Dh^2 + Dv^2 + eps)` grows with *vertical*
//!   edges (which force horizontal interpolation, hence the large `w_h`), and
//!   `H_stat = Dv^2 / (Dh^2 + Dv^2 + eps)` grows with *horizontal* edges
//!   (forcing vertical interpolation, hence the large `w_v`). Squaring
//!   sharpens the discrimination; a linear form is equally valid.

use crate::buffer::{clamp_i, clamp_px, get, neighbor2};
use crate::lab;
use crate::options::ArcdOptions;
use crate::pattern::{BayerPattern, CfaColor};
use crate::stage1::{axial_pair_green, Planes, RegionClass};

/// Mean absolute difference of same-color CFA sample pairs along the
/// horizontal axis inside a 5x5 window centered at (x, y).
fn horizontal_deviation(raw: &[f32], w: usize, h: usize, x: usize, y: usize) -> f32 {
    let mut acc = 0.0;
    for dy in [-2isize, 0] {
        for dx in [-2isize, 0] {
            acc += (get(raw, w, h, x as isize + dx, y as isize + dy)
                - get(raw, w, h, x as isize + dx + 2, y as isize + dy))
            .abs();
        }
    }
    acc / 10.0
}

/// Mean absolute difference of same-color CFA sample pairs along the
/// vertical axis inside a 5x5 window centered at (x, y).
fn vertical_deviation(raw: &[f32], w: usize, h: usize, x: usize, y: usize) -> f32 {
    let mut acc = 0.0;
    for dx in [-2isize, 0] {
        for dy in [-2isize, 0] {
            acc += (get(raw, w, h, x as isize + dx, y as isize + dy)
                - get(raw, w, h, x as isize + dx, y as isize + dy + 2))
            .abs();
        }
    }
    acc / 10.0
}

/// Stage 2: ratio-corrected green interpolation for TEXTURE quarter-color
/// sites. `g_ref` must be pre-initialized (raw green at green sites,
/// pre-interpolated green elsewhere) and is updated in place.
pub fn stage2_texture_green(
    raw: &[f32],
    g_bil: &[f32],
    g_ref: &mut [f32],
    w: usize,
    h: usize,
    opts: &ArcdOptions,
    classes: &[u8],
) {
    let lambda = opts.lambda_ratio;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if classes[i] != RegionClass::Texture as u8 {
                continue;
            }
            if opts.pattern.color_at(x, y) == CfaColor::Green {
                continue; // green sites keep the raw sample
            }
            let s = raw[i];
            // Measured channel/green ratio at this site and at the same-color
            // sites two steps away along each axis (`neighbor2` preserves
            // parity, i.e. the same channel, on both sides; borders fold to
            // the opposite side).
            let ratio = |xx: usize, yy: usize| -> f32 { raw[yy * w + xx] / g_bil[yy * w + xx].max(1.0) };
            let r0 = s / g_bil[i].max(1.0);
            let (lx, rx) = (neighbor2(x, w, true), neighbor2(x, w, false));
            let (uy, dy) = (neighbor2(y, h, true), neighbor2(y, h, false));
            let rho_h = match (lx, rx) {
                (Some(l), Some(r)) => (2.0 * r0 + ratio(l, y) + ratio(r, y)) / 4.0,
                (Some(p), None) | (None, Some(p)) => (r0 + ratio(p, y)) / 2.0,
                (None, None) => r0,
            };
            let rho_v = match (uy, dy) {
                (Some(u), Some(d)) => (2.0 * r0 + ratio(x, u) + ratio(x, d)) / 4.0,
                (Some(p), None) | (None, Some(p)) => (r0 + ratio(x, p)) / 2.0,
                (None, None) => r0,
            };

            // Directional ratio-corrected candidates (RCD-inspired).
            let g_h = s / rho_h.max(1.0e-6);
            let g_v = s / rho_v.max(1.0e-6);

            // Difference-based directional candidates (MHC-style Laplacian
            // correction along each axis). The axial mean is color-filtered
            // so border pixels do not alias neighbor colors; the dist-2
            // Laplacian support uses parity-safe sites only.
            let pair_h = axial_pair_green(raw, w, h, opts.pattern, x, y, true);
            let pair_v = axial_pair_green(raw, w, h, opts.pattern, x, y, false);
            let mean_of = |a: Option<f32>, b: Option<f32>| -> Option<f32> {
                match (a, b) {
                    (Some(p), Some(q)) => Some(0.5 * (p + q)),
                    (Some(p), None) | (None, Some(p)) => Some(p),
                    (None, None) => None,
                }
            };
            let mean_h = mean_of(lx.map(|xx| raw[y * w + xx]), rx.map(|xx| raw[y * w + xx]));
            let mean_v = mean_of(uy.map(|yy| raw[yy * w + x]), dy.map(|yy| raw[yy * w + x]));
            let c_h = pair_h + mean_h.map_or(0.0, |m| 0.5 * (s - m));
            let c_v = pair_v + mean_v.map_or(0.0, |m| 0.5 * (s - m));

            // Discrimination weights (see module docs for the interpretation).
            let dh = horizontal_deviation(raw, w, h, x, y);
            let dv = vertical_deviation(raw, w, h, x, y);
            let denom = dh * dh + dv * dv + opts.eps;
            let v_stat = dh * dh / denom; // large at vertical edges
            let h_stat = dv * dv / denom; // large at horizontal edges
            let w_h = 1.0 / (1.0 + v_stat);
            let w_v = 1.0 / (1.0 + h_stat);
            // Both weights lie in [0.5, 1], so the sum is always >= 1 and the
            // blend needs no epsilon guard (adding one biases the estimate).
            let w_sum = w_h + w_v;

            let g_dir = (w_h * g_h + w_v * g_v) / w_sum;
            let g_diff = (w_h * c_h + w_v * c_v) / w_sum;
            let g_tex = lambda * g_dir + (1.0 - lambda) * g_diff;
            g_ref[i] = clamp_px(g_tex, opts.px_max());
        }
    }
}

/// Full directional RGB candidate image (AHD-style). The direction changes the
/// green plane directly; red/blue follow through color-difference constancy
/// against the directional green, which propagates directionality into every
/// channel.
///
/// Neighbor deltas are collected from both axial and diagonal offsets generically:
/// exactly one family carries the wanted color at each site type (axial pairs
/// at green sites, diagonal quartet at quarter-color sites), so the result is
/// pattern-agnostic.
fn directional_candidates(
    raw: &[f32],
    w: usize,
    h: usize,
    pat: BayerPattern,
    horizontal: bool,
) -> Planes {
    let n = w * h;
    let mut out = Planes {
        r: vec![0.0; n],
        g: vec![0.0; n],
        b: vec![0.0; n],
    };
    // Pass 1: directional green (color-filtered axial pair so borders stay
    // correct).
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            out.g[i] = if pat.color_at(x, y) == CfaColor::Green {
                raw[i]
            } else {
                axial_pair_green(raw, w, h, pat, x, y, horizontal)
            };
        }
    }
    // Pass 2: red/blue via color-difference constancy against the directional
    // green.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let c = pat.color_at(x, y);
            for want in [CfaColor::Red, CfaColor::Blue] {
                let value = if c == want {
                    raw[i]
                } else {
                    let mut acc = 0.0;
                    let mut cnt = 0.0;
                    for (dx, dy) in BayerPattern::AXIAL.iter().chain(BayerPattern::DIAGONAL.iter()) {
                        let jx = clamp_i(x as isize + dx, w);
                        let jy = clamp_i(y as isize + dy, h);
                        if pat.color_at(jx, jy) == want {
                            let j = jy * w + jx;
                            acc += raw[j] - out.g[j];
                            cnt += 1.0;
                        }
                    }
                    if cnt > 0.0 {
                        out.g[i] + acc / cnt
                    } else {
                        out.g[i]
                    }
                };
                if want == CfaColor::Red {
                    out.r[i] = value;
                } else {
                    out.b[i] = value;
                }
            }
        }
    }
    out
}

/// AHD homogeneity count over a 5x5 neighborhood (paper 4.3, "Homogeneity
/// map"): number of neighbors whose CIELAB luminance and chrominance
/// differences are both below the thresholds.
fn homogeneity_counts(lab: &lab::LabPlanes, w: usize, h: usize, eta_l: f32, eta_c: f32) -> Vec<f32> {
    let n = w * h;
    let mut counts = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (l0, a0, b0) = (lab.l[i], lab.a[i], lab.b[i]);
            let mut cnt = 0.0f32;
            for dy in -2..=2isize {
                for dx in -2..=2isize {
                    let j = clamp_i(y as isize + dy, h) * w + clamp_i(x as isize + dx, w);
                    let dl = (lab.l[j] - l0).abs();
                    let da = lab.a[j] - a0;
                    let db = lab.b[j] - b0;
                    if dl < eta_l && (da * da + db * db).sqrt() < eta_c {
                        cnt += 1.0;
                    }
                }
            }
            counts[i] = cnt;
        }
    }
    counts
}

/// Mean of a single-channel map over the clamped 3x3 window at (x, y).
fn mean3(map: &[f32], w: usize, h: usize, x: usize, y: usize) -> f32 {
    let mut acc = 0.0;
    for dy in -1..=1isize {
        for dx in -1..=1isize {
            acc += get(map, w, h, x as isize + dx, y as isize + dy);
        }
    }
    acc / 9.0
}

/// Stage 3: homogeneity-directed refinement for EDGE quarter-color sites
/// (AHD-inspired). `g_ref` is updated in place; green sites keep the raw
/// sample. The full machinery (two directional candidates, CIELAB conversion,
/// homogeneity maps) is skipped entirely when the image has no EDGE pixels.
pub fn stage3_edge_green(
    raw: &[f32],
    g_ref: &mut [f32],
    w: usize,
    h: usize,
    opts: &ArcdOptions,
    classes: &[u8],
) {
    // Count EDGE quarter-color sites; skip everything otherwise.
    let mut edge_q = 0usize;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if classes[i] == RegionClass::Edge as u8
                && opts.pattern.color_at(x, y) != CfaColor::Green
            {
                edge_q += 1;
            }
        }
    }
    if edge_q == 0 {
        return;
    }

    let cand_h = directional_candidates(raw, w, h, opts.pattern, true);
    let cand_v = directional_candidates(raw, w, h, opts.pattern, false);
    let lab_h = lab::rgb_to_lab(&cand_h, opts.px_max());
    let lab_v = lab::rgb_to_lab(&cand_v, opts.px_max());
    let cnt_h = homogeneity_counts(&lab_h, w, h, opts.eta_l, opts.eta_c);
    let cnt_v = homogeneity_counts(&lab_v, w, h, opts.eta_l, opts.eta_c);

    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if classes[i] != RegionClass::Edge as u8 {
                continue;
            }
            if opts.pattern.color_at(x, y) == CfaColor::Green {
                continue;
            }
            // Paper pseudocode: pick the direction whose 3x3-averaged
            // homogeneity is larger (ties resolve to horizontal).
            let m_h = mean3(&cnt_h, w, h, x, y);
            let m_v = mean3(&cnt_v, w, h, x, y);
            let chosen = if m_h >= m_v { cand_h.g[i] } else { cand_v.g[i] };
            g_ref[i] = clamp_px(chosen, opts.px_max());
        }
    }
}

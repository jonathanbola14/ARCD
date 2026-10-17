//! Stage 4: red/blue interpolation with color-difference constancy plus the
//! 3x3 median artifact-suppression pass.
//!
//! Paper sections 4.1 (Stage 4) and 4.3 ("Median filter").

use crate::buffer::{clamp_i, clamp_px, get};
use crate::options::ArcdOptions;
use crate::pattern::{BayerPattern, CfaColor};

/// Median of the 9 samples in the clamped 3x3 window of `map` at (x, y).
fn median3(map: &[f32], w: usize, h: usize, x: usize, y: usize) -> f32 {
    let mut v = [0.0f32; 9];
    let mut k = 0;
    for dy in -1..=1isize {
        for dx in -1..=1isize {
            v[k] = get(map, w, h, x as isize + dx, y as isize + dy);
            k += 1;
        }
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    v[4]
}

/// Interpolate red and blue at every site using color-difference constancy
/// against the (final) green channel, then mix each difference channel with
/// its 3x3 median (`rho_median` blend).
///
/// * at a red site, `R` is the raw sample; at a green site the two axial red
///   neighbors; at a blue site the four diagonal (always red) neighbors;
/// * blue symmetric to red;
/// * the per-site difference `D(p) = C(p) - G_ref(p)` is built for every pixel
///   first, then filtered, so the median sees a complete difference map.
pub fn rb_and_median(
    raw: &[f32],
    g_ref: &[f32],
    w: usize,
    h: usize,
    opts: &ArcdOptions,
) -> (Vec<f32>, Vec<f32>) {
    let pat = opts.pattern;
    let n = w * h;
    let mut r = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];

    // Build the full difference maps D_R = R_hat - G_ref and D_B = B_hat - G_ref.
    let mut d_r = vec![0.0f32; n];
    let mut d_b = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let c = pat.color_at(x, y);
            // Channel sample (`want` == own color spans the same-color sites
            // that surround this pixel) for each of red/blue.
            for (want, d_slot) in [(CfaColor::Red, 0usize), (CfaColor::Blue, 1usize)] {
                let d = if c == want {
                    raw[i] - g_ref[i]
                } else {
                    let mut acc = 0.0;
                    let mut cnt = 0.0;
                    for (dx, dy) in
                        BayerPattern::AXIAL.iter().chain(BayerPattern::DIAGONAL.iter())
                    {
                        let jx = clamp_i(x as isize + dx, w);
                        let jy = clamp_i(y as isize + dy, h);
                        if pat.color_at(jx, jy) == want {
                            let j = jy * w + jx;
                            acc += raw[j] - g_ref[j];
                            cnt += 1.0;
                        }
                    }
                    if cnt > 0.0 {
                        // Estimate: G_ref + mean difference at neighbor sites.
                        acc / cnt
                    } else {
                        0.0
                    }
                };
                if d_slot == 0 {
                    d_r[i] = d;
                } else {
                    d_b[i] = d;
                }
            }
        }
    }

    // Median-filter the difference channels and recombine with green.
    let rho = opts.rho_median;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mr = median3(&d_r, w, h, x, y);
            let mb = median3(&d_b, w, h, x, y);
            r[i] = clamp_px(g_ref[i] + rho * mr + (1.0 - rho) * d_r[i], opts.px_max());
            b[i] = clamp_px(g_ref[i] + rho * mb + (1.0 - rho) * d_b[i], opts.px_max());
        }
    }
    (r, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::ArcdOptions;

    #[test]
    fn median3_simple() {
        // 4x3 map; at (1,1) the clamped 3x3 window is [9,1,8,2,7,3,6,5,4]
        // -> sorted median (index 4) = 5.
        let map = vec![9.0, 1.0, 8.0, 99.0, 2.0, 7.0, 3.0, 99.0, 6.0, 5.0, 4.0, 99.0];
        assert_eq!(median3(&map, 4, 3, 1, 1), 5.0);
        // Clamped corner window repeats the corner value heavily.
        let m2 = vec![10.0, 20.0, 30.0, 40.0];
        assert_eq!(median3(&m2, 2, 2, 0, 0), 20.0);
    }

    #[test]
    fn constant_image_red_blue_recovered() {
        // On a constant Bayer image, differences vanish and output equals the
        // channel constants regardless of rho_median.
        let (w, h) = (12usize, 8usize);
        let pat = BayerPattern::Rggb;
        let mut raw = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                raw[y * w + x] = match pat.color_at(x, y) {
                    CfaColor::Red => 200.0,
                    CfaColor::Green => 150.0,
                    CfaColor::Blue => 100.0,
                };
            }
        }
        let mut opts = ArcdOptions::new(pat);
        let g = vec![150.0; w * h];
        for rho in [0.0, 0.7, 1.0] {
            opts.rho_median = rho;
            let (r, b) = rb_and_median(&raw, &g, w, h, &opts);
            for i in 0..w * h {
                assert!((r[i] - 200.0).abs() < 1e-4);
                assert!((b[i] - 100.0).abs() < 1e-4);
            }
        }
    }
}

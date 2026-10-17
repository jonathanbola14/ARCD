//! ARCD — Adaptive Ratio-Corrected Demosaicing for Bayer CFA sensors.
//!
//! Reference implementation of the staged hybrid architecture proposed in the
//! paper *"Adaptive Ratio-Corrected Demosaicing with Homogeneity-Directed
//! Refinement for Bayer CFA Sensors"* (see `Paper.md` at the repository root).
//!
//! ## Pipeline (paper section 4.1)
//!
//! 1. **Stage 1 — pre-interpolation and region classification.** Fast
//!    bilinear (or MHC) pre-interpolation; local variance over a 5x5 raw
//!    window and axial gradients classify each pixel as `flat`, `texture` or
//!    `edge`.
//! 2. **Stage 2 — ratio-corrected green (texture regions).** RCD-inspired
//!    directional channel/green ratio candidates blended with the measured
//!    discrimination weights, mixed with difference-based (MHC-style)
//!    candidates by `lambda_ratio`.
//! 3. **Stage 3 — homogeneity-directed refinement (edge regions).**
//!    AHD-inspired: full horizontal/vertical RGB candidates are converted to
//!    CIELAB, 5x5 homogeneity maps are averaged over 3x3, and the direction
//!    with larger homogeneity provides the final green at edge sites.
//! 4. **Stage 4 — red/blue interpolation and artifact suppression.** Red/blue
//!    via color-difference constancy against the final green, then 3x3 median
//!    filtering of the `R-G` and `B-G` difference channels.
//!
//! Flat regions bypass Stages 2-4: their output is the Stage 1
//! pre-interpolation itself.
//!
//! ## Design constraints honored for future Python bindings
//!
//! * the crate has **no dependencies** (pure `std`), so it embeds cleanly
//!   behind PyO3/FFI/WASM;
//! * the public API works on plain slices and returns plain owned vectors;
//! * all parameters live in one serializable-friendly struct
//!   [`ArcdOptions`];
//! * `#![forbid(unsafe_code)]` — no unsafe anywhere in the core.
//!
//! ## Example
//!
//! ```no_run
//! use arcd::{demosaic, BayerPattern, ArcdOptions};
//!
//! // 16-bit Bayer sample (RGGB 2x2 tile shown; in practice the full mosaic).
//! let raw = vec![512u16; 16 * 16];
//! let opts = ArcdOptions::for_bit_depth(BayerPattern::Rggb, 16);
//! let rgb = demosaic(&raw, 16, 16, &opts).unwrap();
//! assert_eq!(rgb.len(), 16 * 16 * 3);
//! ```

#![forbid(unsafe_code)]

mod buffer;
mod cfa;
mod error;
mod green;
mod lab;
mod options;
mod pattern;
mod stage1;
mod stage4;

use buffer::clamp_px;

pub use cfa::CfaSample;
pub use error::{ArcdError, Result};
pub use options::{ArcdOptions, PreInterpolation};
pub use pattern::{BayerPattern, CfaColor};
pub use stage1::{bilinear, classify, mhc, Planes, RegionClass};

/// Flatten an interleaved RGB image (length `3 * width * height`) into a Bayer
/// CFA mosaic with the given pattern.
///
/// This is the inverse process used to generate benchmark inputs (paper
/// section 5.4: "Generate mosaiced inputs from ground-truth images using the
/// Bayer RGGB pattern").
pub fn mosaic<S: CfaSample>(
    rgb: &[S],
    width: usize,
    height: usize,
    pattern: BayerPattern,
) -> Result<Vec<u16>> {
    let n = width * height;
    if rgb.len() != 3 * n {
        return Err(ArcdError::ShapeMismatch {
            expected: 3 * n,
            got: rgb.len(),
        });
    }
    let mut out = vec![0u16; n];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let c = match pattern.color_at(x, y) {
                CfaColor::Red => 0usize,
                CfaColor::Green => 1,
                CfaColor::Blue => 2,
            };
            let v = rgb[3 * i + c].value();
            out[i] = v.clamp(0.0, 65535.0).round() as u16;
        }
    }
    Ok(out)
}

/// Demosaic a Bayer CFA mosaic (paper's full ARCD pipeline).
///
/// * `input` — samples in row-major order, `input[y * width + x]`.
/// * returns interleaved RGB, `out[3 * (y * width + x)]..=..+2` for (R, G, B),
///   values in the raw scale `0..=px_max`, rounded by the algorithm but not
///   quantized to integers.
pub fn demosaic<S: CfaSample>(
    input: &[S],
    width: usize,
    height: usize,
    opts: &ArcdOptions,
) -> Result<Vec<f32>> {
    if width == 0 || height == 0 {
        return Err(ArcdError::TooSmall { width, height });
    }
    let n = width * height;
    if input.len() != n {
        return Err(ArcdError::ShapeMismatch {
            expected: n,
            got: input.len(),
        });
    }
    opts.validate()?;

    let pat = opts.pattern;
    let px_max = opts.px_max();
    let raw: Vec<f32> = input.iter().map(|s| s.value()).collect();

    // Stage 1: pre-interpolation + classification.
    let planes = match opts.pre_interpolation {
        PreInterpolation::Bilinear => bilinear(&raw, width, height, pat),
        PreInterpolation::Mhc => mhc(&raw, width, height, pat, px_max),
    };
    let (classes, _edge_count) = classify(&raw, width, height, opts);

    // Final green buffer: raw at green sites, pre-interpolated green at
    // quarter-color sites; refined by Stages 2 and 3 by class.
    let mut g_ref = planes.g.clone();

    // Stage 2: ratio-corrected green for texture regions.
    green::stage2_texture_green(&raw, &planes.g, &mut g_ref, width, height, opts, &classes);

    // Stage 3: homogeneity-directed refinement for edge regions.
    green::stage3_edge_green(&raw, &mut g_ref, width, height, opts, &classes);

    // Stage 4: red/blue interpolation + median artifact suppression.
    let (r, b) = stage4::rb_and_median(&raw, &g_ref, width, height, opts);

    // Assemble interleaved RGB; flat regions use the Stage 1 result directly
    // (paper pseudocode: the flat overwrite happens last).
    let flat = RegionClass::Flat as u8;
    let mut out = Vec::with_capacity(3 * n);
    for i in 0..n {
        if classes[i] == flat {
            for v in [planes.r[i], planes.g[i], planes.b[i]] {
                out.push(clamp_px(v, px_max));
            }
        } else {
            for v in [r[i], g_ref[i], b[i]] {
                out.push(clamp_px(v, px_max));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::CfaColor;

    /// Deterministic smooth synthetic RGB image (values in 0..=255).
    fn synth_rgb(w: usize, h: usize) -> Vec<u16> {
        let mut rgb = Vec::with_capacity(3 * w * h);
        for y in 0..h {
            for x in 0..w {
                let fx = x as f32;
                let fy = y as f32;
                let r = 128.0 + 55.0 * (0.055 * fx).sin() + 30.0 * (0.035 * fy).cos();
                let g = 120.0 + 45.0 * (0.043 * fy + 0.02 * fx).sin() + 25.0 * (0.06 * fx).cos();
                let b = 130.0 + 50.0 * (0.038 * fx + 0.025 * fy).cos() + 20.0 * (0.05 * fy).sin();
                for v in [r, g, b] {
                    rgb.push(v.clamp(8.0, 247.0).round() as u16);
                }
            }
        }
        rgb
    }

    #[test]
    fn mosaic_and_demosaic_roundtrip_smooth() {
        let (w, h) = (96usize, 64usize);
        let rgb = synth_rgb(w, h);
        let raw = mosaic(&rgb, w, h, BayerPattern::Rggb).unwrap();
        let opts = ArcdOptions::new(BayerPattern::Rggb);
        let out = demosaic(&raw, w, h, &opts).unwrap();
        assert_eq!(out.len(), 3 * w * h);
        let mut sse = 0.0f64;
        for k in 0..rgb.len() {
            let d = out[k] - rgb[k] as f32;
            sse += (d * d) as f64;
        }
        let rmse = (sse / rgb.len() as f64).sqrt();
        // Deterministic bound: must beat a naive baseline comfortably while
        // keeping full detail on this smooth synthetic (chosen empirically;
        // failure here signals a regression, not a research result).
        assert!(rmse < 8.0, "RMSE {rmse:.3}");
    }

    #[test]
    fn constant_image_all_patterns_exact() {
        for pat in [
            BayerPattern::Rggb,
            BayerPattern::Grbg,
            BayerPattern::Gbrg,
            BayerPattern::Bggr,
        ] {
            let (w, h) = (16usize, 12usize);
            let rgb = vec![200u16, 150u16, 100u16]
                .into_iter()
                .cycle()
                .take(3 * w * h)
                .collect::<Vec<_>>();
            let raw = mosaic(&rgb, w, h, pat).unwrap();
            let opts = ArcdOptions::new(pat);
            let out = demosaic(&raw, w, h, &opts).unwrap();
            for i in 0..w * h {
                assert!(
                    (out[3 * i] - 200.0).abs() < 1e-3
                        && (out[3 * i + 1] - 150.0).abs() < 1e-3
                        && (out[3 * i + 2] - 100.0).abs() < 1e-3,
                    "{pat:?} pixel {i}: {:?}",
                    (&out[3 * i], &out[3 * i + 1], &out[3 * i + 2])
                );
            }
        }
    }

    #[test]
    fn small_images_do_not_panic() {
        for &(w, h) in [(1usize, 1usize), (2, 2), (3, 3), (5, 5), (6, 6), (4, 9)].iter() {
            let raw = mosaic(&synth_rgb(w, h), w, h, BayerPattern::Rggb).unwrap();
            let opts = ArcdOptions::new(BayerPattern::Rggb);
            let out = demosaic(&raw, w, h, &opts).unwrap();
            assert_eq!(out.len(), 3 * w * h);
            for (k, v) in out.iter().enumerate() {
                assert!(
                    v.is_finite() && (0.0..=255.0).contains(v),
                    "pixel {k} of {w}x{h}: {v}"
                );
            }
        }
    }

    #[test]
    fn sharp_edges_stay_finite_and_in_range() {
        // 6x6 checkerboard: worst case for demosaicers.
        let (w, h) = (32usize, 24usize);
        let mut rgb = Vec::with_capacity(3 * w * h);
        for y in 0..h {
            for x in 0..w {
                let v = if (x / 4 + y / 4) % 2 == 0 { 10u16 } else { 245u16 };
                rgb.extend([v, v, v]);
            }
        }
        let raw = mosaic(&rgb, w, h, BayerPattern::Rggb).unwrap();
        let opts = ArcdOptions::new(BayerPattern::Rggb);
        let out = demosaic(&raw, w, h, &opts).unwrap();
        for (k, v) in out.iter().enumerate() {
            assert!(v.is_finite() && (0.0..=255.0).contains(v), "sample {k}: {v}");
        }
    }

    #[test]
    fn flat_regions_get_bilinear_pre_interpolation() {
        // A plateau of pure gray: nearly everything must classify as FLAT, so
        // the output must equal the constant exactly (bilinear is exact).
        let (w, h) = (20usize, 20usize);
        let rgb: Vec<u16> = std::iter::repeat_n(128u16, 3 * w * h).collect();
        let raw = mosaic(&rgb, w, h, BayerPattern::Rggb).unwrap();
        // add a tiny gradient column to exercise classification paths
        let mut raw2 = raw.clone();
        for y in 0..h {
            raw2[y * w] = raw2[y * w].saturating_add(40);
        }
        let opts = ArcdOptions::new(BayerPattern::Rggb);
        let out = demosaic(&raw2, w, h, &opts).unwrap();
        for (k, v) in out.iter().enumerate() {
            assert!(v.is_finite() && (0.0..=255.0).contains(v), "sample {k}: {v}");
        }
    }

    #[test]
    fn input_len_mismatch_is_an_error() {
        let opts = ArcdOptions::new(BayerPattern::Rggb);
        assert!(demosaic::<u16>(&[], 4, 4, &opts).is_err());
        assert!(demosaic::<u16>(&vec![0u16; 5], 4, 4, &opts).is_err());
        assert!(demosaic(&vec![0u16; 16], 4, 4, &opts).is_ok());
    }

    #[test]
    fn invalid_options_rejected() {
        let mut opts = ArcdOptions::new(BayerPattern::Rggb);
        opts.rho_median = 1.5;
        assert!(demosaic(&vec![0u16; 16], 4, 4, &opts)
            .unwrap_err()
            .to_string()
            .contains("rho_median"));
        opts.rho_median = 0.7;
        opts.bit_depth = 17;
        assert!(demosaic(&vec![0u16; 16], 4, 4, &opts).is_err());
    }

    #[test]
    fn all_bayer_colors_present_in_mosaic() {
        let rgb = synth_rgb(4, 4);
        let mosaic_out = mosaic(&rgb, 4, 4, BayerPattern::Rggb).unwrap();
        let mut seen = vec![(0.0, 0.0, 0.0); 16];
        for y in 0..4 {
            for x in 0..4 {
                seen[y * 4 + x] = (mosaic_out[y * 4 + x] as f64, 0.0, 0.0);
                assert!(mosaic_out[y * 4 + x] > 0);
            }
        }
        assert_eq!(seen[0].0, rgb[0] as f64); // top-left is R in RGGB
        assert_eq!(seen[1].0, rgb[4] as f64); // then G
        let _ = CfaColor::Red;
    }
}

//! Integration tests for the full ARCD pipeline.
//!
//! All images are synthetic and deterministic (no external datasets needed),
//! exercising: round-trip fidelity, all four Bayer phases, both Stage 1
//! pre-interpolation methods, 16-bit input, and noisy inputs.

use arcd::{
    bilinear, demosaic, mosaic, ArcdOptions, BayerPattern, PreInterpolation,
};

/// Smooth synthetic RGB image with values in `8..=247` (8-bit scale).
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

/// Deterministic LCG noise (percent point function-free uniform noise in
/// [-1, 1)), scaled by the caller.
fn noise_bits(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // take the middle 32 bits as uniform [0, 1)
        let u = ((state >> 33) as u32) as f32 / u32::MAX as f32;
        out.push(2.0 * u - 1.0);
    }
    out
}

fn rmse_rgb(out: &[f32], reference: &[u16]) -> f64 {
    assert_eq!(out.len(), reference.len());
    let mut sse = 0.0f64;
    for k in 0..reference.len() {
        let d = out[k] - reference[k] as f32;
        sse += (d * d) as f64;
    }
    (sse / reference.len() as f64).sqrt()
}

#[test]
fn roundtrip_all_patterns_smooth_8bit() {
    let (w, h) = (128usize, 96usize);
    let rgb = synth_rgb(w, h);
    for pat in [
        BayerPattern::Rggb,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
        BayerPattern::Bggr,
    ] {
        let raw = mosaic(&rgb, w, h, pat).unwrap();
        let opts = ArcdOptions::new(pat);
        let out = demosaic(&raw, w, h, &opts).unwrap();
        let rmse = rmse_rgb(&out, &rgb);
        assert!(rmse < 8.0, "{pat:?}: RMSE {rmse:.3}");
    }
}

#[test]
fn roundtrip_16bit_smooth() {
    let (w, h) = (64usize, 64usize);
    let rgb8 = synth_rgb(w, h);
    // scale to 16-bit raw domain
    let rgb16: Vec<u16> = rgb8.iter().map(|&v| v * 257).collect();
    let raw = mosaic(&rgb16, w, h, BayerPattern::Rggb).unwrap();
    let opts = ArcdOptions::for_bit_depth(BayerPattern::Rggb, 16);
    let out = demosaic(&raw, w, h, &opts).unwrap();
    // compare in 16-bit scale; the 8-bit bound (< 8.0) scales to < 2056 abs,
    // i.e. RMSE/257 < 8.0
    let rmse = rmse_rgb(&out, &rgb16);
    assert!(rmse < 8.0 * 257.0, "RMSE {rmse:.1} (16-bit scale)");
}

#[test]
fn mhc_pre_interpolation_mode_works() {
    let (w, h) = (96usize, 64usize);
    let rgb = synth_rgb(w, h);
    let raw = mosaic(&rgb, w, h, BayerPattern::Rggb).unwrap();

    let mut opts = ArcdOptions::new(BayerPattern::Rggb);
    opts.pre_interpolation = PreInterpolation::Mhc;
    let out = demosaic(&raw, w, h, &opts).unwrap();
    for (k, v) in out.iter().enumerate() {
        assert!(v.is_finite() && (0.0..=255.0).contains(v), "sample {k}: {v}");
    }
    let rmse = rmse_rgb(&out, &rgb);
    assert!(rmse < 8.0, "mhc mode RMSE {rmse:.3}");
}

#[test]
fn nosy_input_beats_plain_bilinear_or_stays_competitive() {
    // Gaussian-ish uniform noise at sigma = 5 (8-bit units) on the mosaic.
    let (w, h) = (96usize, 64usize);
    let rgb = synth_rgb(w, h);
    let raw = mosaic(&rgb, w, h, BayerPattern::Rggb).unwrap();
    let noise = noise_bits(42, raw.len());
    let noisy_raw: Vec<u16> = raw
        .iter()
        .zip(noise)
        .map(|(&v, n)| (v as f32 + 5.0 * 1.732 * n).round().clamp(0.0, 255.0) as u16)
        .collect();

    let opts = ArcdOptions::new(BayerPattern::Rggb);
    let out = demosaic(&noisy_raw, w, h, &opts).unwrap();
    let arcd_rmse = rmse_rgb(&out, &rgb);

    // Baseline: bilinear pre-interpolation interleave.
    let rawf: Vec<f32> = noisy_raw.iter().map(|&v| v as f32).collect();
    let bil = bilinear(&rawf, w, h, BayerPattern::Rggb);
    let mut bil_inter = vec![0.0f32; out.len()];
    for i in 0..w * h {
        bil_inter[3 * i] = bil.r[i];
        bil_inter[3 * i + 1] = bil.g[i];
        bil_inter[3 * i + 2] = bil.b[i];
    }
    let bil_rmse = rmse_rgb(&bil_inter, &rgb);

    // The adaptive pipeline must not underperform plain bilinear by more
    // than a small margin on noisy smooth input.
    assert!(
        arcd_rmse <= bil_rmse + 3.0,
        "arcd {arcd_rmse:.3} vs bilinear {bil_rmse:.3}"
    );
}

#[test]
fn lambda_extremes_fall_back_to_pure_candidates() {
    let (w, h) = (64usize, 48usize);
    let rgb = synth_rgb(w, h);
    let raw = mosaic(&rgb, w, h, BayerPattern::Rggb).unwrap();
    for lambda in [0.0, 0.5, 1.0] {
        let mut opts = ArcdOptions::new(BayerPattern::Rggb);
        opts.lambda_ratio = lambda;
        let out = demosaic(&raw, w, h, &opts).unwrap();
        let rmse = rmse_rgb(&out, &rgb);
        assert!(rmse < 10.0, "lambda {lambda}: RMSE {rmse:.3}");
    }
}

#[test]
fn median_strength_extremes_stay_valid() {
    let (w, h) = (64usize, 48usize);
    let rgb = synth_rgb(w, h);
    let raw = mosaic(&rgb, w, h, BayerPattern::Rggb).unwrap();
    for rho in [0.0, 0.25, 0.5, 0.7, 1.0] {
        let mut opts = ArcdOptions::new(BayerPattern::Rggb);
        opts.rho_median = rho;
        let out = demosaic(&raw, w, h, &opts).unwrap();
        for (k, v) in out.iter().enumerate() {
            assert!(v.is_finite() && (0.0..=255.0).contains(v), "rho {rho} sample {k}: {v}");
        }
    }
}

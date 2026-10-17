//! sRGB -> CIELAB conversion (Stage 3 homogeneity metric support).
//!
//! Follows the AHD formulation (Hirakawa & Parks): directional interpolation
//! candidates are converted to CIELAB and compared with per-channel thresholds
//! to build homogeneity maps.

use crate::stage1::Planes;

/// sRGB (D65) to CIEXYZ matrix.
const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175_0],
    [0.019_333_9, 0.119_192, 0.950_304_1],
];

/// D65 reference white.
const XN: f32 = 0.950_47;
const ZN: f32 = 1.088_83;

/// L* piecewise-linear knee: (6/29)^3.
const EPS_LAB: f32 = 0.008_856;
/// Slope below the knee: (1/3) * (29/6)^2.
const KAPPA_LAB: f32 = 7.787_037;

/// CIELAB planes of an image.
#[derive(Debug, Clone)]
pub struct LabPlanes {
    /// L* in 0..100.
    pub l: Vec<f32>,
    /// a* (green..red axis).
    pub a: Vec<f32>,
    /// b* (blue..yellow axis).
    pub b: Vec<f32>,
}

/// sRGB electro-optical transfer (0..1 nonlinear -> 0..1 linear).
#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// CIELAB "f" function.
#[inline]
fn lab_f(t: f32) -> f32 {
    if t > EPS_LAB {
        t.cbrt()
    } else {
        (KAPPA_LAB * t + 16.0) / 116.0
    }
}

/// Convert an RGB image (`Planes`, values in `0..=px_max`) to CIELAB planes.
pub fn rgb_to_lab(rgb: &Planes, px_max: f32) -> LabPlanes {
    let n = rgb.r.len();
    let mut out = LabPlanes {
        l: vec![0.0; n],
        a: vec![0.0; n],
        b: vec![0.0; n],
    };
    for i in 0..n {
        let r = srgb_to_linear((rgb.r[i] / px_max).clamp(0.0, 1.0));
        let g = srgb_to_linear((rgb.g[i] / px_max).clamp(0.0, 1.0));
        let bch = srgb_to_linear((rgb.b[i] / px_max).clamp(0.0, 1.0));

        let x = SRGB_TO_XYZ[0][0] * r + SRGB_TO_XYZ[0][1] * g + SRGB_TO_XYZ[0][2] * bch;
        let y = SRGB_TO_XYZ[1][0] * r + SRGB_TO_XYZ[1][1] * g + SRGB_TO_XYZ[1][2] * bch;
        let z = SRGB_TO_XYZ[2][0] * r + SRGB_TO_XYZ[2][1] * g + SRGB_TO_XYZ[2][2] * bch;

        let fx = lab_f(x / XN);
        let fy = lab_f(y);
        let fz = lab_f(z / ZN);

        out.l[i] = 116.0 * fy - 16.0;
        out.a[i] = 500.0 * (fx - fy);
        out.b[i] = 200.0 * (fy - fz);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planes_at(r: f32, g: f32, b: f32, n: usize) -> Planes {
        Planes {
            r: vec![r; n],
            g: vec![g; n],
            b: vec![b; n],
        }
    }

    #[test]
    fn white_black_gray() {
        let px_max = 255.0;
        let white = rgb_to_lab(&planes_at(255.0, 255.0, 255.0, 4), px_max);
        for i in 0..4 {
            assert!((white.l[i] - 100.0).abs() < 1e-3, "L*={}", white.l[i]);
            assert!(white.a[i].abs() < 1e-3);
            assert!(white.b[i].abs() < 1e-3);
        }
        let black = rgb_to_lab(&planes_at(0.0, 0.0, 0.0, 4), px_max);
        for i in 0..4 {
            assert!(black.l[i].abs() < 1e-4);
        }
        let mid = rgb_to_lab(&planes_at(128.0, 128.0, 128.0, 4), px_max);
        for i in 0..4 {
            // sRGB 128 is slightly below linear mid-gray, so L* near 53.
            assert!((mid.l[i] - 53.0).abs() < 1.0, "L*={}", mid.l[i]);
            assert!(mid.a[i].abs() < 1e-3);
            assert!(mid.b[i].abs() < 1e-3);
        }
    }
}

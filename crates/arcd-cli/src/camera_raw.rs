//! Camera RAW container → Bayer mosaic bridge for the CLI (cargo feature
//! `camera-raw`, backed by the pure-Rust [`rawler`] decoder).
//!
//! Documented pipeline choices, kept deliberately identical to the reference
//! extraction in `scripts/raw_to_ppm.py` (rawpy/LibRaw):
//!
//! * **Active-area crop.** The container's `active_area` is applied (clamped to
//!   the plane) so the masked sensor border stays out of the pipeline, and the
//!   2×2 phase is re-derived for the crop offset — cropping on an odd
//!   row/column rotates the Bayer phase (`CFA::shift`).
//! * **Normalization.** The black level is removed and the `[black, white]`
//!   interval is stretched to the full 16-bit range `[0, 65535]`, which is the
//!   domain `ArcdOptions::for_bit_depth(_, 16)` assumes. Like the script, a
//!   single black level (max over the per-site levels) applies to all sites.
//! * **Phase.** Auto-detected from the container's CFA definition; the CLI's
//!   `--pattern` overrides it.
//! * **Unsupported sensors.** X-Trans (6×6), 4-color (RGBE/CYGM), rotated Fuji
//!   and linear-RGB data error out with a hint; `scripts/raw_to_ppm.py` (which
//!   goes through LibRaw) covers those.

use std::path::Path;

use arcd::BayerPattern;
use rawler::RawImageData;
use rawler::rawimage::RawPhotometricInterpretation;

use crate::CameraRawFrame;

/// Load a camera RAW container (whatever `rawler` supports: DNG, NEF, CR2,
/// ARW, RAF, ORF, RW2, PEF, ...) and return the Bayer mosaic normalized to the
/// full 16-bit scale, i.e. the domain `ArcdOptions::for_bit_depth(_, 16)`
/// expects. `phase_override` (the CLI's `--pattern`) wins over auto-detection.
pub fn load_camera_raw(
    path: &Path,
    phase_override: Option<BayerPattern>,
) -> Result<CameraRawFrame, String> {
    let image =
        rawler::decode_file(path).map_err(|e| format!("cannot decode camera raw file {path:?}: {e}"))?;

    // Rotated Fuji sensors (SuperCCD / EXR) store a diagonal mosaic — not a
    // plain Bayer plane that can be handed to ARCD.
    if image.fuji_rotation_width.is_some() {
        return Err(
            "rotated Fuji sensor layout is not supported by the ARCD pipeline".to_string(),
        );
    }

    let cfa = match &image.photometric {
        RawPhotometricInterpretation::Cfa(config) => config.cfa.clone(),
        other => {
            return Err(format!(
                "unsupported photometric interpretation ({other:?}); ARCD expects a Bayer CFA \
                 sensor (linear-RGB / non-CFA raw files are not covered)"
            ))
        }
    };
    if cfa.width != 2 || cfa.height != 2 {
        return Err(format!(
            "unsupported CFA pattern {cfa} ({0}x{1} grid): ARCD implements the four 2x2 Bayer \
             phases only (X-Trans / 4-color sensors are not covered)",
            cfa.width,
            cfa.height
        ));
    }

    // Active-area crop (clamped to the plane) and the phase re-derived for its
    // offset. `None` (rect fully outside the plane) or an empty crop is an error.
    let rect = image
        .active_area
        .as_ref()
        .map(|r| (r.p.x, r.p.y, r.d.w, r.d.h));
    let (x0, y0, crop_w, crop_h) = crop_bounds(rect, image.width, image.height)?;

    let shifted_cfa = cfa.shift(x0 % 2, y0 % 2);
    let pattern = match phase_override {
        Some(p) => p,
        None => bayer_phase_from_name(&shifted_cfa.to_string())?,
    };

    // Per-site black levels collapse to their maximum, exactly like
    // `scripts/raw_to_ppm.py` (max over `black_level_per_channel`); white level
    // is the sensor's, so the [black, white] interval stretches to [0, 65535].
    let black = image
        .blacklevel
        .as_vec()
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let white = image.whitelevel.0.iter().copied().fold(0_u32, u32::max).max(1) as f32;
    if white <= black {
        return Err(format!("invalid black/white levels (white={white}, black={black})"));
    }

    // The integer CFA decode path: one sample per pixel, row-major, native
    // scale. Float data (linear DNG etc.) is rejected above via the
    // photometric check; this match is the belt-and-suspenders guard.
    let samples_full = match image.data {
        RawImageData::Integer(v) => v,
        RawImageData::Float(_) => {
            return Err("float CFA samples are not supported; ARCD expects integer Bayer data".to_string())
        }
    };
    let full_w = image.width;

    let mut samples = Vec::with_capacity(crop_w * crop_h);
    for row in y0..y0 + crop_h {
        let start = row * full_w + x0;
        samples.extend(samples_full[start..start + crop_w].iter().map(
            |s| rescale_to_u16(*s, black, white),
        ));
    }

    let mut make_model = format!("{} {}", image.clean_make, image.clean_model);
    if make_model.trim().is_empty() {
        make_model = "camera".to_string();
    }

    Ok(CameraRawFrame {
        width: crop_w,
        height: crop_h,
        pattern,
        samples,
        make_model,
        native_bps: image.bps,
        black,
        white,
    })
}

/// Map a rawler CFA pattern string (`"RGGB"`, row-major) to an ARCD phase.
fn bayer_phase_from_name(name: &str) -> Result<BayerPattern, String> {
    match name {
        "RGGB" => Ok(BayerPattern::Rggb),
        "GRBG" => Ok(BayerPattern::Grbg),
        "GBRG" => Ok(BayerPattern::Gbrg),
        "BGGR" => Ok(BayerPattern::Bggr),
        other => Err(format!(
            "unsupported CFA pattern '{other}'; ARCD implements the four 2x2 Bayer phases \
             (rggb, grbg, gbrg, bggr)"
        )),
    }
}

/// Clamp the active-area crop `(x, y, w, h)` to the sensor plane and require a
/// positive intersection with it.
fn crop_bounds(
    active: Option<(usize, usize, usize, usize)>,
    plane_w: usize,
    plane_h: usize,
) -> Result<(usize, usize, usize, usize), String> {
    let (x, y, w, h) = active.unwrap_or((0, 0, plane_w, plane_h));
    let x0 = x.min(plane_w);
    let y0 = y.min(plane_h);
    let crop_w = w.min(plane_w - x0);
    let crop_h = h.min(plane_h - y0);
    if crop_w == 0 || crop_h == 0 {
        return Err(format!(
            "empty active-area crop ({x},{y} {w}x{h}) on a {plane_w}x{plane_h} plane"
        ));
    }
    Ok((x0, y0, crop_w, crop_h))
}

/// Map one raw sample from `[black, white]` into the full 16-bit range.
///
/// Matches `scripts/raw_to_ppm.py`: black removed, interval stretched, then
/// rounded and clipped at both ends. Callers guarantee `white > black`, so the
/// divisor is strictly positive.
fn rescale_to_u16(sample: u16, black: f32, white: f32) -> u16 {
    let stretched = (f32::from(sample) - black) * (65535.0 / (white - black));
    stretched.round().clamp(0.0, 65535.0) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfa_names_map_to_all_four_bayer_phases() {
        assert_eq!(bayer_phase_from_name("RGGB").unwrap(), BayerPattern::Rggb);
        assert_eq!(bayer_phase_from_name("GRBG").unwrap(), BayerPattern::Grbg);
        assert_eq!(bayer_phase_from_name("GBRG").unwrap(), BayerPattern::Gbrg);
        assert_eq!(bayer_phase_from_name("BGGR").unwrap(), BayerPattern::Bggr);
    }

    #[test]
    fn non_bayer_cfa_names_are_rejected() {
        for name in ["", "RGBE", "GRGB", "GRGG", "ABCD"] {
            assert!(bayer_phase_from_name(name).is_err(), "{name} must be rejected");
        }
    }

    #[test]
    fn crop_bounds_without_active_area_is_the_full_plane() {
        assert_eq!(crop_bounds(None, 10, 8).unwrap(), (0, 0, 10, 8));
    }

    #[test]
    fn crop_bounds_clamps_to_the_plane() {
        // Crop extending past plane edges is clamped.
        assert_eq!(crop_bounds(Some((2, 3, 10, 10)), 10, 8).unwrap(), (2, 3, 8, 5));
        // Crop entirely outside the plane is an error, not a zero-sized plane.
        assert!(crop_bounds(Some((10, 0, 4, 4)), 10, 8).is_err());
        assert!(crop_bounds(Some((0, 8, 4, 4)), 10, 8).is_err());
    }

    #[test]
    fn rescale_maps_black_white_and_clamps() {
        assert_eq!(rescale_to_u16(0, 0.0, 65535.0), 0);
        assert_eq!(rescale_to_u16(65535, 0.0, 65535.0), 65535);
        assert_eq!(rescale_to_u16(1024, 1024.0, 16383.0), 0); // == black -> 0
        assert_eq!(rescale_to_u16(16383, 1024.0, 16383.0), 65535); // == white -> max
        assert_eq!(rescale_to_u16(0, 128.0, 4095.0), 0); // below black clamps
        assert_eq!(rescale_to_u16(5000, 128.0, 4095.0), 65535); // above white clamps
    }
}

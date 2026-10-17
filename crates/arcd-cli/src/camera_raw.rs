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
//! * **Orientation.** The container's EXIF orientation is applied as the
//!   display transform (`rawler::Orientation::to_flips`: horizontal/vertical
//!   flips first, then transposition) so portrait shots come out upright; the
//!   Bayer phase is re-derived after the transform. `--orientation none` keeps
//!   the sensor-plane geometry instead.
//! * **White balance (optional).** With `--white-balance`, each site is
//!   multiplied by the camera's WB gain (rawler's RGBE-order `wb_coeffs`,
//!   normalized so the strongest gain is 1 — channel ratios stay exact and a
//!   sample can never clip during the multiply). Without the files' gains
//!   (all zero) it is skipped. This is a preview aid: ARCD itself still
//!   operates in raw scale, and the output is dark/linear (no tone curve).
//! * **Unsupported sensors.** X-Trans (6×6), 4-color (RGBE/CYGM), rotated Fuji
//!   and linear-RGB data error out with a hint; `scripts/raw_to_ppm.py` (which
//!   goes through LibRaw) covers those.

use std::path::Path;

use arcd::BayerPattern;
use rawler::Orientation;
use rawler::RawImageData;
use rawler::rawimage::RawPhotometricInterpretation;

use crate::{CameraRawFrame, CropMode};

/// Load a camera RAW container (whatever `rawler` supports: DNG, NEF, CR2,
/// ARW, RAF, ORF, RW2, PEF, ...) and return the Bayer mosaic normalized to the
/// full 16-bit scale, i.e. the domain `ArcdOptions::for_bit_depth(_, 16)`
/// expects. `phase_override` (the CLI's `--pattern`) wins over auto-detection,
/// `mode` picks which region of the decoded plane becomes the output image:
///
/// * `ActiveArea` — the usable (non-masked) sensor area; the default.
/// * `RecommendedCrop` — the file's own recommended crop (DNG `DefaultCrop`),
///   i.e. the "image size" a viewer shows for the file; falls back to the
///   active area when the container does not declare one.
/// * `FullPlane` — the whole decoded plane, borders included (`dcraw -D`
///   style).
///
/// `white_balance` (the CLI's `--white-balance`) multiplies each site by the
/// camera's WB gain, see the module docs; `orientation` (the CLI's
/// `--orientation auto|none`) applies the container's EXIF orientation.
pub fn load_camera_raw(
    path: &Path,
    phase_override: Option<BayerPattern>,
    mode: CropMode,
    white_balance: bool,
    orientation_auto: bool,
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

    // Region selection (clamped to the plane) and the phase re-derived for
    // its offset. `None` (rect fully outside the plane) or an empty crop is
    // an error.
    let active = image
        .active_area
        .as_ref()
        .map(|r| (r.p.x, r.p.y, r.d.w, r.d.h));
    let recommended = image
        .crop_area
        .as_ref()
        .map(|r| (r.p.x, r.p.y, r.d.w, r.d.h));
    let (x0, y0, crop_w, crop_h) = crop_rect_for_mode(mode, active, recommended, image.width, image.height)?;

    let shifted_cfa = cfa.shift(x0 % 2, y0 % 2);
    let pattern = match phase_override {
        Some(p) => p,
        None => bayer_phase_from_name(&shifted_cfa.to_string())?,
    };
    // 2x2 site letters for the cropped plane's origin — used for the
    // per-site WB gains and re-derived after any orientation transform.
    let mut letters = pattern_letters(pattern);

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
    // Camera WB gains, scaled so the strongest is 1 (see module docs). Sample
    // rescale happens first, then the gain; with max-normalized gains nothing
    // can exceed the 16-bit top, so no extra clamp is needed.
    let wb_gains = if white_balance {
        white_balance_gains(image.wb_coeffs)
    } else {
        None
    };
    for row in y0..y0 + crop_h {
        let start = row * full_w + x0;
        let row_parity = (row - y0) % 2;
        samples.extend(
            samples_full[start..start + crop_w]
                .iter()
                .enumerate()
                .map(|(col, s)| {
                    let stretched = rescale_to_u16(*s, black, white);
                    match wb_gains {
                        None => stretched,
                        Some(gains) => match letters[row_parity][col % 2] {
                            'R' => ((f32::from(stretched)) * gains[0]) as u16,
                            'G' => ((f32::from(stretched)) * gains[1]) as u16,
                            'B' => ((f32::from(stretched)) * gains[2]) as u16,
                            _ => stretched,
                        },
                    }
                }),
        );
    }

    // EXIF orientation as the display transform: flipping first (inside
    // `orient_plane`), transposition last; the phase letters go through the
    // exact same index map, so the phase always matches the transformed
    // plane. Dimensions may swap on 90° orientations.
    let mut width = crop_w;
    let mut height = crop_h;
    let rotated = orientation_auto && {
        let (transpose, hflip, vflip) = image.orientation.to_flips();
        transpose || hflip || vflip
    };
    if rotated {
        letters = orient_letters(letters, width, height, image.orientation);
        let (rotated_samples, w, h) = orient_plane(samples, width, height, image.orientation);
        samples = rotated_samples;
        width = w;
        height = h;
    }
    let pattern = bayer_phase_from_name(&letters_name(&letters))?;

    let mut make_model = format!("{} {}", image.clean_make, image.clean_model);
    if make_model.trim().is_empty() {
        make_model = "camera".to_string();
    }

    Ok(CameraRawFrame {
        width,
        height,
        pattern,
        samples,
        make_model,
        native_bps: image.bps,
        black,
        white,
        white_balanced: wb_gains.is_some(),
        rotated,
    })
}

/// The 2×2 site letters of a Bayer phase at the plane origin (row-major,
/// e.g. `Rggb` → `[['R','G'],['G','B']]`).
fn pattern_letters(pattern: BayerPattern) -> [[char; 2]; 2] {
    match pattern {
        BayerPattern::Rggb => [['R', 'G'], ['G', 'B']],
        BayerPattern::Grbg => [['G', 'R'], ['B', 'G']],
        BayerPattern::Gbrg => [['G', 'B'], ['R', 'G']],
        BayerPattern::Bggr => [['B', 'G'], ['G', 'R']],
    }
}

/// Row-major pattern name of a 2×2 letter grid (`"RGGB"` style).
fn letters_name(letters: &[[char; 2]; 2]) -> String {
    letters.iter().flatten().collect()
}

/// Camera white balance gains from rawler's RGBE-order `wb_coeffs` (the first
/// three entries are R, G, B), normalized dcraw-style so the strongest gain
/// becomes 1: channel ratios — what actually neutralizes the green cast —
/// stay exact and no sample can clip during the multiply. `None` when the
/// container provides no usable coefficients (all zero) or when the camera
/// itself reports a neutral gain (nothing would change).
fn white_balance_gains(coeffs: [f32; 4]) -> Option<[f32; 3]> {
    let max = coeffs[..3].iter().copied().fold(0.0_f32, f32::max);
    if max <= 0.0 {
        return None;
    }
    // NaN coefficients (rawler uses NaN for the unused 4th entry and for
    // missing metadata) never reach the max over the first three; any NaN
    // result here means unusable metadata.
    let gains = [coeffs[0] / max, coeffs[1] / max, coeffs[2] / max];
    if gains.iter().any(|g| !g.is_finite()) {
        return None;
    }
    if gains.iter().all(|g| (g - 1.0).abs() < 1.0e-4) {
        return None; // camera-reported neutral: a no-op is not "wb"
    }
    Some(gains)
}

/// Apply an EXIF orientation to a row-major plane as the display transform:
/// horizontal/vertical mirroring first, then transposition (`Orientation::
/// to_flips` contract). Dimensions swap whenever the transform transposes.
/// The best-effort `Orientation::Unknown` maps to identity, like `Normal`.
fn orient_plane(
    plane: Vec<u16>,
    width: usize,
    height: usize,
    orientation: Orientation,
) -> (Vec<u16>, usize, usize) {
    let (transpose, hflip, vflip) = orientation.to_flips();
    let (out_w, out_h) = if transpose {
        (height, width)
    } else {
        (width, height)
    };
    let mut out = vec![0u16; out_w * out_h];
    for (idx, cell) in out.iter_mut().enumerate() {
        let row = idx / out_w;
        let col = idx % out_w;
        // The flips are applied before transposing, so the inverse index map
        // transposes first, then mirrors — in stored-plane coordinates.
        let (mut sr, mut sc) = if transpose { (col, row) } else { (row, col) };
        if vflip {
            sr = height - 1 - sr;
        }
        if hflip {
            sc = width - 1 - sc;
        }
        *cell = plane[sr * width + sc];
    }
    (out, out_w, out_h)
}

/// The site letters of a plane under [`orient_plane`]: the identical index
/// map applied to the 2×2 phase grid (with the plane's own dimensions, so
/// odd-sized crops stay phase-correct).
fn orient_letters(
    letters: [[char; 2]; 2],
    width: usize,
    height: usize,
    orientation: Orientation,
) -> [[char; 2]; 2] {
    let (transpose, hflip, vflip) = orientation.to_flips();
    let mut out = [['R'; 2]; 2];
    for (row, row_cells) in out.iter_mut().enumerate() {
        for (col, cell) in row_cells.iter_mut().enumerate() {
            // Same index map as `orient_plane`, in stored-plane coordinates:
            // transposition first, then the mirrors (that order matches how
            // rawler's display transform is defined: flips before transpose).
            let (mut sr, mut sc) = if transpose { (col, row) } else { (row, col) };
            if vflip {
                sr = height - 1 - sr;
            }
            if hflip {
                sc = width - 1 - sc;
            }
            *cell = letters[sr % 2][sc % 2];
        }
    }
    out
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

/// Resolve which plane region feeds the pipeline for a `--crop` mode.
fn crop_rect_for_mode(
    mode: CropMode,
    active: Option<(usize, usize, usize, usize)>,
    recommended: Option<(usize, usize, usize, usize)>,
    plane_w: usize,
    plane_h: usize,
) -> Result<(usize, usize, usize, usize), String> {
    match mode {
        CropMode::ActiveArea => crop_bounds(active, plane_w, plane_h),
        // The file's own DefaultCrop ("the image size" viewers show); when the
        // container declares none, the active area is the best default.
        CropMode::RecommendedCrop => {
            let rect = recommended.or(active);
            crop_bounds(rect, plane_w, plane_h)
        }
        CropMode::FullPlane => crop_bounds(None, plane_w, plane_h),
    }
}

/// Clamp an active-area crop `(x, y, w, h)` to the sensor plane and require a
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
    fn crop_mode_picks_the_declared_region() {
        let active = Some((0, 0, 10, 8));
        let recommended = Some((2, 2, 4, 4));

        assert_eq!(
            crop_rect_for_mode(CropMode::ActiveArea, active, recommended, 10, 8).unwrap(),
            (0, 0, 10, 8)
        );
        // The declared (recommended) crop wins when present...
        assert_eq!(
            crop_rect_for_mode(CropMode::RecommendedCrop, active, recommended, 10, 8).unwrap(),
            (2, 2, 4, 4)
        );
        // ...falls back to the active area when the file declares none...
        assert_eq!(
            crop_rect_for_mode(CropMode::RecommendedCrop, active, None, 10, 8).unwrap(),
            (0, 0, 10, 8)
        );
        // ...and `full` ignores both (masked border pixels included).
        assert_eq!(
            crop_rect_for_mode(CropMode::FullPlane, active, recommended, 10, 8).unwrap(),
            (0, 0, 10, 8)
        );
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

    #[test]
    fn pattern_letters_roundtrip_through_the_phase_name() {
        for (name, pattern) in [
            ("RGGB", BayerPattern::Rggb),
            ("GRBG", BayerPattern::Grbg),
            ("GBRG", BayerPattern::Gbrg),
            ("BGGR", BayerPattern::Bggr),
        ] {
            let letters = pattern_letters(pattern);
            assert_eq!(letters_name(&letters), name);
            assert_eq!(bayer_phase_from_name(&letters_name(&letters)).unwrap(), pattern);
        }
    }

    #[test]
    fn orientation_identity_keeps_the_plane_untouched() {
        for o in [Orientation::Normal, Orientation::Unknown] {
            assert_eq!(orient_plane(vec![0, 1, 2, 3, 4, 5], 3, 2, o), (vec![0, 1, 2, 3, 4, 5], 3, 2));
        }
    }

    #[test]
    fn orientation_flips_mirror_the_plane() {
        // 2x3 stored plane, row-major: [0 1 2 / 3 4 5]
        let plane: Vec<u16> = vec![0, 1, 2, 3, 4, 5];
        // horizontally mirrored: [2 1 0 / 5 4 3]
        assert_eq!(orient_plane(plane.clone(), 3, 2, Orientation::HorizontalFlip).0, [2, 1, 0, 5, 4, 3]);
        // vertically mirrored: [3 4 5 / 0 1 2]
        assert_eq!(orient_plane(plane.clone(), 3, 2, Orientation::VerticalFlip).0, [3, 4, 5, 0, 1, 2]);
        // rotated 180: [5 4 3 / 2 1 0]
        assert_eq!(orient_plane(plane.clone(), 3, 2, Orientation::Rotate180).0, [5, 4, 3, 2, 1, 0]);
        // pure transpose keeps rows intact, ordered by column: [0 3 / 1 4 / 2 5]
        assert_eq!(orient_plane(plane.clone(), 3, 2, Orientation::Transpose).0, [0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn orientation_rotates_90_and_270() {
        // EXIF 6 ("rotate 90 CW to display"): 2x3 -> 3x2, new(r,c) = old(H-1-c, r)
        // [0 1 2 / 3 4 5] -> [3 0 / 4 1 / 5 2]
        assert_eq!(
            orient_plane(vec![0, 1, 2, 3, 4, 5], 3, 2, Orientation::Rotate90).0,
            [3, 0, 4, 1, 5, 2]
        );
        // EXIF 8 ("rotate 270 CW to display"): new(r,c) = old(c, W-1-r)
        // [0 1 2 / 3 4 5] -> [2 5 / 1 4 / 0 3]
        assert_eq!(
            orient_plane(vec![0, 1, 2, 3, 4, 5], 3, 2, Orientation::Rotate270).0,
            [2, 5, 1, 4, 0, 3]
        );
    }

    #[test]
    fn orientation_tracks_the_phase_letters() {
        let rggb = pattern_letters(BayerPattern::Rggb);
        // Even-sized planes: flip/rotate of RGGB stays one of the four phases.
        // horizontal flip swaps columns: RGGB -> GRBG
        assert_eq!(letters_name(&orient_letters(rggb, 4, 4, Orientation::HorizontalFlip)), "GRBG");
        // vertical flip swaps rows: RGGB -> GBRG
        assert_eq!(letters_name(&orient_letters(rggb, 4, 4, Orientation::VerticalFlip)), "GBRG");
        // 180 = both flips: RGGB -> BGGR
        assert_eq!(letters_name(&orient_letters(rggb, 4, 4, Orientation::Rotate180)), "BGGR");
        // EXIF 6 (rotate 90 CW): new(r,c) = old(H-1-c, r) -> GRBG
        assert_eq!(letters_name(&orient_letters(rggb, 4, 4, Orientation::Rotate90)), "GRBG");
        // transpose is parity-perfect on the 2x2 grid: RGGB -> RGGB
        assert_eq!(letters_name(&orient_letters(rggb, 3, 3, Orientation::Transpose)), "RGGB");

        // Odd-sized planes shift the parity: e.g. a horizontal flip on an
        // odd-width plane maps column c to W-1-c, whose parity equals c's
        // when W is odd — the phase letters must stay put. Same for a vertical
        // flip on an odd-height plane.
        assert_eq!(letters_name(&orient_letters(rggb, 5, 4, Orientation::HorizontalFlip)), "RGGB");
        assert_eq!(letters_name(&orient_letters(rggb, 4, 5, Orientation::VerticalFlip)), "RGGB");
    }

    #[test]
    fn white_balance_gains_are_max_normalized_or_absent() {
        assert_eq!(
            white_balance_gains([2.0, 1.0, 1.5, 1.0]).unwrap(),
            [1.0, 0.5, 0.75]
        );
        // Coefficients already max-normalized pass through unchanged
        // (non-neutral, unlike a camera-reported [1,1,1]).
        assert_eq!(
            white_balance_gains([1.3, 1.0, 1.0, 1.0]).unwrap(),
            [1.0, 1.0 / 1.3, 1.0 / 1.3]
        );
        // All-zero gains (no WB metadata) disable the step...
        assert_eq!(white_balance_gains([0.0, 0.0, 0.0, 0.0]), None);
        // ...as does a camera-reported neutral gain (a no-op is not "wb")...
        assert_eq!(white_balance_gains([1.0, 1.0, 1.0, f32::NAN]), None);
        // ...and unusable metadata (NaN in the used entries).
        assert_eq!(white_balance_gains([f32::NAN, 2.0, 1.0, 1.0]), None);
    }
}

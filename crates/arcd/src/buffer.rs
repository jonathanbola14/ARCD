//! Small buffer access helpers (clamped indexing at image borders).

/// Clamp a coordinate to `0..len`, so that neighbor lookups stay inside the
/// image (ARCD uses clamped rather than mirrored borders).
#[inline]
pub fn clamp_i(v: isize, len: usize) -> usize {
    if v < 0 {
        0
    } else if v >= len as isize {
        len - 1
    } else {
        v as usize
    }
}

/// Fetch a sample from a single-channel plane with clamped coordinates.
#[inline]
pub fn get(buf: &[f32], w: usize, h: usize, x: isize, y: isize) -> f32 {
    buf[clamp_i(y, h) * w + clamp_i(x, w)]
}

/// In-place write with clamped coordinates (kept for completeness; the
/// algorithm writes planes directly by index).
#[allow(dead_code)]
#[inline]
pub fn set(buf: &mut [f32], w: usize, h: usize, x: isize, y: isize, v: f32) {
    buf[clamp_i(y, h) * w + clamp_i(x, w)] = v;
}

/// Nearest in-bounds coordinate at distance 2 along one axis (preserves the
/// parity, hence the Bayer site color — see [`crate::pattern::BayerPattern`]),
/// preferring the requested side and falling back to the opposite side.
/// Returns `None` when the dimension is too small to contain any same-color
/// axial neighbor.
#[inline]
pub fn neighbor2(c: usize, len: usize, negative: bool) -> Option<usize> {
    if len < 3 {
        return None;
    }
    let lower = c >= 2;
    let upper = c + 2 < len;
    match (negative, lower, upper) {
        (true, true, _) => Some(c - 2),
        (true, false, true) => Some(c + 2),
        (false, _, true) => Some(c + 2),
        (false, true, false) => Some(c - 2),
        _ => None,
    }
}

/// Clamp a value to the representable sample range `[0, px_max]`.
#[inline]
pub fn clamp_px(v: f32, px_max: f32) -> f32 {
    v.max(0.0).min(px_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps() {
        assert_eq!(clamp_i(-5, 4), 0);
        assert_eq!(clamp_i(2, 4), 2);
        assert_eq!(clamp_i(9, 4), 3);
        assert_eq!(clamp_px(-1.0, 255.0), 0.0);
        assert_eq!(clamp_px(300.0, 255.0), 255.0);
    }
}

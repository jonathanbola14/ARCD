//! Error types for the ARCD demosaicing library.

use std::fmt;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ArcdError>;

/// Errors returned by ARCD operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArcdError {
    /// The input is too small to be demosaiced meaningfully.
    TooSmall {
        /// Requested width (in pixels).
        width: usize,
        /// Requested height (in pixels).
        height: usize,
    },
    /// The input buffer length does not match `width * height` (or `3 * width * height`
    /// for mosaic helpers).
    ShapeMismatch {
        /// Number of samples expected.
        expected: usize,
        /// Number of samples provided.
        got: usize,
    },
    /// One or more options are outside their valid range.
    InvalidOptions(String),
}

impl fmt::Display for ArcdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArcdError::TooSmall { width, height } => {
                write!(f, "image is too small to demosaic: {width}x{height}")
            }
            ArcdError::ShapeMismatch { expected, got } => {
                write!(f, "input shape mismatch: expected {expected} samples, got {got}")
            }
            ArcdError::InvalidOptions(msg) => write!(f, "invalid options: {msg}"),
        }
    }
}

impl std::error::Error for ArcdError {}

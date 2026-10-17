//! PyO3 bindings for the ARCD demosaicer.
//!
//! Exposed as the private `arcd._arcd` extension module; the public, typed API
//! lives in `python/arcd/__init__.py` (see `pyproject.toml [tool.maturin]`).
//!
//! * `demosaic_u16(raw2d, ...)` — 16-bit Bayer sample input
//! * `demosaic_u8(raw2d, ...)` — 8-bit Bayer sample input
//! * `mosaic(rgb3d_f32, pattern)` — flatten an RGB image to a Bayer mosaic (the
//!   inverse process used to build benchmark inputs)
//! * `classify_u16(raw2d, ...)` / `classify_u8(raw2d, ...)` — Stage 1 region
//!   classification map (0 = Flat, 1 = Texture, 2 = Edge)
//!
//! Both demosaic functions return a `(height, width, 3)` float32 array in the
//! raw sample scale (`0..=px_max`); the classify functions return a
//! `(height, width)` uint8 array of region codes.

use numpy::{PyArray1, PyArray2, PyArrayLike2, PyArrayLike3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use arcd::{classify, demosaic, mosaic, ArcdOptions, BayerPattern, PreInterpolation};

fn pattern_from_str(s: &str) -> PyResult<BayerPattern> {
    BayerPattern::parse(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn pre_interpolation_from_str(s: Option<&str>) -> PyResult<Option<PreInterpolation>> {
    let Some(s) = s else { return Ok(None) };
    match s.trim().to_ascii_lowercase().as_str() {
        "bilinear" => Ok(Some(PreInterpolation::Bilinear)),
        "mhc" | "malvar" => Ok(Some(PreInterpolation::Mhc)),
        other => Err(PyValueError::new_err(format!(
            "unknown pre_interpolation '{other}' (expected 'bilinear' or 'mhc')"
        ))),
    }
}

/// Build validated options from the Python-facing kwargs; `None` keeps the
/// paper defaults (except bit depth, which defaults to the sample container).
fn build_options(
    pattern: &str,
    bit_depth: Option<u8>,
    tau_flat: Option<f32>,
    tau_anis: Option<f32>,
    lambda_ratio: Option<f32>,
    eta_l: Option<f32>,
    eta_c: Option<f32>,
    rho_median: Option<f32>,
    pre_interpolation: Option<&str>,
    fallback_depth: u8,
) -> PyResult<ArcdOptions> {
    let pat = pattern_from_str(pattern)?;
    let mut opts = ArcdOptions::for_bit_depth(pat, bit_depth.unwrap_or(fallback_depth));
    if let Some(v) = tau_flat {
        opts.tau_flat = v;
    }
    if let Some(v) = tau_anis {
        opts.tau_anis = v;
    }
    if let Some(v) = lambda_ratio {
        opts.lambda_ratio = v;
    }
    if let Some(v) = eta_l {
        opts.eta_l = v;
    }
    if let Some(v) = eta_c {
        opts.eta_c = v;
    }
    if let Some(v) = rho_median {
        opts.rho_median = v;
    }
    if let Some(p) = pre_interpolation_from_str(pre_interpolation)? {
        opts.pre_interpolation = p;
    }
    opts.validate().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(opts)
}

fn pyerr(e: arcd::ArcdError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn demosaic_impl<'py>(
    py: Python<'py>,
    data: Vec<u16>,
    shape: (usize, usize),
    opts: &ArcdOptions,
) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
    let (h, w) = shape;
    let out = demosaic(&data, w, h, opts).map_err(pyerr)?;
    let flat = PyArray1::from_vec(py, out);
    flat.reshape([h, w, 3]).map_err(|e: PyErr| e)
}

#[pyfunction]
#[pyo3(signature = (
    raw,
    pattern = "rggb",
    bit_depth = None,
    tau_flat = None,
    tau_anis = None,
    lambda_ratio = None,
    eta_l = None,
    eta_c = None,
    rho_median = None,
    pre_interpolation = None,
))]
fn demosaic_u16<'py>(
    py: Python<'py>,
    raw: PyArrayLike2<u16>,
    pattern: &str,
    bit_depth: Option<u8>,
    tau_flat: Option<f32>,
    tau_anis: Option<f32>,
    lambda_ratio: Option<f32>,
    eta_l: Option<f32>,
    eta_c: Option<f32>,
    rho_median: Option<f32>,
    pre_interpolation: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
    let (h, w) = (raw.shape()[0], raw.shape()[1]);
    if h == 0 || w == 0 {
        return Err(PyValueError::new_err("image must not be empty"));
    }
    let data: Vec<u16> = raw.as_array().iter().copied().collect();
    let opts = build_options(
        pattern,
        bit_depth,
        tau_flat,
        tau_anis,
        lambda_ratio,
        eta_l,
        eta_c,
        rho_median,
        pre_interpolation,
        16,
    )?;
    demosaic_impl(py, data, (h, w), &opts)
}

#[pyfunction]
#[pyo3(signature = (
    raw,
    pattern = "rggb",
    bit_depth = None,
    tau_flat = None,
    tau_anis = None,
    lambda_ratio = None,
    eta_l = None,
    eta_c = None,
    rho_median = None,
    pre_interpolation = None,
))]
fn demosaic_u8<'py>(
    py: Python<'py>,
    raw: PyArrayLike2<u8>,
    pattern: &str,
    bit_depth: Option<u8>,
    tau_flat: Option<f32>,
    tau_anis: Option<f32>,
    lambda_ratio: Option<f32>,
    eta_l: Option<f32>,
    eta_c: Option<f32>,
    rho_median: Option<f32>,
    pre_interpolation: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
    let (h, w) = (raw.shape()[0], raw.shape()[1]);
    if h == 0 || w == 0 {
        return Err(PyValueError::new_err("image must not be empty"));
    }
    let data: Vec<u16> = raw.as_array().iter().map(|&v| u16::from(v)).collect();
    let opts = build_options(
        pattern,
        bit_depth,
        tau_flat,
        tau_anis,
        lambda_ratio,
        eta_l,
        eta_c,
        rho_median,
        pre_interpolation,
        8,
    )?;
    demosaic_impl(py, data, (h, w), &opts)
}

fn classify_impl<'py>(
    py: Python<'py>,
    data: Vec<u16>,
    shape: (usize, usize),
    opts: &ArcdOptions,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let (h, w) = shape;
    // classify() takes the raw samples at f32 precision, exactly like the
    // demosaic pipeline does (`CfaSample::value`).
    let raw: Vec<f32> = data.iter().map(|&v| f32::from(v)).collect();
    let (classes, _edge_count) = classify(&raw, w, h, opts);
    let flat = PyArray1::from_vec(py, classes);
    flat.reshape([h, w]).map_err(|e: PyErr| e)
}

#[pyfunction]
#[pyo3(signature = (raw, pattern = "rggb", bit_depth = None, tau_flat = None, tau_anis = None))]
fn classify_u16<'py>(
    py: Python<'py>,
    raw: PyArrayLike2<u16>,
    pattern: &str,
    bit_depth: Option<u8>,
    tau_flat: Option<f32>,
    tau_anis: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let (h, w) = (raw.shape()[0], raw.shape()[1]);
    if h == 0 || w == 0 {
        return Err(PyValueError::new_err("image must not be empty"));
    }
    // Classification only depends on tau_flat, tau_anis and bit_depth; the
    // remaining options are kept at their defaults (they must still validate).
    let opts = build_options(
        pattern,
        bit_depth,
        tau_flat,
        tau_anis,
        None,
        None,
        None,
        None,
        None,
        16,
    )?;
    classify_impl(py, raw.as_array().iter().copied().collect(), (h, w), &opts)
}

#[pyfunction]
#[pyo3(signature = (raw, pattern = "rggb", bit_depth = None, tau_flat = None, tau_anis = None))]
fn classify_u8<'py>(
    py: Python<'py>,
    raw: PyArrayLike2<u8>,
    pattern: &str,
    bit_depth: Option<u8>,
    tau_flat: Option<f32>,
    tau_anis: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let (h, w) = (raw.shape()[0], raw.shape()[1]);
    if h == 0 || w == 0 {
        return Err(PyValueError::new_err("image must not be empty"));
    }
    let opts = build_options(
        pattern,
        bit_depth,
        tau_flat,
        tau_anis,
        None,
        None,
        None,
        None,
        None,
        8,
    )?;
    let data: Vec<u16> = raw.as_array().iter().map(|&v| u16::from(v)).collect();
    classify_impl(py, data, (h, w), &opts)
}

#[pyfunction]
#[pyo3(signature = (rgb, pattern = "rggb"))]
fn mosaic_api<'py>(
    py: Python<'py>,
    rgb: PyArrayLike3<f32>,
    pattern: &str,
) -> PyResult<Bound<'py, numpy::PyArray2<u16>>> {
    let shape = rgb.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err(
            "rgb must have shape (height, width, 3)",
        ));
    }
    let (h, w) = (shape[0], shape[1]);
    let data: Vec<f32> = rgb.as_array().iter().copied().collect();
    let out = mosaic(&data, w, h, pattern_from_str(pattern)?).map_err(pyerr)?;
    Ok(PyArray1::from_vec(py, out).reshape([h, w]).map_err(|e: PyErr| e)?)
}

/// Package metadata, mirroring the Python wrapper.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _arcd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(demosaic_u16, m)?)?;
    m.add_function(wrap_pyfunction!(demosaic_u8, m)?)?;
    m.add_function(wrap_pyfunction!(classify_u16, m)?)?;
    m.add_function(wrap_pyfunction!(classify_u8, m)?)?;
    m.add_function(wrap_pyfunction!(mosaic_api, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

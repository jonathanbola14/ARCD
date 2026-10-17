"""Pytest smoke tests for the `arcd` Python bindings.

They require the compiled extension: install it with maturin first, e.g.

    python -m venv .venv && . .venv/bin/activate
    pip install maturin numpy pytest
    maturin develop -m crates/arcd-py/Cargo.toml
    pytest crates/arcd-py/tests
"""

from __future__ import annotations

import numpy as np
import pytest

import arcd


def synth(h: int, w: int) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    fx, fy = xs.astype(np.float64), ys.astype(np.float64)
    r = 128 + 55 * np.sin(0.055 * fx) + 30 * np.cos(0.035 * fy)
    g = 120 + 45 * np.sin(0.043 * fy + 0.02 * fx) + 25 * np.cos(0.06 * fx)
    b = 130 + 50 * np.cos(0.038 * fx + 0.025 * fy) + 20 * np.sin(0.05 * fy)
    return np.clip(np.stack([r, g, b], axis=-1), 8, 247).astype(np.float32)


@pytest.fixture(scope="module")
def rgb() -> np.ndarray:
    return synth(64, 96)


def test_mosaic_roundtrip(rgb: np.ndarray) -> None:
    raw = arcd.mosaic(rgb, "rggb")
    assert raw.dtype == np.uint16
    assert raw.shape == rgb.shape[:2]
    out = arcd.demosaic(raw, "rggb")
    assert out.dtype == np.float32 and out.shape == rgb.shape
    rmse = float(np.sqrt(np.mean((out - rgb) ** 2)))
    assert rmse < 8.0, rmse


@pytest.mark.parametrize("pattern", ["rggb", "grbg", "gbrg", "bggr"])
def test_all_patterns_finite(rgb: np.ndarray, pattern: str) -> None:
    out = arcd.demosaic(arcd.mosaic(rgb, pattern), pattern)
    assert np.isfinite(out).all()
    assert (out >= 0).all() and (out <= 65535).all()


def test_mhc_mode_and_kwargs(rgb: np.ndarray) -> None:
    out = arcd.demosaic(
        arcd.mosaic(rgb, "rggb"),
        "rggb",
        pre_interpolation="mhc",
        tau_flat=650.25,
        lambda_ratio=0.5,
        rho_median=0.7,
    )
    assert np.isfinite(out).all()


def test_error_paths(rgb: np.ndarray) -> None:
    with pytest.raises(ValueError):
        arcd.demosaic(arcd.mosaic(rgb, "rggb"), pattern="qqqq")
    with pytest.raises(ValueError):
        arcd.demosaic(arcd.mosaic(rgb, "rggb"), rho_median=1.5)
    with pytest.raises(ValueError):
        arcd.demosaic(arcd.mosaic(rgb, "rggb"), bit_depth=32)
    with pytest.raises(TypeError):
        arcd.demosaic(np.zeros((4, 4)))  # float64 is not a CFA sample dtype


def test_classify_constant_image_is_all_flat() -> None:
    # A pure gray plateau has zero variance: every pixel must classify FLAT.
    rgb = np.full((24, 32, 3), 130.0, dtype=np.float32)
    regions = arcd.classify(arcd.mosaic(rgb, "rggb"))
    assert regions.shape == (24, 32)
    assert regions.dtype == np.uint8
    assert (regions == arcd.RegionClass.FLAT).all()


def test_classify_checkerboard_has_edges() -> None:
    # Worst-case checkerboard; values at the full 16-bit scale (tau_flat is in
    # 8-bit units and is rescaled to the container depth by default).
    (h, w) = (32, 48)
    ys, xs = np.mgrid[0:h, 0:w]
    v = np.where((xs // 4 + ys // 4) % 2 == 0, 0.0, 65535.0)
    rgb = np.repeat(v[..., None], 3, axis=2).astype(np.float32)
    regions = arcd.classify(arcd.mosaic(rgb, "rggb"))
    assert regions.dtype == np.uint8
    assert set(np.unique(regions)).issubset({0, 1, 2})
    assert (regions == arcd.RegionClass.EDGE).any()


def test_classify_u8_input_and_thresholds() -> None:
    # uint8 mosaic; a tighter tau_flat re-marks more pixels as non-Flat.
    gray = np.full((16, 16), np.uint8(0))  # constant black mosaic
    assert (arcd.classify(gray, "bggr") == arcd.RegionClass.FLAT).all()
    with pytest.raises(ValueError):
        arcd.classify(gray, tau_anis=0.5)  # below the valid range


def test_classify_shared_error_paths(rgb: np.ndarray) -> None:
    raw = arcd.mosaic(rgb, "rggb")
    with pytest.raises(ValueError):
        arcd.classify(raw, pattern="qqqq")
    with pytest.raises(ValueError):
        arcd.classify(raw, bit_depth=32)
    with pytest.raises(TypeError):
        arcd.classify(np.zeros((4, 4)))  # float64 is not a CFA sample dtype

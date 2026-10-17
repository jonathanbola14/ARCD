"""Public API for the `arcd` Python package.

The heavy lifting is performed by the compiled `arcd._arcd` extension module,
which wraps the dependency-free Rust core in `crates/arcd`.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from ._arcd import classify_u8 as _classify_u8
from ._arcd import classify_u16 as _classify_u16
from ._arcd import demosaic_u8 as _demosaic_u8
from ._arcd import demosaic_u16 as _demosaic_u16
from ._arcd import mosaic_api as _mosaic
from ._arcd import version as _version

__version__: str = _version()

__all__ = ["ArcdOptions", "RegionClass", "classify", "demosaic", "mosaic", "__version__"]

_KNOWN_PATTERNS = ("rggb", "grbg", "gbrg", "bggr")
_PRE_INTERPOLATIONS = (None, "bilinear", "mhc")


class RegionClass(IntEnum):
    """Region codes of the Stage 1 classification map (paper section 4.1).

    Returned by :func:`classify`: ``0`` = Flat (served directly by the Stage 1
    pre-interpolation), ``1`` = Texture (Stage 2 ratio-corrected green),
    ``2`` = Edge (Stage 3 homogeneity-directed refinement).
    """

    FLAT = 0
    TEXTURE = 1
    EDGE = 2


class ArcdOptions:
    """Parameter bundle for :func:`demosaic` (paper section 4.4).

    Attributes mirror the paper's tunables; ``None`` defers to the built-in
    default. ``tau_flat`` is expressed in 8-bit squared units and is rescaled
    automatically according to ``bit_depth``.

    Parameters
    ----------
    tau_flat:
        Flat-region variance threshold (default 650.25 = 0.01 * 255**2;
        paper range [0, 0.1 * 255**2]).
    tau_anis:
        Edge anisotropy threshold ``max(Gh, Gv) / (min(Gh, Gv) + eps)``
        (default 2.0; paper range [1.5, 5.0]).
    lambda_ratio:
        Ratio/difference blend of Stage 2 (default 0.5, range [0, 1]).
    eta_l, eta_c:
        AHD homogeneity thresholds in CIELAB units (defaults 5.0 / 3.0,
        from the paper's ``eta_homog = (5, 3)``).
    rho_median:
        Median-filter strength for Stage 4 (default 0.7, range [0, 1]).
    """

    __slots__ = (
        "tau_flat",
        "tau_anis",
        "lambda_ratio",
        "eta_l",
        "eta_c",
        "rho_median",
    )

    def __init__(
        self,
        *,
        tau_flat: float | None = None,
        tau_anis: float | None = None,
        lambda_ratio: float | None = None,
        eta_l: float | None = None,
        eta_c: float | None = None,
        rho_median: float | None = None,
    ) -> None:
        self.tau_flat = tau_flat
        self.tau_anis = tau_anis
        self.lambda_ratio = lambda_ratio
        self.eta_l = eta_l
        self.eta_c = eta_c
        self.rho_median = rho_median


def demosaic(
    raw: np.ndarray,
    pattern: str = "rggb",
    *,
    bit_depth: int | None = None,
    tau_flat: float | None = None,
    tau_anis: float | None = None,
    lambda_ratio: float | None = None,
    eta_l: float | None = None,
    eta_c: float | None = None,
    rho_median: float | None = None,
    pre_interpolation: str | None = None,
) -> np.ndarray:
    """Demosaic a Bayer CFA mosaic with the ARCD pipeline.

    Parameters
    ----------
    raw:
        2-D array (``height x width``) of CFA samples, dtype ``uint16`` or
        ``uint8``, row-major (``raw[y, x]``).
    pattern:
        Bayer phase: ``"rggb"`` (default), ``"grbg"``, ``"gbrg"``, ``"bggr"``.
    bit_depth:
        Effective sample depth (1..16). Defaults to 8 for ``uint8`` input and
        16 for ``uint16`` input; scales ``tau_flat`` automatically.
    tau_flat, tau_anis, lambda_ratio, eta_l, eta_c, rho_median:
        Tunable parameters (see :class:`ArcdOptions`); ``None`` keeps defaults.
    pre_interpolation:
        Stage 1 method: ``"bilinear"`` (default) or ``"mhc"``.

    Returns
    -------
    numpy.ndarray
        ``(height, width, 3)`` float32 RGB image, interleaved R, G, B, in the
        raw sample scale (``0..=px_max``). Values are clamped by the algorithm.

    Raises
    ------
    TypeError
        If ``raw`` is neither ``uint16`` nor ``uint8``.
    ValueError
        On empty images, out-of-range parameters or unknown pattern names.
    """
    if not isinstance(pattern, str) or pattern.lower() not in _KNOWN_PATTERNS:
        raise ValueError(f"unknown Bayer pattern {pattern!r}; expected one of {_KNOWN_PATTERNS}")
    if pre_interpolation not in _PRE_INTERPOLATIONS:
        raise ValueError(
            f"unknown pre_interpolation {pre_interpolation!r}; expected one of "
            f"'bilinear' or 'mhc' (or None for the default)"
        )
    arr = np.ascontiguousarray(raw)
    if arr.ndim != 2:
        raise ValueError(f"raw must be 2-D (height, width); got shape {arr.shape}")
    if bit_depth is not None and not 1 <= int(bit_depth) <= 16:
        raise ValueError(f"bit_depth must be in 1..16; got {bit_depth}")
    opts = dict(
        pattern=pattern.lower(),
        bit_depth=None if bit_depth is None else int(bit_depth),
        tau_flat=tau_flat,
        tau_anis=tau_anis,
        lambda_ratio=lambda_ratio,
        eta_l=eta_l,
        eta_c=eta_c,
        rho_median=rho_median,
        pre_interpolation=pre_interpolation,
    )
    if arr.dtype == np.uint16:
        return np.asarray(_demosaic_u16(arr, **opts))
    if arr.dtype == np.uint8:
        return np.asarray(_demosaic_u8(arr, **opts))
    raise TypeError(f"raw dtype must be uint16 or uint8; got {arr.dtype}")


def classify(
    raw: np.ndarray,
    pattern: str = "rggb",
    *,
    bit_depth: int | None = None,
    tau_flat: float | None = None,
    tau_anis: float | None = None,
) -> np.ndarray:
    """Run the Stage 1 region classification of a Bayer CFA mosaic.

    Classification is purely local (paper section 4.1): the 5x5 raw-window
    variance against ``tau_flat`` decides Flat, and the axial-gradient
    anisotropy ``max(Gh, Gv) / (min(Gh, Gv) + eps)`` against ``tau_anis``
    decides Edge vs Texture. This is the same classifier :func:`demosaic`
    applies internally — handy for debugging and mask visualization.

    Parameters
    ----------
    raw:
        2-D array (``height x width``) of CFA samples, dtype ``uint16`` or
        ``uint8``, row-major (``raw[y, x]``).
    pattern:
        Bayer phase: ``"rggb"`` (default), ``"grbg"``, ``"gbrg"``, ``"bggr"``.
        It is part of the option set for symmetry with :func:`demosaic` but
        does not influence the classification thresholds.
    bit_depth:
        Effective sample depth (1..16). Defaults to 8 for ``uint8`` input and
        16 for ``uint16`` input; scales ``tau_flat`` automatically.
    tau_flat, tau_anis:
        The only tunables the classifier reads (see :class:`ArcdOptions`);
        ``None`` keeps the paper defaults.

    Returns
    -------
    numpy.ndarray
        ``(height, width)`` uint8 map of :class:`RegionClass` codes.

    Raises
    ------
    TypeError
        If ``raw`` is neither ``uint16`` nor ``uint8``.
    ValueError
        On empty images, out-of-range parameters or unknown pattern names.
    """
    if not isinstance(pattern, str) or pattern.lower() not in _KNOWN_PATTERNS:
        raise ValueError(f"unknown Bayer pattern {pattern!r}; expected one of {_KNOWN_PATTERNS}")
    arr = np.ascontiguousarray(raw)
    if arr.ndim != 2:
        raise ValueError(f"raw must be 2-D (height, width); got shape {arr.shape}")
    if bit_depth is not None and not 1 <= int(bit_depth) <= 16:
        raise ValueError(f"bit_depth must be in 1..16; got {bit_depth}")
    opts = dict(
        pattern=pattern.lower(),
        bit_depth=None if bit_depth is None else int(bit_depth),
        tau_flat=tau_flat,
        tau_anis=tau_anis,
    )
    if arr.dtype == np.uint16:
        return np.asarray(_classify_u16(arr, **opts))
    if arr.dtype == np.uint8:
        return np.asarray(_classify_u8(arr, **opts))
    raise TypeError(f"raw dtype must be uint16 or uint8; got {arr.dtype}")


def mosaic(rgb: np.ndarray, pattern: str = "rggb") -> np.ndarray:
    """Flatten an RGB image into a Bayer CFA mosaic.

    This is the inverse process used in the paper's experimental protocol to
    generate inputs from ground-truth images ("Generate mosaiced inputs from
    ground-truth images using the Bayer RGGB pattern").

    Parameters
    ----------
    rgb:
        ``(height, width, 3)`` float32 array in the raw sample scale
        (``0..=65535``); values are rounded and clamped to 16-bit.
    pattern:
        Bayer phase of the produced mosaic.

    Returns
    -------
    numpy.ndarray
        ``(height, width)`` uint16 CFA mosaic, feed it to :func:`demosaic`.
    """
    if not isinstance(pattern, str) or pattern.lower() not in _KNOWN_PATTERNS:
        raise ValueError(f"unknown Bayer pattern {pattern!r}; expected one of {_KNOWN_PATTERNS}")
    arr = np.ascontiguousarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"rgb must have shape (height, width, 3); got {arr.shape}")
    return np.asarray(_mosaic(arr, pattern.lower()))

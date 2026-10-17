#!/usr/bin/env python
"""Demosaic a camera RAW file with ARCD, using the `rawpy` toolkit.

`rawpy` wraps LibRaw and can decode sensor formats (DNG, NEF, CR2, ...). This
script extracts the *Bayer mosaic* (what `arcd.demosaic` wants), runs the ARCD
pipeline on it and saves the result as PPM. Optionally it also saves LibRaw's
own processed output as a visual reference.

Note on the comparison: LibFull's postprocess applies white balance, sRGB
conversion and tone curve — the ARCD output stays in the raw sample scale —
so the two images are not algorithmically like-for-like; the reference is
only for eyeballing whether the mosaic was extracted correctly.

Usage (inside ARCD/.venv):

    pip install rawpy                # the only extra dependency
    python scripts/raw_to_ppm.py IMG.RAW             # -> IMG.arcd.ppm (+ .libraw.ppm)
    python scripts/raw_to_ppm.py IMG.RAW -o my --no-reference
    python scripts/raw_to_ppm.py IMG.RAW --pattern gbrg --depth 14

Exit codes: 0 success, 1 argument/processing errors, 2 missing dependencies.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

try:
    import rawpy
except ImportError:
    rawpy = None

import numpy as np

try:
    import arcd
except ImportError:
    arcd = None

_KNOWN_PATTERNS = ("rggb", "grbg", "gbrg", "bggr")

# LibRaw raw_colors codes: 0=Red, 1=First Green, 2=Blue, 3=Second Green.
COLOR_LETTER = {0: "r", 1: "g", 2: "b", 3: "g"}


# ---------------------------------------------------------------------------
# rawpy -> ARCD inputs
# ---------------------------------------------------------------------------

def phase_from_raw_pattern(raw_pattern: np.ndarray) -> str:
    """Map LibRaw's CFA pattern to an ARCD Bayer phase name ('rggb', ...)."""
    pat = np.asarray(raw_pattern)
    if pat.shape != (2, 2):
        raise ValueError(
            f"unsupported CFA pattern shape {pat.shape}; ARCD implements the "
            "4 Bayer phases (2x2). X-Trans / paneled sensors are not covered."
        )
    letters = "".join(COLOR_LETTER[int(c)] for c in pat.flat)  # row-major
    if letters in _KNOWN_PATTERNS:
        return letters
    raise ValueError(f"unexpected CFA pattern {pat.tolist()} ('{letters}')")


def raw_to_bayer(path: str | Path):
    """Read a camera RAW file and return (mosaic_u16, phase, white, black).

    The export normalizes samples to the *full 16-bit scale* by removing the
    black level and stretching the [black, white] interval to [0, 65535], so
    `arcd.demosaic`/`arcd.classify` defaults (bit_depth 16) are meaningful no
    matter the sensor depth (10/12/14 bits). Pixel aspect is NOT applied.
    """
    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible.astype(np.float32)
        phase = phase_from_raw_pattern(raw.raw_pattern)
        white = int(raw.white_level or (1 << 16) - 1)
        black = int(np.max(raw.black_level_per_channel)) if raw.black_level_per_channel is not None else 0
        bits = float(math.ceil(math.log2(white + 1)))
    span = float(max(white - black, 1))
    mosaic = np.clip((mosaic - black) * (65535.0 / span), 0.0, 65535.0)
    return mosaic.astype(np.uint16), phase, white, black


# ---------------------------------------------------------------------------
# Output helper (binary PPM P6, 8-bit; matches arcd-cli's output format family)
# ---------------------------------------------------------------------------

def save_ppm(path: str | Path, rgb: np.ndarray) -> None:
    """Save (H, W, 3) uint8 as a binary PPM (P6)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"save_ppm expects (H, W, 3); got {rgb.shape}")
    with open(path, "wb") as fh:
        fh.write(b"P6\n%d %d\n255\n" % (rgb.shape[1], rgb.shape[0]))
        fh.write(rgb.tobytes())


def to_u8(img: np.ndarray, scale: float | None) -> np.ndarray:
    """Map an image to 8-bit output range (0..=255, rounded and clipped)."""
    img = np.asarray(img, dtype=np.float32)
    if scale is not None:
        img = img * (255.0 / scale)
    return np.clip(img, 0.0, 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demosaic a camera RAW (DNG/NEF/CR2/...) with ARCD."
    )
    p.add_argument("input", help="camera RAW file readable by rawpy/LibRaw")
    p.add_argument("-o", "--output-prefix", default=None,
                   help="output file prefix (default: input name without suffix)")
    p.add_argument("--pattern", default=None, metavar="PHASE",
                   help="force Bayer phase instead of auto-detecting from raw_pattern")
    p.add_argument("--no-reference", action="store_true",
                   help="skip saving LibRaw's own processed output")
    p.add_argument("--quiet", action="store_true", help="only print the file paths")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if rawpy is None or arcd is None:
        missing = "rawpy" if rawpy is None else "arcd"
        print(f"error: missing dependency '{missing}'; activate ARCD/.venv first "
              f"and run 'pip install rawpy'", file=sys.stderr)
        return 2

    args = parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        print(f"error: input file {src} not found", file=sys.stderr)
        return 1
    prefix = Path(args.output_prefix) if args.output_prefix else src.with_suffix("")

    t0 = time.perf_counter()
    mosaic, phase, white, black = raw_to_bayer(src)
    if args.pattern:
        if args.pattern.lower() not in _KNOWN_PATTERNS:
            print(f"error: unknown pattern {args.pattern!r}; expected one of {_KNOWN_PATTERNS}",
                  file=sys.stderr)
            return 1
        phase = args.pattern.lower()
    if not args.quiet:
        print(f"read {src}: {mosaic.shape[0]}x{mosaic.shape[1]} {phase} "
              f"(white={white}, black={black}), {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    rgb = arcd.demosaic(mosaic, phase)
    dt = time.perf_counter() - t1
    mpx = mosaic.size / 1e6
    if not args.quiet:
        print(f"arcd.demosaic: {dt:.3f}s ({dt / mpx:.0f} ms/MPx)")

    out_arcd = prefix.with_suffix(".arcd.ppm")
    save_ppm(out_arcd, to_u8(rgb, scale=65535.0))
    print(out_arcd)

    if not args.no_reference:
        ref_prefix = prefix.with_suffix(".libraw.ppm")
        with rawpy.imread(str(src)) as raw:
            ref = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)
        save_ppm(ref_prefix, to_u8(ref, scale=None))
        print(ref_prefix)

    return 0


if __name__ == "__main__":
    sys.exit(main())

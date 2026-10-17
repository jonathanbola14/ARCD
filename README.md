# ARCD — Adaptive Ratio-Corrected Demosaicing

Reference Rust implementation of **ARCD** (*Adaptive Ratio-Corrected
Demosaicing with Homogeneity-Directed Refinement for Bayer CFA Sensors*), the
staged hybrid demosaicing algorithm proposed in [`Paper.md`](Paper.md).

ARCD demosaices single-sensor Bayer CFA images by classifying each local
region and routing it to the best-suited strategy:

1. **Stage 1 — pre-interpolation & classification.** Fast bilinear (or
   Malvar-He-Cutler) pre-interpolation; local variance (5×5) and axial
   gradients classify every pixel as **flat**, **texture**, or **edge**.
2. **Stage 2 — ratio-corrected green (texture).** RCD-inspired directional
   channel/green ratio candidates blended by measured discrimination weights,
   mixed with difference-based (MHC-style) candidates via `lambda_ratio`.
3. **Stage 3 — homogeneity-directed refinement (edge).** AHD-inspired: full
   horizontal/vertical RGB candidates are converted to CIELAB, 5×5 homogeneity
   maps are averaged over 3×3, and the more homogeneous direction wins.
4. **Stage 4 — red/blue & artifact suppression.** Red/blue by
   color-difference constancy against the final green, followed by a 3×3
   median filter mixed into the `R−G` / `B−G` difference channels
   (`rho_median`).

Flat regions skip Stages 2–4 entirely and receive the Stage 1 result.

## Repository layout

```
Paper.md              the algorithm specification (problem definition)
crates/arcd/          core library: dependency-free Rust, #[forbid(unsafe_code)]
crates/arcd-cli/      `arcd` binary: raw Bayer -> PPM (plain dump or camera RAW)
crates/arcd-py/       PyO3 bindings (built with maturin; Python numpy API)
scripts/              helper scripts (camera RAW files via `rawpy`)
```

## Rust core (`crates/arcd`)

The core crate has **no dependencies** and works on plain slices, so it can be
embedded behind any FFI (PyO3, C ABI, WASM).

```rust
use arcd::{demosaic, mosaic, ArcdOptions, BayerPattern};

// Build a mosaic from an interleaved RGB image (paper's benchmark protocol).
let rgb = vec![200u16, 150, 100]; // one RGGB tile worth of samples
let raw = mosaic(&rgb, 1, 1, BayerPattern::Rggb)?;

// Demosaic: returns interleaved RGB f32 in the raw sample scale.
let opts = ArcdOptions::for_bit_depth(BayerPattern::Rggb, 16);
let out = demosaic(&raw, 1, 1, &opts)?;
assert_eq!(out.len(), 3); // R, G, B
```

Public surface: `demosaic`, `mosaic`, `ArcdOptions`, `BayerPattern`,
`CfaSample`, `bilinear`, `mhc`, `classify`, `RegionClass`, `ArcdError`.

### Parameters (paper §4.4)

| Parameter     | Default              | Range (paper)      | Function                                    |
|---------------|----------------------|--------------------|---------------------------------------------|
| `tau_flat`    | `650.25`             | `[0, 0.1·255²]`    | flat-region variance threshold (8-bit units)|
| `tau_anis`    | `2.0`                | `[1.5, 5.0]`       | edge anisotropy threshold                   |
| `lambda_ratio`| `0.5`                | `[0, 1]`           | ratio ↔ difference blend (Stage 2)          |
| `eta_l/eta_c` | `5.0 / 3.0`          | —                  | CIELAB homogeneity thresholds (Stage 3)     |
| `rho_median`  | `0.7`                | `[0, 1]`           | median-filter strength (Stage 4)            |

## CLI (`crates/arcd-cli`)

```console
$ cargo test --workspace          # 26 unit/integration tests
$ cargo run -p arcd-cli --release -- in.raw \
    --width 1024 --height 768 --depth 16 --pattern rggb -o out.ppm
demosaiced 1024x768 in 126.4 ms (161 ms/MPx)

# camera RAW containers: geometry comes from the file, no --width/--height
$ cargo run -p arcd-cli --release -- IMG_0001.NEF -o out.ppm
camera raw: Nikon ...: 6000x4000, pattern rggb (black 0, white 16383, ...)
demosaiced 6000x4000 in ...
```

Input is either a headerless raw Bayer dump (`u8` with `--depth 8`, `u16` LE
with `--depth 16`, add `--endian be` for big-endian; auto-detected, force with
`--format plain`) or a camera RAW container decoded by the pure-Rust `rawler`
crate (cargo feature `camera-raw`, on by default; `--format camera` to force).
For camera files the active-area crop is applied, the Bayer phase is
auto-detected from the CFA definition (`--pattern` overrides it) and samples
are normalized to the full 16-bit scale (black level removed, so the Stage 1
thresholds are meaningful for any sensor depth) — the same pipeline as
`scripts/raw_to_ppm.py`. Output is binary PPM (P6), 8 bits.

Bayer-only: X-Trans, 4-color (RGBE/CYGM), rotated-Fuji and linear-RGB DNG
inputs are rejected with a pointer to the scripts; those can still be
demosaiced via `scripts/raw_to_ppm.py` (LibRaw). Build the std-only frontend
with `cargo build -p arcd-cli --no-default-features`.

## Python bindings (`crates/arcd-py`)

```console
$ python -m venv .venv && . .venv/bin/activate
$ pip install maturin numpy pytest
$ maturin develop -m crates/arcd-py/Cargo.toml     # or: maturin build --release
$ pytest crates/arcd-py/tests                      # 11 binding tests
```

```python
import numpy as np
import arcd

rgb = ...                                   # (H, W, 3) float32, raw scale
raw = arcd.mosaic(rgb, "rggb")              # (H, W) uint16 Bayer mosaic
out = arcd.demosaic(
    raw, "rggb",
    tau_flat=650.25, tau_anis=2.0,          # paper defaults
    lambda_ratio=0.5, eta_l=5.0, eta_c=3.0, rho_median=0.7,
)                                           # (H, W, 3) float32 RGB

regions = arcd.classify(raw, "rggb")        # (H, W) uint8 Stage 1 region map
n_edges = int((regions == arcd.RegionClass.EDGE).sum())  # flat/texture/edge codes
```

`demosaic` and `classify` accept `uint8` or `uint16` 2-D arrays; `bit_depth`
defaults to the dtype's natural depth and scales `tau_flat` automatically. The
wrapper is typed (`py.typed`) so mypy sees the signatures.

### Camera RAW files (`scripts/raw_to_ppm.py`, needs `rawpy`)

```console
$ . .venv/bin/activate
$ pip install rawpy
$ python scripts/raw_to_ppm.py IMG_0001.RAW           # -> IMG_0001.arcd.ppm (+ .libraw.ppm)
$ python scripts/raw_to_ppm.py IMG_0001.RAW --no-reference
```

Extracts the Bayer mosaic via rawpy/LibRaw (DNG, NEF, CR2, ...), normalizes it
to the 16-bit scale (black level removed), auto-detects the Bayer phase from
the file's CFA pattern (`--pattern` to override), runs `arcd.demosaic` and
saves an 8-bit PPM. With no `--no-reference`, LibRaw's own processed output is
saved alongside for eyeballing (it applies white balance/sRGB, so the two are
not algorithmically comparable).

## Fidelity to the paper and disambiguation notes

The pseudocode in §4.2 is followed stage by stage; a few details that the paper
deliberately leaves open ("see the RCD source code for exact coefficients")
are closed here and documented in the source:

* **Measured ratios** (Stage 2): `rho(p) = S(p)/G_bil(p)` per quarter-color
  site, sampled parity-preserving at distance 2 (`crates/arcd/src/green.rs`,
  module docs). Directional candidates are `S/rho_h`, `S/rho_v` with
  `rho_{h,v} = (2·rho_c + rho_l + rho_r)/4`; `S/rho_c` alone collapses to the
  bilinear estimate, so the neighbor ratios are what make it directional.
* **Discrimination weights**: `w_h = 1/(1+V_stat)`, `w_v = 1/(1+H_stat)` with
  `V_stat = Dh²/(Dh²+Dv²+eps)`, `H_stat = Dv²/(Dh²+Dv²+eps)` — consistent with
  the paper's formulas (a *vertical* edge penalizes horizontal interpolation).
* **Stage 3 candidates**: full-image horizontal/vertical green candidates with
  red/blue propagated by color-difference constancy; computed only when at
  least one EDGE pixel exists (fast path per §4.5).
* **MHC kernels**: Malvar-He-Cutler Table 1 coefficients, normalized to sum 1
  (verified by a constant-image test); borders fall back to color-filtered
  bilinear where the 5×5 kernel would alias neighbor colors.
* **Mosaic phases**: the paper assumes RGGB, but all four Bayer phases are
  supported (the algorithm only uses phase-independent geometry invariants).

## Performance notes

Single-threaded scalar Rust reaches ~107 ms/MPx release-built on a 1024×768
16-bit image — in line with the paper's projected ~120 ms/MPx before
SIMD/GPU/tiling optimizations (§4.5 reports "Optimizations" as future work
here, e.g. rayon tiling or AVX2 for the 5×5/3×3 filters).

## Limitations

Matches §8.1 of the paper: no experimental validation on Kodak/McMaster yet
(see §5 for the protocol; this implementation provides `mosaic()` to generate
the inputs), Bayer-only, fixed default parameters, no noise estimation.

## License

MIT (matches the workspace `license` field).

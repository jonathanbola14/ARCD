# AGENTS.md

Rust implementation of the ARCD demosaicing algorithm specified by `Paper.md`.
`Paper.md` is the spec authored by the user — never edit it. Where the paper
leaves details open, the choices made are documented in code (see "Where the
truth lives" below), not invented ad hoc.

## Layout & boundaries

- Workspace (`Cargo.toml`): `crates/arcd` = dep-free core library,
  `crates/arcd-cli` = the `arcd` binary (headerless dump **and** camera RAW →
  PPM).
- `crates/arcd-cli` decodes camera RAW via `rawler 0.8` behind the default-on
  `camera-raw` feature; `--no-default-features` keeps the binary std-only (plain
  dumps still work, camera files error with a hint). Only the core must stay
  dependency-free — CLI-only deps are fine.
- `crates/arcd-py` (PyO3/numpy bindings) is **excluded** from the workspace on
  purpose so plain `cargo build/test` never needs a Python toolchain. It has
  its own `Cargo.lock` and must be built via maturin only.
- The git repo root is `ARCD/` itself (standalone repo; the parent
  `~/Documentos/Projetos` has a separate repo holding many unrelated projects —
  ignore it). Work from `ARCD/` and keep paths relative to it.

## Commands (all verified)

```bash
cargo test --workspace                #core + cli only; python NOT covered
cargo test -p arcd constant_image_all_patterns_exact   # single-test filter
cargo clippy --workspace              # keep at 0 warnings
# std-only CLI (no rawler): camera RAW paths compiled out, plain dumps only
cargo build -p arcd-cli --no-default-features
cargo test  -p arcd-cli --no-default-features

# Python bindings (venv: ARCD/.venv — ignored by .gitignore)
python3 -m venv .venv && . .venv/bin/activate
pip install maturin numpy pytest
maturin develop -m crates/arcd-py/Cargo.toml    # rebuild+reinstall after Rust edits
pytest crates/arcd-py/tests                     # 11 binding tests
```

`maturin develop` requires the venv active or `VIRTUAL_ENV` set. Alternative:
`maturin build --release -m crates/arcd-py/Cargo.toml` then pip-install the
wheel from `crates/arcd-py/target/wheels/`.

## Module naming is a 3-place lockstep

`crates/arcd-py/pyproject.toml [tool.maturin] module-name = "arcd._arcd"` ↔
`crates/arcd-py/Cargo.toml [lib] name = "_arcd"` ↔ `#[pymodule] fn _arcd`.
Renames require updating all three plus the imports in
`crates/arcd-py/python/arcd/__init__.py` (the public typed API wrapper). Keep
the `numpy` crate version in lockstep with `pyo3` (both 0.30 today).

## Algorithm invariants (the test suite enforces them)

- Bayer geometry is the contract: a distance-2 axial move preserves site
  color; axial neighbors of R/B sites are green. Border code MUST sample
  parity-preserving sites (`buffer::neighbor2`) or filter neighbors by the
  color of the **clamped** coordinates (`axial_mean`, `axial_pair_green` in
  `stage1.rs`). Naive clamped indexing aliases the wrong channel at borders —
  constant-image tests then fail with small channel offsets (e.g. R=175 where
  150 was expected).
- MHC 5×5 kernels in `stage1.rs` are normalized to sum 1 and must be exact on
  constant images. `mhc_kernels_sum_to_one`,
  `bilinear_and_mhc_exact_on_constant_image` and
  `constant_image_all_patterns_exact` are the canaries for any kernel edit.
- `ArcdOptions::tau_flat` is always expressed in 8-bit squared units;
  `demosaic`/`classify` rescale it internally by `bit_depth` via `scale()`.
  Never pre-scale it (double-scaling is a known bug class here).
- Stage 3 (AHD candidate + homogeneity machinery) is skipped entirely when the
  image has no EDGE pixel — keep that fast path.

## CLI / camera RAW (`crates/arcd-cli`)

- **The dark/green PPM is expected.** The camera path normalizes to 16-bit
  (`ArcdOptions::for_bit_depth(_, 16)`) and outputs raw sample scale; without
  `--white-balance --gamma srgb` it is dark and green-cast (the sensor's own G
  response). That is not a demosaic bug — do not compensate in the core.
- **Phase is re-derived twice, and parity is dimension-sensitive.** Cropping on
  an odd row/column rotates the Bayer phase (`CFA::shift`); the EXIF orientation
  rotates it again. Any crop/rotate must transform the 2×2 phase letters with the
  *same* index map as the plane, using the plane's actual (possibly odd)
  dimensions — odd sizes flip parity differently. Tests live in `camera_raw.rs`.
- **`rawler::Orientation::to_flips() -> (transpose, hflip, vflip)` applies the
  flips BEFORE the transpose** (EXIF 6 / `Rotate90` = transpose∘vflip = 90° CW).
  Preserve that order or portrait files come out mirrored.
- **Match `scripts/raw_to_ppm.py`** (rawpy/LibRaw) for extraction choices; the
  list lives in `camera_raw.rs` module docs. Unsupported sensors (X-Trans,
  4-color RGBE/CYGM, rotated Fuji, linear-RGB DNG) error with a pointer to it.

## Conventions

- Core crate stays dependency-free, `#![forbid(unsafe_code)]`, plain-slice API
  returning owned `Vec<f32>`; `f32` math; values in raw scale `0..=px_max`.
  No rayon/SIMD on purpose (single-threaded scalar, ~100–160 ms/MPx release).
- All tests are deterministic (synthetic images + LCG noise): no fixtures, no
  network, no snapshot files. A large relative jump in performance or RMSE on
  the smooth-image round-trip is an algorithmic regression, not tuning.

## Where the truth lives

- `Paper.md` §4.1–4.5 — spec.
- `crates/arcd/src/green.rs` module docs — "Interpretation notes" resolving
  paper ambiguities (ratio candidates, discrimination weights); README
  §"Fidelity to the paper" summarizes them. Change code + notes together.
- `crates/arcd-cli/src/camera_raw.rs` module docs — camera RAW pipeline choices
  (crop, normalization, phase, orientation, WB) and their `rawler` contract;
  README §CLI summarizes them. Change code + notes together.
- `README.md` — user-facing commands; re-verify any command before updating it.

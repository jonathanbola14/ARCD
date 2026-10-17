//! `arcd` — command-line frontend for the ARCD demosaicer.
//!
//! Reads a raw Bayer mosaic and writes a binary PPM (P6) RGB image. Two input
//! kinds are supported:
//!
//! * **Headerless raw data** (8- or 16-bit samples, no header): the `std`-only
//!   path; it doubles as the simplest example of embedding the `arcd` core.
//! * **Camera RAW containers** (DNG, NEF, CR2, ARW, RAF, ...): extracted to a
//!   Bayer mosaic and normalized to the full 16-bit scale (black level
//!   removed). Requires the optional cargo feature `camera-raw` (enabled by
//!   default); built with `--no-default-features` the binary is std-only and
//!   camera files error out with a hint.
//!
//! ```text
//! # headerless input: the geometry is part of the contract
//! arcd input.raw --width 512 --height 384 --pattern rggb --depth 16 -o out.ppm
//! # camera RAW input: the geometry comes from the container
//! arcd IMG_0001.NEF -o out.ppm
//! ```

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use arcd::{demosaic, ArcdOptions, BayerPattern, PreInterpolation};

#[cfg(feature = "camera-raw")]
mod camera_raw;

/// A Bayer mosaic extracted from a camera RAW container, normalized to the
/// full 16-bit scale (black level removed, `[black, white]` → `[0, 65535]`).
struct CameraRawFrame {
    width: usize,
    height: usize,
    pattern: BayerPattern,
    samples: Vec<u16>,
    /// cleaned-up make/model, for the status line
    make_model: String,
    /// sensor bit depth before normalization, for the status line
    native_bps: usize,
    /// black/white levels used for the normalization, for the status line
    black: f32,
    white: f32,
}

#[cfg(feature = "camera-raw")]
fn load_camera_frame(
    path: &Path,
    phase_override: Option<BayerPattern>,
) -> Result<CameraRawFrame, String> {
    camera_raw::load_camera_raw(path, phase_override)
}

#[cfg(not(feature = "camera-raw"))]
fn load_camera_frame(
    _path: &Path,
    _phase_override: Option<BayerPattern>,
) -> Result<CameraRawFrame, String> {
    Err("this binary was built without camera RAW support; rebuild with \
         `cargo build -p arcd-cli --features camera-raw` (enabled by default)"
        .to_string())
}

/// How the input file should be interpreted.
#[derive(Clone, Copy)]
enum InputFormat {
    /// headerless raw Bayer dump (--width/--height required)
    Plain,
    /// camera RAW container (DNG, NEF, CR2, ...), handled by `rawler`
    Camera,
}

impl InputFormat {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "plain" => Ok(Self::Plain),
            "camera" => Ok(Self::Camera),
            other => Err(format!("unknown --format '{other}' (plain or camera)")),
        }
    }
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    /// plain input only; container-provided for camera RAW
    width: Option<usize>,
    height: Option<usize>,
    /// plain default: rggb; camera RAW default: CFA-derived (--pattern overrides)
    pattern: Option<BayerPattern>,
    /// plain input only
    depth: Option<u8>,
    /// plain input only
    endian_le: Option<bool>,
    format: Option<InputFormat>,
    mhc: bool,
    tau_flat: Option<f32>,
    tau_anis: Option<f32>,
    lambda_ratio: Option<f32>,
    rho_median: Option<f32>,
}

const USAGE: &str = "\
arcd - Adaptive Ratio-Corrected Demosaicing

USAGE:
    arcd <input> --width W --height H [options]

    <input> is either a headerless raw Bayer dump (--width/--height
    required) or a camera RAW container (DNG, NEF, CR2, ARW, RAF, ...
    auto-detected; geometry comes from the container).

OPTIONS:
    -o, --out <file>        output file (PPM P6); default: <input>.ppm
    --format <plain|camera> force the input kind; default: auto-detect
    --width <n>             image width in pixels (plain input only)
    --height <n>            image height in pixels (plain input only)
    --pattern <p>           Bayer phase: rggb (plain default), grbg, gbrg,
                            bggr; for camera input it overrides the CFA
    --depth <bits>          sample depth for plain input: 8 or 16 (default 16)
    --endian <le|be>        byte order for 16-bit plain input (default le)
    --mhc                   use Malvar-He-Cutler pre-interpolation
    --tau-flat <f>          flat-region variance threshold (8-bit units)
    --tau-anis <f>          edge anisotropy threshold
    --lambda-ratio <f>      ratio/difference blend (Stages 2)
    --rho-median <f>        median filter strength (Stage 4)
    -h, --help              this help
";

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0usize;
    let take = |i: &mut usize, flag: &str| -> Result<&str, String> {
        let r = argv
            .get(*i)
            .map(|s| s.as_str())
            .ok_or_else(|| format!("missing value for {flag}"));
        if r.is_ok() {
            *i += 1; // consume the value token
        }
        r
    };

    let mut args = Args {
        input: PathBuf::new(),
        output: PathBuf::new(),
        width: None,
        height: None,
        pattern: None,
        depth: None,
        endian_le: None,
        format: None,
        mhc: false,
        tau_flat: None,
        tau_anis: None,
        lambda_ratio: None,
        rho_median: None,
    };
    let mut have_out = false;

    while i < argv.len() {
        let arg = argv[i].clone();
        i += 1;
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "-o" | "--out" => {
                args.output = PathBuf::from(take(&mut i, "--out")?);
                have_out = true;
            }
            "--format" => {
                args.format = Some(InputFormat::parse(take(&mut i, "--format")?)?);
            }
            "--width" => {
                args.width = Some(take(&mut i, "--width")?.parse().map_err(|_| "invalid --width".to_string())?);
            }
            "--height" => {
                args.height = Some(take(&mut i, "--height")?.parse().map_err(|_| "invalid --height".to_string())?);
            }
            "--pattern" => {
                args.pattern = Some(
                    BayerPattern::parse(take(&mut i, "--pattern")?).map_err(|e| e.to_string())?,
                );
            }
            "--depth" => {
                args.depth = Some(take(&mut i, "--depth")?.parse().map_err(|_| "invalid --depth".to_string())?);
            }
            "--endian" => match take(&mut i, "--endian")? {
                "le" => args.endian_le = Some(true),
                "be" => args.endian_le = Some(false),
                other => return Err(format!("unknown --endian '{other}' (le or be)")),
            },
            "--mhc" => args.mhc = true,
            "--tau-flat" => {
                args.tau_flat =
                    Some(take(&mut i, "--tau-flat")?.parse().map_err(|_| "invalid --tau-flat")?)
            }
            "--tau-anis" => {
                args.tau_anis =
                    Some(take(&mut i, "--tau-anis")?.parse().map_err(|_| "invalid --tau-anis")?)
            }
            "--lambda-ratio" => {
                args.lambda_ratio = Some(
                    take(&mut i, "--lambda-ratio")?
                        .parse()
                        .map_err(|_| "invalid --lambda-ratio")?,
                )
            }
            "--rho-median" => {
                args.rho_median = Some(
                    take(&mut i, "--rho-median")?
                        .parse()
                        .map_err(|_| "invalid --rho-median")?,
                )
            }
            other => {
                if other.starts_with('-') {
                    return Err(format!("unknown option '{other}'"));
                }
                if !args.input.as_os_str().is_empty() {
                    return Err(format!("unexpected extra argument '{other}'"));
                }
                args.input = PathBuf::from(other);
            }
        }
    }
    if args.input.as_os_str().is_empty() {
        return Err("missing input file (positional argument)".to_string());
    }
    if !have_out {
        let mut p = args.input.clone().into_os_string();
        p.push(".ppm");
        args.output = PathBuf::from(p);
    }
    Ok(args)
}

/// Cheap magic-byte sniff to pick the default path for the input file. This is
/// deliberately coarse: `--format plain` is the escape hatch for a plain dump
/// whose first bytes happen to match a magic (e.g. `II`), and `--format
/// camera` for exotic containers the table does not know.
fn looks_like_camera_raw(head: &[u8]) -> bool {
    match head {
        // little/big-endian TIFF family: DNG, NEF, CR2, ARW, ORF, PEF, RW2, ...
        [b'I', b'I', ..] | [b'M', b'M', ..] => true,
        // Fujifilm RAF ("FUJIFILMCCD-RAW")
        [b'F', b'U', b'J', b'I', ..] => true,
        // Minolta MRW / Sigma X3F
        [0x00, b'M', b'R', b'M', ..] | [b'F', b'O', b'V', b'b', ..] => true,
        // Canon CR3: ISO-BMFF box with the 'ftyp' brand at offset 4
        [_, _, _, _, b'f', b't', b'y', b'p', ..] => true,
        _ => false,
    }
}

fn sniff_input_format(path: &PathBuf) -> InputFormat {
    let mut buf = [0u8; 16];
    // Best effort: an unreadable file falls through to the plain path, whose
    // error messages ("cannot read", size mismatch) are the actionable ones.
    let head: &[u8] = match fs::File::open(path) {
        Ok(mut f) => match f.read(&mut buf) {
            Ok(n) => &buf[..n],
            Err(_) => &[],
        },
        Err(_) => &[],
    };
    if looks_like_camera_raw(head) {
        InputFormat::Camera
    } else {
        InputFormat::Plain
    }
}

/// Assemble the `arcd` core options from CLI flags (pattern/depth differ per
/// input kind, the rest is shared).
fn build_opts(pattern: BayerPattern, depth: u8, args: &Args) -> Result<ArcdOptions, String> {
    let mut opts = ArcdOptions::for_bit_depth(pattern, depth);
    opts.pre_interpolation = if args.mhc {
        PreInterpolation::Mhc
    } else {
        PreInterpolation::Bilinear
    };
    if let Some(v) = args.tau_flat {
        opts.tau_flat = v;
    }
    if let Some(v) = args.tau_anis {
        opts.tau_anis = v;
    }
    if let Some(v) = args.lambda_ratio {
        opts.lambda_ratio = v;
    }
    if let Some(v) = args.rho_median {
        opts.rho_median = v;
    }
    opts.validate().map_err(|e| format!("invalid options: {e}"))?;
    Ok(opts)
}

fn read_samples(path: &PathBuf, depth: u8, endian_le: bool, len: usize) -> Result<Vec<u16>, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path:?}: {e}"))?;
    match depth {
        8 => {
            if bytes.len() != len {
                return Err(format!(
                    "expected {len} bytes (8-bit samples), got {}",
                    bytes.len()
                ));
            }
            Ok(bytes.into_iter().map(u16::from).collect())
        }
        16 => {
            if bytes.len() != 2 * len {
                return Err(format!(
                    "expected {len} 16-bit samples ({} bytes), got {} bytes",
                    len,
                    bytes.len()
                ));
            }
            let mut out = Vec::with_capacity(len);
            for idx in 0..len {
                let b0 = bytes[2 * idx];
                let b1 = bytes[2 * idx + 1];
                let v = if endian_le {
                    u16::from_le_bytes([b0, b1])
                } else {
                    u16::from_be_bytes([b0, b1])
                };
                out.push(v);
            }
            Ok(out)
        }
        d => Err(format!("unsupported depth {d} (use 8 or 16)")),
    }
}

fn write_ppm(path: &PathBuf, width: usize, height: usize, rgb: &[f32], px_max: f32) -> Result<(), String> {
    let mut data = Vec::with_capacity(3 * width * height);
    for v in rgb {
        let u = (v * 255.0 / px_max).round().clamp(0.0, 255.0) as u8;
        data.push(u);
    }
    let file = fs::File::create(path).map_err(|e| format!("cannot create {path:?}: {e}"))?;
    let mut w = std::io::BufWriter::new(file);
    write!(w, "P6\n{width} {height}\n255\n").map_err(|e| e.to_string())?;
    w.write_all(&data).map_err(|e| e.to_string())?;
    Ok(())
}

/// Shared tail of the pipeline: demosaic, timing line, PPM write.
fn demosaic_and_write(
    width: usize,
    height: usize,
    opts: &ArcdOptions,
    samples: &[u16],
    output: &PathBuf,
) -> ExitCode {
    let t0 = Instant::now();
    let rgb = match demosaic(samples, width, height, opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: demosaic failed: {e}");
            return ExitCode::from(2);
        }
    };
    let elapsed = t0.elapsed().as_secs_f64();
    let len = width * height;
    let mpix = len as f64 / 1.0e6;
    eprintln!(
        "demosaiced {}x{} in {:.1} ms ({:.0} ms/MPx)",
        width,
        height,
        elapsed * 1000.0,
        elapsed * 1000.0 / mpix
    );

    if let Err(e) = write_ppm(output, width, height, &rgb, opts.px_max()) {
        eprintln!("error: {e}");
        return ExitCode::from(2);
    }
    println!("wrote {}", output.display());
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let format = args.format.unwrap_or_else(|| sniff_input_format(&args.input));

    match format {
        InputFormat::Plain => {
            let width = match args.width {
                Some(w) => w,
                None => {
                    eprintln!("error: missing --width");
                    return ExitCode::from(2);
                }
            };
            let height = match args.height {
                Some(h) => h,
                None => {
                    eprintln!("error: missing --height");
                    return ExitCode::from(2);
                }
            };
            let pattern = args.pattern.unwrap_or(BayerPattern::Rggb);
            let depth = args.depth.unwrap_or(16);
            let endian_le = args.endian_le.unwrap_or(true);
            let opts = match build_opts(pattern, depth, &args) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let len = width * height;
            let samples = match read_samples(&args.input, depth, endian_le, len) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            demosaic_and_write(width, height, &opts, &samples, &args.output)
        }
        InputFormat::Camera => {
            if args.width.is_some()
                || args.height.is_some()
                || args.depth.is_some()
                || args.endian_le.is_some()
            {
                eprintln!(
                    "error: --width/--height/--depth/--endian only apply to plain (headerless) \
                     input; a camera RAW container provides them"
                );
                return ExitCode::from(2);
            }
            let frame = match load_camera_frame(&args.input, args.pattern) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("error: {e}\n(if this is a headerless dump, retry with --format plain)");
                    return ExitCode::from(2);
                }
            };
            let pattern_name = match frame.pattern {
                BayerPattern::Rggb => "rggb",
                BayerPattern::Grbg => "grbg",
                BayerPattern::Gbrg => "gbrg",
                BayerPattern::Bggr => "bggr",
            };
            eprintln!(
                "camera raw: {}: {}x{}, pattern {pattern_name} (black {:.0}, white {:.0}, \
                 {}-bit native -> 16-bit)",
                frame.make_model, frame.width, frame.height, frame.black, frame.white,
                frame.native_bps
            );
            let opts = match build_opts(frame.pattern, 16, &args) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            demosaic_and_write(
                frame.width,
                frame.height,
                &opts,
                &frame.samples,
                &args.output,
            )
        }
    }
}

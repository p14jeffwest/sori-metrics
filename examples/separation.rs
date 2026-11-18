//! Source-separation scores of WAV files: BSS Eval v4 image criteria per window and their
//! medians, and SI-SDR per source.
//!
//! ```text
//! cargo run --release -p sori-metrics --example separation -- [--tsv] REF1 EST1 [REF2 EST2 ...]
//! ```
//!
//! Each `REF`/`EST` pair is one source (reference image and estimate), all of the same rate,
//! channel count and length. With `--tsv`, prints every window
//! (`source window start sdr isr sir sar`) and then the SI-SDR lines
//! (`si source channel si_sdr si_sdr_zero_mean`); otherwise one summary line per source.

use sori_metrics::separation::{bss_eval, scale_invariant, Options};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let tsv = args.first().is_some_and(|a| a == "--tsv");
    if tsv {
        args.remove(0);
    }
    if args.is_empty() || args.len() % 2 != 0 {
        eprintln!("usage: separation [--tsv] REF1 EST1 [REF2 EST2 ...]");
        std::process::exit(2);
    }
    let read = |p: &String| sori_metrics::wav::read(std::path::Path::new(p)).expect(p);
    let refs: Vec<_> = args.iter().step_by(2).map(read).collect();
    let ests: Vec<_> = args.iter().skip(1).step_by(2).map(read).collect();
    let (rate, ch) = (refs[0].sample_rate, refs[0].channels);
    let r: Vec<&[f32]> = refs.iter().map(|a| a.samples.as_slice()).collect();
    let e: Vec<&[f32]> = ests.iter().map(|a| a.samples.as_slice()).collect();
    let before = peak_mib();
    let t0 = std::time::Instant::now();
    let bss = bss_eval(&r, &e, ch, &Options::new(rate)).expect("bss_eval");
    let t_bss = t0.elapsed();
    let after_bss = peak_mib();
    let t1 = std::time::Instant::now();
    let si = scale_invariant(&r, &e, ch, false).expect("si-sdr");
    let si0 = scale_invariant(&r, &e, ch, true).expect("si-sdr");
    let t_si = t1.elapsed() / 2;
    if tsv {
        for (j, s) in bss.iter().enumerate() {
            for (w, f) in s.frames.iter().enumerate() {
                println!(
                    "{j}\t{w}\t{}\t{}\t{}\t{}\t{}",
                    f.start, f.sdr, f.isr, f.sir, f.sar
                );
            }
        }
        for (j, (a, b)) in si.iter().zip(&si0).enumerate() {
            for (c, (x, y)) in a.channels.iter().zip(&b.channels).enumerate() {
                println!("si\t{j}\t{c}\t{}\t{}", x.si_sdr, y.si_sdr);
            }
        }
    } else {
        for (j, (s, v)) in bss.iter().zip(&si).enumerate() {
            let m = s.median();
            println!(
                "source {j}: SDR {:.3} ISR {:.3} SIR {:.3} SAR {:.3} dB (median of {} windows), SI-SDR {:.3} dB",
                m.sdr,
                m.isr,
                m.sir,
                m.sar,
                s.frames.len(),
                v.mean.si_sdr
            );
        }
        eprintln!(
            "bss_eval {:.2} s, SI-SDR {:.2} s; peak resident {before} MiB with the inputs loaded, \
             {after_bss} MiB after bss_eval, {} MiB after SI-SDR",
            t_bss.as_secs_f64(),
            t_si.as_secs_f64(),
            peak_mib()
        );
    }
}

/// Peak resident memory of the process in MiB (Linux `VmHWM`; 0 elsewhere).
fn peak_mib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .unwrap_or(0)
        / 1024
}

//! Time and peak memory of the metrics on one reference and one or more test signals.
//!
//! ```text
//! cargo run --release --example footprint -- <op> <reference> <test>... [--channels N] [--runs N]
//! ```
//!
//! Inputs are WAV files or raw interleaved `f32` little-endian files (`.f32`, 48 kHz,
//! `--channels`, default 2). Operations:
//!
//! - `twof`, `visqol`: one-shot score of the reference against each test, one after another;
//! - `twof-ref`, `visqol-ref`: build a `Reference` once, then score every test with it;
//! - `twof-vs-ref`, `visqol-vs-ref`: the two above alternately in one process (one-shot,
//!   then reference, `--runs` times each), so both see the same machine state; prints the
//!   two medians and their ratio.
//!
//! Each operation runs `--runs` times (default 5). Prints one tab-separated line: operation,
//! median / min / max milliseconds, peak resident memory of the process in KiB (Linux
//! `VmHWM`, 0 elsewhere) and the scores. Run one operation per process to get its peak.

#![allow(clippy::doc_markdown)]

use std::io::Read;
use std::time::Instant;

fn load(path: &str, channels: usize) -> (Vec<f32>, usize) {
    if std::path::Path::new(path)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("f32"))
    {
        // Read straight into the sample buffer, a megabyte at a time.
        let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let n = usize::try_from(f.metadata().expect("metadata").len()).expect("size") / 4;
        let mut out = Vec::with_capacity(n);
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let got = f.read(&mut buf).expect("read");
            if got == 0 {
                break;
            }
            assert_eq!(got % 4, 0, "partial sample (short read) in {path}");
            out.extend(
                buf[..got]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        (out, channels)
    } else {
        let a = sori_metrics::wav::read(path.as_ref()).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!(a.sample_rate, 48_000, "{path}: 48 kHz required");
        (a.samples, a.channels)
    }
}

fn peak_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut opt = |name: &str, default: usize| -> usize {
        match args.iter().position(|a| a == name) {
            Some(i) => {
                let v = args.remove(i + 1).parse().expect("number");
                args.remove(i);
                v
            }
            None => default,
        }
    };
    let (channels, runs) = (opt("--channels", 2), opt("--runs", 5));
    assert!(
        args.len() >= 3,
        "usage: footprint <op> <reference> <test>..."
    );
    let op = args[0].clone();
    let (reference, ch) = load(&args[1], channels);
    let tests: Vec<Vec<f32>> = args[2..].iter().map(|p| load(p, ch).0).collect();
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    if let Some(base) = op.strip_suffix("-vs-ref") {
        let (mut one, mut reuse) = (Vec::new(), Vec::new());
        for _ in 0..runs {
            for (name, times) in [
                (base.to_string(), &mut one),
                (format!("{base}-ref"), &mut reuse),
            ] {
                let t0 = Instant::now();
                std::hint::black_box(run(&name, &reference, &tests, ch));
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        let (a, b) = (median(&mut one), median(&mut reuse));
        println!("{op}\t{a:.1}\t{b:.1}\t{:.3}\t{}", b / a, peak_kib());
        return;
    }
    let mut times = Vec::new();
    let mut scores = Vec::new();
    for _ in 0..runs {
        let t0 = Instant::now();
        scores = run(&op, &reference, &tests, ch);
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mid = median(&mut times);
    let values: Vec<String> = scores.iter().map(|v| format!("{v:.17e}")).collect();
    println!(
        "{op}\t{mid:.1}\t{:.1}\t{:.1}\t{}\t{}",
        times[0],
        times[times.len() - 1],
        peak_kib(),
        values.join("\t")
    );
}

/// Scores of every test against the reference with operation `op`.
fn run(op: &str, reference: &[f32], tests: &[Vec<f32>], ch: usize) -> Vec<f64> {
    match op {
        "twof" => tests
            .iter()
            .map(|t| {
                sori_metrics::twof_score(reference, t, 48_000, ch)
                    .unwrap()
                    .mushra
            })
            .collect(),
        "visqol" => tests
            .iter()
            .map(|t| {
                sori_metrics::visqol_score(reference, t, 48_000, ch)
                    .unwrap()
                    .moslqo
            })
            .collect(),
        "twof-ref" => {
            let r = sori_metrics::twof::Reference::new(reference, 48_000, ch).unwrap();
            tests.iter().map(|t| r.score(t).unwrap().mushra).collect()
        }
        "visqol-ref" => {
            let r = sori_metrics::visqol::Reference::new(reference, 48_000, ch).unwrap();
            tests.iter().map(|t| r.score(t).unwrap().moslqo).collect()
        }
        o => panic!("unknown operation {o}"),
    }
}

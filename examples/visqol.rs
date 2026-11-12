//! ViSQOL v3 audio-mode compatible MOS on WAV pairs (48 kHz, channels averaged to mono).
//!
//! ```text
//! cargo run --release --example visqol -- <reference.wav> <test.wav>
//! cargo run --release --example visqol -- --batch <pairs.tsv>
//! ```
//!
//! `pairs.tsv` holds one `reference<TAB>test` pair per line. Prints one tab-separated line per
//! pair: `moslqo  vnsim  lag_samples  seconds`; with `VISQOL_FV=1` the 32 per-band
//! similarities (`fvnsim`) follow.

#![allow(clippy::doc_markdown)]

use sori_metrics::visqol::Analyzer;
use std::fmt::Write as _;
use std::time::Instant;

fn mono(path: &str) -> Result<Vec<f64>, String> {
    let a = sori_metrics::wav::read(path.as_ref()).map_err(|e| e.to_string())?;
    if a.sample_rate != 48_000 {
        return Err(format!("{path}: {} Hz (48000 required)", a.sample_rate));
    }
    #[allow(clippy::cast_precision_loss)]
    let n = a.channels as f64;
    Ok(a.samples
        .chunks_exact(a.channels)
        .map(|f| f.iter().map(|&v| f64::from(v)).sum::<f64>() / n)
        .collect())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pairs: Vec<(String, String)> = if args.len() == 2 && args[0] == "--batch" {
        std::fs::read_to_string(&args[1])
            .expect("pairs file")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let mut it = l.split('\t');
                (
                    it.next().unwrap_or("").to_string(),
                    it.next().unwrap_or("").to_string(),
                )
            })
            .collect()
    } else if args.len() == 2 {
        vec![(args[0].clone(), args[1].clone())]
    } else {
        eprintln!("usage: visqol <reference.wav> <test.wav> | --batch <pairs.tsv>");
        std::process::exit(2);
    };
    let analyzer = Analyzer::audio();
    let with_bands = std::env::var("VISQOL_FV").is_ok();
    for (r, t) in pairs {
        let t0 = Instant::now();
        let res = mono(&r)
            .and_then(|x| mono(&t).map(|y| (x, y)))
            .and_then(|(x, y)| analyzer.score_mono(&x, &y).map_err(|e| e.to_string()));
        match res {
            Ok(v) => {
                let mut line = format!(
                    "{:.6}\t{:.6}\t{}\t{:.3}",
                    v.moslqo,
                    v.vnsim,
                    v.lag,
                    t0.elapsed().as_secs_f64()
                );
                if with_bands {
                    for x in &v.fvnsim {
                        let _ = write!(line, "\t{x:.6}");
                    }
                }
                println!("{line}");
            }
            Err(e) => println!("error\t{e}"),
        }
    }
}

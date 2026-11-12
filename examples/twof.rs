//! 2f on WAV pairs.
//!
//! ```text
//! cargo run --release -p sori-metrics --example twof -- <ref.wav> <test.wav>
//! cargo run --release -p sori-metrics --example twof -- --batch <pairs.tsv>
//! ```
//!
//! Prints one tab-separated line per pair: `mushra  adb  avg_mod_diff1  seconds`.

use std::time::Instant;

fn one(ref_path: &str, test_path: &str) -> Result<String, String> {
    let a = sori_metrics::wav::read(ref_path.as_ref()).map_err(|e| e.to_string())?;
    let b = sori_metrics::wav::read(test_path.as_ref()).map_err(|e| e.to_string())?;
    if a.channels != b.channels || a.sample_rate != b.sample_rate {
        return Err("format mismatch".into());
    }
    let t0 = Instant::now();
    let s = sori_metrics::twof_score(&a.samples, &b.samples, a.sample_rate, a.channels)
        .map_err(|e| e.to_string())?;
    Ok(format!(
        "{:.6}\t{:.6}\t{:.6}\t{:.3}",
        s.mushra,
        s.adb,
        s.avg_mod_diff1,
        t0.elapsed().as_secs_f64()
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pairs: Vec<(String, String)> = if args.len() == 2 && args[0] == "--batch" {
        std::fs::read_to_string(&args[1])
            .expect("pairs file")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let mut p = l.split('\t');
                (
                    p.next().unwrap_or("").to_string(),
                    p.next().unwrap_or("").to_string(),
                )
            })
            .collect()
    } else if args.len() == 2 {
        vec![(args[0].clone(), args[1].clone())]
    } else {
        eprintln!("usage: twof <ref.wav> <test.wav> | --batch <pairs.tsv>");
        std::process::exit(2);
    };
    for (r, t) in pairs {
        match one(&r, &t) {
            Ok(s) => println!("{s}"),
            Err(e) => println!("error\t{e}"),
        }
    }
}

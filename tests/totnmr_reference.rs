//! `Total NMR_B` against the outputs of the project's Python measurement (`peaq_nmr.py`,
//! task M-08).
//!
//! 1. Synthetic pairs (`common/signals.rs`): expected values are the Python outputs on the
//!    WAVs written by `examples/write_fixtures.rs`.
//! 2. The public validation set (`data/totnmr-public.tsv`: 14 encoder settings × 23 public
//!    items, no speech): runs only when `SORI_TOTNMR_PAIRS_DIR` points at the measurement work
//!    directory holding the aligned 48 kHz pairs (`<arm>/<item>.<tag>.ref48.wav` /
//!    `.dec48.wav`, as `conformance quality --work` leaves them). Audio is never committed.

mod common;

use common::signals;
use sori_metrics::twof_score;

/// Python `peaq_nmr.py` outputs (dB) for the synthetic pairs, in `signals::all()` order.
const SYNTHETIC_PY: [(&str, f64); 3] = [
    ("chord-quantised", -4.581_587_007_839_453),
    ("noise-lowpass", -6.923_588_725_374_021),
    ("clicks-noise", 22.279_714_423_624_306),
];

#[test]
fn synthetic_pairs_match_python_outputs() {
    for (p, (name, py)) in signals::all().iter().zip(SYNTHETIC_PY) {
        assert_eq!(p.name, name);
        let s = twof_score(&p.reference, &p.test, 48_000, p.channels).unwrap();
        assert!(
            (s.total_nmr - py).abs() < 1e-4,
            "{name}: {} vs {py}",
            s.total_nmr
        );
    }
}

#[test]
fn identical_signals_have_no_noise() {
    let p = &signals::all()[0];
    let s = twof_score(&p.reference, &p.reference, 48_000, p.channels).unwrap();
    // Only the 1e-12 floor of the noise pattern remains.
    assert!(s.total_nmr < -100.0, "{}", s.total_nmr);
}

#[test]
fn reference_object_gives_the_same_value() {
    for p in signals::all() {
        let one = twof_score(&p.reference, &p.test, 48_000, p.channels).unwrap();
        let r = sori_metrics::twof::Reference::new(&p.reference, 48_000, p.channels).unwrap();
        assert_eq!(r.score(&p.test).unwrap(), one, "{}", p.name);
    }
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn public_pairs_match_python_outputs() {
    let Ok(dir) = std::env::var("SORI_TOTNMR_PAIRS_DIR") else {
        eprintln!("SORI_TOTNMR_PAIRS_DIR not set: public-pair comparison skipped");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let table = include_str!("data/totnmr-public.tsv");
    let mut diffs = Vec::new();
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();
        let (arm, item, py): (&str, &str, f64) = (f[0], f[1], f[2].parse().unwrap());
        let dec = std::fs::read_dir(dir.join(arm))
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| {
                let n = p.file_name().unwrap().to_string_lossy();
                n.starts_with(&format!("{item}.")) && n.ends_with(".dec48.wav")
            })
            .unwrap_or_else(|| panic!("{arm}/{item}: no dec48.wav"));
        let r_path = dec.to_string_lossy().replace(".dec48.wav", ".ref48.wav");
        let r = sori_metrics::wav::read(r_path.as_ref()).unwrap();
        let t = sori_metrics::wav::read(&dec).unwrap();
        let s = twof_score(&r.samples, &t.samples, 48_000, r.channels).unwrap();
        diffs.push((s.total_nmr - py).abs());
    }
    diffs.sort_by(f64::total_cmp);
    let (med, max) = (diffs[diffs.len() / 2], diffs[diffs.len() - 1]);
    eprintln!(
        "{} pairs: |Δ| median {med:.3e} dB, max {max:.3e} dB",
        diffs.len()
    );
    assert!(diffs.len() >= 300);
    assert!(med <= 0.02 && max <= 0.1, "median {med}, max {max}");
}

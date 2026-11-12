//! 2f against reference outputs of the Python `seabass-2f` tool (outputs only — task M-01).
//!
//! 1. Synthetic pairs (`common/signals.rs`, rebuilt here bit for bit): expected values are the
//!    Python outputs on the WAVs written by `examples/write_fixtures.rs`.
//! 2. The public validation set (`data/twof-public.tsv`, public items × 5 arms): runs only
//!    when `SORI_TWOF_PAIRS_DIR` points at the measurement work directory holding the aligned
//!    48 kHz pairs (`<arm>/<item>.<tag>.ref48.wav` / `.dec48.wav`, as `conformance quality
//!    --work` leaves them). Audio is never committed.

mod common;

use common::signals;
use sori_metrics::twof_score;

/// Python outputs (`mms`, `adb`, `avg_mod_diff1`) for the synthetic pairs, in `signals::all()` order.
const SYNTHETIC_PY: [(&str, f64, f64, f64); 3] = [
    ("chord-quantised", 52.092_051, 1.936_315, 20.388_269),
    ("noise-lowpass", 48.414_771, 2.568_733, 0.433_986),
    ("clicks-noise", 25.277_776, 2.924_299, 20.378_634),
];

/// Rust outputs at the time of writing, pinned to catch unintended changes (determinism and
/// regression; the tolerance allows for libm differences between platforms).
const SYNTHETIC_RS: [(f64, f64, f64); 3] = [
    (52.061_307, 1.935_773, 20.482_340),
    (48.418_421, 2.568_598, 0.434_000),
    (25.947_244, 2.924_053, 19.039_564),
];

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-9)
}

#[test]
fn synthetic_pairs_match_python_outputs() {
    for ((p, py), pinned) in signals::all().iter().zip(SYNTHETIC_PY).zip(SYNTHETIC_RS) {
        assert_eq!(p.name, py.0);
        let s = twof_score(&p.reference, &p.test, 48_000, p.channels).unwrap();
        // Pinned values.
        assert!(
            (s.mushra - pinned.0).abs() < 1e-4,
            "{} mushra {}",
            p.name,
            s.mushra
        );
        assert!((s.adb - pinned.1).abs() < 1e-5, "{} adb {}", p.name, s.adb);
        assert!(
            (s.avg_mod_diff1 - pinned.2).abs() < 1e-4,
            "{} amd1 {}",
            p.name,
            s.avg_mod_diff1
        );
        // Against the Python tool. `clicks-noise` (6 ms clicks) is the known open difference
        // of `AvgModDiff1` on very short transients (README §남은 차이): wider bound there.
        let (mms_tol, amd_tol) = if p.name == "clicks-noise" {
            (1.0, 0.08)
        } else {
            (0.2, 0.01)
        };
        assert!((s.mushra - py.1).abs() <= mms_tol, "{} {s:?}", p.name);
        assert!(rel(s.adb, py.2) <= 0.01, "{} {s:?}", p.name);
        assert!(rel(s.avg_mod_diff1, py.3) <= amd_tol, "{} {s:?}", p.name);
    }
}

#[test]
fn deterministic() {
    let p = &signals::all()[0];
    let a = twof_score(&p.reference, &p.test, 48_000, p.channels).unwrap();
    let b = twof_score(&p.reference, &p.test, 48_000, p.channels).unwrap();
    assert_eq!(a, b);
}

#[test]
fn public_set_matches_python_outputs() {
    let Some(dir) = std::env::var_os("SORI_TWOF_PAIRS_DIR") else {
        eprintln!("SORI_TWOF_PAIRS_DIR not set: public-set comparison skipped");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let data = include_str!("data/twof-public.tsv");
    let mut d_mms = Vec::new();
    let mut r_adb = Vec::new();
    let mut r_amd = Vec::new();
    for line in data
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();
        let (arm, item, tag) = (f[0], f[1], f[2]);
        let (mms, adb, amd): (f64, f64, f64) = (
            f[3].parse().unwrap(),
            f[4].parse().unwrap(),
            f[5].parse().unwrap(),
        );
        let r = sori_metrics::wav::read(&dir.join(arm).join(format!("{item}.{tag}.ref48.wav")));
        let t = sori_metrics::wav::read(&dir.join(arm).join(format!("{item}.{tag}.dec48.wav")));
        let (Ok(r), Ok(t)) = (r, t) else {
            panic!("missing pair for {arm} {item} under {}", dir.display());
        };
        let s = twof_score(&r.samples, &t.samples, r.sample_rate, r.channels).unwrap();
        d_mms.push((s.mushra - mms).abs());
        if adb.abs() > 0.05 {
            r_adb.push(rel(s.adb, adb));
        }
        r_amd.push(rel(s.avg_mod_diff1, amd));
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let max = d_mms.iter().copied().fold(0.0, f64::max);
    let (m, a, b) = (median(&mut d_mms), median(&mut r_adb), median(&mut r_amd));
    eprintln!(
        "pairs {} |dMMS| median {m:.3} max {max:.3}; MOV rel median {a:.4} {b:.4}",
        d_mms.len()
    );
    // Acceptance of task M-01.
    // 5 arms per public item (24 here, 23 in the public export).
    assert!(d_mms.len() >= 5 * 23, "{}", d_mms.len());
    assert!(m <= 0.2 && max <= 1.0);
    assert!(a <= 0.01 && b <= 0.01);
}

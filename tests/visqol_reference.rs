//! ViSQOL against the reference tool's outputs (task M-03; values only, no audio committed).
//!
//! - `data/visqol-intermediate/` (task T-113): `visqol-python` 3.8.0 intermediate outputs.
//!   Stage checks that need no audio: band centres, and the SVR stage (the reference
//!   per-band similarities through our SVR give the reference MOS).
//! - Synthetic pairs of T-113, rebuilt here (xorshift32 noise as described in its README).
//! - `data/visqol-t97-public.tsv` (task M-02): public items × 16 decodes; runs only when
//!   `SORI_VISQOL_PAIRS_DIR` points at the 48 kHz aligned pairs (`<item>.ref48.wav`,
//!   `<item>.<decode>.dec48.wav`), as M-02 rebuilt them.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::type_complexity
)]

use std::collections::BTreeMap;

use sori_metrics::visqol::{gammatone::erb_space, svr, Analyzer};

const PAIRS: &str = include_str!("data/visqol-intermediate/pairs.tsv");
const BANDS: &str = include_str!("data/visqol-intermediate/bands.tsv");

/// `pair → (moslqo, vnsim)` and `pair → (centre freqs, fvnsim)` from the T-113 table.
fn table() -> (
    BTreeMap<String, (f64, f64)>,
    BTreeMap<String, (Vec<f64>, Vec<f64>)>,
) {
    let mut p = BTreeMap::new();
    for l in PAIRS.lines().skip(1) {
        let c: Vec<&str> = l.split('\t').collect();
        p.insert(
            c[0].to_string(),
            (c[7].parse().unwrap(), c[8].parse().unwrap()),
        );
    }
    let mut b: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for l in BANDS.lines().skip(1) {
        let c: Vec<&str> = l.split('\t').collect();
        let e = b.entry(c[0].to_string()).or_default();
        e.0.push(c[2].parse().unwrap());
        e.1.push(c[3].parse().unwrap());
    }
    (p, b)
}

#[test]
fn band_centres_match_reference() {
    let (_, b) = table();
    let ours = erb_space(50.0, 24_000.0, 32);
    for (pair, (cf, _)) in &b {
        assert_eq!(cf.len(), 32, "{pair}");
        for (a, c) in cf.iter().zip(&ours) {
            assert!((a - c).abs() < 1e-9, "{pair}: {a} vs {c}");
        }
    }
}

#[test]
fn svr_stage_matches_reference() {
    let (p, b) = table();
    let m = svr::SvrModel::parse(svr::AUDIO_MODEL).unwrap();
    // Public items x 2 decodes + 3 synthetic (the public export has no speech item).
    assert!(p.len() >= 2 * 23 + 3 && p.len() % 2 == 1, "{}", p.len());
    for (pair, (mos, vnsim)) in &p {
        let fv = &b[pair].1;
        assert!((m.predict(fv) - mos).abs() < 1e-9, "{pair}");
        assert!(
            (fv.iter().sum::<f64>() / 32.0 - vnsim).abs() < 1e-12,
            "{pair}"
        );
    }
}

/// xorshift32 in [−1, 1) as T-113 defines it.
fn xorshift(seed: u32, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            f64::from(s) / 2_147_483_648.0 - 1.0
        })
        .collect()
}

fn sine(f: f64, i: usize) -> f64 {
    let t = i as f64 / 48_000.0;
    (2.0 * std::f64::consts::PI * f * t).sin()
}

/// T-113 synthetic signals, rounded to f32 as they were stored.
fn synthetic() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 5 * 48_000;
    let u1 = xorshift(1, n);
    let identical: Vec<f64> = (0..n)
        .map(|i| {
            let tone = 0.08 * sine(220.0, i)
                + 0.05 * sine(440.0, i)
                + 0.04 * sine(1000.0, i)
                + 0.02 * sine(3000.0, i)
                + 0.01 * sine(8000.0, i);
            f64::from((tone + 0.01 * u1[i]) as f32)
        })
        .collect();
    let u2 = xorshift(2, n);
    let amp = 0.1 / 2f64.sqrt() * 0.1 * 3f64.sqrt();
    let sine_ref: Vec<f64> = (0..n)
        .map(|i| f64::from((0.1 * sine(1000.0, i)) as f32))
        .collect();
    let sine_deg: Vec<f64> = (0..n)
        .map(|i| f64::from((0.1 * sine(1000.0, i) + u2[i] * amp) as f32))
        .collect();
    let mut padded = vec![0.0; 24_000];
    padded.extend_from_slice(&identical);
    padded.extend(std::iter::repeat_n(0.0, 24_000));
    (identical, sine_ref, sine_deg, padded)
}

#[test]
fn synthetic_pairs() {
    let (p, _) = table();
    let an = Analyzer::audio();
    let (idn, sr, sd, pad) = synthetic();
    let a = an.score_mono(&idn, &idn).unwrap();
    assert_eq!(a.patches.len(), 7);
    assert!(
        (a.moslqo - p["syn-identical"].0).abs() < 1e-6,
        "{}",
        a.moslqo
    );
    assert!((a.vnsim - 1.0).abs() < 1e-12);
    let b = an.score_mono(&idn, &pad).unwrap();
    assert_eq!(b.lag, 24_000);
    assert!(
        (b.moslqo - p["syn-pad-silence"].0).abs() < 0.002,
        "{}",
        b.moslqo
    );
    // Pure-tone reference with added noise: the known open difference (README §남은 차이) —
    // the reference tool scores 3.38. Pinned so a change is noticed.
    let c = an.score_mono(&sr, &sd).unwrap();
    assert!((c.moslqo - 1.582_929).abs() < 1e-4, "{}", c.moslqo);
}

#[test]
fn public_pairs_match_reference() {
    let Some(dir) = std::env::var_os("SORI_VISQOL_PAIRS_DIR") else {
        eprintln!("SORI_VISQOL_PAIRS_DIR not set: public-pair comparison skipped");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let mono = |p: std::path::PathBuf| -> Vec<f64> {
        let a = sori_metrics::wav::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        a.samples
            .chunks_exact(a.channels)
            .map(|f| f.iter().map(|&v| f64::from(v)).sum::<f64>() / a.channels as f64)
            .collect()
    };
    let an = Analyzer::audio();
    let mut refs: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut d = Vec::new();
    for l in include_str!("data/visqol-t97-public.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let c: Vec<&str> = l.split('\t').collect();
        let (item, dec, py): (&str, &str, f64) = (c[0], c[1], c[2].parse().unwrap());
        let r = refs
            .entry(item.to_string())
            .or_insert_with(|| mono(dir.join(format!("{item}.ref48.wav"))))
            .clone();
        let t = mono(dir.join(format!("{item}.{dec}.dec48.wav")));
        let v = an.score_mono(&r, &t).unwrap();
        d.push((v.moslqo - py).abs());
    }
    d.sort_by(f64::total_cmp);
    let (med, max) = (d[d.len() / 2], d[d.len() - 1]);
    eprintln!("pairs {} |dMOS| median {med:.4} max {max:.4}", d.len());
    // Adoption criteria of M-02 / M-03.
    // 16 decodes per public item (24 items here, 23 in the public export).
    assert!(d.len() >= 16 * 23 && d.len() % 16 == 0, "{}", d.len());
    assert!(med <= 0.02 && max <= 0.1);
}

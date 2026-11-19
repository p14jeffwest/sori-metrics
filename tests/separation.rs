//! Source-separation criteria against the outputs (only) of `museval` (BSS Eval v4, image
//! criteria) and `pb_bss_eval` (SI-SDR).
//!
//! 1. Synthetic sets (`common/separation.rs`, 8 kHz): expected values are the Python outputs
//!    on the WAVs written by `examples/write_fixtures.rs` (`data/separation-synthetic.tsv`).
//! 2. The public validation set (`data/separation-public.tsv`: 10 mixtures of public clips,
//!    4-source and vocals/accompaniment set-ups, 60 source estimates): runs only when
//!    `SORI_SEPARATION_DIR` points at the directory written by `scripts/separation-data.py`
//!    (`syn-mixNN/{ref,est}-{vocals,drums,bass,other,accompaniment}.wav`). Audio is never
//!    committed. Run it with `--release`: each mixture solves a 4096-unknown system.

// `0.5 * (x + y)` as in the crate (newer Clippy calls it a manual midpoint).
#![allow(unknown_lints, clippy::manual_midpoint)]

mod common;

use common::separation::{self as sets, Set};
use sori_metrics::separation::{bss_eval, scale_invariant, Options, SourceScores};
use sori_metrics::Error;

fn slices(v: &[Vec<f32>]) -> Vec<&[f32]> {
    v.iter().map(Vec::as_slice).collect()
}

/// `pb_bss_eval` stabilises SI-SDR with `+1e-8` on the energy ratio (and on `‖s‖²` in the
/// scale, negligible here); both were found from its outputs. Our value is the plain
/// definition, so convert before comparing.
fn as_reference_tool(si_sdr: f64) -> f64 {
    10.0 * (10f64.powf(si_sdr / 10.0) + 1e-8).log10()
}

fn parse(v: &str) -> f64 {
    if v == "nan" {
        f64::NAN
    } else {
        v.parse().unwrap()
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol
}

/// Rows `kind source index values...` of one item.
fn rows<'a>(table: &'a str, item: &'a str, setup: Option<&'a str>) -> Vec<Vec<&'a str>> {
    table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .filter(|f| f[0] == item && setup.is_none_or(|s| f[1] == s))
        .map(|f| f[if setup.is_some() { 2 } else { 1 }..].to_vec())
        .collect()
}

/// Largest per-window difference of the four criteria (and checks the unscored windows).
fn compare_windows(scores: &[SourceScores], rows: &[Vec<&str>], what: &str) -> f64 {
    let mut worst: f64 = 0.0;
    let mut seen = 0;
    for f in rows.iter().filter(|f| f[0] == "bss") {
        let (j, w): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
        let got = &scores[j].frames[w];
        for (g, e) in [got.sdr, got.isr, got.sir, got.sar]
            .into_iter()
            .zip(&f[3..7])
        {
            let e = parse(e);
            assert_eq!(
                g.is_nan(),
                e.is_nan(),
                "{what} source {j} window {w}: {g} vs {e}"
            );
            if !e.is_nan() {
                worst = worst.max((g - e).abs());
            }
        }
        seen += 1;
    }
    let total: usize = scores.iter().map(|s| s.frames.len()).sum();
    assert_eq!(seen, total, "{what}: window count");
    worst
}

fn synthetic_options(set: &Set) -> Options {
    Options::new(set.rate).with_filter_len(64)
}

#[test]
fn synthetic_sets_match_reference_outputs() {
    let table = include_str!("data/separation-synthetic.tsv");
    for set in sets::all() {
        let (r, e) = (slices(&set.references), slices(&set.estimates));
        let scores = bss_eval(&r, &e, set.channels, &synthetic_options(&set)).unwrap();
        let expected = rows(table, set.name, None);
        let worst = compare_windows(&scores, &expected, set.name);
        assert!(worst < 1e-6, "{}: window |Δ| {worst} dB", set.name);
        let si = scale_invariant(&r, &e, set.channels, true).unwrap();
        for f in expected.iter().filter(|f| f[0] == "si") {
            let (j, c): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
            let got = as_reference_tool(si[j].channels[c].si_sdr);
            assert!(
                close(got, parse(f[3]), 1e-6),
                "{} si {j}/{c}: {got} vs {}",
                set.name,
                f[3]
            );
        }
    }
}

#[test]
fn silent_reference_window_is_unscored_for_every_source() {
    let set = sets::three_stereo();
    let s = bss_eval(
        &slices(&set.references),
        &slices(&set.estimates),
        2,
        &synthetic_options(&set),
    )
    .unwrap();
    for source in &s {
        assert!(source.frames[1].sdr.is_nan() && !source.frames[0].sdr.is_nan());
        assert_eq!(source.frames[1].start, 8_000);
        let m = source.median();
        let mid = 0.5 * (source.frames[0].sar + source.frames[2].sar);
        assert!((m.sar - mid).abs() < 1e-12);
    }
}

#[test]
fn sdr_is_the_plain_energy_ratio() {
    // The total distortion of the decomposition is ŝ − s, so SDR needs no projection.
    let set = sets::two_mono_clipped();
    let s = bss_eval(
        &slices(&set.references),
        &slices(&set.estimates),
        1,
        &synthetic_options(&set),
    )
    .unwrap();
    for (j, src) in s.iter().enumerate() {
        for f in &src.frames {
            let (mut num, mut den) = (0.0, 0.0);
            for t in f.start..f.start + 8_000 {
                let (r, e) = (
                    f64::from(set.references[j][t]),
                    f64::from(set.estimates[j][t]),
                );
                num += r * r;
                den += (e - r) * (e - r);
            }
            assert!((f.sdr - 10.0 * (num / den).log10()).abs() < 1e-9);
        }
    }
}

#[test]
fn scale_invariant_parts_add_up() {
    // e_res = e_interf + e_artif with orthogonal parts: 1/SDR = 1/SIR + 1/SAR (linear).
    for set in sets::all() {
        let si = scale_invariant(
            &slices(&set.references),
            &slices(&set.estimates),
            set.channels,
            true,
        )
        .unwrap();
        for v in si.iter().flat_map(|s| &s.channels) {
            let inv = |d: f64| 10f64.powf(-d / 10.0);
            let rel = (inv(v.si_sdr) - inv(v.si_sir) - inv(v.si_sar)).abs() / inv(v.si_sdr);
            assert!(rel < 1e-9, "{v:?}");
        }
    }
}

#[test]
fn scale_invariant_ignores_gain_and_offset_with_zero_mean() {
    let set = sets::two_mono_clipped();
    let r = slices(&set.references);
    let base = scale_invariant(&r, &slices(&set.estimates), 1, true).unwrap();
    let scaled: Vec<Vec<f32>> = set
        .estimates
        .iter()
        .map(|e| e.iter().map(|x| 0.5 * x + 0.01).collect())
        .collect();
    let other = scale_invariant(&r, &slices(&scaled), 1, true).unwrap();
    for (a, b) in base.iter().zip(&other) {
        assert!((a.mean.si_sdr - b.mean.si_sdr).abs() < 1e-4);
    }
}

#[test]
fn identical_estimate_scores_infinite() {
    let set = sets::two_mono_clipped();
    let r = slices(&set.references);
    let si = scale_invariant(&r, &r, 1, false).unwrap();
    assert!(si[0].mean.si_sdr > 200.0);
    let s = bss_eval(&r, &r, 1, &synthetic_options(&set)).unwrap();
    assert!(s[0].frames[0].sdr.is_infinite());
}

#[test]
fn shape_errors() {
    let a = vec![0.1f32; 16_000];
    let b = vec![0.1f32; 15_998];
    let o = Options::new(8_000);
    assert!(matches!(
        bss_eval(&[&a], &[&a, &a], 1, &o),
        Err(Error::Mismatch(_))
    ));
    assert!(matches!(
        bss_eval(&[&a, &b], &[&a, &b], 1, &o),
        Err(Error::Mismatch(_))
    ));
    assert!(matches!(
        bss_eval(&[&a[..15_999]], &[&a[..15_999]], 2, &o),
        Err(Error::Mismatch(_))
    ));
    assert!(matches!(
        bss_eval(&[&a[..7_000]], &[&a[..7_000]], 1, &o),
        Err(Error::TooShort(7_000))
    ));
    assert!(matches!(
        bss_eval(&[&a], &[&a], 0, &o),
        Err(Error::Channels(0))
    ));
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn public_mixtures_match_reference_outputs() {
    let Ok(dir) = std::env::var("SORI_SEPARATION_DIR") else {
        eprintln!("SORI_SEPARATION_DIR not set: public-mixture comparison skipped");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let table = include_str!("data/separation-public.tsv");
    let setups: [(&str, &[&str]); 2] = [
        ("four", &["vocals", "drums", "bass", "other"]),
        ("voc", &["vocals", "accompaniment"]),
    ];
    let (mut worst_window, mut medians, mut si_worst, mut estimates) = (0f64, Vec::new(), 0f64, 0);
    for k in 0..10 {
        let item = format!("syn-mix{k:02}");
        for (setup, names) in setups {
            let read = |kind: &str, n: &str| {
                sori_metrics::wav::read(&dir.join(&item).join(format!("{kind}-{n}.wav")))
                    .unwrap()
                    .samples
            };
            let refs: Vec<Vec<f32>> = names.iter().map(|n| read("ref", n)).collect();
            let ests: Vec<Vec<f32>> = names.iter().map(|n| read("est", n)).collect();
            let (r, e) = (slices(&refs), slices(&ests));
            let scores = bss_eval(&r, &e, 2, &Options::new(44_100)).unwrap();
            let expected = rows(table, &item, Some(setup));
            worst_window = worst_window.max(compare_windows(&scores, &expected, &item));
            for (j, s) in scores.iter().enumerate() {
                let rows_j: Vec<f64> = expected
                    .iter()
                    .filter(|f| f[0] == "bss" && f[1] == j.to_string())
                    .map(|f| parse(f[3]))
                    .collect();
                let mut v: Vec<f64> = rows_j.into_iter().filter(|x| !x.is_nan()).collect();
                v.sort_by(f64::total_cmp);
                let n = v.len();
                let want = if n % 2 == 1 {
                    v[n / 2]
                } else {
                    0.5 * (v[n / 2 - 1] + v[n / 2])
                };
                medians.push((s.median().sdr - want).abs());
                estimates += 1;
            }
            let si = scale_invariant(&r, &e, 2, true).unwrap();
            for f in expected.iter().filter(|f| f[0] == "si") {
                let (j, c): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
                let got = as_reference_tool(si[j].channels[c].si_sdr);
                si_worst = si_worst.max((got - parse(f[3])).abs());
            }
        }
    }
    medians.sort_by(f64::total_cmp);
    let (med, max) = (medians[medians.len() / 2], medians[medians.len() - 1]);
    eprintln!(
        "{estimates} estimates: track SDR |Δ| median {med:.2e}, max {max:.2e} dB; \
         window |Δ| max {worst_window:.2e} dB; SI-SDR |Δ| max {si_worst:.2e} dB"
    );
    assert_eq!(estimates, 60);
    assert!(med <= 0.01 && max <= 0.05);
    assert!(worst_window <= 1e-6 && si_worst <= 1e-6);
}

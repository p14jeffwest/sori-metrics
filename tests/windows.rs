//! Window/block outputs and the piece-by-piece (`Source`) paths (task M-07).

#![allow(clippy::many_single_char_names)] // short signal-generation names

use sori_metrics::source::{Interleaved, Source};
use sori_metrics::{twof, visqol, Error};

/// A two-tone reference, and a test equal to it plus a noise burst at 2.0–2.3 s and slight
/// quantisation (stereo if `ch` = 2).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn pair(seconds: f32, ch: usize) -> (Vec<f32>, Vec<f32>) {
    let n = (seconds * 48_000.0) as usize;
    let mut s = 1u32;
    let (mut r, mut t) = (Vec::with_capacity(n * ch), Vec::with_capacity(n * ch));
    let q = 1.0 / 4096.0;
    for i in 0..n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let x = i as f32 / 48_000.0;
        let burst = if (2.0..2.3).contains(&x) {
            0.05 * (s as f32 / 4.3e9 - 0.5)
        } else {
            0.0
        };
        for c in 0..ch {
            let f = 330.0 * (c + 1) as f32;
            let v = 0.2 * (2.0 * std::f32::consts::PI * f * x).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 2.7 * f * x).sin();
            r.push(v);
            t.push(((v + burst) / q).round() * q);
        }
    }
    (r, t)
}

/// A source that counts reads and fails on request.
struct Counting<'a> {
    inner: Interleaved<'a>,
    reads: usize,
    fail: bool,
}

impl Source for Counting<'_> {
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }
    fn channels(&self) -> usize {
        self.inner.channels()
    }
    fn frames(&self) -> usize {
        self.inner.frames()
    }
    fn read(&mut self, start: usize, out: &mut [f32]) -> Result<(), Error> {
        self.reads += 1;
        if self.fail {
            return Err(Error::Wav("read failed".into()));
        }
        self.inner.read(start, out)
    }
}

#[test]
fn source_paths_equal_slice_paths() {
    for ch in [1, 2] {
        let (r, t) = pair(12.0, ch);
        let src = |x| Interleaved::new(x, 48_000, ch).unwrap();
        let a = twof::detail(&r, &t, 48_000, ch).unwrap();
        let b = twof::detail_source(src(&r), src(&t)).unwrap();
        assert_eq!(a, b, "2f, {ch} ch");
        assert_eq!(a.score, twof::score(&r, &t, 48_000, ch).unwrap());
        let v = visqol::score(&r, &t, 48_000, ch).unwrap();
        let w = visqol::score_source(src(&r), src(&t)).unwrap();
        assert_eq!(v, w, "ViSQOL, {ch} ch");
    }
}

#[test]
fn sources_are_read_in_pieces() {
    let (r, t) = pair(12.0, 2);
    let src = |x| Counting {
        inner: Interleaved::new(x, 48_000, 2).unwrap(),
        reads: 0,
        fail: false,
    };
    let (mut a, mut b) = (src(&r), src(&t));
    twof::detail_source(&mut a, &mut b).unwrap();
    // 12 s = 2 pieces of 10 s (and a few re-reads for the data-range scan).
    assert!(a.reads >= 2 && b.reads >= 2, "{} {}", a.reads, b.reads);
    let mut c = src(&t);
    c.fail = true;
    assert!(twof::detail_source(src(&r), &mut c).is_err());
    assert!(visqol::score_source(src(&r), &mut c).is_err());
}

#[test]
#[allow(clippy::float_cmp)]
fn windows_cover_the_signal() {
    let (r, t) = pair(12.0, 2);
    let d = twof::detail(&r, &t, 48_000, 2).unwrap();
    // A window longer than the signal is the whole comparison.
    let all = d.windows(60.0, 1.0);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].score, Some(d.score));
    assert_eq!(all[0].blocks, d.len());
    // 3 s every 1 s over ~12 s: windows 0–3 … 9–12 (as many as fit).
    let w = d.windows(3.0, 1.0);
    assert!(w.len() >= 9 && w.len() <= 10, "{}", w.len());
    assert_eq!(w[1].start, 1.0);
    assert!(w.iter().all(|w| (w.end - w.start - 3.0).abs() < 1e-9));
    // The noise burst (2.0–2.3 s) is the worst part.
    let worst = w
        .iter()
        .min_by(|a, b| a.score.unwrap().mushra.total_cmp(&b.score.unwrap().mushra))
        .unwrap();
    assert!(worst.start <= 2.0 && worst.end >= 2.3, "{worst:?}");
    // Every block counted once per window it is centred in.
    let blocks: Vec<twof::Block> = d.blocks().collect();
    assert_eq!(blocks.len(), d.len());
    assert!(blocks.iter().all(|b| (0.0..=1.0).contains(&b.detection)));
    assert!(blocks.iter().filter(|b| b.valid).count() > 500);

    let v = visqol::score(&r, &t, 48_000, 2).unwrap();
    let all = v.windows(60.0, 1.0);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].patches, v.patches.len());
    assert!((all[0].moslqo.unwrap() - v.moslqo).abs() < 1e-12);
    let w = v.windows(3.0, 1.0);
    assert!(w.iter().all(|w| w.patches == 4 || w.patches == 5), "{w:?}");
}

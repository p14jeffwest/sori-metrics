//! Source-separation quality for music: BSS Eval image metrics in the version-4 form used by
//! the SiSEC 2018 evaluation (SDR, ISR, SIR, SAR per window) and the scale-invariant SDR.
//!
//! For evaluating **music source separation** (for example vocals, drums, bass and other, or
//! a target against its accompaniment), following the conventions of BSS Eval v4 and SiSEC
//! 2018. Not validated for speech separation.
//!
//! # BSS Eval v4, image form ([`bss_eval`])
//!
//! Each estimated source image `ŝ_j` (all channels) is decomposed as in Vincent, Gribonval &
//! Févotte, "Performance measurement in blind audio source separation", IEEE TASLP 14(4),
//! 2006, and Vincent, Sawada, Bofill, Makino & Rosca, "First stereo audio source separation
//! evaluation campaign", ICA 2007 (the *image* criteria):
//!
//! - `P_j(ŝ_ij)`: the least-squares fit of channel `i` of the estimate by all channels of the
//!   reference image of source `j`, each through an FIR filter of `filter_len` taps;
//! - `P(ŝ_ij)`: the same with the channels of **all** reference sources;
//! - `e_spat = P_j − s_ij`, `e_interf = P − P_j`, `e_artif = ŝ_ij − P`.
//!
//! As in SiSEC 2018 (Stöter, Liutkus & Ito, "The 2018 Signal Separation Evaluation Campaign",
//! LVA/ICA 2018), the filters are **time-invariant**: they are fitted once on the whole track
//! (on zero-padded signals, so the fit covers `len + filter_len − 1` samples), and the scores
//! are computed on windows. In each window the reference segments are filtered with the
//! track-wide filters (again zero-padded), and the energies are summed over all channels:
//!
//! ```text
//! SDR = 10 log10 Σ s²        / Σ (ŝ − s)²
//! ISR = 10 log10 Σ s²        / Σ e_spat²
//! SIR = 10 log10 Σ P_j²      / Σ e_interf²
//! SAR = 10 log10 Σ P²        / Σ e_artif²
//! ```
//!
//! A window in which **any** reference source is silent (all its samples zero) gets `NaN` for
//! every source; [`SourceScores::median`] ignores those windows, which gives the usual track
//! value (median over windows). Only full windows are scored.
//!
//! The paper leaves the window framing, the zero padding and the silent-window rule open; they
//! were fixed by comparing outputs with the `museval` Python package (v4 mode), run as a
//! black box, and agree with it to about `1e-9` dB on the bundled validation set.
//!
//! # Scale-invariant SDR ([`scale_invariant`])
//!
//! Le Roux, Wisdom, Erdogan & Hershey, "SDR – half-baked or well done?", ICASSP 2019, per
//! channel: `e_target = (ŝᵀs / ‖s‖²) s`, `e_res = ŝ − e_target`;
//! `SI-SDR = 10 log10 ‖e_target‖² / ‖e_res‖²`. `e_res` is split into its orthogonal
//! projection on the span of all reference sources (`e_interf`) and the rest (`e_artif`),
//! giving SI-SIR and SI-SAR. With `zero_mean`, the mean of each channel is removed first.
//!
//! # Example
//!
//! ```
//! use sori_metrics::separation::{bss_eval, scale_invariant, Options};
//! // Two mono sources (2 s at 8 kHz) and estimates that leak 10 % of each other.
//! let tone = |f: f32| -> Vec<f32> {
//!     (0..16_000).map(|i| (i as f32 * f * 0.001).sin() * (1.0 + (i % 7) as f32 * 0.01)).collect()
//! };
//! let (a, b) = (tone(0.9), tone(2.3));
//! let mix = |x: &[f32], y: &[f32]| -> Vec<f32> { x.iter().zip(y).map(|(p, q)| p + 0.1 * q).collect() };
//! let (ea, eb) = (mix(&a, &b), mix(&b, &a));
//! let scores = bss_eval(&[&a, &b], &[&ea, &eb], 1, &Options::new(8_000))?;
//! let m = scores[0].median();
//! assert!((m.sir - 20.0).abs() < 1.0 && m.sar > 60.0); // leak only: no artifacts
//! let si = scale_invariant(&[&a, &b], &[&ea, &eb], 1, true)?;
//! assert!(si[0].mean.si_sdr > 15.0);
//! # Ok::<(), sori_metrics::Error>(())
//! ```
//!
//! # Cost
//!
//! The track-wide filter fit solves one symmetric positive definite system of size
//! `sources × channels × filter_len` (4096 for four stereo sources), held as a packed
//! triangle: 64 MiB, and a few seconds of Cholesky factorisation. The correlations behind it
//! and the windowed filtering use block FFTs, so the rest grows linearly with the length and
//! the inputs are never copied.

// Linear algebra in the papers' notation: short names (l, m, n, a, b, j, t) on purpose.
#![allow(clippy::many_single_char_names, clippy::similar_names)]

use crate::fft::{Complex, Fft};
use crate::Error;

/// Analysis settings of [`bss_eval`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Options {
    /// Window length in samples (per channel).
    pub window: usize,
    /// Distance between window starts in samples.
    pub hop: usize,
    /// Length of the distortion filters in taps.
    pub filter_len: usize,
}

impl Options {
    /// The SiSEC 2018 settings at `sample_rate`: one-second windows without overlap and
    /// 512-tap filters.
    #[must_use]
    pub const fn new(sample_rate: u32) -> Self {
        Self {
            window: sample_rate as usize,
            hop: sample_rate as usize,
            filter_len: 512,
        }
    }

    /// The same with another window and hop (in samples).
    #[must_use]
    pub const fn with_window(mut self, window: usize, hop: usize) -> Self {
        self.window = window;
        self.hop = hop;
        self
    }

    /// The same with another filter length.
    #[must_use]
    pub const fn with_filter_len(mut self, filter_len: usize) -> Self {
        self.filter_len = filter_len;
        self
    }
}

/// The four image criteria of one source in one window, in dB. `NaN` when the window was not
/// scored (a silent reference source).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Frame {
    /// Start of the window in samples.
    pub start: usize,
    /// Source to distortion ratio.
    pub sdr: f64,
    /// Source image to spatial distortion ratio.
    pub isr: f64,
    /// Source to interference ratio.
    pub sir: f64,
    /// Source to artifacts ratio.
    pub sar: f64,
}

/// Track values: the median of each criterion over the scored windows.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Medians {
    /// Median SDR (dB); `NaN` if no window was scored.
    pub sdr: f64,
    /// Median ISR (dB).
    pub isr: f64,
    /// Median SIR (dB).
    pub sir: f64,
    /// Median SAR (dB).
    pub sar: f64,
}

/// The windows of one estimated source.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SourceScores {
    /// One entry per window, in time order.
    pub frames: Vec<Frame>,
}

impl SourceScores {
    /// Median over the windows, ignoring `NaN` (unscored) windows; for an even count the
    /// mean of the two middle values.
    #[must_use]
    pub fn median(&self) -> Medians {
        let m = |f: fn(&Frame) -> f64| nan_median(self.frames.iter().map(f));
        Medians {
            sdr: m(|f| f.sdr),
            isr: m(|f| f.isr),
            sir: m(|f| f.sir),
            sar: m(|f| f.sar),
        }
    }
}

/// Median of the finite-or-infinite (non-`NaN`) values; `NaN` if there are none.
fn nan_median(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// BSS Eval v4 image criteria of every estimated source, per window.
///
/// `references[j]` and `estimates[j]` are the reference and estimated images of source `j`,
/// interleaved with `channels` channels, all of the same length. For a target against its
/// accompaniment, pass two sources: the target and the sum of the other sources (for both
/// references and estimates).
///
/// # Errors
/// [`Error::Channels`] for zero channels; [`Error::Mismatch`] if there are no sources, the
/// counts differ, or any signal has another length or is not a whole number of frames;
/// [`Error::TooShort`] if the signals are shorter than one window or the filter, or the
/// window, hop or filter length is zero.
pub fn bss_eval(
    references: &[&[f32]],
    estimates: &[&[f32]],
    channels: usize,
    options: &Options,
) -> Result<Vec<SourceScores>, Error> {
    let sig = Signals::new(references, estimates, channels)?;
    let (win, hop, l) = (options.window, options.hop, options.filter_len);
    if win == 0 || hop == 0 || l == 0 || sig.len < win || sig.len < l {
        return Err(Error::TooShort(sig.len));
    }
    let frames = (sig.len - win) / hop + 1;
    let n_src = references.len();
    let starts: Vec<usize> = (0..frames).map(|w| w * hop).collect();
    let scored: Vec<bool> = starts.iter().map(|&a0| sig.all_sounding(a0, win)).collect();
    let mut out: Vec<SourceScores> = (0..n_src)
        .map(|_| SourceScores {
            frames: starts
                .iter()
                .map(|&start| Frame {
                    start,
                    sdr: f64::NAN,
                    isr: f64::NAN,
                    sir: f64::NAN,
                    sar: f64::NAN,
                })
                .collect(),
        })
        .collect();
    if !scored.iter().any(|&s| s) {
        return Ok(out);
    }
    let corr = Correlations::compute(&sig, l);
    let filters = Filters::fit(&sig, &corr);
    drop(corr);
    let fft = filters.fft();
    let mut work = FrameWork::new(&sig, win, l);
    for (w, &a0) in starts.iter().enumerate() {
        if !scored[w] {
            continue;
        }
        work.filter_frame(&sig, &filters, &fft, a0);
        for (j, scores) in out.iter_mut().enumerate() {
            let f = &mut scores.frames[w];
            let e = work.energies(&sig, j, a0);
            f.sdr = db(e.target, e.error);
            f.isr = db(e.target, e.spatial);
            f.sir = db(e.image, e.interference);
            f.sar = db(e.projection, e.artifacts);
        }
    }
    Ok(out)
}

fn db(num: f64, den: f64) -> f64 {
    10.0 * (num / den).log10()
}

/// Validated view of the interleaved inputs. Channel `a = j * channels + c` of the flattened
/// list is channel `c` of source `j`.
struct Signals<'a> {
    refs: &'a [&'a [f32]],
    ests: &'a [&'a [f32]],
    ch: usize,
    len: usize,
}

impl<'a> Signals<'a> {
    fn new(refs: &'a [&'a [f32]], ests: &'a [&'a [f32]], ch: usize) -> Result<Self, Error> {
        if ch == 0 {
            return Err(Error::Channels(0));
        }
        if refs.is_empty() || refs.len() != ests.len() {
            return Err(Error::Mismatch(format!(
                "{} reference and {} estimated sources",
                refs.len(),
                ests.len()
            )));
        }
        let total = refs[0].len();
        if total % ch != 0 {
            return Err(Error::Mismatch(format!(
                "{total} samples are not a whole number of {ch}-channel frames"
            )));
        }
        if let Some(bad) = refs.iter().chain(ests).find(|s| s.len() != total) {
            return Err(Error::Mismatch(format!(
                "signal lengths differ: {} and {total} samples",
                bad.len()
            )));
        }
        Ok(Self {
            refs,
            ests,
            ch,
            len: total / ch,
        })
    }

    /// Number of flattened channels (sources × channels).
    fn count(&self) -> usize {
        self.refs.len() * self.ch
    }

    fn reference(&self, a: usize, t: usize) -> f64 {
        f64::from(self.refs[a / self.ch][t * self.ch + a % self.ch])
    }

    fn estimate(&self, a: usize, t: usize) -> f64 {
        f64::from(self.ests[a / self.ch][t * self.ch + a % self.ch])
    }

    /// Writes samples `from..from + dst.len()` of channel `a` (estimate or reference) into
    /// the real (`im == false`) or imaginary parts of `dst`, zero outside `0..upto`.
    fn segment(
        &self,
        estimate: bool,
        a: usize,
        from: isize,
        upto: usize,
        dst: &mut [Complex],
        im: bool,
    ) {
        let data = if estimate { self.ests } else { self.refs }[a / self.ch];
        let c = a % self.ch;
        let n = isize::try_from(dst.len()).expect("block fits isize");
        let upto = isize::try_from(upto.min(self.len)).expect("length fits isize");
        let lo = (-from).clamp(0, n);
        let hi = (upto - from).clamp(lo, n);
        let (lo, hi) = (lo.unsigned_abs(), hi.unsigned_abs());
        let set = |z: &mut Complex, v: f64| if im { z.1 = v } else { z.0 = v };
        for z in &mut dst[..lo] {
            set(z, 0.0);
        }
        for z in &mut dst[hi..] {
            set(z, 0.0);
        }
        if hi > lo {
            let t0 = (from + isize::try_from(lo).expect("fits")).unsigned_abs();
            let src = data[t0 * self.ch + c..].iter().step_by(self.ch);
            for (z, &v) in dst[lo..hi].iter_mut().zip(src) {
                set(z, f64::from(v));
            }
        }
    }

    /// True if every reference source has a non-zero sample in `[a0, a0 + win)`.
    fn all_sounding(&self, a0: usize, win: usize) -> bool {
        self.refs.iter().all(|r| {
            r[a0 * self.ch..(a0 + win) * self.ch]
                .iter()
                .any(|&x| x != 0.0)
        })
    }
}

/// Transform size of the block FFTs: at least four filter lengths, at least 8192.
fn block_fft_len(l: usize) -> usize {
    (4 * l).next_power_of_two().max(8192)
}

/// Forward transform of two real signals at once; returns their half spectra (`0..=n/2`).
fn forward_pair(fft: &Fft, buf: &mut [Complex], x: &mut [Complex], y: &mut [Complex]) {
    fft.forward(buf);
    let n = buf.len();
    for k in 0..=n / 2 {
        let (zr, zi) = buf[k];
        let (mr, mi) = buf[(n - k) % n];
        x[k] = (0.5 * (zr + mr), 0.5 * (zi - mi));
        y[k] = (0.5 * (zi + mi), -0.5 * (zr - mr));
    }
}

/// Inverse transform of two real signals given by their half spectra; real outputs go to
/// `buf[t].0` (from `x`) and `buf[t].1` (from `y`), scaled by `1/n`.
fn inverse_pair(fft: &Fft, x: &[Complex], y: &[Complex], buf: &mut [Complex]) {
    let n = buf.len();
    // Z = X + iY on the full circle, conjugated for the inverse-by-forward trick.
    for k in 0..=n / 2 {
        let (xr, xi) = x[k];
        let (yr, yi) = y[k];
        buf[k] = (xr - yi, -(xi + yr));
        if k > 0 && k < n / 2 {
            // Z[n-k] = conj(X[k]) + i conj(Y[k]).
            buf[n - k] = (xr + yi, -(-xi + yr));
        }
    }
    fft.forward(buf);
    #[allow(clippy::cast_precision_loss)]
    let s = 1.0 / n as f64;
    for z in buf.iter_mut() {
        *z = (z.0 * s, -z.1 * s);
    }
}

/// Cross-correlations of all reference channels with each other and with all estimate
/// channels, over the zero-padded signals, for lags `-(l-1)..=l-1`.
struct Correlations {
    l: usize,
    m: usize,
    /// `rr[pair(a, b)][τ + l - 1] = Σ_t r_a(t) r_b(t + τ)` for `a ≤ b`.
    rr: Vec<Vec<f64>>,
    /// `re[a * m + e][τ] = Σ_t r_a(t) x_e(t + τ)` for `τ` in `0..l`.
    re: Vec<Vec<f64>>,
}

impl Correlations {
    fn pair(m: usize, a: usize, b: usize) -> usize {
        a * m - a * (a + 1) / 2 + b
    }

    /// Block FFT correlation: each block of `b` samples of `r_a` (zero-padded to `n`) is
    /// correlated with the segment of the other signal starting `l - 1` samples earlier; the
    /// products are summed over blocks in the frequency domain and transformed back once.
    fn compute(sig: &Signals<'_>, l: usize) -> Self {
        let m = sig.count();
        let n = block_fft_len(l);
        let half = n / 2 + 1;
        let block = n - 2 * (l - 1);
        let fft = Fft::new(n);
        let pairs = m * (m + 1) / 2;
        let mut acc_rr = vec![vec![(0.0, 0.0); half]; pairs];
        let mut acc_re = vec![vec![(0.0, 0.0); half]; m * m];
        // Half spectra of the 3m inputs of a block: reference blocks (A, `0..m`), reference
        // segments (B, `m..2m`), estimate segments (X, `2m..3m`); one spare slot pads an odd
        // count.
        let total = 3 * m;
        let mut spec = vec![vec![(0.0, 0.0); half]; total + 1];
        let mut buf = vec![(0.0, 0.0); n];
        let shift = isize::try_from(l - 1).expect("filter length fits isize");
        let mut start = 0;
        while start < sig.len {
            let s0 = isize::try_from(start).expect("length fits isize");
            let end = (start + block).min(sig.len);
            let fill = |idx: usize, buf: &mut [Complex], im: bool| {
                let (kind, a) = (idx / m, idx % m);
                match kind {
                    0 => sig.segment(false, a, s0, end, buf, im),
                    1 => sig.segment(false, a, s0 - shift, sig.len, buf, im),
                    2 => sig.segment(true, a, s0 - shift, sig.len, buf, im),
                    _ => buf.iter_mut().for_each(|z| z.1 = 0.0),
                }
            };
            let mut k = 0;
            while k < total {
                fill(k, &mut buf, false);
                fill(k + 1, &mut buf, true);
                let (lo, hi) = spec.split_at_mut(k + 1);
                forward_pair(&fft, &mut buf, &mut lo[k], &mut hi[0]);
                k += 2;
            }
            for a in 0..m {
                for b in a..m {
                    mul_conj_acc(&mut acc_rr[Self::pair(m, a, b)], &spec[a], &spec[m + b]);
                }
                for e in 0..m {
                    mul_conj_acc(&mut acc_re[a * m + e], &spec[a], &spec[2 * m + e]);
                }
            }
            start = end;
        }
        let lags = 2 * l - 1;
        let mut back = |acc: &[Vec<Complex>], keep: usize, offset: usize| -> Vec<Vec<f64>> {
            let mut out = Vec::with_capacity(acc.len());
            let mut i = 0;
            while i < acc.len() {
                let zero = vec![(0.0, 0.0); half];
                let y = if i + 1 < acc.len() {
                    &acc[i + 1]
                } else {
                    &zero
                };
                inverse_pair(&fft, &acc[i], y, &mut buf);
                out.push(buf[offset..offset + keep].iter().map(|z| z.0).collect());
                if i + 1 < acc.len() {
                    out.push(buf[offset..offset + keep].iter().map(|z| z.1).collect());
                }
                i += 2;
            }
            out
        };
        // Circular index τ' = τ + l - 1 holds lag τ.
        let rr = back(&acc_rr, lags, 0);
        let re = back(&acc_re, l, l - 1);
        Self { l, m, rr, re }
    }

    /// `Σ_t r_a(t) r_b(t + τ)`.
    fn cross(&self, a: usize, b: usize, tau: isize) -> f64 {
        let off = isize::try_from(self.l - 1).expect("filter length fits isize");
        let (i, j, t) = if a <= b { (a, b, tau) } else { (b, a, -tau) };
        self.rr[Self::pair(self.m, i, j)][usize::try_from(t + off).expect("lag in range")]
    }
}

/// `acc += conj(a) · b`.
fn mul_conj_acc(acc: &mut [Complex], a: &[Complex], b: &[Complex]) {
    for ((s, &(ar, ai)), &(br, bi)) in acc.iter_mut().zip(a).zip(b) {
        s.0 += ar * br + ai * bi;
        s.1 += ar * bi - ai * br;
    }
}

/// Lower triangle of a symmetric matrix, row by row (`(i, j)`, `j ≤ i`, at `i(i+1)/2 + j`).
struct Packed {
    n: usize,
    v: Vec<f64>,
}

impl Packed {
    fn row(i: usize) -> usize {
        i * (i + 1) / 2
    }

    /// In-place Cholesky factorisation `A = L Lᵀ`. False if a pivot is not positive.
    fn cholesky(&mut self) -> bool {
        // Rows are factorised in tiles of four that share each loaded row `j` (left-looking).
        const TILE: usize = 4;
        let mut i0 = 0;
        while i0 < self.n {
            let i1 = (i0 + TILE).min(self.n);
            let r0 = Self::row(i0);
            let (head, tail) = self.v.split_at_mut(r0);
            let start = |i: usize| Self::row(i) - r0;
            // Columns left of the tile: rows of the tile against finished rows.
            for j in 0..i0 {
                let rj = Self::row(j);
                let lj = &head[rj..rj + j];
                let mut d = [0.0; TILE];
                {
                    let rows: [&[f64]; TILE] = core::array::from_fn(|k| {
                        let i = (i0 + k).min(i1 - 1);
                        &tail[start(i)..start(i) + j]
                    });
                    dot_tile(&rows, lj, &mut d);
                }
                let pivot = head[rj + j];
                for (k, &dk) in d.iter().enumerate().take(i1 - i0) {
                    let p = start(i0 + k) + j;
                    tail[p] = (tail[p] - dk) / pivot;
                }
            }
            // The triangle inside the tile.
            for i in i0..i1 {
                let ri = start(i);
                for j in i0..=i {
                    let rj = start(j);
                    let s = tail[ri + j] - dot(&tail[ri..ri + j], &tail[rj..rj + j]);
                    if j == i {
                        if s.is_nan() || s <= 0.0 {
                            return false;
                        }
                        tail[ri + j] = s.sqrt();
                    } else {
                        tail[ri + j] = s / tail[rj + j];
                    }
                }
            }
            i0 = i1;
        }
        true
    }

    /// Solves `L Lᵀ x = b` in place with the factor from [`Packed::cholesky`].
    fn solve(&self, b: &mut [f64]) {
        for i in 0..self.n {
            let ri = Self::row(i);
            b[i] = (b[i] - dot(&self.v[ri..ri + i], &b[..i])) / self.v[ri + i];
        }
        for i in (0..self.n).rev() {
            let ri = Self::row(i);
            b[i] /= self.v[ri + i];
            let xi = b[i];
            for (bk, &lk) in b[..i].iter_mut().zip(&self.v[ri..ri + i]) {
                *bk -= lk * xi;
            }
        }
    }
}

/// Four dot products against the same vector `b` (rows of equal length), each with the
/// same four partial sums as [`dot`].
fn dot_tile(rows: &[&[f64]; 4], b: &[f64], out: &mut [f64; 4]) {
    let mut s = [[0.0; 4]; 4];
    let n4 = b.len() / 4 * 4;
    let mut t = 0;
    while t < n4 {
        let y = &b[t..t + 4];
        for (acc, r) in s.iter_mut().zip(rows) {
            let x = &r[t..t + 4];
            acc[0] += x[0] * y[0];
            acc[1] += x[1] * y[1];
            acc[2] += x[2] * y[2];
            acc[3] += x[3] * y[3];
        }
        t += 4;
    }
    for ((o, acc), r) in out.iter_mut().zip(&s).zip(rows) {
        let mut v = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for (x, y) in r[n4..].iter().zip(&b[n4..]) {
            v += x * y;
        }
        *o = v;
    }
}

/// Dot product with four partial sums (fixed order, so results are reproducible).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = [0.0; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        s[0] += x[0] * y[0];
        s[1] += x[1] * y[1];
        s[2] += x[2] * y[2];
        s[3] += x[3] * y[3];
    }
    let mut t = (s[0] + s[1]) + (s[2] + s[3]);
    for (x, y) in ra.iter().zip(rb) {
        t += x * y;
    }
    t
}

/// Solves the normal equations of the projection of estimate channels on the reference
/// channels `sel` (filters of `l` taps each); returns one coefficient vector per estimate
/// channel in `ests`, laid out as `sel.len()` filters of `l` taps.
fn fit(corr: &Correlations, sel: &[usize], ests: &[usize]) -> Vec<Vec<f64>> {
    let l = corr.l;
    let n = sel.len() * l;
    let gram = |load: f64| {
        let mut v = Vec::with_capacity(n * (n + 1) / 2);
        for r in 0..n {
            let (p, t1) = (r / l, r % l);
            for c in 0..=r {
                let (q, t2) = (c / l, c % l);
                let tau = isize::try_from(t1).expect("tap fits isize")
                    - isize::try_from(t2).expect("tap fits isize");
                let mut g = corr.cross(sel[p], sel[q], tau);
                if r == c {
                    g += load;
                }
                v.push(g);
            }
        }
        Packed { n, v }
    };
    let mut g = gram(0.0);
    if !g.cholesky() {
        // Singular system (for example a channel that is silent for the whole track, or two
        // exactly proportional channels): add a tiny diagonal load so a solution exists. The
        // scores of such inputs are not comparable with other implementations.
        #[allow(clippy::cast_precision_loss)]
        let trace = (0..n).map(|i| g_diag(corr, sel, i)).sum::<f64>() / n as f64;
        let mut load = trace.max(f64::MIN_POSITIVE) * 1e-10;
        loop {
            g = gram(load);
            if g.cholesky() {
                break;
            }
            load *= 100.0;
        }
    }
    ests.iter()
        .map(|&e| {
            let mut b: Vec<f64> = sel
                .iter()
                .flat_map(|&a| corr.re[a * corr.m + e].iter().copied())
                .collect();
            g.solve(&mut b);
            b
        })
        .collect()
}

fn g_diag(corr: &Correlations, sel: &[usize], i: usize) -> f64 {
    let a = sel[i / corr.l];
    corr.cross(a, a, 0)
}

/// Half spectra of the fitted filters, for block convolution.
struct Filters {
    n: usize,
    l: usize,
    /// `all[e][a]`: filter from reference channel `a` to estimate channel `e` (all sources).
    all: Vec<Vec<Vec<Complex>>>,
    /// `own[e][c]`: filter from channel `c` of the same source to estimate channel `e`.
    own: Vec<Vec<Vec<Complex>>>,
}

impl Filters {
    fn fit(sig: &Signals<'_>, corr: &Correlations) -> Self {
        let (m, ch, l) = (sig.count(), sig.ch, corr.l);
        let n = block_fft_len(l);
        let fft = Fft::new(n);
        let every: Vec<usize> = (0..m).collect();
        let ests: Vec<usize> = (0..m).collect();
        let coef_all = fit(corr, &every, &ests);
        let mut coef_own = Vec::with_capacity(m);
        for j in 0..sig.refs.len() {
            let sel: Vec<usize> = (j * ch..(j + 1) * ch).collect();
            coef_own.extend(fit(corr, &sel, &sel));
        }
        let spectra = |coef: &[f64], count: usize| -> Vec<Vec<Complex>> {
            let mut out = vec![vec![(0.0, 0.0); n / 2 + 1]; count];
            let mut buf = vec![(0.0, 0.0); n];
            let mut spare = vec![(0.0, 0.0); n / 2 + 1];
            let mut a = 0;
            while a < count {
                buf.fill((0.0, 0.0));
                for t in 0..l {
                    buf[t].0 = coef[a * l + t];
                    if a + 1 < count {
                        buf[t].1 = coef[(a + 1) * l + t];
                    }
                }
                if a + 1 < count {
                    let (lo, hi) = out.split_at_mut(a + 1);
                    forward_pair(&fft, &mut buf, &mut lo[a], &mut hi[0]);
                } else {
                    forward_pair(&fft, &mut buf, &mut out[a], &mut spare);
                }
                a += 2;
            }
            out
        };
        Self {
            n,
            l,
            all: coef_all.iter().map(|c| spectra(c, m)).collect(),
            own: coef_own.iter().map(|c| spectra(c, ch)).collect(),
        }
    }

    fn fft(&self) -> Fft {
        Fft::new(self.n)
    }
}

/// Per-window buffers: the projections `P` and `P_j` of every estimate channel.
struct FrameWork {
    win: usize,
    l: usize,
    /// `proj[e]`: `(P, P_j)` of estimate channel `e`, `win + l - 1` samples.
    proj: Vec<Vec<(f64, f64)>>,
    spec: Vec<Vec<Complex>>,
    buf: Vec<Complex>,
    ya: Vec<Complex>,
    yj: Vec<Complex>,
}

/// Energy sums of one source in one window.
struct Energies {
    target: f64,
    error: f64,
    spatial: f64,
    image: f64,
    interference: f64,
    projection: f64,
    artifacts: f64,
}

impl FrameWork {
    fn new(sig: &Signals<'_>, win: usize, l: usize) -> Self {
        let m = sig.count();
        let n = block_fft_len(l);
        Self {
            win,
            l,
            proj: vec![vec![(0.0, 0.0); win + l - 1]; m],
            spec: vec![vec![(0.0, 0.0); n / 2 + 1]; m + 1],
            buf: vec![(0.0, 0.0); n],
            ya: vec![(0.0, 0.0); n / 2 + 1],
            yj: vec![(0.0, 0.0); n / 2 + 1],
        }
    }

    /// Filters the reference segments `[a0, a0 + win)` (zero-padded) with the track-wide
    /// filters, by overlap-add of FFT blocks.
    fn filter_frame(&mut self, sig: &Signals<'_>, f: &Filters, fft: &Fft, a0: usize) {
        let (m, ch, n, l) = (sig.count(), sig.ch, f.n, f.l);
        let block = n - l + 1;
        for p in &mut self.proj {
            p.fill((0.0, 0.0));
        }
        let mut q0 = 0;
        while q0 < self.win {
            let q1 = (q0 + block).min(self.win);
            let mut a = 0;
            while a < m {
                for (u, z) in self.buf.iter_mut().enumerate() {
                    let t = a0 + q0 + u;
                    let inside = q0 + u < q1;
                    let x = if inside { sig.reference(a, t) } else { 0.0 };
                    let y = if inside && a + 1 < m {
                        sig.reference(a + 1, t)
                    } else {
                        0.0
                    };
                    *z = (x, y);
                }
                let (lo, hi) = self.spec.split_at_mut(a + 1);
                forward_pair(fft, &mut self.buf, &mut lo[a], &mut hi[0]);
                a += 2;
            }
            let len = q1 - q0 + l - 1;
            for e in 0..m {
                let j = e / ch;
                self.ya.fill((0.0, 0.0));
                self.yj.fill((0.0, 0.0));
                for (a, h) in f.all[e].iter().enumerate() {
                    mul_acc(&mut self.ya, &self.spec[a], h);
                }
                for (c, h) in f.own[e].iter().enumerate() {
                    mul_acc(&mut self.yj, &self.spec[j * ch + c], h);
                }
                inverse_pair(fft, &self.ya, &self.yj, &mut self.buf);
                for (p, z) in self.proj[e][q0..q0 + len].iter_mut().zip(&self.buf) {
                    p.0 += z.0;
                    p.1 += z.1;
                }
            }
            q0 = q1;
        }
    }

    fn energies(&self, sig: &Signals<'_>, j: usize, a0: usize) -> Energies {
        let mut e = Energies {
            target: 0.0,
            error: 0.0,
            spatial: 0.0,
            image: 0.0,
            interference: 0.0,
            projection: 0.0,
            artifacts: 0.0,
        };
        for c in 0..sig.ch {
            let a = j * sig.ch + c;
            for (t, &(pa, pj)) in self.proj[a].iter().enumerate() {
                let (s, x) = if t < self.win {
                    (sig.reference(a, a0 + t), sig.estimate(a, a0 + t))
                } else {
                    (0.0, 0.0)
                };
                e.target += s * s;
                e.error += (x - s) * (x - s);
                e.spatial += (pj - s) * (pj - s);
                e.image += pj * pj;
                e.interference += (pa - pj) * (pa - pj);
                e.projection += pa * pa;
                e.artifacts += (x - pa) * (x - pa);
            }
        }
        let _ = self.l;
        e
    }
}

/// `acc += a · b`.
fn mul_acc(acc: &mut [Complex], a: &[Complex], b: &[Complex]) {
    for ((s, &(ar, ai)), &(br, bi)) in acc.iter_mut().zip(a).zip(b) {
        s.0 += ar * br - ai * bi;
        s.1 += ar * bi + ai * br;
    }
}

/// Scale-invariant criteria of one channel, in dB.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ScaleInvariant {
    /// Scale-invariant source to distortion ratio.
    pub si_sdr: f64,
    /// Scale-invariant source to interference ratio (`+∞` with a single source).
    pub si_sir: f64,
    /// Scale-invariant source to artifacts ratio.
    pub si_sar: f64,
}

/// Scale-invariant criteria of one estimated source.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SiScores {
    /// One entry per channel.
    pub channels: Vec<ScaleInvariant>,
    /// Arithmetic mean of the channel values (in dB).
    pub mean: ScaleInvariant,
}

/// SI-SDR, SI-SIR and SI-SAR of every estimated source, per channel and averaged.
///
/// Inputs as for [`bss_eval`]. The interference subspace of each channel is spanned by the
/// same channel of all reference sources. With `zero_mean`, each channel of every signal has
/// its mean removed first (a common pre-processing step; Le Roux et al. assume zero-mean
/// signals).
///
/// # Errors
/// As [`bss_eval`] for the shape of the inputs; [`Error::TooShort`] for empty signals.
pub fn scale_invariant(
    references: &[&[f32]],
    estimates: &[&[f32]],
    channels: usize,
    zero_mean: bool,
) -> Result<Vec<SiScores>, Error> {
    let sig = Signals::new(references, estimates, channels)?;
    if sig.len == 0 {
        return Err(Error::TooShort(0));
    }
    let n_src = references.len();
    let mut out: Vec<SiScores> = (0..n_src)
        .map(|_| SiScores {
            channels: Vec::with_capacity(channels),
            mean: ScaleInvariant {
                si_sdr: 0.0,
                si_sir: 0.0,
                si_sar: 0.0,
            },
        })
        .collect();
    for c in 0..channels {
        for (scores, v) in out.iter_mut().zip(si_channel(&sig, c, zero_mean)) {
            scores.channels.push(v);
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let k = channels as f64;
    for s in &mut out {
        let mean = |f: fn(&ScaleInvariant) -> f64| s.channels.iter().map(f).sum::<f64>() / k;
        s.mean = ScaleInvariant {
            si_sdr: mean(|v| v.si_sdr),
            si_sir: mean(|v| v.si_sir),
            si_sar: mean(|v| v.si_sar),
        };
    }
    Ok(out)
}

/// Scale-invariant criteria of every source in channel `c`. Two passes over the inputs,
/// without copying them: inner products first, then the energies of the decomposition.
fn si_channel(sig: &Signals<'_>, c: usize, zero_mean: bool) -> Vec<ScaleInvariant> {
    let n_src = sig.refs.len();
    let channels = sig.ch;
    #[allow(clippy::cast_precision_loss)]
    let len = sig.len as f64;
    let mut r = vec![0.0; n_src];
    let mut x = vec![0.0; n_src];
    let at = |j: usize| j * channels + c;
    let (mut mr, mut mx) = (vec![0.0; n_src], vec![0.0; n_src]);
    if zero_mean {
        for j in 0..n_src {
            mr[j] = (0..sig.len).map(|t| sig.reference(at(j), t)).sum::<f64>() / len;
            mx[j] = (0..sig.len).map(|t| sig.estimate(at(j), t)).sum::<f64>() / len;
        }
    }
    let load = |t: usize, r: &mut [f64], x: &mut [f64]| {
        for j in 0..n_src {
            r[j] = sig.reference(at(j), t) - mr[j];
            x[j] = sig.estimate(at(j), t) - mx[j];
        }
    };
    // gram[i][k] = <r_i, r_k>; cross[j][i] = <x_j, r_i>.
    let mut gram = vec![vec![0.0; n_src]; n_src];
    let mut cross = vec![vec![0.0; n_src]; n_src];
    for t in 0..sig.len {
        load(t, &mut r, &mut x);
        for i in 0..n_src {
            for k in 0..=i {
                gram[i][k] += r[i] * r[k];
            }
            for j in 0..n_src {
                cross[j][i] += x[j] * r[i];
            }
        }
    }
    let mut g = Packed {
        n: n_src,
        v: (0..n_src)
            .flat_map(|i| (0..=i).map(move |k| (i, k)))
            .map(|(i, k)| gram[i][k])
            .collect(),
    };
    let full_rank = g.cholesky();
    let alpha: Vec<f64> = (0..n_src).map(|j| cross[j][j] / gram[j][j]).collect();
    let beta: Vec<Vec<f64>> = cross
        .iter()
        .map(|b| {
            let mut b = b.clone();
            if full_rank {
                g.solve(&mut b);
            } else {
                b.fill(f64::NAN);
            }
            b
        })
        .collect();
    // Per estimate: target, residual, interference, artifacts energies.
    let mut e = vec![[0.0; 4]; n_src];
    for t in 0..sig.len {
        load(t, &mut r, &mut x);
        for j in 0..n_src {
            let et = alpha[j] * r[j];
            let p: f64 = r.iter().zip(&beta[j]).map(|(r, b)| b * r).sum();
            let s = &mut e[j];
            s[0] += et * et;
            s[1] += (x[j] - et) * (x[j] - et);
            s[2] += (p - et) * (p - et);
            s[3] += (x[j] - p) * (x[j] - p);
        }
    }
    e.iter()
        .map(|s| ScaleInvariant {
            si_sdr: db(s[0], s[1]),
            si_sir: db(s[0], s[2]),
            si_sar: db(s[0], s[3]),
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn pair_transforms_round_trip() {
        let n = 16;
        let fft = Fft::new(n);
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 1.3).cos() + 0.1).collect();
        let mut buf: Vec<Complex> = x.iter().zip(&y).map(|(&a, &b)| (a, b)).collect();
        let (mut sx, mut sy) = (vec![(0.0, 0.0); n / 2 + 1], vec![(0.0, 0.0); n / 2 + 1]);
        forward_pair(&fft, &mut buf, &mut sx, &mut sy);
        inverse_pair(&fft, &sx, &sy, &mut buf);
        for t in 0..n {
            assert!((buf[t].0 - x[t]).abs() < 1e-12 && (buf[t].1 - y[t]).abs() < 1e-12);
        }
    }

    #[test]
    fn cholesky_solves() {
        // A = [[4, 2], [2, 3]], b = [2, 1] -> x = [0.5, 0].
        let mut p = Packed {
            n: 2,
            v: vec![4.0, 2.0, 3.0],
        };
        assert!(p.cholesky());
        let mut b = [2.0, 1.0];
        p.solve(&mut b);
        assert!((b[0] - 0.5).abs() < 1e-15 && b[1].abs() < 1e-15);
    }

    #[test]
    fn median_ignores_nan() {
        assert_eq!(nan_median([3.0, f64::NAN, 1.0, 2.0].into_iter()), 2.0);
        assert_eq!(nan_median([4.0, 1.0].into_iter()), 2.5);
        assert!(nan_median([f64::NAN].into_iter()).is_nan());
    }
}

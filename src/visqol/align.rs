//! Level matching and global alignment (stage ②).
//!
//! ViSQOL scales the degraded signal to the reference's level and aligns the two globally
//! before building spectrograms (EURASIP 2015 §2 pre-processing; v3 Fig. 6 "global
//! alignment"). Level: the degraded signal is scaled so its mean square equals the
//! reference's (fixed against the reference tool's outputs). Alignment: the lag maximising the cross-correlation,
//! found coarsely on 16× block-averaged signals with an FFT, then refined at the full rate
//! within ±32 samples.

// Signal-processing code: short symbols follow the formulas; lags are signed sample counts
// far below 2^52; exact float comparison only breaks ties between equal correlation sums.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]

use super::source::{mean_square, Blocks, BLOCK};
use crate::fft::{forward_untabled, Complex};

/// Block-average decimation factor of the coarse search.
const DECIM: usize = 16;
/// Refinement half-width at the full rate, samples.
const REFINE: isize = 32;

/// Block averages of `DECIM` samples (the last one shorter), written to `out[..]` through
/// `put(index, value)`. Returns the number of averages.
#[allow(clippy::cast_precision_loss)]
fn decimate(x: &impl Blocks, mut put: impl FnMut(usize, f64)) -> usize {
    let len = x.len();
    let mut buf = vec![0.0; BLOCK.min(len)];
    let mut k = 0;
    let mut s = 0;
    while s < len {
        let n = BLOCK.min(len - s);
        x.fill(s, &mut buf[..n]);
        for c in buf[..n].chunks(DECIM) {
            put(k, c.iter().sum::<f64>() / c.len() as f64);
            k += 1;
        }
        s += n;
    }
    k
}

/// Lag of the cross-correlation maximum `c[L] = Σ rc[n] · dc[n + L]` of the decimated
/// signals, via one complex FFT holding both (real/imaginary packing), computed in place:
/// the only buffer is the transform itself. Ties go to the most negative lag.
// `0.5 * (x + y)` kept as written (same rounding; `f64::midpoint` would raise the MSRV).
#[allow(unknown_lints, clippy::manual_midpoint)]
#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
fn coarse_lag(r: &impl Blocks, d: &impl Blocks) -> isize {
    let (rn, dn) = (r.len().div_ceil(DECIM), d.len().div_ceil(DECIM));
    let n = (rn + dn).next_power_of_two();
    let mut z: Vec<Complex> = vec![(0.0, 0.0); n];
    decimate(r, |i, v| z[i].0 = v);
    decimate(d, |i, v| z[i].1 = v);
    forward_untabled(&mut z);
    // R[k] = (Z[k] + conj Z[n−k]) / 2, D[k] = (Z[k] − conj Z[n−k]) / 2i; C = conj(R) · D.
    // Pairs (k, n−k) are updated together; the inverse transform is taken as
    // conj(fft(conj C)) / n, so conj C is stored.
    let c_of = |(zr, zi): Complex, (cr, ci): Complex| -> Complex {
        let (rr, ri) = (0.5 * (zr + cr), 0.5 * (zi - ci));
        let (dr, di) = (0.5 * (zi + ci), -0.5 * (zr - cr));
        (rr * dr + ri * di, -(rr * di - ri * dr))
    };
    for k in 0..=n / 2 {
        let k2 = (n - k) % n;
        let (a, b) = (z[k], z[k2]);
        z[k] = c_of(a, b);
        if k2 != k {
            z[k2] = c_of(b, a);
        }
    }
    forward_untabled(&mut z);
    let nf = n as f64;
    // Lags −rn..0 sit at the end of the buffer, 0..dn at the start.
    let value = |lag: isize| -> f64 {
        let i = if lag < 0 {
            (n as isize + lag) as usize
        } else {
            lag as usize
        };
        z[i].0 / nf
    };
    let mut best = (-(rn as isize), value(-(rn as isize)));
    for lag in -(rn as isize) + 1..dn as isize {
        let v = value(lag);
        if v.total_cmp(&best.1).is_gt() {
            best = (lag, v);
        }
    }
    best.0
}

/// Lag `L` (samples) such that `d[n + L]` best matches `r[n]`: coarse search on 16×
/// block averages, then the full-rate correlation within ±32 samples (ties: smallest |L|).
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
pub fn global_lag(r: &impl Blocks, d: &impl Blocks) -> isize {
    if r.len() == 0 || d.len() == 0 {
        return 0;
    }
    let coarse = coarse_lag(r, d) * DECIM as isize;
    // Σ_i r[i] · d[i + lag] for every lag of the window, each sum accumulated in i order.
    let lags: Vec<isize> = (coarse - REFINE..=coarse + REFINE).collect();
    let mut acc = vec![0.0f64; lags.len()];
    let (rlen, dlen) = (r.len() as isize, d.len() as isize);
    let mut rb = vec![0.0; BLOCK.min(r.len())];
    let mut db = vec![0.0; BLOCK.min(r.len()) + 2 * REFINE as usize];
    let mut s = 0isize;
    while s < rlen {
        let n = (BLOCK as isize).min(rlen - s);
        r.fill(s as usize, &mut rb[..n as usize]);
        // Degraded indices needed: s + lags[0] .. s + n + lags[last], clipped.
        let (ds, de) = (
            (s + lags[0]).clamp(0, dlen),
            (s + n + coarse + REFINE).clamp(0, dlen),
        );
        if de > ds {
            d.fill(ds as usize, &mut db[..(de - ds) as usize]);
        }
        // For lag l, i runs over s..s+n with 0 <= i + l < dlen, in increasing i. Inside
        // [i0, i1) every lag is valid; there the lags are interleaved (independent sums, so
        // the CPU can overlap them) while each sum still adds its terms in increasing i.
        let (lmin, lmax) = (lags[0], lags[lags.len() - 1]);
        let i0 = s.max(-lmin);
        let i1 = (s + n).min(dlen - lmax).max(i0);
        let one = |a: &mut f64, lag: isize, lo: isize, hi: isize| {
            for i in lo.max(s).max(-lag)..hi.min(s + n).min(dlen - lag) {
                *a += rb[(i - s) as usize] * db[(i + lag - ds) as usize];
            }
        };
        for (a, &lag) in acc.iter_mut().zip(&lags) {
            one(a, lag, s, i0);
        }
        for i in i0..i1 {
            let x = rb[(i - s) as usize];
            let base = (i + lmin - ds) as usize;
            for (a, y) in acc.iter_mut().zip(&db[base..base + lags.len()]) {
                *a += x * y;
            }
        }
        for (a, &lag) in acc.iter_mut().zip(&lags) {
            one(a, lag, i1, s + n);
        }
        s += n;
    }
    let mut best = (coarse, f64::NEG_INFINITY);
    for (&lag, &v) in lags.iter().zip(&acc) {
        if v > best.1 || (v == best.1 && lag.abs() < best.0.abs()) {
            best = (lag, v);
        }
    }
    best.0
}

/// Gain that brings `d` to the mean square of the reference (`ref_ms`); `None` for a
/// silent `d` (left unchanged).
pub fn level_gain(ref_ms: f64, d: &impl Blocks) -> Option<f64> {
    let b = mean_square(d);
    (b > 0.0).then(|| (ref_ms / b).sqrt())
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::super::source::Mono;
    use super::*;

    fn noise(n: usize, seed: u32) -> Vec<f64> {
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

    #[test]
    fn finds_delays_both_ways() {
        let r = noise(48_000, 7);
        for lag in [0isize, 5, 480, 24_000, -333] {
            let d = if lag >= 0 {
                let mut v = vec![0.0; lag as usize];
                v.extend_from_slice(&r);
                v
            } else {
                r[(-lag) as usize..].to_vec()
            };
            assert_eq!(global_lag(&Mono::F64(&r), &Mono::F64(&d)), lag);
            let a = super::super::source::Aligned {
                src: &Mono::F64(&d),
                lag,
                len: r.len(),
                gain: None,
            };
            let mut out = vec![0.0; r.len()];
            a.fill(0, &mut out);
            assert!(out.iter().zip(&r).all(|(x, y)| x == y || lag < 0));
        }
    }

    #[test]
    fn level_matching() {
        let r = noise(1000, 3);
        let d: Vec<f64> = r.iter().map(|v| v * 0.25).collect();
        let g = level_gain(mean_square(&Mono::F64(&r)), &Mono::F64(&d)).unwrap();
        assert!(d.iter().zip(&r).all(|(a, b)| (a * g - b).abs() < 1e-12));
        assert!(level_gain(1.0, &Mono::F64(&[0.0; 10])).is_none());
    }
}

//! ERB-spaced 4th-order gammatone filterbank (stage ③, filter part).
//!
//! ViSQOL v3 builds its spectrogram with gammatone ERB filters reaching 24 kHz in audio mode
//! (Chinen et al. 2020, §II, Fig. 6). The filters here follow the published Patterson–
//! Holdsworth design as described by M. Slaney, "An Efficient Implementation of the
//! Patterson-Holdsworth Auditory Filter Bank", Apple Computer Technical Report #35 (1993):
//!
//! - `ERB(f) = f / EarQ + minBW` (Glasberg & Moore, EarQ = 9.26449, minBW = 24.7 Hz, order 1)
//! - centre frequencies equally spaced on the ERB-rate scale between `low` and `high`
//!   (`cf_k = −EarQ·minBW + (high + EarQ·minBW) · exp(−k/N · ln((high + EarQ·minBW) /
//!   (low + EarQ·minBW)))`, `k = 1..=N`, so the lowest band sits exactly at `low`)
//! - impulse response `t³ e^{−2πbt} cos(2πf t)` with `b = 1.019·ERB(f)`, realised as four
//!   cascaded second-order sections sharing one denominator (impulse-invariant design), each
//!   cascade normalised to unit gain at its centre frequency.
//!
//! Bands are returned low → high.

use std::f64::consts::PI;

const EAR_Q: f64 = 9.264_49;
const MIN_BW: f64 = 24.7;

/// Equivalent rectangular bandwidth (Hz) at `f` (Glasberg & Moore 1990, order 1).
pub fn erb(f: f64) -> f64 {
    f / EAR_Q + MIN_BW
}

/// `n` centre frequencies equally spaced on the ERB-rate scale, ascending, the lowest equal
/// to `low` and the highest just below `high` (Slaney 1993 spacing).
#[allow(clippy::cast_precision_loss)]
pub fn erb_space(low: f64, high: f64, n: usize) -> Vec<f64> {
    let c = EAR_Q * MIN_BW;
    let step = ((high + c) / (low + c)).ln() / n as f64;
    let mut cf: Vec<f64> = (1..=n)
        .map(|k| -c + (high + c) * (-(k as f64) * step).exp())
        .collect();
    cf.reverse();
    cf
}

/// One band: four second-order sections `(b0, b1k, b2) / (1, a1, a2)` and a gain.
#[derive(Debug, Clone)]
pub struct Band {
    /// Centre frequency, Hz.
    pub cf: f64,
    b0: f64,
    b1: [f64; 4],
    a1: f64,
    a2: f64,
    /// Multiplier applied once to the cascade so `|H(cf)| = 1`.
    gain: f64,
}

impl Band {
    /// Design the band for centre frequency `cf` at sample rate `fs`.
    pub fn new(cf: f64, fs: f64) -> Self {
        let t = 1.0 / fs;
        let bw = 1.019 * 2.0 * PI * erb(cf);
        let (c, s) = ((2.0 * PI * cf * t).cos(), (2.0 * PI * cf * t).sin());
        let e = (bw * t).exp();
        let r_plus = (3.0 + 2f64.powf(1.5)).sqrt();
        let r_minus = (3.0 - 2f64.powf(1.5)).sqrt();
        let a1k = |r: f64| -(2.0 * t * c / e + 2.0 * r * t * s / e) / 2.0;
        let mut band = Self {
            cf,
            b0: t,
            b1: [a1k(r_plus), a1k(-r_plus), a1k(r_minus), a1k(-r_minus)],
            a1: -2.0 * c / e,
            a2: (-2.0 * bw * t).exp(),
            gain: 1.0,
        };
        band.gain = 1.0 / band.response(cf, fs);
        band
    }

    /// `|H(e^{jω})|` of the cascade (including the current gain) at frequency `f`.
    pub fn response(&self, f: f64, fs: f64) -> f64 {
        let w = 2.0 * PI * f / fs;
        // z^-1 and z^-2 as complex numbers.
        let (z1r, z1i) = (w.cos(), -w.sin());
        let (z2r, z2i) = ((2.0 * w).cos(), -(2.0 * w).sin());
        let den = (
            1.0 + self.a1 * z1r + self.a2 * z2r,
            self.a1 * z1i + self.a2 * z2i,
        );
        let den_mag = (den.0 * den.0 + den.1 * den.1).sqrt();
        let mut mag = self.gain;
        for &b1 in &self.b1 {
            let num = (self.b0 + b1 * z1r, b1 * z1i);
            mag *= (num.0 * num.0 + num.1 * num.1).sqrt() / den_mag;
        }
        mag
    }

    /// Filter `x`, calling `out(i, y_i)` for every output sample (no allocation).
    pub fn run(&self, x: &[f64], mut out: impl FnMut(usize, f64)) {
        let mut st = State::default();
        for (i, &v) in x.iter().enumerate() {
            out(i, self.step(&mut st, v));
        }
    }

    /// One output sample for input `v`, advancing the filter state `st`.
    #[inline]
    pub(crate) fn step(&self, st: &mut State, v: f64) -> f64 {
        // Transposed direct form II per section; state (s1, s2) per section.
        let mut y = v * self.gain;
        for (k, s) in st.0.iter_mut().enumerate() {
            let o = self.b0 * y + s[0];
            s[0] = self.b1[k] * y - self.a1 * o + s[1];
            s[1] = -self.a2 * o; // b2 = 0
            y = o;
        }
        y
    }
}

/// Filter state of one band (four sections, two values each), zero at rest.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct State([[f64; 2]; 4]);

/// A bank of [`Band`]s, low → high.
#[derive(Debug, Clone)]
pub struct FilterBank {
    /// The bands, ascending centre frequency.
    pub bands: Vec<Band>,
}

impl FilterBank {
    /// `n` bands between `low` and `high` Hz at sample rate `fs`.
    pub fn new(fs: f64, n: usize, low: f64, high: f64) -> Self {
        Self {
            bands: erb_space(low, high, n)
                .into_iter()
                .map(|cf| Band::new(cf, fs))
                .collect(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    #[test]
    fn erb_spacing() {
        let cf = erb_space(50.0, 24_000.0, 32);
        assert_eq!(cf.len(), 32);
        assert!((cf[0] - 50.0).abs() < 1e-9);
        assert!(cf[31] < 24_000.0 && cf[31] > 15_000.0);
        // Equal steps on the ERB-rate scale ln(f + EarQ·minBW).
        let r = |f: f64| (f + EAR_Q * MIN_BW).ln();
        let d0 = r(cf[1]) - r(cf[0]);
        for w in cf.windows(2) {
            assert!(((r(w[1]) - r(w[0])) - d0).abs() < 1e-12);
        }
    }

    #[test]
    fn unit_gain_at_centre_and_peak_near_it() {
        for &cf in &[50.0, 300.0, 1000.0, 5000.0, 15_000.0, 22_000.0] {
            let b = Band::new(cf, FS);
            assert!((b.response(cf, FS) - 1.0).abs() < 1e-12);
            // The peak of a gammatone lies within a fraction of an ERB of cf.
            let (mut best, mut fbest) = (0.0, 0.0);
            let e = erb(cf);
            let mut f = (cf - 2.0 * e).max(1.0);
            while f < (cf + 2.0 * e).min(FS / 2.0) {
                let m = b.response(f, FS);
                if m > best {
                    (best, fbest) = (m, f);
                }
                f += e / 200.0;
            }
            assert!((fbest - cf).abs() < 0.15 * e, "cf {cf} peak {fbest}");
        }
    }

    #[test]
    fn equivalent_rectangular_bandwidth_is_erb() {
        // ∫|H(f)|² df / |H(cf)|² equals the ERB for a 4th-order gammatone with b = 1.019 ERB.
        for &cf in &[500.0, 1000.0, 4000.0] {
            let b = Band::new(cf, FS);
            let df = 0.5;
            let mut s = 0.0;
            let mut f = df / 2.0;
            while f < FS / 2.0 {
                s += b.response(f, FS).powi(2) * df;
                f += df;
            }
            let e = erb(cf);
            assert!((s / e - 1.0).abs() < 0.02, "cf {cf}: {s} vs {e}");
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn impulse_response_is_a_gammatone() {
        // Compare with the sampled analog gammatone t³ e^{-2πbt} cos(2πf t) by correlation.
        let cf = 1000.0;
        let b = Band::new(cf, FS);
        let n = 4800;
        let mut x = vec![0.0; n];
        x[0] = 1.0;
        let mut h = vec![0.0; n];
        b.run(&x, |i, y| h[i] = y);
        let bw = 2.0 * PI * 1.019 * erb(cf);
        let g: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / FS;
                t.powi(3) * (-bw * t).exp() * (2.0 * PI * cf * t).cos()
            })
            .collect();
        let dot = |a: &[f64], c: &[f64]| a.iter().zip(c).map(|(p, q)| p * q).sum::<f64>();
        let rho = dot(&h, &g) / (dot(&h, &h) * dot(&g, &g)).sqrt();
        assert!(rho > 0.999, "correlation {rho}");
    }

    #[test]
    fn sine_lands_in_its_band() {
        let fb = FilterBank::new(FS, 32, 50.0, 24_000.0);
        let k0 = 12;
        let f = fb.bands[k0].cf;
        let x: Vec<f64> = (0..9600)
            .map(|i| (2.0 * PI * f * i as f64 / FS).sin())
            .collect();
        let energy: Vec<f64> = fb
            .bands
            .iter()
            .map(|b| {
                let mut e = 0.0;
                b.run(&x, |i, y| {
                    if i >= 4800 {
                        e += y * y;
                    }
                });
                e
            })
            .collect();
        let kmax = (0..32)
            .max_by(|&a, &b| energy[a].total_cmp(&energy[b]))
            .unwrap();
        assert_eq!(kmax, k0);
        // Steady state of a unit sine through a unit-gain filter: mean square 0.5.
        assert!(
            (energy[k0] / 4800.0 - 0.5).abs() < 0.01,
            "{}",
            energy[k0] / 4800.0
        );
    }
}

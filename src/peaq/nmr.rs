//! `Total NMR_B` (BS.1387 Annex 2 §4.5.1, Basic Version): noise-to-mask ratio of the test
//! signal against the reference's masked threshold, averaged over frames.
//!
//! The reference values are the outputs of the sori-mp3 project's own Python implementation
//! (`peaq_nmr.py`, written from BS.1387-1 Annex 2, task T-04b), which the measurement harness
//! used until this module. Its constants are reproduced here exactly, including its three
//! documented choices (they move results by ≪ 0.1 dB):
//!
//! - the level scaling `fac` is taken from one frame of the 1019.5 Hz sine (§2.1.3 eqs 5–6
//!   use the maximum over 10 frames);
//! - band edges and centres come from the Bark formula `z = 7·asinh(f/650)` (the last centre
//!   is 17698.877 Hz; Table 6 lists 17690.045 Hz);
//! - the data boundary of §5.2.4.4 is taken per channel (the Recommendation uses one boundary
//!   over all channels), and one frame before the first data frame is included.
//!
//! The steps (Annex 2 clause numbers): §2.1.3 FFT of the Hann-windowed frame with level
//! scaling; §2.1.4 outer/middle-ear weighting; §2.1.5 grouping into 109 bands of 0.25 Bark
//! (80 Hz – 18 kHz) by fractional overlap of the FFT lines, floor 1e-12; §2.1.6 internal noise;
//! §2.1.7 level-dependent frequency spreading normalised by `NormSP`; §2.1.8 time spreading;
//! §3 / §4.5 masked threshold `M = Es / 10^(m/10)` with m = 3 dB up to 12 Bark and
//! 0.25·k·Δz dB above; the noise pattern is the squared difference of the weighted magnitude
//! spectra grouped into the same bands (no internal noise, no spreading); NMR of a frame is
//! the band mean of noise/mask; `Total NMR = 10·log10(frame mean)`, channels averaged.

use super::{spread_raw, N, SAMPLE_RATE, STEP, Z};
use crate::fft::Fft;

/// FFT lines `0..=N/2`.
pub(crate) const BINS: usize = N / 2 + 1;
/// Pitch resolution, Bark.
const DZ: f64 = 0.25;
/// Full-scale sine level, dB SPL (§2.1.3).
const LP: f64 = 92.0;

/// Constants of the NMR measurement.
pub(crate) struct NmrModel {
    /// Per FFT line: weighted magnitude = `sqrt(|FFT(x·hw)|²) · mag_gain`, with `x` on the
    /// 16-bit scale of the ear model input (hence the 1/32768).
    pub(crate) mag_gain: Vec<f64>,
    /// Band grouping: `(first line, weights)`.
    groups: Vec<(usize, Vec<f64>)>,
    p_thres: [f64; Z],
    up_base: [f64; Z],
    low_step: f64,
    norm: [f64; Z],
    a_time: [f64; Z],
    mask_offset: [f64; Z],
}

impl NmrModel {
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn new(window: &[f64]) -> Self {
        let fs = f64::from(SAMPLE_RATE);
        let n = N as f64;
        // §2.1.3: one frame of a full-scale 1019.5 Hz sine; largest |X|/N over the lines.
        let fft = Fft::new(N);
        let mut buf: Vec<(f64, f64)> = (0..N)
            .map(|t| {
                let x = (2.0 * core::f64::consts::PI * 1019.5 * t as f64 / fs).sin();
                (window[t] * x, 0.0)
            })
            .collect();
        fft.forward(&mut buf);
        let xmax = buf[..BINS]
            .iter()
            .map(|&(re, im)| (re * re + im * im).sqrt() / n)
            .fold(0.0f64, f64::max);
        let fac = 10f64.powf(LP / 20.0) / xmax;
        // §2.1.4 outer/middle ear; no weight at 0 Hz.
        let mag_gain = (0..BINS)
            .map(|k| {
                if k == 0 {
                    return 0.0;
                }
                let f = (k as f64 * fs / n).max(1.0) / 1000.0;
                let w_db = -0.6 * 3.64 * f.powf(-0.8) + 6.5 * (-0.6 * (f - 3.3).powi(2)).exp()
                    - 1e-3 * f.powf(3.6);
                fac / n / 32768.0 * 10f64.powf(w_db / 10.0).sqrt()
            })
            .collect();
        // §2.1.5 bands from the Bark formula; fractional overlap of each line [k·df ± df/2].
        let zl = 7.0 * (80.0f64 / 650.0).asinh();
        let edge = |z: f64| 650.0 * (z / 7.0).sinh();
        let df = fs / n;
        let mut fc = [0.0; Z];
        let groups = (0..Z)
            .map(|i| {
                let fl = edge(zl + i as f64 * DZ);
                fc[i] = edge(zl + (i as f64 + 0.5) * DZ);
                let fu = edge(zl + (i + 1) as f64 * DZ).min(18_000.0);
                let mut first = None;
                let mut w = Vec::new();
                for k in 0..BINS {
                    let (lo, hi) = (k as f64 * df - df / 2.0, k as f64 * df + df / 2.0);
                    let v = (hi.min(fu) - lo.max(fl)).max(0.0) / df;
                    if v > 0.0 {
                        let f0 = *first.get_or_insert(k);
                        w.resize(k - f0, 0.0);
                        w.push(v);
                    }
                }
                (first.unwrap_or(0), w)
            })
            .collect();
        let p_thres =
            core::array::from_fn(|k| 10f64.powf(0.4 * 0.364 * (fc[k] / 1000.0).powf(-0.8)));
        let up_base = core::array::from_fn(|k| -24.0 - 230.0 / fc[k]);
        let low_step = 10f64.powf(-27.0 * DZ / 10.0);
        let mut norm = [0.0; Z];
        spread_raw(&[1.0; Z], &[0.0; Z], low_step, &up_base, &mut norm);
        let dt = STEP as f64 / fs;
        let a_time = core::array::from_fn(|k| {
            let tau = 0.008 + (100.0 / fc[k]) * (0.030 - 0.008);
            (-dt / tau).exp()
        });
        let mask_offset = core::array::from_fn(|k| {
            let z = k as f64 * DZ;
            let m = if z <= 12.0 { 3.0 } else { 0.25 * z };
            10f64.powf(m / 10.0)
        });
        Self {
            mag_gain,
            groups,
            p_thres,
            up_base,
            low_step,
            norm,
            a_time,
            mask_offset,
        }
    }

    /// Band sums `Σ_k U[i,k]·f(k)`.
    fn bands(&self, f: impl Fn(usize) -> f64) -> [f64; Z] {
        core::array::from_fn(|i| {
            let (first, w) = &self.groups[i];
            w.iter().enumerate().map(|(j, u)| u * f(first + j)).sum()
        })
    }

    /// Masked threshold of one reference frame (weighted magnitudes `mag`); `ef` is the
    /// time-spreading state of the channel.
    pub(crate) fn mask(&self, mag: &[f64; BINS], ef: &mut [f64; Z]) -> [f64; Z] {
        let e = self.bands(|k| mag[k] * mag[k]);
        let energy: [f64; Z] = core::array::from_fn(|i| e[i].max(1e-12) + self.p_thres[i]);
        let level: [f64; Z] = energy.map(|v| 10.0 * v.log10());
        let mut s = [0.0; Z];
        spread_raw(&energy, &level, self.low_step, &self.up_base, &mut s);
        core::array::from_fn(|k| {
            let e2 = s[k] / self.norm[k];
            ef[k] = self.a_time[k] * ef[k] + (1.0 - self.a_time[k]) * e2;
            ef[k].max(e2) / self.mask_offset[k]
        })
    }

    /// NMR of one frame: band mean of noise / mask.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn local(&self, r: &[f64; BINS], t: &[f64; BINS], mask: &[f64; Z]) -> f64 {
        let noise = self.bands(|k| (r[k] - t[k]) * (r[k] - t[k]));
        noise
            .iter()
            .zip(mask)
            .map(|(n, m)| n.max(1e-12) / m)
            .sum::<f64>()
            / Z as f64
    }
}

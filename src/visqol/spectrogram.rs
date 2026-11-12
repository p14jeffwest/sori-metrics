//! Gammatone spectrogram (stage ③, framing part).
//!
//! ViSQOL v3: 80 ms windows with a 20 ms hop over the gammatone filter outputs, NSIM on log
//! amplitudes (Chinen et al. 2020, §III-A/§III-B). The numbers the paper leaves open were
//! fixed against the reference tool's intermediate outputs (see the crate README):
//!
//! - frame value: `10·log10(RMS)` of the filter output weighted by a periodic Hann window,
//!   the mean square taken over the frame length (`Σ (w·y)² / N`) — "log amplitude";
//! - absolute floor (v3 eq. 1, `Y_min`): [`FLOOR`] = −45, i.e. RMS −90 dB re full scale.
//!
//! Values are returned relative to the floor (floored values are 0), the zero point the
//! NSIM constants are defined against.

use super::gammatone::{FilterBank, State};
use super::source::{Blocks, BLOCK};

/// Absolute floor of the log-amplitude spectrogram (`10·log10(RMS)` units).
pub const FLOOR: f64 = -45.0;

/// Framing parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Framing {
    /// Frame length, seconds (v3: 0.080).
    pub window_s: f64,
    /// Frame advance, seconds (v3: 0.020).
    pub hop_s: f64,
}

impl Default for Framing {
    fn default() -> Self {
        Self {
            window_s: 0.080,
            hop_s: 0.020,
        }
    }
}

/// Band × frame log amplitudes above the floor, band-major, low band first.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrogram {
    /// Number of bands (rows).
    pub bands: usize,
    /// Number of frames (columns).
    pub frames: usize,
    /// Centre frequency of every band, Hz.
    pub cf: Vec<f64>,
    /// `data[band * frames + frame]` = `max(10·log10(RMS), FLOOR) − FLOOR`.
    pub data: Vec<f64>,
}

impl Spectrogram {
    /// Value at (`band`, `frame`).
    #[cfg(test)]
    pub fn at(&self, band: usize, frame: usize) -> f64 {
        self.data[band * self.frames + frame]
    }
}

/// Frame length and hop in samples for `fs`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn frame_sizes(framing: &Framing, fs: f64) -> (usize, usize) {
    (
        (framing.window_s * fs).round() as usize,
        (framing.hop_s * fs).round() as usize,
    )
}

/// Number of whole frames in `len` samples.
pub fn frame_count(len: usize, framing: &Framing, fs: f64) -> usize {
    let (n, hop) = frame_sizes(framing, fs);
    if len < n || hop == 0 {
        0
    } else {
        (len - n) / hop + 1
    }
}

/// Log amplitude `10·log10(RMS) = 5·log10(mean square)` of one frame, unfloored.
fn level(ms: f64) -> f64 {
    5.0 * ms.max(1e-300).log10()
}

/// Build the spectrogram of `x` (mono, sample rate `fs`) with `bank`.
///
/// The signal is read block by block; every band keeps its filter state and a ring buffer of
/// one frame of squared outputs, so memory does not grow with the signal length. Each frame
/// value is the same sum, in the same order, as over a whole-signal output buffer.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
pub fn build(x: &impl Blocks, fs: f64, bank: &FilterBank, framing: &Framing) -> Spectrogram {
    let (len, hop) = frame_sizes(framing, fs);
    let frames = frame_count(x.len(), framing, fs);
    let nb = bank.bands.len();
    // Periodic Hann weighting inside each frame.
    let w: Vec<f64> = (0..len)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / len as f64).cos())
        .collect();
    let w2: Vec<f64> = w.iter().map(|v| v * v).collect();
    let mut data = vec![0.0; nb * frames];
    let mut states = vec![State::default(); nb];
    // ring[b * len + (i % len)] = y_b[i]² for the last `len` samples.
    let mut ring = vec![0.0; nb * len];
    // Samples beyond the last frame do not change any value; stop reading there.
    let used = if frames == 0 {
        0
    } else {
        (frames - 1) * hop + len
    };
    let mut buf = vec![0.0; BLOCK.min(used)];
    let mut s = 0;
    while s < used {
        let n = BLOCK.min(used - s);
        x.fill(s, &mut buf[..n]);
        // Frame f ends at sample f·hop + len − 1; the first end at or after s:
        let first_end = if s < len {
            len - 1
        } else {
            len - 1 + (s - (len - 1)).div_ceil(hop) * hop
        };
        for (b, band) in bank.bands.iter().enumerate() {
            let st = &mut states[b];
            let ring = &mut ring[b * len..(b + 1) * len];
            let row = &mut data[b * frames..(b + 1) * frames];
            let mut end = first_end;
            let mut pos = s % len; // = i % len
            for (j, &v) in buf[..n].iter().enumerate() {
                let i = s + j;
                let y = band.step(st, v);
                ring[pos] = y * y;
                pos += 1;
                if pos == len {
                    pos = 0;
                }
                if i == end {
                    end += hop;
                    let start = i + 1 - len;
                    // ring[(start + t) % len] for t = 0..len: two contiguous runs, same order.
                    let p = start % len;
                    let ms = ring[p..]
                        .iter()
                        .chain(&ring[..p])
                        .zip(&w2)
                        .map(|(v, g)| v * g)
                        .sum::<f64>()
                        / len as f64;
                    row[start / hop] = level(ms).max(FLOOR) - FLOOR;
                }
            }
        }
        s += n;
    }
    Spectrogram {
        bands: nb,
        frames,
        cf: bank.bands.iter().map(|b| b.cf).collect(),
        data,
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::super::source::Mono;
    use super::*;

    const FS: f64 = 48_000.0;

    #[test]
    fn frame_count_and_silence_is_floor() {
        let fb = FilterBank::new(FS, 4, 100.0, 8000.0);
        let x = vec![0.0; 48_000];
        let s = build(&Mono::F64(&x), FS, &fb, &Framing::default());
        assert_eq!(frame_sizes(&Framing::default(), FS), (3840, 960));
        assert_eq!(s.frames, (48_000 - 3840) / 960 + 1);
        assert_eq!(s.bands, 4);
        assert!(s.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn sine_level() {
        // Unit sine at a band centre: RMS 1/√2; Hann weighting keeps 3/8 of the mean square.
        let fb = FilterBank::new(FS, 32, 50.0, 24_000.0);
        let f = fb.bands[20].cf;
        let x: Vec<f64> = (0..48_000)
            .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / FS).sin())
            .collect();
        let s = build(&Mono::F64(&x), FS, &fb, &Framing::default());
        let want = 5.0 * (0.5f64 * 3.0 / 8.0).log10() - FLOOR;
        let v = s.at(20, s.frames - 1);
        assert!((v - want).abs() < 0.03, "{v} vs {want}");
        assert!(s.at(5, s.frames - 1) < v - 20.0);
    }

    #[test]
    fn level_change_shifts_values() {
        let fb = FilterBank::new(FS, 8, 100.0, 10_000.0);
        let x: Vec<f64> = (0..24_000)
            .map(|i| ((i * 7919) % 2001) as f64 / 1000.0 - 1.0)
            .collect();
        let x2: Vec<f64> = x.iter().map(|v| v * 0.1).collect();
        let a = build(&Mono::F64(&x), FS, &fb, &Framing::default());
        let b = build(&Mono::F64(&x2), FS, &fb, &Framing::default());
        // −20 dB amplitude = −10 in 10·log10(RMS) units, as long as both stay above the floor.
        for (p, q) in a.data.iter().zip(&b.data) {
            if *q > 0.0 {
                assert!((p - q - 10.0).abs() < 1e-9);
            }
        }
    }
}

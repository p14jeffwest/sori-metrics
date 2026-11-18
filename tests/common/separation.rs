//! Deterministic synthetic source-separation sets (8 kHz) shared by `tests/separation.rs` and
//! the fixture writer example. Integer LCG noise through fixed `f64` recursions, rounded to
//! `f32` once, so the WAV files written for the Python tools and the signals rebuilt in the
//! test agree to the last bit.

#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

/// References and estimates of every source, interleaved.
pub struct Set {
    pub name: &'static str,
    pub rate: u32,
    pub channels: usize,
    pub references: Vec<Vec<f32>>,
    pub estimates: Vec<Vec<f32>>,
}

/// 32-bit LCG (Numerical Recipes constants), uniform in [-1, 1).
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(self.0 >> 8) / f64::from(1u32 << 23) - 1.0
    }
}

/// Noise through a two-pole resonator: a different "instrument" per `(pole, angle)`.
fn coloured(seed: u32, len: usize, radius: f64, cos_w: f64, gain: f64) -> Vec<f64> {
    let mut g = Lcg(seed);
    let (mut y1, mut y2) = (0.0, 0.0);
    (0..len)
        .map(|_| {
            let y = g.next() + 2.0 * radius * cos_w * y1 - radius * radius * y2;
            y2 = y1;
            y1 = y;
            gain * y
        })
        .collect()
}

/// Builds stereo or mono source images and estimates with filtering, delay, leakage, noise
/// and (optionally) clipping.
fn build(
    name: &'static str,
    channels: usize,
    seconds: usize,
    sources: &[(f64, f64, f64)],
    silent: Option<(usize, usize, usize)>,
    clip: Option<f64>,
) -> Set {
    let rate = 8_000;
    let len = seconds * rate / 2; // half seconds
    let mut images: Vec<Vec<Vec<f64>>> = Vec::new();
    for (j, &(r, c, g)) in sources.iter().enumerate() {
        let base = coloured(11 + j as u32, len, r, c, g);
        let mut chans = Vec::new();
        for ch in 0..channels {
            // A pan per channel plus an independent floor, so channels are not collinear.
            let floor = coloured(101 + (j * 8 + ch) as u32, len, 0.0, 0.0, g * 0.003);
            let pan = 1.0 - 0.3 * ch as f64 + 0.1 * j as f64;
            chans.push(base.iter().zip(&floor).map(|(b, f)| pan * b + f).collect());
        }
        images.push(chans);
    }
    if let Some((j, a, b)) = silent {
        for ch in &mut images[j] {
            ch[a * rate..b * rate].iter_mut().for_each(|x| *x = 0.0);
        }
    }
    // Round the references to f32 first: the estimates are built from the stored values.
    let images: Vec<Vec<Vec<f64>>> = images
        .into_iter()
        .map(|s| {
            s.into_iter()
                .map(|c| c.into_iter().map(|x| f64::from(x as f32)).collect())
                .collect()
        })
        .collect();
    let n = sources.len();
    let mut estimates = Vec::new();
    for j in 0..n {
        let delay = 3 + 7 * j;
        let h = [0.9, 0.15 - 0.05 * j as f64, -0.08];
        let mut chans = Vec::new();
        for ch in 0..channels {
            let s = &images[j][ch];
            let noise = coloured(501 + (j * 8 + ch) as u32, len, 0.0, 0.0, 0.01);
            let est: Vec<f64> = (0..len)
                .map(|t| {
                    let mut v = 0.0;
                    for (k, hk) in h.iter().enumerate() {
                        if t >= delay + k {
                            v += hk * s[t - delay - k];
                        }
                    }
                    for (m, other) in images.iter().enumerate() {
                        if m != j {
                            v += (0.05 + 0.04 * ((j + 2 * m) % 3) as f64) * other[ch][t];
                        }
                    }
                    v += noise[t];
                    clip.map_or(v, |c| v.clamp(-c, c))
                })
                .collect();
            chans.push(est);
        }
        estimates.push(chans);
    }
    let interleave = |s: &[Vec<f64>]| -> Vec<f32> {
        (0..len)
            .flat_map(|t| s.iter().map(move |c| c[t] as f32))
            .collect()
    };
    Set {
        name,
        rate: rate as u32,
        channels,
        references: images.iter().map(|s| interleave(s)).collect(),
        estimates: estimates.iter().map(|s| interleave(s)).collect(),
    }
}

/// Three stereo sources, 3.5 s; source 0 is silent from 1 s to 2 s (one unscored window).
pub fn three_stereo() -> Set {
    build(
        "syn-sep-three",
        2,
        7,
        &[(0.95, 0.3, 0.05), (0.7, -0.5, 0.2), (0.99, 0.95, 0.01)],
        Some((0, 1, 2)),
        None,
    )
}

/// Two mono sources, 4 s, clipped estimates.
pub fn two_mono_clipped() -> Set {
    build(
        "syn-sep-mono",
        1,
        8,
        &[(0.9, 0.0, 0.1), (0.98, 0.8, 0.02)],
        None,
        Some(0.25),
    )
}

/// All synthetic sets.
pub fn all() -> Vec<Set> {
    vec![three_stereo(), two_mono_clipped()]
}

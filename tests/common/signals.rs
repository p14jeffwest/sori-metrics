//! Deterministic synthetic test pairs (48 kHz, interleaved) shared by the reference test and
//! the fixture writer example. The generator uses only integer arithmetic and `sin` of
//! exactly representable arguments, so the WAV files written once for the Python tool and
//! the signals rebuilt in the test agree to the last bit of `f32` on common platforms.

#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

/// One synthetic comparison.
pub struct Pair {
    pub name: &'static str,
    pub channels: usize,
    pub reference: Vec<f32>,
    pub test: Vec<f32>,
}

const RATE: f64 = 48_000.0;

/// 32-bit LCG (Numerical Recipes constants), uniform in [-1, 1).
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (f64::from(self.0 >> 8) / f64::from(1u32 << 23) - 1.0) as f32
    }
}

fn sine(f: f64, i: usize) -> f64 {
    (2.0 * std::f64::consts::PI * f * i as f64 / RATE).sin()
}

/// Quantise to `bits`-bit integers (round half away from zero).
fn quantise(x: f32, bits: u32) -> f32 {
    let q = f64::from(1u32 << (bits - 1));
    ((f64::from(x) * q).round() / q) as f32
}

/// Chord with a slow tremolo after 0.6 s of digital silence, stereo; test = 9-bit quantised.
fn chord_quantised() -> Pair {
    let n = 3 * 48_000;
    let mut noise = Lcg(1);
    let mut reference = Vec::with_capacity(2 * n);
    for i in 0..n {
        let on = i >= 28_800;
        let t = 0.5 + 0.5 * sine(3.0, i);
        let l = if on {
            0.25 * (sine(220.0, i) + 0.5 * sine(277.18, i) + 0.3 * sine(329.63, i)) * t
                + 0.003 * f64::from(noise.next())
        } else {
            0.0
        };
        let r = if on {
            0.25 * (sine(110.0, i) + 0.6 * sine(440.0, i) + 0.2 * sine(659.26, i)) * (1.0 - 0.5 * t)
                + 0.003 * f64::from(noise.next())
        } else {
            0.0
        };
        reference.push(l as f32);
        reference.push(r as f32);
    }
    let test = reference.iter().map(|&x| quantise(x, 9)).collect();
    Pair {
        name: "chord-quantised",
        channels: 2,
        reference,
        test,
    }
}

/// Amplitude-modulated noise, mono; test = one-pole low-pass (a = 0.6).
fn noise_lowpass() -> Pair {
    let n = 3 * 48_000;
    let mut noise = Lcg(7);
    let reference: Vec<f32> = (0..n)
        .map(|i| (0.2 * (0.6 + 0.4 * sine(5.0, i)) * f64::from(noise.next())) as f32)
        .collect();
    let mut y = 0.0f32;
    let test = reference
        .iter()
        .map(|&x| {
            y = 0.6 * y + 0.4 * x;
            y
        })
        .collect();
    Pair {
        name: "noise-lowpass",
        channels: 1,
        reference,
        test,
    }
}

/// Decaying noise clicks every 0.25 s over a quiet hum, stereo; test = right channel at
/// 0.8 gain plus added noise.
fn clicks_noise() -> Pair {
    let n = 3 * 48_000;
    let mut noise = Lcg(42);
    let mut add = Lcg(99);
    let mut reference = Vec::with_capacity(2 * n);
    let mut test = Vec::with_capacity(2 * n);
    for i in 0..n {
        let k = i % 12_000;
        let env = if k < 2400 {
            (-(k as f64) / 300.0).exp()
        } else {
            0.0
        };
        let hum = 0.01 * sine(100.0, i);
        let l = (0.6 * env * f64::from(noise.next()) + hum) as f32;
        let r = (0.5 * env * f64::from(noise.next()) + hum) as f32;
        reference.push(l);
        reference.push(r);
        test.push(l + 0.004 * add.next());
        test.push(0.8 * r + 0.004 * add.next());
    }
    Pair {
        name: "clicks-noise",
        channels: 2,
        reference,
        test,
    }
}

/// All synthetic pairs.
pub fn all() -> Vec<Pair> {
    vec![chord_quantised(), noise_lowpass(), clicks_noise()]
}

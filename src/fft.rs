//! Iterative radix-2 complex FFT (decimation in time) in `f64`.
//!
//! Small and dependency-free on purpose. Only the 2048-point
//! transform of the PEAQ® FFT ear model needs to be fast; any power-of-two size works.
//! The operation order is fixed, so the same input always gives the same output.

/// A complex number as `(re, im)`.
pub type Complex = (f64, f64);

/// Precomputed twiddle factors and bit-reversal table for one transform size.
#[derive(Debug, Clone)]
pub struct Fft {
    n: usize,
    /// `exp(-2πi k / n)` for `k` in `0..n/2`.
    twiddle: Vec<Complex>,
    /// Bit-reversed index of every position.
    rev: Vec<usize>,
}

impl Fft {
    /// Plan a forward transform of size `n` (a power of two, at least 2).
    ///
    /// # Panics
    /// If `n` is not a power of two or smaller than 2 (a programming error in this crate).
    pub fn new(n: usize) -> Self {
        assert!(
            n >= 2 && n.is_power_of_two(),
            "FFT size must be a power of two"
        );
        let bits = n.trailing_zeros();
        let rev = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        #[allow(clippy::cast_precision_loss)] // n ≤ 2^16 here: exact in f64
        let twiddle = (0..n / 2)
            .map(|k| {
                let w = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                (w.cos(), w.sin())
            })
            .collect();
        Self { n, twiddle, rev }
    }

    /// In-place forward transform `X[k] = Σ x[t] e^{-2πi kt/n}` (no scaling).
    ///
    /// # Panics
    /// If `buf.len()` differs from the planned size.
    pub fn forward(&self, buf: &mut [Complex]) {
        assert_eq!(buf.len(), self.n, "FFT buffer length");
        for i in 0..self.n {
            let j = self.rev[i];
            if j > i {
                buf.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= self.n {
            let half = len / 2;
            let step = self.n / len;
            for start in (0..self.n).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = self.twiddle[k * step];
                    let (ar, ai) = buf[start + k];
                    let (br, bi) = buf[start + k + half];
                    let tr = br * wr - bi * wi;
                    let ti = br * wi + bi * wr;
                    buf[start + k] = (ar + tr, ai + ti);
                    buf[start + k + half] = (ar - tr, ai - ti);
                }
            }
            len *= 2;
        }
    }
}

/// Twiddle factors computed at a time by [`forward_untabled`].
const RUN: usize = 4096;

/// In-place forward transform of any power-of-two length without precomputed tables.
///
/// Gives exactly the same output as [`Fft::forward`] (same butterflies, same twiddle
/// expressions; only the loop order inside a stage differs, and butterflies of one stage are
/// independent). Used for the one long transform of the ViSQOL alignment, where the tables
/// of [`Fft`] would cost as much memory as the data. Each twiddle factor is computed once
/// per stage, `n` evaluations in all.
///
/// # Panics
/// If the length is not a power of two of at least 2.
pub fn forward_untabled(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(
        n >= 2 && n.is_power_of_two(),
        "FFT size must be a power of two"
    );
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            buf.swap(i, j);
        }
    }
    // Twiddles are made a run of `RUN` at a time and reused across the blocks of the stage,
    // keeping the memory walk sequential; each is computed once per stage (`n` in all).
    let mut tw: Vec<Complex> = Vec::with_capacity(RUN.min(n / 2));
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        let mut k0 = 0;
        while k0 < half {
            let k1 = (k0 + RUN).min(half);
            tw.clear();
            tw.extend((k0..k1).map(|k| {
                // Same expression as the table entry `twiddle[k * step]` of `Fft::new(n)`.
                #[allow(clippy::cast_precision_loss)]
                let w = -2.0 * std::f64::consts::PI * (k * step) as f64 / n as f64;
                (w.cos(), w.sin())
            }));
            for start in (0..n).step_by(len) {
                for (k, &(wr, wi)) in (k0..k1).zip(&tw) {
                    let (ar, ai) = buf[start + k];
                    let (br, bi) = buf[start + k + half];
                    let tr = br * wr - bi * wi;
                    let ti = br * wi + bi * wr;
                    buf[start + k] = (ar + tr, ai + ti);
                    buf[start + k + half] = (ar - tr, ai - ti);
                }
            }
            k0 = k1;
        }
        len *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn untabled_is_bit_identical() {
        for n in [2usize, 16, 1024, 1 << 15] {
            let x: Vec<Complex> = (0..n)
                .map(|i| {
                    (
                        ((i * 7919) % 1013) as f64 / 97.0,
                        ((i * 31) % 17) as f64 - 8.0,
                    )
                })
                .collect();
            let (mut a, mut b) = (x.clone(), x);
            Fft::new(n).forward(&mut a);
            forward_untabled(&mut b);
            assert!(a
                .iter()
                .zip(&b)
                .all(|(p, q)| p.0.to_bits() == q.0.to_bits() && p.1.to_bits() == q.1.to_bits()));
        }
    }

    /// Direct O(n²) DFT for checking the fast transform.
    #[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
    fn dft(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        (0..n)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(sr, si), (t, &(xr, xi))| {
                        let w = -2.0 * std::f64::consts::PI * ((k * t) % n) as f64 / n as f64;
                        let (c, s) = (w.cos(), w.sin());
                        (sr + xr * c - xi * s, si + xr * s + xi * c)
                    })
            })
            .collect()
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn matches_direct_dft() {
        for n in [2usize, 8, 64, 256] {
            let x: Vec<Complex> = (0..n)
                .map(|i| {
                    (
                        ((i * 7 + 3) % 11) as f64 - 5.0,
                        ((i * 5 + 1) % 13) as f64 - 6.0,
                    )
                })
                .collect();
            let mut y = x.clone();
            Fft::new(n).forward(&mut y);
            for (a, b) in y.iter().zip(dft(&x)) {
                assert!((a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
            }
        }
    }
}

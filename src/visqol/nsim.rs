//! Neurogram similarity index measure (stage ⑤, similarity part).
//!
//! NSIM (Hines & Harte; ViSQOL, EURASIP JASMP 2015, §2) is SSIM without the contrast term:
//! `Q = l · s`, `l = (2μ_r μ_d + C1) / (μ_r² + μ_d² + C1)`,
//! `s = (σ_rd + C3) / (σ_r σ_d + C3)`, with local statistics from a 3 × 3 Gaussian window.
//! Fixed against the reference tool's intermediate outputs: Gaussian σ = 0.5, edges handled by
//! repeating the border row/column, `C1 = (0.01·L)²`, `C3 = (0.03·L)² / 2` with `L = 1` in the
//! spectrogram's `10·log10(RMS)` units.

#![allow(clippy::many_single_char_names)] // the symbols of the NSIM formula

/// `C1 = (0.01 L)²`, `L = 1`.
const C1: f64 = 1e-4;
/// `C3 = (0.03 L)² / 2`, `L = 1`.
const C3: f64 = 4.5e-4;

/// 3 × 3 Gaussian weights (σ = 0.5), normalised to sum 1; `W[i][j]` for offsets −1, 0, +1.
fn weights() -> [[f64; 3]; 3] {
    let g = |d: f64| (-(d * d) / (2.0 * 0.25)).exp();
    let mut w = [[0.0; 3]; 3];
    let mut sum = 0.0;
    for (i, row) in w.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let (di, dj) = (i as f64 - 1.0, j as f64 - 1.0);
            *v = g(di) * g(dj);
            sum += *v;
        }
    }
    for row in &mut w {
        for v in row {
            *v /= sum;
        }
    }
    w
}

/// Per-band mean of the NSIM map of two equally sized patches.
///
/// `r` and `d` are `bands × frames` (band-major, row length `frames`). Returns `bands` values.
///
/// # Panics
/// If the slices do not have `bands * frames` values.
pub fn band_means(r: &[f64], d: &[f64], bands: usize, frames: usize) -> Vec<f64> {
    assert_eq!(r.len(), bands * frames);
    assert_eq!(d.len(), bands * frames);
    let w = weights();
    let at = |a: &[f64], b: isize, f: isize| -> f64 {
        // Repeat the border (edge replication).
        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
        let bb = b.clamp(0, bands as isize - 1) as usize;
        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
        let ff = f.clamp(0, frames as isize - 1) as usize;
        a[bb * frames + ff]
    };
    let mut out = vec![0.0; bands];
    for (b, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for f in 0..frames {
            let (mut mr, mut md, mut rr, mut dd, mut rd) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for (i, row) in w.iter().enumerate() {
                for (j, &g) in row.iter().enumerate() {
                    #[allow(clippy::cast_possible_wrap)]
                    let (bi, fj) = (b as isize + i as isize - 1, f as isize + j as isize - 1);
                    let (x, y) = (at(r, bi, fj), at(d, bi, fj));
                    mr += g * x;
                    md += g * y;
                    rr += g * x * x;
                    dd += g * y * y;
                    rd += g * x * y;
                }
            }
            let vr = (rr - mr * mr).max(0.0);
            let vd = (dd - md * md).max(0.0);
            let cov = rd - mr * md;
            let l = (2.0 * mr * md + C1) / (mr * mr + md * md + C1);
            let s = (cov + C3) / ((vr * vd).sqrt() + C3);
            acc += l * s;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            *o = acc / frames as f64;
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    #[test]
    fn identical_patches_score_one() {
        let r: Vec<f64> = (0..32 * 30).map(|i| ((i * 37) % 41) as f64 * 0.7).collect();
        for v in band_means(&r, &r, 32, 30) {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn floor_against_signal_is_near_zero() {
        // Reference with content, degraded at the floor (0): l = C1 / (μ² + C1) ≈ 0.
        let r: Vec<f64> = (0..4 * 10).map(|i| 10.0 + (i % 3) as f64).collect();
        let d = vec![0.0; 40];
        for v in band_means(&r, &d, 4, 10) {
            assert!(v.abs() < 1e-5, "{v}");
        }
    }

    #[test]
    fn weights_sum_to_one_and_are_symmetric() {
        let w = weights();
        let s: f64 = w.iter().flatten().sum();
        assert!((s - 1.0).abs() < 1e-15);
        assert!((w[0][0] - w[2][2]).abs() < 1e-15 && (w[0][1] - w[1][0]).abs() < 1e-15);
        assert!(w[1][1] > w[0][1] && w[0][1] > w[0][0]);
    }
}

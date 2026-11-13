//! The 2f model: an estimate of the mean MUSHRA score from two PEAQ® Basic MOVs.
//!
//! Source: T. Kastner and J. Herre, "An efficient model for estimating subjective quality of
//! separated audio source signals", IEEE WASPAA 2019 (doi:10.1109/WASPAA.2019.8937179),
//! the model equation with the parameter set published by `AudioLabs` Erlangen
//! (resource page "2019-WASPAA-SEBASS"):
//!
//! `MMS = 56.1345 / (1 + (-0.0282·AvgModDiff1 - 0.8628)²) - 27.1451·ADB + 86.3515`,
//! limited to 0..=100.
//!
//! The MOVs come from [`crate::peaq`] (ITU-R BS.1387 Basic Version, FFT ear model).

use crate::peaq::{self, Blocks, Input};
use crate::source::Source;
use crate::Error;

/// Result of one 2f comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TwoF {
    /// Estimated mean MUSHRA score, 0..=100 (higher is better).
    pub mushra: f64,
    /// `ADBB` (average distorted block, §4.7.2 of BS.1387 Annex 2).
    pub adb: f64,
    /// `AvgModDiff1B` (§4.2.3 of BS.1387 Annex 2).
    pub avg_mod_diff1: f64,
    /// `Total NMR_B` in dB (§4.5.1 of BS.1387 Annex 2; lower is better), computed in the same
    /// pass from the same frames — see [`crate::peaq`] ("Total NMR"). Not part of the 2f
    /// score. NaN for a window ([`Detail::windows`]) whose frames all lie outside the
    /// reference's data range.
    pub total_nmr: f64,
}

/// Map the two MOVs to the MUSHRA-like score (Kastner & Herre 2019, published parameters).
pub fn mushra_from_movs(avg_mod_diff1: f64, adb: f64) -> f64 {
    let x = -0.0282 * avg_mod_diff1 - 0.8628;
    let mms = 56.1345 / (1.0 + x * x) - 27.1451 * adb + 86.3515;
    mms.clamp(0.0, 100.0)
}

/// 2f score of `test` against `reference`.
///
/// Both are interleaved `f32` samples in [-1, 1] with `channels` (1 or 2) channels at
/// `sample_rate`, already time aligned. Only 48 kHz is
/// accepted — the model is defined at 48 kHz and resampling is the caller's job (the
/// reference tool also expects 48 kHz input). The shorter length is
/// used. Stereo: left and right are analysed separately; `AvgModDiff1` is the channel mean
/// and `ADB` uses the binaural (per-band maximum) detection values (BS.1387 Annex 2 §4.7, §5.3).
///
/// The signals are read in place, frame by frame: memory use does not grow with their
/// length. Same result as `Reference::new(reference, ..)?.score(test)`.
///
/// # Errors
/// [`Error::SampleRate`] if not 48 kHz, [`Error::Channels`] if not 1 or 2 channels,
/// [`Error::TooShort`] if fewer than one 2048-sample frame.
pub fn score(
    reference: &[f32],
    test: &[f32],
    sample_rate: u32,
    channels: usize,
) -> Result<TwoF, Error> {
    check(sample_rate, channels)?;
    let len = (reference.len() / channels).min(test.len() / channels);
    if len < 2048 {
        return Err(Error::TooShort(len));
    }
    Ok(detail(reference, test, sample_rate, channels)?.score)
}

/// Like [`score`], with the per-block values ([`Detail::blocks`], [`Detail::windows`]).
///
/// # Errors
/// As [`score`].
pub fn detail(
    reference: &[f32],
    test: &[f32],
    sample_rate: u32,
    channels: usize,
) -> Result<Detail, Error> {
    check(sample_rate, channels)?;
    let len = (reference.len() / channels).min(test.len() / channels);
    if len < 2048 {
        return Err(Error::TooShort(len));
    }
    let b = peaq::blocks_of(
        Input::Interleaved(reference, channels),
        Input::Interleaved(test, channels),
    )?;
    Ok(Detail::new(b))
}

/// [`detail`] on two [`Source`]s, read 10 s at a time: memory use does not grow with the
/// signal length beyond the per-block values (about 50 bytes per 1024 samples). Same values
/// as [`detail`] on the whole signals.
///
/// # Errors
/// As [`score`] (for the sources' rate and channel count; the two must have the same
/// channel count), and any error of [`Source::read`].
pub fn detail_source(reference: impl Source, test: impl Source) -> Result<Detail, Error> {
    let channels = reference.channels();
    check(reference.sample_rate(), channels)?;
    check(test.sample_rate(), test.channels())?;
    if test.channels() != channels {
        return Err(Error::Channels(test.channels()));
    }
    let len = reference.frames().min(test.frames());
    if len < 2048 {
        return Err(Error::TooShort(len));
    }
    Ok(Detail::new(peaq::blocks_source(reference, test)?))
}

fn check(sample_rate: u32, channels: usize) -> Result<(), Error> {
    if sample_rate != peaq::SAMPLE_RATE {
        return Err(Error::SampleRate(sample_rate));
    }
    if !(1..=2).contains(&channels) {
        return Err(Error::Channels(channels));
    }
    Ok(())
}

impl From<peaq::Movs> for TwoF {
    fn from(m: peaq::Movs) -> Self {
        Self {
            mushra: mushra_from_movs(m.avg_mod_diff1, m.adb),
            adb: m.adb,
            avg_mod_diff1: m.avg_mod_diff1,
            total_nmr: m.total_nmr,
        }
    }
}

/// A reference signal run through the ear model once, for scoring any number of test
/// signals against it.
///
/// Gives exactly the same results as [`score`]; it saves the reference half of the ear-model
/// work on every further test signal. It borrows the reference samples and holds the
/// per-frame reference values, about 82 KB per second per channel.
///
/// ```
/// # fn main() -> Result<(), sori_metrics::Error> {
/// # let tone = |a: f32| -> Vec<f32> { (0..96_000)
/// #     .map(|i| a * (i as f32 * 0.0575).sin()).collect() };
/// # let (reference, a, b) = (tone(0.3), tone(0.29), tone(0.2));
/// let r = sori_metrics::twof::Reference::new(&reference, 48_000, 1)?;
/// for candidate in [&a, &b] {
///     println!("{:.1}", r.score(candidate)?.mushra);
/// }
/// # Ok(())
/// # }
/// ```
pub struct Reference<'a> {
    model: peaq::RefModel<'a>,
    channels: usize,
}

impl core::fmt::Debug for Reference<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Reference")
            .field("channels", &self.channels)
            .field("len", &self.model.len())
            .finish_non_exhaustive()
    }
}

impl<'a> Reference<'a> {
    /// Analyse a reference: interleaved `f32` in [-1, 1] at 48 kHz, 1 or 2 channels.
    ///
    /// # Errors
    /// [`Error::SampleRate`] if not 48 kHz, [`Error::Channels`] if not 1 or 2 channels,
    /// [`Error::TooShort`] if fewer than one 2048-sample frame.
    pub fn new(reference: &'a [f32], sample_rate: u32, channels: usize) -> Result<Self, Error> {
        check(sample_rate, channels)?;
        let len = reference.len() / channels;
        if len < 2048 {
            return Err(Error::TooShort(len));
        }
        Ok(Self {
            model: peaq::RefModel::new(Input::Interleaved(reference, channels)),
            channels,
        })
    }

    /// Score interleaved `test` samples with the reference's channel count, time aligned
    /// with the reference (the shorter length is used).
    ///
    /// # Errors
    /// [`Error::TooShort`] if the common length is under one 2048-sample frame.
    pub fn score(&self, test: &[f32]) -> Result<TwoF, Error> {
        Ok(self.detail(test)?.score)
    }

    /// Like [`Reference::score`], with the per-block values.
    ///
    /// # Errors
    /// As [`Reference::score`].
    pub fn detail(&self, test: &[f32]) -> Result<Detail, Error> {
        let len = self.model.len().min(test.len() / self.channels);
        if len < 2048 {
            return Err(Error::TooShort(len));
        }
        Ok(Detail::new(
            self.model.blocks(Input::Interleaved(test, self.channels)),
        ))
    }
}

/// Per-block values of one 2f comparison and the overall score.
///
/// A block is one analysis frame of the ear model: 2048 samples, advanced by 1024 (21.3 ms
/// at 48 kHz). Windows ([`Detail::windows`]) apply the same averaging as the overall score
/// to the blocks whose centre lies in the window.
#[derive(Debug, Clone, PartialEq)]
pub struct Detail {
    /// The overall result, as [`score`] returns it.
    pub score: TwoF,
    blocks: Blocks,
}

/// Values of one block (one analysis frame).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Block {
    /// Block index.
    pub index: usize,
    /// Start of the frame, seconds.
    pub start: f64,
    /// Inside the reference's data range (BS.1387 Annex 2 §5.2.4.4); only valid blocks
    /// count in the averages.
    pub valid: bool,
    /// Binaural detection probability of the block, `1 − Π_k (1 − Pbin[k])` (§4.7, eq 80).
    pub detection: f64,
    /// `detection > 0.5`: a distorted block in the sense of `ADB` (§4.7.2).
    pub distorted: bool,
    /// Steps above threshold summed over the bands, `Qbin` (§4.7, eq 81).
    pub steps: f64,
    /// `ModDiff1` per channel (eq 63); the second entry is 0 for mono.
    pub mod_diff: [f64; 2],
    /// `TempWt` per channel (eq 65), the weight of `ModDiff1` in `AvgModDiff1`.
    pub weight: [f64; 2],
}

/// Values of one time window.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Window {
    /// Window start, seconds.
    pub start: f64,
    /// Window end, seconds.
    pub end: f64,
    /// Blocks whose frame centre lies in the window.
    pub blocks: usize,
    /// Of those, the valid ones.
    pub valid_blocks: usize,
    /// Of those, the distorted ones (`detection > 0.5`).
    pub distorted_blocks: usize,
    /// `distorted_blocks / valid_blocks` (0 when no block is valid).
    pub distorted_share: f64,
    /// 2f of the window: `AvgModDiff1` and `ADB` over its valid blocks, mapped as in
    /// [`mushra_from_movs`]; `None` when no block is valid. The first 24 blocks of the
    /// signal (0.5 s) stay out of `AvgModDiff1`, as in the overall score.
    pub score: Option<TwoF>,
}

impl Detail {
    fn new(blocks: Blocks) -> Self {
        let (m, _, _) = blocks.movs(0..blocks.frames);
        Self {
            score: TwoF::from(m),
            blocks,
        }
    }

    /// Number of blocks.
    pub fn len(&self) -> usize {
        self.blocks.frames
    }

    /// `true` if there are no blocks (never for a result of this crate's functions).
    pub fn is_empty(&self) -> bool {
        self.blocks.frames == 0
    }

    /// Values of block `n`.
    ///
    /// # Panics
    /// If `n >= self.len()`.
    #[allow(clippy::cast_precision_loss)]
    pub fn block(&self, n: usize) -> Block {
        let b = &self.blocks;
        assert!(n < b.frames, "block index");
        let ch = b.channels;
        let pick = |v: &[f64], c: usize| if c < ch { v[n * ch + c] } else { 0.0 };
        Block {
            index: n,
            start: (n * peaq::STEP) as f64 / f64::from(peaq::SAMPLE_RATE),
            valid: b.valid(n),
            detection: b.detection[n],
            distorted: b.detection[n] > 0.5,
            steps: b.steps[n],
            mod_diff: [pick(&b.mod_diff, 0), pick(&b.mod_diff, 1)],
            weight: [pick(&b.weight, 0), pick(&b.weight, 1)],
        }
    }

    /// All blocks in order.
    pub fn blocks(&self) -> impl Iterator<Item = Block> + '_ {
        (0..self.blocks.frames).map(|n| self.block(n))
    }

    /// Windows of `length` seconds every `hop` seconds from the start, as many as fit in the
    /// compared length (one window over everything if the signal is shorter than `length`).
    /// R-36 (B) uses 3 s windows with a 1 s hop.
    ///
    /// # Panics
    /// If `length` or `hop` is not positive.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn windows(&self, length: f64, hop: f64) -> Vec<Window> {
        assert!(length > 0.0 && hop > 0.0, "window length and hop");
        let fs = f64::from(peaq::SAMPLE_RATE);
        let (wl, wh) = (
            (length * fs).round() as usize,
            (hop * fs).round().max(1.0) as usize,
        );
        let n = self.blocks.frames;
        // Samples covered by the compared frames.
        let total = if n == 0 {
            0
        } else {
            (n - 1) * peaq::STEP + peaq::N
        };
        let mut out = Vec::new();
        let mut s = 0;
        loop {
            let e = if total < wl { total } else { s + wl };
            // Blocks whose centre n·STEP + N/2 lies in [s, e).
            let first = s.saturating_sub(peaq::N / 2).div_ceil(peaq::STEP);
            let last = e.saturating_sub(peaq::N / 2).div_ceil(peaq::STEP).min(n);
            let (m, valid, distorted) = self.blocks.movs(first..last);
            out.push(Window {
                start: s as f64 / fs,
                end: e as f64 / fs,
                blocks: last.saturating_sub(first),
                valid_blocks: valid,
                distorted_blocks: distorted,
                distorted_share: if valid == 0 {
                    0.0
                } else {
                    distorted as f64 / valid as f64
                },
                score: (valid > 0).then(|| TwoF::from(m)),
            });
            s += wh;
            if total < wl || s + wl > total {
                break;
            }
        }
        out
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)] // exact clamp values; small sample counts
mod tests {
    use super::*;

    #[test]
    fn mapping_endpoints() {
        // No modulation difference and no distortion: 56.1345 / (1 + 0.8628²) + 86.3515 > 100.
        assert_eq!(mushra_from_movs(0.0, 0.0), 100.0);
        assert!(mushra_from_movs(50.0, 2.0) < 60.0);
    }

    #[test]
    fn identical_signals_score_high() {
        let x: Vec<f32> = (0..48_000 * 2)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin() * (1.0 + 0.5 * (7.0 * t).sin())
            })
            .collect();
        let s = score(&x, &x, 48_000, 1).unwrap();
        assert_eq!(s.adb, 0.0);
        assert!(s.avg_mod_diff1.abs() < 1e-6, "{s:?}");
        assert_eq!(s.mushra, 100.0);
    }

    #[test]
    fn rejects_other_rates() {
        assert!(matches!(
            score(&[0.0; 4096], &[0.0; 4096], 44_100, 1),
            Err(Error::SampleRate(44_100))
        ));
    }
}

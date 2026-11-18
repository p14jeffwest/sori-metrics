//! Objective perceptual audio quality metrics in pure Rust: no dependencies, no `unsafe`.
//!
//! - [`twof`]: the **2f model** (Kastner & Herre, WASPAA 2019), an estimate of the mean
//!   MUSHRA score from two Model Output Variables of the PEAQ® Basic Version.
//! - [`peaq`]: the parts of an **ITU-R BS.1387 (PEAQ®) Basic Version compatible** FFT ear
//!   model that 2f needs (`AvgModDiff1B`, `ADBB`).
//! - [`visqol`]: a **ViSQOL v3 audio-mode compatible** implementation (MOS-LQO 1–5) for
//!   music and general audio — **audio mode only, not for speech quality estimation**.
//! - [`separation`]: **music source separation** scores: BSS Eval image criteria (SDR, ISR,
//!   SIR, SAR) per window in the version-4 form of SiSEC 2018, and SI-SDR.
//! - [`source`]: signals read piece by piece (WAV files on disk, slices, your own storage)
//!   for [`twof::detail_source`] and [`visqol::score_source`], which never hold a whole
//!   signal.
//! - [`wav`]: a minimal WAV reader/writer for tools and tests.
//!
//! These are independent implementations written from the published Recommendation and
//! papers (for ViSQOL chiefly Chinen et al. 2020). They are **not affiliated with or endorsed
//! by ITU, OPTICOM, Google, or the authors of the reference tools**. PEAQ® is a registered
//! trademark of OPTICOM GmbH; "PEAQ" and "ViSQOL" are used only to say which published method
//! the code follows. Their outputs were checked against the outputs (only) of the Python
//! reference tools `seabass-2f` and `visqol-python`; see the README for the agreement tables
//! and known differences.
//!
//! # Input
//!
//! The two perceptual metrics take interleaved `f32` samples in `[-1, 1]` at **48 kHz**, already time
//! aligned with the reference (ViSQOL also aligns globally itself). Resample beforehand; other
//! rates are rejected with [`Error::SampleRate`]. 2f analyses up to two channels separately;
//! ViSQOL averages all channels to mono.
//!
//! # Example
//!
//! ```
//! # fn main() -> Result<(), sori_metrics::Error> {
//! let rate = 48_000;
//! // Two seconds of a stereo tone, and two coarser copies of it (12-bit and 8-bit).
//! let reference: Vec<f32> = (0..2 * rate)
//!     .flat_map(|i| {
//!         let s = 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin();
//!         [s, s]
//!     })
//!     .collect();
//! let quantise = |step: f32| -> Vec<f32> {
//!     reference.iter().map(|s| (s / step).round() * step).collect()
//! };
//! let (fine, coarse) = (quantise(1.0 / 2048.0), quantise(1.0 / 128.0));
//!
//! // 2f: estimated mean MUSHRA score, 0–100.
//! let a = sori_metrics::twof_score(&reference, &fine, 48_000, 2)?;
//! let b = sori_metrics::twof_score(&reference, &coarse, 48_000, 2)?;
//! assert!(a.mushra > b.mushra);
//!
//! // ViSQOL-compatible MOS-LQO, 1–5.
//! let a = sori_metrics::visqol_score(&reference, &fine, 48_000, 2)?;
//! let b = sori_metrics::visqol_score(&reference, &coarse, 48_000, 2)?;
//! assert!(a.moslqo > b.moslqo);
//! # Ok(())
//! # }
//! ```
//!
//! # Stability
//!
//! Version 0.x: the API may change between minor versions. Result and error types are
//! `#[non_exhaustive]` so fields and variants can be added.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(clippy::doc_markdown)] // proper names (ViSQOL, PEAQ, MUSHRA) in prose

pub(crate) mod fft;
pub mod peaq;
pub mod separation;
pub mod source;
pub mod twof;
pub mod visqol;
pub mod wav;

pub use twof::{score as twof_score, TwoF};
pub use visqol::{score as visqol_score, Visqol};

/// Errors of the metric entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The metric is defined at another sampling rate (both metrics: 48 kHz).
    SampleRate(u32),
    /// Unsupported channel count (2f: 1 or 2; ViSQOL: at least 1).
    Channels(usize),
    /// Fewer samples per channel than the metric needs (2f: one 2048-sample frame; ViSQOL:
    /// one 0.6 s patch after the first 0.28 s).
    TooShort(usize),
    /// Malformed or unsupported WAV data.
    Wav(String),
    /// Malformed or unsupported SVR model data.
    Model(String),
    /// Inputs that do not fit together (source counts, signal lengths, channel layout).
    Mismatch(String),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SampleRate(r) => write!(f, "unsupported sample rate {r} Hz (48000 required)"),
            Self::Channels(c) => write!(f, "unsupported channel count {c}"),
            Self::TooShort(n) => write!(f, "signal too short: {n} samples per channel"),
            Self::Wav(m) => write!(f, "WAV: {m}"),
            Self::Model(m) => write!(f, "model: {m}"),
            Self::Mismatch(m) => write!(f, "inputs do not match: {m}"),
        }
    }
}

impl std::error::Error for Error {}

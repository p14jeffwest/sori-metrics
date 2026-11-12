//! A ViSQOL v3 audio-mode compatible implementation (MOS-LQO, 48 kHz).
//!
//! **Audio (music) mode only.** This module estimates the quality of coded music and general
//! audio; it is not meant for, and has not been validated for, speech quality estimation.
//! There is no speech mode (no voice activity detection, no speech model).
//!
//! An independent implementation, not affiliated with or endorsed by Google or the ViSQOL
//! authors, written from the published descriptions (chiefly Chinen et al. 2020, below); no
//! reference implementation source code was used:
//! - M. Chinen et al., "ViSQOL v3: An Open Source Production Ready Objective Speech and Audio
//!   Metric", QoMEX 2020 (arXiv:2004.09584): gammatone ERB spectrogram up to 24 kHz, 80 ms
//!   windows with a 20 ms hop, silence thresholds, patch alignment, NSIM, libsvm SVR mapping.
//! - A. Hines et al., "ViSQOL: an objective speech quality model", EURASIP JASMP 2015
//!   (doi:10.1186/s13636-015-0054-9): level matching, patches, NSIM definition.
//! - A. Hines et al., "ViSQOLAudio: An objective audio quality metric for low bitrate
//!   codecs", JASA Express Letters 137(6), 2015: audio mode without VAD, bands from 50 Hz.
//! - M. Slaney, "An Efficient Implementation of the Patterson-Holdsworth Auditory Filter
//!   Bank", Apple Computer Technical Report #35, 1993: gammatone filter design.
//!
//! The constants the papers leave open were fixed against the intermediate *outputs* of
//! `visqol-python` 3.8.0 (each is documented where it is defined; the README lists them with
//! the evidence). The SVR model is Google's audio-mode model file, shipped as data under the
//! Apache License 2.0 (see `NOTICE`).
//!
//! Pipeline: level matching and global alignment → gammatone spectrograms ([`gammatone`]) →
//! 0.6 s patches → NSIM per band → mean over patches (`fvnsim`) → SVR ([`svr`]) → MOS.

#![allow(clippy::doc_markdown)] // proper names (ViSQOL, NSIM, libsvm) in prose

pub(crate) mod align;
pub mod gammatone;
pub(crate) mod nsim;
pub(crate) mod source;
pub(crate) mod spectrogram;
pub mod svr;

use crate::source::Source;
use crate::Error;
use source::{Aligned, Blocks, Mono, SourceMono};
use spectrogram::{build, frame_count, Framing, Spectrogram};

/// Number of bands in audio mode: the feature count of the shipped SVR model.
pub const AUDIO_BANDS: usize = 32;
/// Lowest band centre, Hz (ViSQOLAudio: bands from 50 Hz).
pub const AUDIO_MIN_FREQ: f64 = 50.0;
/// Upper edge of the analysed range in audio mode, Hz (v3: "up to 24 kHz").
pub const AUDIO_MAX_FREQ: f64 = 24_000.0;
/// Sample rate of audio mode, Hz.
pub const AUDIO_SAMPLE_RATE: u32 = 48_000;
/// Patch length in frames (0.6 s at a 20 ms hop; the reference tool's patch duration).
pub const PATCH_FRAMES: usize = 30;
/// First patch start frame (0.28 s at a 20 ms hop; the reference tool's first patch).
pub const FIRST_PATCH_FRAME: usize = 14;

/// Half-width, in frames, of the search for the best-matching degraded patch. Wider searches
/// (±1, ±2, ±5 frames) gave identical results on the validation pairs, so none is done.
const SEARCH_FRAMES: usize = 0;

/// One reference patch and its matched degraded patch.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PatchSimilarity {
    /// Reference patch start, seconds.
    pub ref_start: f64,
    /// Matched degraded patch start, seconds (after global alignment).
    pub deg_start: f64,
    /// Mean NSIM of the patch over its 32 bands.
    pub similarity: f64,
    /// Per-band mean NSIM over the patch frames.
    pub band_means: Vec<f64>,
}

/// Result of one ViSQOL comparison.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Visqol {
    /// Predicted MOS (listening quality, objective).
    pub moslqo: f64,
    /// Mean of `fvnsim`.
    pub vnsim: f64,
    /// Per-band NSIM averaged over patches (the SVR features), low band first.
    pub fvnsim: Vec<f64>,
    /// Band centre frequencies, Hz.
    pub center_freq: Vec<f64>,
    /// Global lag applied to the degraded signal, samples.
    pub lag: isize,
    /// Every patch.
    pub patches: Vec<PatchSimilarity>,
    /// Analysed length (the reference length), seconds.
    pub duration: f64,
}

/// Reusable analysis state (filterbank and SVR model). Build once with [`Analyzer::audio`]
/// when scoring many pairs.
#[derive(Debug, Clone)]
pub struct Analyzer {
    bank: gammatone::FilterBank,
    model: svr::SvrModel,
    framing: Framing,
}

impl Analyzer {
    /// Audio mode at 48 kHz with the shipped SVR model.
    ///
    /// # Panics
    /// Never in practice: the shipped model is checked by the unit tests.
    pub fn audio() -> Self {
        Self {
            bank: gammatone::FilterBank::new(
                f64::from(AUDIO_SAMPLE_RATE),
                AUDIO_BANDS,
                AUDIO_MIN_FREQ,
                AUDIO_MAX_FREQ,
            ),
            model: svr::SvrModel::parse(svr::AUDIO_MODEL).expect("shipped model parses"),
            framing: Framing::default(),
        }
    }

    /// Score mono signals at 48 kHz (`reference`, `degraded`; any lengths). The degraded
    /// signal is shifted by the global lag, cut or zero-padded to the reference length and
    /// scaled to the reference's mean square before analysis.
    ///
    /// Same as `self.reference_mono(reference)?.score_mono(degraded)`.
    ///
    /// # Errors
    /// [`Error::TooShort`] if the reference does not hold one patch.
    pub fn score_mono(&self, reference: &[f64], degraded: &[f64]) -> Result<Visqol, Error> {
        self.reference_mono(reference)?.score_mono(degraded)
    }

    /// Analyse a mono 48 kHz reference once, for scoring several degraded signals against it
    /// with [`Reference::score_mono`].
    ///
    /// # Errors
    /// [`Error::TooShort`] if the reference does not hold one patch.
    pub fn reference_mono<'a>(&self, reference: &'a [f64]) -> Result<Reference<'a>, Error> {
        self.reference(Mono::F64(reference), 1)
    }

    fn reference<'a>(&self, signal: Mono<'a>, channels: usize) -> Result<Reference<'a>, Error> {
        let fs = f64::from(AUDIO_SAMPLE_RATE);
        if frame_count(signal.len(), &self.framing, fs) < FIRST_PATCH_FRAME + PATCH_FRAMES {
            return Err(Error::TooShort(signal.len()));
        }
        Ok(Reference {
            analyzer: self.clone(),
            signal,
            channels,
            mean_square: source::mean_square(&signal),
            spectrogram: build(&signal, fs, &self.bank, &self.framing),
        })
    }
}

/// A reference signal analysed once — its spectrogram and level — for scoring any number of
/// degraded signals against it.
///
/// Scoring through a `Reference` gives exactly the same results as the one-shot functions
/// ([`score`], [`Analyzer::score_mono`]), which use it internally; it saves the reference
/// half of the filterbank work on every further degraded signal. It borrows the reference
/// samples (the alignment reads them) and holds about 13 KB per second of signal.
///
/// ```
/// # fn main() -> Result<(), sori_metrics::Error> {
/// # let tone = |a: f32| -> Vec<f32> { (0..96_000)
/// #     .map(|i| a * (i as f32 * 0.0575).sin()).collect() };
/// # let (reference, a, b) = (tone(0.3), tone(0.29), tone(0.2));
/// let r = sori_metrics::visqol::Reference::new(&reference, 48_000, 1)?;
/// for candidate in [&a, &b] {
///     println!("{:.3}", r.score(candidate)?.moslqo);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Reference<'a> {
    analyzer: Analyzer,
    signal: Mono<'a>,
    channels: usize,
    mean_square: f64,
    spectrogram: Spectrogram,
}

impl<'a> Reference<'a> {
    /// Analyse a reference: interleaved `f32` at 48 kHz with `channels` channels, averaged
    /// to mono.
    ///
    /// # Errors
    /// [`Error::SampleRate`] unless 48 kHz, [`Error::Channels`] for 0 channels,
    /// [`Error::TooShort`] if shorter than one patch.
    pub fn new(reference: &'a [f32], sample_rate: u32, channels: usize) -> Result<Self, Error> {
        if sample_rate != AUDIO_SAMPLE_RATE {
            return Err(Error::SampleRate(sample_rate));
        }
        if channels == 0 {
            return Err(Error::Channels(channels));
        }
        Analyzer::audio().reference(Mono::Interleaved(reference, channels), channels)
    }

    /// Score interleaved `f32` samples with the reference's channel count (1 for a reference
    /// made by [`Analyzer::reference_mono`]), averaged to mono.
    ///
    /// # Errors
    /// None at present (kept for future checks); the reference was validated when built.
    pub fn score(&self, test: &[f32]) -> Result<Visqol, Error> {
        self.score_signal(Mono::Interleaved(test, self.channels))
    }

    /// Score a mono 48 kHz degraded signal.
    ///
    /// # Errors
    /// None at present (kept for future checks); the reference was validated when built.
    pub fn score_mono(&self, degraded: &[f64]) -> Result<Visqol, Error> {
        self.score_signal(Mono::F64(degraded))
    }

    #[allow(
        clippy::many_single_char_names,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::unnecessary_wraps
    )]
    fn score_signal(&self, degraded: Mono) -> Result<Visqol, Error> {
        Ok(score_core(
            &self.analyzer,
            &self.signal,
            self.mean_square,
            &self.spectrogram,
            &degraded,
        ))
    }
}

/// Align `degraded` to the reference signal, build its spectrogram and compare.
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn score_core(
    an: &Analyzer,
    signal: &impl Blocks,
    mean_square: f64,
    r: &Spectrogram,
    degraded: &impl Blocks,
) -> Visqol {
    {
        let fs = f64::from(AUDIO_SAMPLE_RATE);
        let lag = align::global_lag(signal, degraded);
        let mut deg = Aligned {
            src: degraded,
            lag,
            len: signal.len(),
            gain: None,
        };
        deg.gain = align::level_gain(mean_square, &deg);
        let d = build(&deg, fs, &an.bank, &an.framing);
        let npatch = (r.frames - FIRST_PATCH_FRAME) / PATCH_FRAMES;
        let hop = an.framing.hop_s;
        let mut patches = Vec::with_capacity(npatch);
        for k in 0..npatch {
            let s = FIRST_PATCH_FRAME + PATCH_FRAMES * k;
            let rp = patch(r, s);
            let mut best: Option<(isize, Vec<f64>, f64)> = None;
            let w = SEARCH_FRAMES as isize;
            for o in 0..=2 * w {
                // Offsets in the order 0, −1, +1, −2, +2, … so ties keep the smallest shift.
                let off = if o % 2 == 0 { o / 2 } else { -(o + 1) / 2 };
                let ds = s as isize + off;
                if ds < 0 || ds as usize + PATCH_FRAMES > d.frames {
                    continue;
                }
                let m = nsim::band_means(&rp, &patch(&d, ds as usize), r.bands, PATCH_FRAMES);
                let mean = m.iter().sum::<f64>() / m.len() as f64;
                if best.as_ref().is_none_or(|b| mean > b.2) {
                    best = Some((off, m, mean));
                }
            }
            let (off, band_means, similarity) = best.expect("offset 0 is always valid");
            patches.push(PatchSimilarity {
                ref_start: s as f64 * hop,
                deg_start: (s as isize + off) as f64 * hop,
                similarity,
                band_means,
            });
        }
        let mut fvnsim = vec![0.0; r.bands];
        for p in &patches {
            for (a, b) in fvnsim.iter_mut().zip(&p.band_means) {
                *a += b;
            }
        }
        for a in &mut fvnsim {
            *a /= patches.len() as f64;
        }
        let vnsim = fvnsim.iter().sum::<f64>() / fvnsim.len() as f64;
        let moslqo = an.model.predict(&fvnsim);
        Visqol {
            moslqo,
            vnsim,
            fvnsim,
            center_freq: r.cf.clone(),
            lag,
            duration: signal.len() as f64 / fs,
            patches,
        }
    }
}

/// Copy frames `s..s + PATCH_FRAMES` of every band (band-major).
fn patch(sp: &Spectrogram, s: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(sp.bands * PATCH_FRAMES);
    for b in 0..sp.bands {
        out.extend_from_slice(&sp.data[b * sp.frames + s..b * sp.frames + s + PATCH_FRAMES]);
    }
    out
}

/// [`score`] on two [`Source`]s (any channel counts, each averaged to mono), read in pieces:
/// memory use does not grow with the signal length beyond the two spectrograms (13 KB per
/// second each) and the alignment transform (16 bytes per 16 samples, rounded up to a power
/// of two). Same values as [`score`] on the whole signals.
///
/// # Errors
/// As [`score`], and any error of [`Source::read`].
pub fn score_source(reference: impl Source, test: impl Source) -> Result<Visqol, Error> {
    for s in [reference.sample_rate(), test.sample_rate()] {
        if s != AUDIO_SAMPLE_RATE {
            return Err(Error::SampleRate(s));
        }
    }
    let an = Analyzer::audio();
    let fs = f64::from(AUDIO_SAMPLE_RATE);
    let r = SourceMono::new(reference);
    if frame_count(r.len(), &an.framing, fs) < FIRST_PATCH_FRAME + PATCH_FRAMES {
        return Err(Error::TooShort(r.len()));
    }
    let ms = source::mean_square(&r);
    let spec = build(&r, fs, &an.bank, &an.framing);
    let d = SourceMono::new(test);
    let v = score_core(&an, &r, ms, &spec, &d);
    match r.error().or_else(|| d.error()) {
        Some(e) => Err(e),
        None => Ok(v),
    }
}

/// Values of one time window of a ViSQOL comparison.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Window {
    /// Window start, seconds.
    pub start: f64,
    /// Window end, seconds.
    pub end: f64,
    /// Patches lying entirely inside the window.
    pub patches: usize,
    /// Per-band NSIM averaged over those patches; empty when there are none.
    pub fvnsim: Vec<f64>,
    /// Mean of `fvnsim` (`None` without patches).
    pub vnsim: Option<f64>,
    /// The SVR mapping applied to `fvnsim` (`None` without patches). The model was trained
    /// on whole files; over a few patches it is an indication, not a calibrated MOS.
    pub moslqo: Option<f64>,
}

impl Analyzer {
    /// Windows of `length` seconds every `hop` seconds over `v` (as many as fit in its
    /// duration; one over everything if shorter), each summarising the patches (0.6 s)
    /// that lie entirely inside it.
    ///
    /// # Panics
    /// If `length` or `hop` is not positive.
    #[allow(clippy::cast_precision_loss)]
    pub fn windows(&self, v: &Visqol, length: f64, hop: f64) -> Vec<Window> {
        assert!(length > 0.0 && hop > 0.0, "window length and hop");
        let patch_s = PATCH_FRAMES as f64 * self.framing.hop_s;
        let mut out = Vec::new();
        let mut k = 0usize;
        loop {
            let s = k as f64 * hop;
            let e = if v.duration < length {
                v.duration
            } else {
                s + length
            };
            let inside: Vec<&PatchSimilarity> = v
                .patches
                .iter()
                .filter(|p| p.ref_start >= s - 1e-9 && p.ref_start + patch_s <= e + 1e-9)
                .collect();
            let fvnsim: Vec<f64> = if inside.is_empty() {
                Vec::new()
            } else {
                (0..v.fvnsim.len())
                    .map(|b| {
                        inside.iter().map(|p| p.band_means[b]).sum::<f64>() / inside.len() as f64
                    })
                    .collect()
            };
            let some = !fvnsim.is_empty();
            out.push(Window {
                start: s,
                end: e,
                patches: inside.len(),
                vnsim: some.then(|| fvnsim.iter().sum::<f64>() / fvnsim.len() as f64),
                moslqo: some.then(|| self.model.predict(&fvnsim)),
                fvnsim,
            });
            k += 1;
            if v.duration < length || (k as f64 * hop) + length > v.duration + 1e-9 {
                break;
            }
        }
        out
    }
}

impl Visqol {
    /// [`Analyzer::windows`] with the audio-mode analyser.
    ///
    /// # Panics
    /// If `length` or `hop` is not positive.
    pub fn windows(&self, length: f64, hop: f64) -> Vec<Window> {
        Analyzer::audio().windows(self, length, hop)
    }
}

/// ViSQOL v3 audio-mode compatible MOS of `test` against `reference`: interleaved `f32` at
/// 48 kHz with `channels` channels, averaged to mono.
///
/// # Errors
/// [`Error::SampleRate`] unless 48 kHz, [`Error::Channels`] for 0 channels,
/// [`Error::TooShort`] if shorter than one patch.
pub fn score(
    reference: &[f32],
    test: &[f32],
    sample_rate: u32,
    channels: usize,
) -> Result<Visqol, Error> {
    Reference::new(reference, sample_rate, channels)?.score(test)
}

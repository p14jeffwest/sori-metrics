//! Mono views of the caller's samples, read block by block.
//!
//! The analysis never copies a whole signal: every stage pulls blocks of [`BLOCK`] mono
//! samples from a [`Mono`] view (and the degraded side through [`Aligned`]), so the memory
//! used beyond the caller's own buffers does not grow with the signal length. Each value is
//! computed with exactly the expression the whole-signal version used, so results are
//! unchanged.

use crate::source::{Source, Window};
use crate::Error;
use core::cell::RefCell;

/// Samples per block. A multiple of the alignment decimation factor (16).
pub const BLOCK: usize = 1 << 16;

/// A mono signal: `f64` samples, or interleaved `f32` frames averaged over the channels.
#[derive(Debug, Clone, Copy)]
pub enum Mono<'a> {
    /// Mono samples as given.
    F64(&'a [f64]),
    /// Interleaved samples and their channel count (≥ 1); trailing partial frames ignored.
    Interleaved(&'a [f32], usize),
}

impl Mono<'_> {
    /// Number of mono samples.
    pub fn len(&self) -> usize {
        match *self {
            Mono::F64(x) => x.len(),
            Mono::Interleaved(x, ch) => x.len() / ch,
        }
    }

    /// Write samples `start..start + out.len()` (all inside the signal) to `out`.
    pub fn fill(&self, start: usize, out: &mut [f64]) {
        match *self {
            Mono::F64(x) => out.copy_from_slice(&x[start..start + out.len()]),
            Mono::Interleaved(x, ch) => {
                #[allow(clippy::cast_precision_loss)]
                let n = ch as f64;
                for (o, f) in out.iter_mut().zip(x[start * ch..].chunks_exact(ch)) {
                    *o = f.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
                }
            }
        }
    }
}

/// The degraded signal as analysed: shifted by the global lag (`d[i + lag]`, zero outside),
/// cut or zero-padded to `len`, and multiplied by `gain` when one is set.
#[derive(Debug, Clone, Copy)]
pub struct Aligned<'b, B: Blocks> {
    /// The degraded mono signal.
    pub src: &'b B,
    /// Global lag, samples.
    pub lag: isize,
    /// Analysed length (the reference length).
    pub len: usize,
    /// Level-matching gain, if any.
    pub gain: Option<f64>,
}

impl<B: Blocks> Aligned<'_, B> {
    /// Write samples `start..start + out.len()` (all below `len`) to `out`.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn fill(&self, start: usize, out: &mut [f64]) {
        let n = self.src.len() as isize;
        // Source indices start + lag .. start + lag + out.len(), clipped to 0..n.
        let s = start as isize + self.lag;
        let e = s + out.len() as isize;
        let (cs, ce) = (s.clamp(0, n), e.clamp(0, n));
        out.fill(0.0);
        if ce > cs {
            let at = (cs - s) as usize;
            self.src
                .fill(cs as usize, &mut out[at..at + (ce - cs) as usize]);
        }
        if let Some(g) = self.gain {
            for v in out.iter_mut() {
                *v *= g;
            }
        }
    }
}

/// A [`crate::source::Source`] averaged to mono, read through a window of [`crate::source::PIECE_FRAMES`].
/// A read error is kept and reported by [`SourceMono::error`]; the failed samples read as 0.
pub struct SourceMono<S: Source> {
    window: RefCell<Window<S>>,
    error: RefCell<Option<Error>>,
}

impl<S: Source> SourceMono<S> {
    pub fn new(src: S) -> Self {
        Self {
            window: RefCell::new(Window::new(src)),
            error: RefCell::new(None),
        }
    }

    /// Take the first read error, if any.
    pub fn error(&self) -> Option<Error> {
        self.error.borrow_mut().take()
    }
}

impl<S: Source> Blocks for SourceMono<S> {
    fn len(&self) -> usize {
        self.window.borrow().frames()
    }
    fn fill(&self, start: usize, out: &mut [f64]) {
        let mut w = self.window.borrow_mut();
        let end = start + out.len();
        if let Err(e) = w.prepare(start, end) {
            out.fill(0.0);
            self.error.borrow_mut().get_or_insert(e);
            return;
        }
        let ch = w.channels();
        #[allow(clippy::cast_precision_loss)]
        let n = ch as f64;
        // Same expression as `Mono::Interleaved`.
        for (o, f) in out.iter_mut().zip(w.slice(start, end).chunks_exact(ch)) {
            *o = f.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        }
    }
}

/// Anything the spectrogram can read block by block.
pub trait Blocks {
    /// Number of samples.
    fn len(&self) -> usize;
    /// Write samples `start..start + out.len()` to `out`.
    fn fill(&self, start: usize, out: &mut [f64]);
}

impl Blocks for Mono<'_> {
    fn len(&self) -> usize {
        Mono::len(self)
    }
    fn fill(&self, start: usize, out: &mut [f64]) {
        Mono::fill(self, start, out);
    }
}

impl<B: Blocks> Blocks for Aligned<'_, B> {
    fn len(&self) -> usize {
        self.len
    }
    fn fill(&self, start: usize, out: &mut [f64]) {
        Aligned::fill(self, start, out);
    }
}

/// Mean square of a block source, accumulated sample by sample in order
/// (`Σ v² / max(len, 1)`).
#[allow(clippy::cast_precision_loss)]
pub fn mean_square(x: &impl Blocks) -> f64 {
    let len = x.len();
    let mut buf = vec![0.0; BLOCK.min(len)];
    let mut acc = -0.0f64; // the start value of `Iterator::sum` for f64
    let mut s = 0;
    while s < len {
        let n = BLOCK.min(len - s);
        x.fill(s, &mut buf[..n]);
        for v in &buf[..n] {
            acc += v * v;
        }
        s += n;
    }
    acc / len.max(1) as f64
}

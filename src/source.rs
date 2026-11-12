//! Signals read piece by piece.
//!
//! A [`Source`] gives the metrics random access to an interleaved signal without holding it
//! in memory: a WAV file on disk ([`crate::wav::WavReader`]), audio kept by a host
//! (a browser `Blob`, a decoder that can seek), or a slice ([`Interleaved`]). The
//! `*_source` functions of [`crate::twof`] and [`crate::visqol`] read their inputs in pieces
//! of [`PIECE_FRAMES`] (10 s at 48 kHz), so their memory use does not depend on the signal
//! length beyond small per-frame results. They give the same values as the slice functions.

use crate::Error;

/// Sample frames per read: 10 s at 48 kHz.
pub const PIECE_FRAMES: usize = 480_000;

/// An interleaved `f32` signal in [-1, 1] that can be read at any position.
pub trait Source {
    /// Sampling rate, Hz.
    fn sample_rate(&self) -> u32;
    /// Channel count (≥ 1).
    fn channels(&self) -> usize;
    /// Length in sample frames.
    fn frames(&self) -> usize;
    /// Fill `out` with the interleaved frames `start..start + out.len() / channels()`.
    /// Callers only ask for whole frames inside `0..frames()`.
    ///
    /// # Errors
    /// Whatever reading the underlying storage can fail with.
    fn read(&mut self, start: usize, out: &mut [f32]) -> Result<(), Error>;
}

impl<S: Source + ?Sized> Source for &mut S {
    fn sample_rate(&self) -> u32 {
        (**self).sample_rate()
    }
    fn channels(&self) -> usize {
        (**self).channels()
    }
    fn frames(&self) -> usize {
        (**self).frames()
    }
    fn read(&mut self, start: usize, out: &mut [f32]) -> Result<(), Error> {
        (**self).read(start, out)
    }
}

impl<S: Source + ?Sized> Source for Box<S> {
    fn sample_rate(&self) -> u32 {
        (**self).sample_rate()
    }
    fn channels(&self) -> usize {
        (**self).channels()
    }
    fn frames(&self) -> usize {
        (**self).frames()
    }
    fn read(&mut self, start: usize, out: &mut [f32]) -> Result<(), Error> {
        (**self).read(start, out)
    }
}

/// A slice of interleaved samples as a [`Source`] (trailing partial frames ignored).
#[derive(Debug, Clone, Copy)]
pub struct Interleaved<'a> {
    samples: &'a [f32],
    sample_rate: u32,
    channels: usize,
}

impl<'a> Interleaved<'a> {
    /// Wrap `samples` with `channels` channels at `sample_rate`.
    ///
    /// # Errors
    /// [`Error::Channels`] for 0 channels.
    pub fn new(samples: &'a [f32], sample_rate: u32, channels: usize) -> Result<Self, Error> {
        if channels == 0 {
            return Err(Error::Channels(0));
        }
        Ok(Self {
            samples,
            sample_rate,
            channels,
        })
    }
}

impl Source for Interleaved<'_> {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn channels(&self) -> usize {
        self.channels
    }
    fn frames(&self) -> usize {
        self.samples.len() / self.channels
    }
    fn read(&mut self, start: usize, out: &mut [f32]) -> Result<(), Error> {
        let s = start * self.channels;
        out.copy_from_slice(&self.samples[s..s + out.len()]);
        Ok(())
    }
}

/// A window of a [`Source`] kept in memory: frames `start..start + len` (interleaved).
pub(crate) struct Window<S: Source> {
    pub(crate) src: S,
    buf: Vec<f32>,
    start: usize,
    len: usize,
    ch: usize,
}

impl<S: Source> Window<S> {
    pub(crate) fn new(src: S) -> Self {
        let ch = src.channels();
        Self {
            src,
            buf: Vec::new(),
            start: 0,
            len: 0,
            ch,
        }
    }

    pub(crate) fn channels(&self) -> usize {
        self.ch
    }

    pub(crate) fn frames(&self) -> usize {
        self.src.frames()
    }

    /// Make frames `from..to` (inside the signal) available, reading ahead up to
    /// [`PIECE_FRAMES`] beyond `from` when `to` is closer.
    pub(crate) fn prepare(&mut self, from: usize, to: usize) -> Result<(), Error> {
        if from >= self.start && to <= self.start + self.len {
            return Ok(());
        }
        let end = (from + PIECE_FRAMES.max(to - from)).min(self.src.frames());
        let ch = self.ch;
        // Keep what is already there and overlaps (reading forward is the common case).
        if from >= self.start && from < self.start + self.len {
            let keep_from = from - self.start;
            self.buf.drain(..keep_from * ch);
            let have = self.len - keep_from;
            self.start = from;
            self.buf.resize((end - from) * ch, 0.0);
            self.src.read(from + have, &mut self.buf[have * ch..])?;
        } else {
            self.start = from;
            self.buf.resize((end - from) * ch, 0.0);
            self.src.read(from, &mut self.buf)?;
        }
        self.len = end - from;
        Ok(())
    }

    /// Sample of channel `c` at frame `i` (inside the prepared range).
    #[inline]
    pub(crate) fn at(&self, c: usize, i: usize) -> f32 {
        self.buf[(i - self.start) * self.ch + c]
    }

    /// The prepared frames `from..to` as an interleaved slice.
    pub(crate) fn slice(&self, from: usize, to: usize) -> &[f32] {
        &self.buf[(from - self.start) * self.ch..(to - self.start) * self.ch]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_reads_what_the_source_holds() {
        #[allow(clippy::cast_precision_loss)]
        let x: Vec<f32> = (0..2 * 1_200_000).map(|i| (i % 977) as f32).collect();
        let mut w = Window::new(Interleaved::new(&x, 48_000, 2).unwrap());
        for (from, to) in [
            (0, 10),
            (5, 600_000),
            (599_999, 1_100_000),
            (3, 4),
            (1_199_990, 1_200_000),
        ] {
            w.prepare(from, to).unwrap();
            for i in from..to {
                assert_eq!(w.at(1, i).to_bits(), x[2 * i + 1].to_bits());
            }
            assert_eq!(w.slice(from, to), &x[2 * from..2 * to]);
        }
    }
}

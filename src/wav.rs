//! Minimal RIFF/WAVE reader: PCM 16/24/32-bit integer and 32/64-bit float, also inside
//! `WAVE_FORMAT_EXTENSIBLE`. Enough for the measurement pairs (ffmpeg `pcm_f32le`) and the
//! corpus files.

use crate::Error;

/// Decoded audio: interleaved samples in [-1, 1].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Audio {
    /// Sampling rate, Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: usize,
    /// Interleaved samples.
    pub samples: Vec<f32>,
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Bytes per sample of a supported `(format tag, bits)` pair.
fn sample_bytes(tag: u16, bits: u16) -> Result<usize, Error> {
    match (tag, bits) {
        (1, 16) => Ok(2),
        (1, 24) => Ok(3),
        (1 | 3, 32) => Ok(4),
        (3, 64) => Ok(8),
        _ => Err(Error::Wav(format!(
            "unsupported format tag {tag} / {bits} bits"
        ))),
    }
}

/// Append the samples in `data` (whole samples only; a trailing partial one is ignored).
fn convert(tag: u16, bits: u16, data: &[u8], out: &mut Vec<f32>) -> Result<(), Error> {
    let n = data.len() / sample_bytes(tag, bits)?;
    let old = out.len();
    out.resize(old + n, 0.0);
    convert_into(tag, bits, data, &mut out[old..])
}

/// Convert the first `out.len()` samples of `data` into `out`.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn convert_into(tag: u16, bits: u16, data: &[u8], out: &mut [f32]) -> Result<(), Error> {
    let size = sample_bytes(tag, bits)?;
    let pairs = out.iter_mut().zip(data.chunks_exact(size));
    match (tag, bits) {
        (1, 16) => {
            for (o, c) in pairs {
                *o = f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0;
            }
        }
        (1, 24) => {
            for (o, c) in pairs {
                *o = (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8_388_608.0;
            }
        }
        (1, 32) => {
            for (o, c) in pairs {
                *o = (f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])) / 2_147_483_648.0)
                    as f32;
            }
        }
        (3, 32) => {
            for (o, c) in pairs {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        _ => {
            for (o, c) in pairs {
                *o = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32;
            }
        }
    }
    Ok(())
}

/// Parse a WAV file image.
///
/// # Errors
/// [`Error::Wav`] for anything that is not a supported PCM or float WAV.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn parse(bytes: &[u8]) -> Result<Audio, Error> {
    let bad = |m: &str| Error::Wav(m.to_string());
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let mut pos = 12;
    let mut fmt: Option<(u16, usize, u32, u16)> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_at(bytes, pos + 4) as usize;
        let body = pos + 8;
        let end = body.saturating_add(size).min(bytes.len());
        if id == b"fmt " {
            if size < 16 || body + 16 > bytes.len() {
                return Err(bad("short fmt chunk"));
            }
            let mut tag = u16_at(bytes, body);
            if tag == 0xFFFE && size >= 26 {
                tag = u16_at(bytes, body + 24); // sub-format GUID starts with the format tag
            }
            let channels = usize::from(u16_at(bytes, body + 2));
            let rate = u32_at(bytes, body + 4);
            let bits = u16_at(bytes, body + 14);
            fmt = Some((tag, channels, rate, bits));
        } else if id == b"data" {
            let (tag, channels, rate, bits) = fmt.ok_or_else(|| bad("data before fmt"))?;
            if channels == 0 {
                return Err(bad("zero channels"));
            }
            let data = &bytes[body..end];
            let mut samples = Vec::new();
            convert(tag, bits, data, &mut samples)?;
            let whole = samples.len() / channels * channels;
            samples.truncate(whole);
            return Ok(Audio {
                sample_rate: rate,
                channels,
                samples,
            });
        }
        pos = body + size + (size & 1);
    }
    Err(bad("no data chunk"))
}

/// Read and parse a WAV file.
///
/// The data chunk is read and converted piece by piece, so the only large allocation is the
/// returned sample buffer. Same result as [`parse`] on the whole file image.
///
/// # Errors
/// [`Error::Wav`] on I/O failure or an unsupported file.
pub fn read(path: &std::path::Path) -> Result<Audio, Error> {
    let mut r = WavReader::open(path)?;
    let mut samples = vec![0.0; r.frames * r.channels];
    let piece = crate::source::PIECE_FRAMES;
    let mut s = 0;
    while s < r.frames {
        let n = piece.min(r.frames - s);
        let ch = r.channels;
        crate::source::Source::read(&mut r, s, &mut samples[s * ch..(s + n) * ch])?;
        s += n;
    }
    Ok(Audio {
        sample_rate: r.rate,
        channels: r.channels,
        samples,
    })
}

/// A WAV file opened for reading at any position (a [`Source`](crate::source::Source)),
/// without loading it.
#[derive(Debug)]
pub struct WavReader {
    file: std::fs::File,
    name: String,
    data: u64,
    frames: usize,
    channels: usize,
    rate: u32,
    tag: u16,
    bits: u16,
    bytes: Vec<u8>,
}

impl WavReader {
    /// Open `path` and read its header.
    ///
    /// # Errors
    /// [`Error::Wav`] on I/O failure or an unsupported file.
    pub fn open(path: &std::path::Path) -> Result<Self, Error> {
        use std::io::{Read, Seek, SeekFrom};
        let io = |e: std::io::Error| Error::Wav(format!("{}: {e}", path.display()));
        let bad = |m: &str| Error::Wav(m.to_string());
        let mut f = std::fs::File::open(path).map_err(io)?;
        let file_len = f.metadata().map_err(io)?.len();
        // Read up to n bytes (fewer at the end of the file).
        let read_upto = |f: &mut std::fs::File, n: usize| -> Result<Vec<u8>, Error> {
            let mut v = Vec::with_capacity(n);
            f.take(n as u64).read_to_end(&mut v).map_err(io)?;
            Ok(v)
        };
        let head = read_upto(&mut f, 12)?;
        if head.len() < 12 || &head[0..4] != b"RIFF" || &head[8..12] != b"WAVE" {
            return Err(bad("not a RIFF/WAVE file"));
        }
        let mut pos: u64 = 12;
        let mut fmt: Option<(u16, usize, u32, u16)> = None;
        while pos + 8 <= file_len {
            let h = read_upto(&mut f, 8)?;
            let size = u64::from(u32_at(&h, 4));
            let body = pos + 8;
            let end = body.saturating_add(size).min(file_len);
            if &h[0..4] == b"fmt " {
                let b = read_upto(&mut f, usize::try_from(end - body).unwrap_or(0))?;
                if size < 16 || b.len() < 16 {
                    return Err(bad("short fmt chunk"));
                }
                let mut tag = u16_at(&b, 0);
                if tag == 0xFFFE && size >= 26 && b.len() >= 26 {
                    tag = u16_at(&b, 24); // sub-format GUID starts with the format tag
                }
                fmt = Some((
                    tag,
                    usize::from(u16_at(&b, 2)),
                    u32_at(&b, 4),
                    u16_at(&b, 14),
                ));
            } else if &h[0..4] == b"data" {
                let (tag, channels, rate, bits) = fmt.ok_or_else(|| bad("data before fmt"))?;
                if channels == 0 {
                    return Err(bad("zero channels"));
                }
                let ss = sample_bytes(tag, bits)?;
                let total = usize::try_from(end - body).map_err(|_| bad("data chunk too large"))?;
                return Ok(Self {
                    file: f,
                    name: path.display().to_string(),
                    data: body,
                    frames: total / ss / channels,
                    channels,
                    rate,
                    tag,
                    bits,
                    bytes: Vec::new(),
                });
            }
            pos = body + size + (size & 1);
            f.seek(SeekFrom::Start(pos.min(file_len))).map_err(io)?;
        }
        Err(bad("no data chunk"))
    }
}

impl crate::source::Source for WavReader {
    fn sample_rate(&self) -> u32 {
        self.rate
    }
    fn channels(&self) -> usize {
        self.channels
    }
    fn frames(&self) -> usize {
        self.frames
    }
    fn read(&mut self, start: usize, out: &mut [f32]) -> Result<(), Error> {
        use std::io::{Read, Seek, SeekFrom};
        let io = |e: std::io::Error| Error::Wav(format!("{}: {e}", self.name));
        let ss = sample_bytes(self.tag, self.bits)?;
        let off = self.data + (start * self.channels * ss) as u64;
        self.bytes.resize(out.len() * ss, 0);
        self.file.seek(SeekFrom::Start(off)).map_err(io)?;
        self.file.read_exact(&mut self.bytes).map_err(io)?;
        convert_into(self.tag, self.bits, &self.bytes, out)
    }
}

/// Encode interleaved samples as a 32-bit float WAV image.
///
/// # Panics
/// If the data does not fit a RIFF file (more than 4 GiB) or `channels` is not 1..=65535.
pub fn encode_f32(sample_rate: u32, channels: usize, samples: &[f32]) -> Vec<u8> {
    let ch = u16::try_from(channels).expect("channel count");
    let data_len = u32::try_from(samples.len() * 4).expect("WAV too large");
    let mut b = Vec::with_capacity(44 + samples.len() * 4);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes());
    b.extend_from_slice(&ch.to_le_bytes());
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&(sample_rate * u32::from(ch) * 4).to_le_bytes());
    b.extend_from_slice(&(ch * 4).to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for x in samples {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pcm16_stereo() {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&44u32.to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        for v in [1u16, 2] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&48_000u32.to_le_bytes());
        b.extend_from_slice(&192_000u32.to_le_bytes());
        for v in [4u16, 16] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(b"data");
        b.extend_from_slice(&8u32.to_le_bytes());
        for v in [16384i16, -16384, 0, 32767] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let a = parse(&b).unwrap();
        assert_eq!((a.sample_rate, a.channels), (48_000, 2));
        assert_eq!(a.samples[..2], [0.5, -0.5]);
    }

    #[test]
    fn read_matches_parse() {
        // Odd-length data with a padding byte, an extra chunk before it, 24-bit PCM.
        let x: Vec<u8> = (0..=3 * 2 * 5000u32)
            .map(|i| u8::try_from(i * 37 % 251).unwrap())
            .collect();
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF\0\0\0\0WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        for v in [1u16, 2] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&48_000u32.to_le_bytes());
        b.extend_from_slice(&288_000u32.to_le_bytes());
        for v in [6u16, 24] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(b"LIST");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(b"abc\0");
        b.extend_from_slice(b"data");
        b.extend_from_slice(&u32::try_from(x.len()).unwrap().to_le_bytes());
        b.extend_from_slice(&x);
        let path =
            std::env::temp_dir().join(format!("sori-metrics-wav-{}.wav", std::process::id()));
        std::fs::write(&path, &b).unwrap();
        let got = read(&path);
        std::fs::remove_file(&path).unwrap();
        let (got, want) = (got.unwrap(), parse(&b).unwrap());
        assert_eq!(got, want);
        assert_eq!(want.samples.len(), 2 * 5000);
    }

    #[test]
    fn float_round_trip() {
        let x = [0.25f32, -1.0, 0.125, 3.0e-5];
        let a = parse(&encode_f32(48_000, 2, &x)).unwrap();
        assert_eq!((a.sample_rate, a.channels), (48_000, 2));
        assert_eq!(a.samples, x);
    }
}

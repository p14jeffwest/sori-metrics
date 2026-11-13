//! Window-by-window 2f and ViSQOL of one pair, read from disk in 10 s pieces.
//!
//! ```text
//! cargo run --release --example windows -- <reference> <test> [--window 3] [--hop 1]
//!     [--metrics twof,visqol] [--blocks] [--channels 2]
//! ```
//!
//! Inputs are WAV files (read with `wav::WavReader`) or raw interleaved `f32` little-endian
//! files (`.f32`, 48 kHz, `--channels`, default 2). Neither file is loaded whole.
//!
//! Output (tab-separated):
//! - `total  twof  <mushra> <adb> <avg_mod_diff1> <total_nmr dB>` and `total  visqol  <moslqo> <vnsim> <lag>`;
//! - one `window` line per window: start and end (s), then for 2f the valid and distorted
//!   blocks, the distorted share and the window 2f (empty if no valid block), then for ViSQOL
//!   the patches and the window MOS (empty if none);
//! - with `--blocks`, one `block` line per 2f block: index, start (s), valid, detection
//!   probability, steps, `ModDiff1` per channel;
//! - last, `peak_rss_kib <n>` (Linux `VmHWM`, 0 elsewhere).

#![allow(clippy::doc_markdown)]

use sori_metrics::source::Source;
use sori_metrics::Error;
use std::fmt::Write as _;
use std::io::{Read, Seek, SeekFrom};

/// A raw interleaved `f32le` file read on demand.
struct RawF32 {
    file: std::fs::File,
    frames: usize,
    channels: usize,
    bytes: Vec<u8>,
}

impl Source for RawF32 {
    fn sample_rate(&self) -> u32 {
        48_000
    }
    fn channels(&self) -> usize {
        self.channels
    }
    fn frames(&self) -> usize {
        self.frames
    }
    fn read(&mut self, start: usize, out: &mut [f32]) -> Result<(), Error> {
        let io = |e: std::io::Error| Error::Wav(e.to_string());
        self.bytes.resize(out.len() * 4, 0);
        self.file
            .seek(SeekFrom::Start((start * self.channels * 4) as u64))
            .map_err(io)?;
        self.file.read_exact(&mut self.bytes).map_err(io)?;
        for (o, c) in out.iter_mut().zip(self.bytes.chunks_exact(4)) {
            *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        Ok(())
    }
}

fn open(path: &str, channels: usize) -> Box<dyn Source> {
    let is_raw = std::path::Path::new(path)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("f32"));
    if is_raw {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let len = usize::try_from(file.metadata().expect("metadata").len()).expect("size");
        Box::new(RawF32 {
            file,
            frames: len / 4 / channels,
            channels,
            bytes: Vec::new(),
        })
    } else {
        Box::new(
            sori_metrics::wav::WavReader::open(path.as_ref())
                .unwrap_or_else(|e| panic!("{path}: {e}")),
        )
    }
}

fn peak_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut opt = |name: &str| -> Option<String> {
        let i = args.iter().position(|a| a == name)?;
        let v = args.remove(i + 1);
        args.remove(i);
        Some(v)
    };
    let num = |v: Option<String>, d: f64| v.map_or(d, |s| s.parse().expect("number"));
    let (window, hop) = (num(opt("--window"), 3.0), num(opt("--hop"), 1.0));
    let channels: usize = opt("--channels").map_or(2, |s| s.parse().expect("number"));
    let metrics = opt("--metrics").unwrap_or_else(|| "twof,visqol".into());
    let dump_blocks = if let Some(i) = args.iter().position(|a| a == "--blocks") {
        args.remove(i);
        true
    } else {
        false
    };
    assert!(
        args.len() == 2,
        "usage: windows <reference> <test> [options]"
    );
    let (rp, tp) = (&args[0], &args[1]);

    let twof = metrics.contains("twof").then(|| {
        sori_metrics::twof::detail_source(open(rp, channels), open(tp, channels))
            .unwrap_or_else(|e| panic!("2f: {e}"))
    });
    let visqol = metrics.contains("visqol").then(|| {
        sori_metrics::visqol::score_source(open(rp, channels), open(tp, channels))
            .unwrap_or_else(|e| panic!("ViSQOL: {e}"))
    });

    if let Some(d) = &twof {
        let s = d.score;
        println!(
            "total\ttwof\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            s.mushra, s.adb, s.avg_mod_diff1, s.total_nmr
        );
    }
    if let Some(v) = &visqol {
        println!("total\tvisqol\t{:.6}\t{:.6}\t{}", v.moslqo, v.vnsim, v.lag);
    }
    let tw = twof
        .as_ref()
        .map(|d| d.windows(window, hop))
        .unwrap_or_default();
    let vw = visqol
        .as_ref()
        .map(|v| v.windows(window, hop))
        .unwrap_or_default();
    for i in 0..tw.len().max(vw.len()) {
        let (start, end) = tw
            .get(i)
            .map(|w| (w.start, w.end))
            .or_else(|| vw.get(i).map(|w| (w.start, w.end)))
            .expect("one of the two");
        let mut line = format!("window\t{start:.3}\t{end:.3}");
        match tw.get(i) {
            Some(w) => {
                let f = w
                    .score
                    .map(|s| format!("{:.3}", s.mushra))
                    .unwrap_or_default();
                let _ = write!(
                    line,
                    "\t{}\t{}\t{:.4}\t{f}",
                    w.valid_blocks, w.distorted_blocks, w.distorted_share
                );
            }
            None => line += "\t\t\t\t",
        }
        match vw.get(i) {
            Some(w) => {
                let m = w.moslqo.map(|m| format!("{m:.4}")).unwrap_or_default();
                let _ = write!(line, "\t{}\t{m}", w.patches);
            }
            None => line += "\t\t",
        }
        println!("{line}");
    }
    if dump_blocks {
        if let Some(d) = &twof {
            for b in d.blocks() {
                println!(
                    "block\t{}\t{:.5}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                    b.index,
                    b.start,
                    u8::from(b.valid),
                    b.detection,
                    b.steps,
                    b.mod_diff[0],
                    b.mod_diff[1]
                );
            }
        }
    }
    println!("peak_rss_kib\t{}", peak_kib());
}

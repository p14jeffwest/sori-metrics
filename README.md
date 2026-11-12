# sori-metrics

Objective perceptual audio quality metrics in pure Rust — no dependencies, no `unsafe`.

- **2f model** (Kastner & Herre, 2019): estimated mean MUSHRA score (0–100) from two Model Output
  Variables of an **ITU-R BS.1387 (PEAQ®) Basic Version compatible** FFT ear model.
- **ViSQOL v3 audio-mode compatible** MOS-LQO (1–5) for music and general audio.

**ViSQOL here is the audio (music) mode only. It is not intended or validated for speech
quality estimation**, and there is no speech mode (no voice activity detection, no speech
model).

These are independent implementations written from the published Recommendation and papers —
for ViSQOL chiefly the description in Chinen et al., 2020. They are **not affiliated with or
endorsed by ITU, OPTICOM, Google, or the authors of the reference tools**. PEAQ® is a
registered trademark of OPTICOM GmbH. "PEAQ" and "ViSQOL" are used only to say which published
method the code follows.

## Usage

```toml
[dependencies]
sori-metrics = "0.1"
```

```rust
// Interleaved f32 samples in [-1, 1], 48 kHz, time aligned.
let twof = sori_metrics::twof_score(&reference, &test, 48_000, channels)?;
println!("2f MUSHRA estimate {:.1}", twof.mushra); // also twof.adb, twof.avg_mod_diff1

let vq = sori_metrics::visqol_score(&reference, &test, 48_000, channels)?;
println!("MOS-LQO {:.3}", vq.moslqo); // also vq.vnsim, vq.fvnsim (32 bands), vq.patches
```

To compare several candidates with the same reference, analyse the reference once. The
results are identical to the one-shot functions (which use the same path); the reference half
of the work is done only once:

```rust
let r2f = sori_metrics::twof::Reference::new(&reference, 48_000, channels)?;
let rvq = sori_metrics::visqol::Reference::new(&reference, 48_000, channels)?;
for test in &candidates {
    println!("{:.1} {:.3}", r2f.score(test)?.mushra, rvq.score(test)?.moslqo);
}
```

Mono `f64` input: `visqol::Analyzer::audio().reference_mono(&r)?.score_mono(&d)?` (or
`Analyzer::score_mono(&r, &d)` for one pair).

### Blocks and windows

To find *where* a test signal is worst, both metrics also report per-block and per-window
values:

```rust
let d = sori_metrics::twof::detail(&reference, &test, 48_000, channels)?;
for b in d.blocks() {            // one ear-model frame per 1024 samples (21.3 ms)
    // b.start (s), b.valid, b.detection (binaural detection probability), b.distorted,
    // b.steps, b.mod_diff[c], b.weight[c]
}
for w in d.windows(3.0, 1.0) {   // 3 s windows, 1 s hop
    // w.distorted_share (distorted / valid blocks), w.score (2f of the window, Option)
}
let v = sori_metrics::visqol_score(&reference, &test, 48_000, channels)?;
for w in v.windows(3.0, 1.0) {   // patches (0.6 s) inside each window
    // w.patches, w.fvnsim, w.moslqo (SVR on the window's patches: an indication only)
}
```

A window's 2f applies the same averaging as the overall score to the blocks centred in it.
`d.score` equals `twof_score`.

### Reading from files or other storage

`twof::detail_source` and `visqol::score_source` take two `source::Source`s — anything that
can read interleaved `f32` at a position: `wav::WavReader` (a WAV file on disk),
`source::Interleaved` (a slice), or your own type (a decoder, a browser `Blob`). The inputs
are read 10 s at a time and never held whole; the values are the same as from slices.

```text
cargo run --release --example windows -- reference.wav test.wav --window 3 --hop 1 [--blocks]
```

Command-line examples (WAV in, PCM 16/24/32-bit or float 32/64):

```text
cargo run --release --example twof   -- reference.wav test.wav
cargo run --release --example visqol -- reference.wav test.wav
cargo run --release --example visqol -- --batch pairs.tsv     # reference<TAB>test per line
```

### Input requirements

| | 2f | ViSQOL |
|---|---|---|
| Sample rate | 48 kHz only (`Error::SampleRate` otherwise — resample first) | 48 kHz only |
| Channels | 1 or 2; analysed separately, `AvgModDiff1` averaged, `ADB` binaural (per-band maximum) | any; averaged to mono |
| Alignment | caller aligns; the shorter length is used | global lag found internally; level matched to the reference |
| Minimum length | one 2048-sample frame (useful results need ≥ 1 s) | 0.88 s (first patch at 0.28 s, 0.6 s long) |

## Accuracy

Agreement with the Python reference tools, measured on their **outputs** on the same 48 kHz
time-aligned WAV pairs. The items are 23 public music clips: ODAQ (10), vocadito (4, solo
singing) and OpenGameArt (9). (The development set also holds one speech clip; it is left out
here and from the bundled test data, since the ViSQOL part is not meant for speech.)

### 2f vs `seabass-2f` (92 pairs)

23 items × MP3 encodes at 64/96/128 kbit/s (an in-development encoder) and LAME 3.100 at
128 kbit/s CBR.

| | value |
|---|---|
| \|ΔMUSHRA\| median / 90 % / max | 0.011 / 0.040 / 0.531 |
| `ADB` relative error, median | 0.043 % |
| `AvgModDiff1` relative error, median / max | 0.081 % / 0.98 % |
| Pearson / Spearman | 0.999994 / 0.99998 |

A paired comparison of two settings of that encoder at 128 kbit/s (23 items, mean MUSHRA
difference [95 % CI]: Python +0.96 [+0.17, +1.75], Rust +0.95 [+0.16, +1.73]) reaches the same
conclusion.

### ViSQOL vs `visqol-python` 3.8.0 (368 pairs)

23 items × 16 decodes: the MP3 encoder above and LAME at 64/96/128 kbit/s, Opus VBR and CBR at
64/96/128 kbit/s, ffmpeg AAC at 64/96/128 kbit/s and AAC 128 kbit/s without PNS/IS.

| | this crate | `visqol-rs` 0.4.1 (for comparison) |
|---|---|---|
| \|ΔMOS\| median / 90 % / max | **0.0054 / 0.018 / 0.053** | 0.094 / 0.600 / 1.418 |
| mean difference | −0.006 | +0.189 |
| Pearson / Spearman | 0.99983 / 0.99974 | 0.883 / 0.907 |

Seven paired codec comparisons (23 items, mean MOS difference [95 % CI]) all keep their
conclusion:

| comparison | `visqol-python` | this crate |
|---|---|---|
| MP3 encoder − LAME, 64 kbit/s | −0.071 [−0.169, +0.027] | −0.071 [−0.171, +0.029] |
| MP3 encoder − LAME, 96 kbit/s | −0.176 [−0.269, −0.083] | −0.179 [−0.272, −0.086] |
| MP3 encoder − LAME, 128 kbit/s | −0.191 [−0.275, −0.107] | −0.194 [−0.278, −0.109] |
| MP3 encoder 128 − Opus 128 | −0.365 [−0.503, −0.227] | −0.368 [−0.507, −0.229] |
| MP3 encoder 128 − AAC 128 | −0.130 [−0.190, −0.070] | −0.136 [−0.197, −0.074] |
| AAC without PNS/IS − AAC, 128 | +0.093 [+0.012, +0.174] | +0.094 [+0.013, +0.175] |
| Opus VBR − CBR, 128 | +0.005 [−0.015, +0.026] | +0.004 [−0.017, +0.025] |

Stage by stage against `visqol-python`'s intermediate outputs (two decodes of each item plus
three synthetic pairs, shipped in `tests/data/visqol-intermediate/`): band centre frequencies
within 4e-12 Hz; the SVR stage (reference per-band similarities in, MOS out) within 3e-14;
per-band similarity `fvnsim` median |Δ| about 0.0003.

### Speed (one 3-minute stereo pair, 48 kHz, release build, one thread, 2-core cloud VM)

| | time |
|---|---|
| 2f, this crate | 2.7 s |
| 2f, `seabass-2f` (warm / first call) | 54 s / 151 s |
| ViSQOL, this crate | 7.9 s |
| ViSQOL, `visqol-python` (≈ 2 s per 10–15 s pair, scaled) | ≈ 26 s (estimate) |
| ViSQOL, `visqol-rs` 0.4.1 | 124 s |

Results are deterministic (same input → bit-identical output). On the 211 s pair of the
memory table below the current version takes 2.1 s (2f) and 6.8 s (ViSQOL).

### Memory

Both metrics read the caller's sample buffers in place, block by block (no deinterleaved, mono
or `f64` copies of the signals): the memory they add does not grow with the signal length,
except for ViSQOL's alignment transform (16 bytes per 16 input samples, rounded up to a power
of two: 32 MiB for 3.5 minutes). `wav::read` reads the data chunk piecewise, so it holds only
the returned samples.

Peak resident memory of the whole process, including the two input buffers (interleaved
`f32` stereo at 48 kHz, 0.37 MiB per second each), release build:

| signal length | inputs | 2f | ViSQOL |
|---|---|---|---|
| 30.5 s | 22 MiB | 26 MiB | 31 MiB |
| 89.4 s | 65 MiB | 69 MiB | 87 MiB |
| 211 s | 155 MiB | 158 MiB | 194 MiB |

(Version 0.1 of this crate: 107/71, 307/202 and 721/476 MiB.) A `twof::Reference` keeps
82 KB per second per channel of reference values; a `visqol::Reference` 13 KB per second.

Reading the inputs from a `Source` instead (`twof::detail_source`, `visqol::score_source`)
leaves only the metrics' own memory. 211 s pair, whole process, release build:

| | 2f | ViSQOL |
|---|---|---|
| native (`examples/windows.rs`, inputs read from disk) | 18 MiB | 53 MiB |
| wasm32, linear memory (inputs kept by the host) | 9 MiB | 45 MiB |
| wasm32, linear memory, slice API (inputs copied in) | 156 MiB | 192 MiB |

Reference reuse, one reference and three candidates, time relative to three one-shot calls
(median of 5, alternating in one process): 2f 62 / 71 / 74 %, ViSQOL 71 / 71 / 70 % for
30.5 / 89.4 / 211 s.

Measure on your own files with
`cargo run --release --example footprint -- <op> <reference> <test>...` (see its docs).

## Known differences and limitations

- **ViSQOL, pure tone plus noise**: for a reference that is a single sine and a test with added
  white noise, this crate gives 1.58 and `visqol-python` 3.38. The reference tool appears to
  apply a per-frame relative silence threshold there that could not be pinned down from its
  outputs. Music pairs are not affected (the 368-pair results above).
- **ViSQOL, slight negative bias**: −0.006 MOS on average, mostly from the lowest band (50 Hz)
  and bands 26–31 (≥ 13 kHz).
- **2f, onsets right after digital silence**: frames at the start of a signal that begins after
  digital silence differ slightly (largest effect on the 92 pairs: +0.53 MUSHRA).
- **2f, very short transients**: `AvgModDiff1` is 5–7 % lower on 6 ms clicks (synthetic);
  within 1 % on music.
- Only what these two metrics need is implemented: no other PEAQ® MOVs, no ODG, no ViSQOL
  speech mode, no resampling.
- ViSQOL fields `fvdegenergy`, `fstdnsim` and `fvnsim10` are not computed.

## Sources and provenance

Implemented from:

- ITU-R Recommendation BS.1387-2 (2023), *Method for objective measurements of perceived audio
  quality*, Annex 2 (Basic Version, FFT ear model).
- T. Kastner and J. Herre, "An efficient model for estimating subjective quality of separated
  audio source signals", IEEE WASPAA 2019 (doi:10.1109/WASPAA.2019.8937179), with the parameter
  set published by AudioLabs Erlangen.
- A. Hines, J. Skoglund, A. Kokaram, N. Harte, "ViSQOL: an objective speech quality model",
  EURASIP J. Audio, Speech, and Music Processing 2015:13 (doi:10.1186/s13636-015-0054-9).
- A. Hines, E. Gillen, D. Kelly, J. Skoglund, A. Kokaram, N. Harte, "ViSQOLAudio: An objective
  audio quality metric for low bitrate codecs", J. Acoust. Soc. Am. 137(6), EL449, 2015.
- M. Chinen, F. S. C. Lim, J. Skoglund, N. Gureev, F. O'Gorman, A. Hines, "ViSQOL v3: An Open
  Source Production Ready Objective Speech and Audio Metric", QoMEX 2020 (arXiv:2004.09584).
- M. Slaney, "An Efficient Implementation of the Patterson-Holdsworth Auditory Filter Bank",
  Apple Computer Technical Report #35, 1993.
- The libsvm model-file format and nu-SVR prediction as documented by libsvm.

Where these texts leave a choice open, the choice was fixed by comparing with the **outputs**
of `seabass-2f` and `visqol-python` (3.8.0) on test signals; each such choice is marked in the
code (`INTERPRETATION` comments for 2f; the module and constant docs for ViSQOL). The source
code of `seabass-2f`, `visqol-python`, Google's ViSQOL (C++), GstPEAQ and peaqb was **not**
read or used.

Transparency note: while evaluating the `visqol-rs` crate (0.4.1, Apache-2.0) as a possible
dependency, before this implementation started, parts of its source were read (configuration
constants, the choice of the gammatone upper frequency, mono loading, result fields). It was
not adopted, nothing was copied from it, and every constant here is justified by the papers,
the shipped model data, or the reference tool's outputs (for example the 0.6 s patch length
comes from those outputs and differs from the value `visqol-rs` uses).

The ViSQOL SVR model file (`data/visqol/libsvm_nu_svr_model.txt`) is Google's audio-mode model,
included unchanged as data under the Apache License 2.0 — see [NOTICE](NOTICE).

## License

Code: MIT OR Apache-2.0, at your option. The bundled SVR model file: Apache-2.0 (Google LLC),
see [NOTICE](NOTICE).

MSRV: Rust 1.85.

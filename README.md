# sori-metrics

Objective perceptual audio quality metrics in pure Rust — no dependencies, no `unsafe`.

- **2f model** (Kastner & Herre, 2019): estimated mean MUSHRA score (0–100) from two Model Output
  Variables of an **ITU-R BS.1387 (PEAQ®) Basic Version compatible** FFT ear model.
- **ViSQOL v3 audio-mode compatible** MOS-LQO (1–5) for music and general audio.
- **Total NMR** (`Total NMR_B`, dB): noise-to-mask ratio of the same ear model, reported with
  2f (`twof.total_nmr`) from the same pass.
- **Source separation (music)**: the BSS Eval image criteria SDR, ISR, SIR and SAR per window
  with track medians, in the version-4 form used by the SiSEC 2018 evaluation (time-invariant
  512-tap distortion filters, 1 s windows), and SI-SDR / SI-SIR / SI-SAR (`separation`). For
  evaluating music source separation; not validated for speech separation.

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
sori-metrics = "0.3"
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

### Source separation

Pass the reference and estimated image of every source (interleaved, same length and channel
count, any sample rate). For a target against its accompaniment, pass two sources: the target
and the sum of the others.

```rust
use sori_metrics::separation::{bss_eval, scale_invariant, Options};

let refs = [&vocals[..], &drums[..], &bass[..], &other[..]];
let ests = [&vocals_est[..], &drums_est[..], &bass_est[..], &other_est[..]];
let scores = bss_eval(&refs, &ests, 2, &Options::new(44_100))?;
for (j, s) in scores.iter().enumerate() {
    let m = s.median(); // track value: median over the scored 1 s windows
    println!("source {j}: SDR {:.2} ISR {:.2} SIR {:.2} SAR {:.2}", m.sdr, m.isr, m.sir, m.sar);
    // s.frames: per-window values (NaN where a reference source is silent)
}
let si = scale_invariant(&refs, &ests, 2, true)?; // zero-mean SI-SDR, per channel and mean
println!("SI-SDR of source 0: {:.2} dB", si[0].mean.si_sdr);
```

`cargo run --release --example separation -- REF1 EST1 REF2 EST2 ...` prints the same for WAV
files.

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

### Total NMR vs the reference Python measurement (322 pairs)

`Total NMR_B` reproduces a Python implementation of BS.1387 Annex 2 written for the
measurement harness this crate grew out of (same constants, including three documented
simplifications worth ≪ 0.1 dB each — see the `peaq` module docs). On 23 items × 14 encoder
settings (MP3 CBR 96–192 and VBR V0/V2/V5 of the in-development encoder, LAME 3.100 at 128 and
192 kbit/s CBR and V0/V2/V5; 320 distinct decodes): |Δ| at most 1.8e-14 dB. Eight paired
comparisons give the same mean differences and confidence intervals to three decimals.
On a 211 s stereo pair the Python tool takes 8.3 s and 3.5 GB; here Total NMR adds about 25 %
to the 2f time (3.9 s for both) and no memory.

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

### Source separation vs `museval` 0.4.1 and `pb_bss_eval` 0.0.2 (60 estimates)

Ten 8.5 s stereo mixtures at 44.1 kHz built from the public clips (vocals, drums, bass, other;
`scripts/separation-data.py`), with estimates made by filtering, delay, leakage of the other
sources, noise and clipping; scored as four sources and as vocals against accompaniment
(40 + 20 estimates, 432 windows). The reference outputs ship in
`tests/data/separation-public.tsv`.

| | max \|Δ\| |
|---|---|
| BSS Eval v4 track medians (SDR, ISR, SIR, SAR) | 2.0e-9 dB |
| BSS Eval v4, every window and criterion (NaN pattern identical) | 2.9e-9 dB |
| SI-SDR (zero-mean) | 2.7e-8 dB, after the reference tool's `+1e-8` floor (see below) |

On a 3.5-minute 44.1 kHz stereo song with four sources (release build, one thread, 2-core cloud
VM): BSS Eval 10.5–15 s, SI-SDR 1.4 s; the process peak is 640 MiB, of which 571 MiB are the
eight input signals, so the metric adds 69 MiB (mostly the 4096 × 4096 normal equations, kept
as a packed triangle). In wasm32 (inputs copied into linear memory) 21 s and 639 MiB of linear
memory. `museval` took 40 s and 2.0 GiB for the first minute of that song and was killed for
lack of memory (8 GiB machine) on the whole song.

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
- Only what these metrics need is implemented: of the PEAQ® Basic MOVs only `AvgModDiff1B`,
  `ADBB` and `Total NMR_B`; no ODG, no ViSQOL speech mode, no resampling.
- ViSQOL fields `fvdegenergy`, `fstdnsim` and `fvnsim10` are not computed.
- **Source separation**: only the image criteria in the v4 form (time-invariant filters); no
  framewise-filter (v3) or "sources" variant, no permutation search, and only full windows are
  scored. A window in which any reference source is digitally silent is not scored, for every
  source. When the normal equations are singular (a reference channel silent for the whole
  track, or two exactly proportional channels such as a mono source copied to both channels),
  a tiny diagonal load is added and the scores are not comparable with other implementations;
  add a little independent noise to such references, or score the channels separately.
- **SI-SDR** follows the definition exactly. `pb_bss_eval` adds `1e-8` to the energy ratio
  (and to `‖s‖²`), which matters only below about −50 dB (0.01 dB at −53.8 dB); it always
  removes the mean, which corresponds to `zero_mean = true` here.

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
- E. Vincent, R. Gribonval, C. Févotte, "Performance measurement in blind audio source
  separation", IEEE Trans. Audio, Speech, and Language Processing 14(4), 1462–1469, 2006.
- E. Vincent, H. Sawada, P. Bofill, S. Makino, J. P. Rosca, "First stereo audio source
  separation evaluation campaign: data, algorithms and results", ICA 2007, LNCS 4666 (image
  criteria ISR, SIR, SAR).
- F.-R. Stöter, A. Liutkus, N. Ito, "The 2018 Signal Separation Evaluation Campaign", LVA/ICA
  2018, LNCS 10891 (BSS Eval v4: time-invariant filters, 1 s windows, median per track).
- J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR – half-baked or well done?", ICASSP
  2019 (SI-SDR, SI-SIR, SI-SAR).

Where these texts leave a choice open, the choice was fixed by comparing with the **outputs**
of `seabass-2f` and `visqol-python` (3.8.0) on test signals; each such choice is marked in the
code (`INTERPRETATION` comments for 2f; the module and constant docs for ViSQOL). The source
code of `seabass-2f`, `visqol-python`, Google's ViSQOL (C++), GstPEAQ and peaqb was **not**
read or used. For source separation, the window framing, zero padding and silent-window rule
that the papers leave open were fixed from the outputs of `museval` (run as a black box); the
code of `museval`, `fast_bss_eval`, `mir_eval` and `pb_bss_eval` was not read.

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

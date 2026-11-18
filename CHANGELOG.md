# Changelog

## 0.3.0 (2026-11-18)

Additive release: no existing API changed, scores of 2f, ViSQOL and Total NMR are identical to
0.2.0. MSRV stays 1.85; still no dependencies and no `unsafe`.

New public API: module `separation` (`bss_eval`, `scale_invariant`, `Options`, `Frame`,
`Medians`, `SourceScores`, `ScaleInvariant`, `SiScores`) and `Error::Mismatch`.

- **Source separation** (new module `separation`): BSS Eval image criteria SDR, ISR, SIR and
  SAR per window, in the version-4 form of the SiSEC 2018 evaluation (distortion filters fitted
  once per track, 512 taps, 1 s windows), with track medians (`SourceScores::median`); and
  SI-SDR, SI-SIR and SI-SAR per channel and averaged, optionally zero-mean
  (`separation::scale_invariant`). For music source separation. Agrees with the outputs of
  `museval` 0.4.1 to 3e-9 dB per window and `pb_bss_eval` 0.0.2 to 3e-8 dB on 60 estimates
  (`tests/data/separation-public.tsv`). A 3.5-minute four-source stereo song takes about 11 s
  and 69 MiB beyond the inputs.
- New error variant `Error::Mismatch` (inputs that do not fit together); `Error` is
  `#[non_exhaustive]`, so this is additive.
- Example `separation` scores WAV files; `scripts/separation-data.py` rebuilds the validation
  mixtures and reference outputs.
- SI-SDR follows the definition of Le Roux et al. exactly; `pb_bss_eval` adds `1e-8` to the
  energy ratio, which differs by up to 0.01 dB only below about −50 dB.
- `CITATION.cff` for citing the crate (GitHub, Zenodo).
- Newer Clippy lints fixed (`manual_midpoint`, `needless_range_loop`); no behaviour change.

## 0.2.0 (2026-11-13)

- **Total NMR** (`Total NMR_B`, dB) computed in the 2f pass: new field `total_nmr` on
  `twof::TwoF` and `peaq::Movs` (both `#[non_exhaustive]`, so this is additive). It reproduces
  the reference Python measurement it replaces to 1e-14 dB on 322 music pairs; see the
  `peaq` module docs for its three documented simplifications. About 25 % more 2f time, no
  extra memory. 2f, `ADB` and `AvgModDiff1` values are unchanged.
- Examples `twof` and `windows` print `total_nmr` as a new last column.

## 0.1.0

First public release: 2f (`AvgModDiff1B`, `ADBB` and the Kastner & Herre mapping), ViSQOL v3
audio-mode compatible MOS-LQO, `Reference` objects for scoring many candidates against one
reference, per-block and per-window outputs, and `source::Source` piece-by-piece reading.

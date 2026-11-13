# Changelog

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

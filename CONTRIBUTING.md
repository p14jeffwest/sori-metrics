# Contributing to sori-metrics

Thank you for your interest. Bug reports, questions, validation results on your own material and
pull requests are welcome.

## Reporting a problem or asking a question

Open an issue at <https://github.com/p14jeffwest/sori-metrics/issues>. For a wrong or surprising
score, please include:

- the crate version and how you called it (function, sample rate, channel count);
- the score you got and, if you have it, the score of the reference tool (`visqol-python`,
  `seabass-2f`, or another PEAQ implementation) on the same 48 kHz, time-aligned pair;
- if you can share it, the audio pair or a short excerpt that shows the difference.

Questions about using the library are welcome as issues too.

## Pull requests

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Keep the crate free of dependencies and `unsafe` code, and keep its MSRV (see `Cargo.toml`).
3. Run before submitting:

   ```sh
   cargo test
   cargo clippy --all-targets -- -D warnings
   cargo fmt --check
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
   python3 scripts/check-test-data.py --no-speech
   ```

4. A change that alters any score must say why, and show the effect on the bundled reference
   data (the tests in `tests/` compare with the reference tools' outputs).

## Rules that protect the implementation's provenance

This crate is an independent implementation written from the published Recommendation and
papers. To keep it that way:

- do not copy or translate code from other implementations of PEAQ, 2f or ViSQOL (including
  Google's ViSQOL, `visqol-python`, `visqol-rs`, `seabass-2f`, GstPEAQ, PEAQb); cite the
  published text you worked from instead;
- where a text leaves a choice open, settle it by comparing **outputs** with a reference tool,
  and mark the choice in the code;
- test data may hold only public items and plain numbers; no speech material (the ViSQOL part
  implements the audio/music mode only).

## How changes reach a release

[AUTHOR TO CONFIRM — depends on the development-history decision.] Pull requests are reviewed
and merged here. Releases are tagged in this repository and published to crates.io by the
maintainer; `CHANGELOG.md` lists the changes of each release.

## Conduct

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

By contributing you agree that your contribution is licensed under the crate's licence
(MIT OR Apache-2.0, at the user's option).

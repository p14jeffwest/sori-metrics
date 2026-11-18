---
title: 'sori-metrics: dependency-free Rust implementations of the 2f model, ViSQOL v3 (audio mode) and Total NMR for objective music quality measurement'
tags:
  - Rust
  - audio
  - audio quality
  - perceptual metrics
  - codec evaluation
  - WebAssembly
authors:
  - name: Jeff
    orcid: 0000-0000-0000-0000 # [ORCID]
    affiliation: 1
affiliations:
  - name: Independent researcher, Republic of Korea # [AUTHOR TO CONFIRM]
    index: 1
date: 17 November 2026
bibliography: paper.bib
---

# Summary

When an audio codec shrinks a music file, it removes information that listeners are not
supposed to notice. Whether they notice is ultimately decided by listening tests, which are
slow and expensive, so codec developers also use *objective* metrics: programs that compare a
coded signal with its original through a model of human hearing and predict how a listening
panel would rate it. `sori-metrics` is a Rust library that computes three such metrics for
music: the 2f model, which predicts a MUSHRA listening-test score from two outputs of the
ITU-R BS.1387 (PEAQ®) ear model [@BS1387; @Kastner2019]; a ViSQOL v3 audio-mode compatible
MOS-LQO [@Hines2015ViSQOL; @Hines2015Audio; @Chinen2020]; and the BS.1387 noise-to-mask ratio
`Total NMR`. It has no dependencies and no `unsafe` code, reads its inputs piece by piece so
that memory use does not grow with the signal length, and compiles to WebAssembly. Its results
agree with the established Python reference tools to within a few thousandths of a score point
on several hundred coded music items.

# Statement of need

Objective quality metrics sit inside the development loop of audio codecs: every encoder
change is scored on a test set, and the paired differences decide what is kept. That loop
needs metrics that are (i) faithful to the published methods, because conclusions are drawn
from differences of a few tenths of a point; (ii) fast enough to score hundreds of items per
change; and (iii) embeddable in the encoder's own toolchain, including a browser, where a codec
may run as WebAssembly and pick among candidate encodings on the user's machine.

The reference implementations are Python or C++ programs that run as separate tools. In Rust,
no implementation of the PEAQ ear model or the 2f model existed, and the one ViSQOL port we
found, `visqol-rs` [@visqolrs], disagrees with the Python reference `visqol-python`
[@visqolpython] by a median of 0.094 and up to 1.42 MOS on our test set (Pearson 0.883), which
is larger than the codec differences it would be used to resolve. `sori-metrics` targets
researchers and engineers who evaluate music codecs, enhancement or restoration systems and
want these metrics as a library: in a Rust or WebAssembly code base, inside an encoder's
decision loop, or as a fast replacement for an external tool in a test harness.

The library implements music (audio-mode) quality only. ViSQOL's speech mode is not
implemented, and the metrics are not validated for speech.

# State of the field

PEAQ (ITU-R BS.1387) has open implementations accompanying the analysis of @Kabal2002 and as
a GStreamer plug-in (GstPEAQ, @Holters2015); for the 2f model of @Kastner2019
a Python implementation is available (`seabass-2f`, @seabass2f). ViSQOL v3 is maintained by
Google as C++ [@Chinen2020] with a Python port (`visqol-python`, @visqolpython). None of these
is a library for Rust or WebAssembly, and wrapping them would bring a C/C++ or Python runtime
into every consumer. Adopting `visqol-rs` was considered first: on the pairs described below
its scores were systematically compressed (one of seven codec comparisons changed its
conclusion) and it ran about 16 times slower than the implementation presented here. Rather
than build on it, we wrote an independent implementation from the published descriptions,
validated stage by stage, which also gave us the PEAQ ear model, 2f and Total NMR in one code
base with one validation set.

# Software design

`sori-metrics` is organised around the two ear models. The `peaq` module implements the FFT
ear model of BS.1387 Annex 2 (Basic Version) up to the model output variables that 2f needs
(`AvgModDiff1`, `ADB`) and `Total NMR`; `twof` maps the first two to a MUSHRA estimate with the
published regression [@Kastner2019]. The `visqol` module implements the v3 audio mode: global
alignment by cross-correlation, level matching, a 32-band gammatone filter bank
[@Slaney1993], 0.6 s spectrogram patches compared with the neurogram similarity index (NSIM),
and the published support-vector regression model, which is bundled unchanged under its
Apache-2.0 licence [@Chinen2020].

Three design decisions shape the library.

*Fidelity is defined by outputs.* Where the texts leave a choice open (frame boundaries, edge
handling of the 3×3 similarity window, silence floors), the choice was fixed by comparing with
the reference tools' outputs on test signals, not by reading their source code (parts of
`visqol-rs` were read while evaluating it as a dependency; the README records this); every
such choice is marked in the code. The bundled test data hold reference outputs and intermediate
values per stage, so each stage is tested on its own.

*Memory does not grow with the signal.* Both ear models are frame-by-frame filters. The
library reads its inputs through a `Source` trait in 10 s pieces, so a 3.5-minute stereo pair
is scored in 18 MiB (2f) and 53 MiB (ViSQOL) natively and 9/45 MiB of WebAssembly linear
memory, compared with 721/476 MiB in our first version. The only length-dependent buffer is
ViSQOL's alignment transform, computed in place without tables.

*One reference, many candidates.* An encoder that tries several settings compares each with the
same original. `Reference` objects analyse the original once; one-shot functions use the same
path, so results are identical either way. With three candidates the total time drops to
62–74 % of three separate calls. Per-block and per-window outputs (`Detail::windows`) locate
where in a track a candidate is worst, which matters for music whose coding errors are local.

The trade-off of these choices is scope: only the outputs these metrics need are computed, and
only 48 kHz input is accepted (resampling is left to the caller).

# Validation

All comparisons use 23 public music clips (ODAQ, vocadito, OpenGameArt) coded by MP3, AAC and
Opus encoders at 64–192 kbit/s, aligned at 48 kHz.

- **2f** against `seabass-2f` on 92 pairs: median |ΔMUSHRA| 0.011, maximum 0.531, Pearson
  0.999994.
- **ViSQOL** against `visqol-python` 3.8.0 on 368 pairs: median |ΔMOS| 0.0054, 90th percentile
  0.018, maximum 0.053, Pearson 0.99983. Seven paired codec comparisons (mean difference and
  95 % confidence interval over the 23 items) reach the same conclusion as with the reference.
- **Total NMR** against a Python implementation of BS.1387 Annex 2 on 322 pairs (320 distinct
  decodes): maximum difference 1.8 × 10⁻¹⁴ dB.

Results are deterministic. On a 2-core cloud machine a 3-minute stereo pair takes 2.7 s for 2f
(`seabass-2f`: 54 s) and 7.9 s for ViSQOL (`visqol-python`: about 26 s; `visqol-rs`: 124 s).
Known differences are documented: a pure sine plus white noise scores 1.58 here and 3.38 in
`visqol-python`, and ViSQOL scores are 0.006 lower on average.

# Research impact statement

`sori-metrics` is the measurement back end of an ongoing MP3 encoder research project by the
author, where it replaced the external Python tools in the test harness and has scored [N —
AUTHOR TO FILL] encoder comparisons; its per-window outputs drive a listening-test item
selection and a quality-guaranteed encoding mode that picks among candidate encodings with
these metrics. Release 0.3.0 is published on crates.io. The validation tables above, with
their bundled reference data, also serve as a reproducible benchmark of 2f and ViSQOL
implementations.

# AI usage disclosure

[AUTHOR TO CONFIRM AND COMPLETE.] Generative AI was used extensively. The code, tests,
documentation and the first draft of this paper were written by Anthropic's Claude models
(Claude Opus, through Claude Code) working as agents on tasks specified by the author in a
project repository with recorded decisions. The author defined the goals and constraints
(independent implementation from published descriptions, no reading of reference source code,
validation against reference outputs, music-only scope), chose among the alternatives at each
decision point, reviewed the results and validation data, and is responsible for the content.
[Describe the author's own review and edits of code and text here.]

# Acknowledgements

The ViSQOL support-vector regression model file is Google LLC's audio-mode model, included
unchanged under the Apache License 2.0. The test items come from ODAQ, vocadito and
OpenGameArt under their respective licences. PEAQ® is a registered trademark of OPTICOM GmbH;
this work is not affiliated with or endorsed by OPTICOM, ITU, Google, or the authors of the
reference tools. [Funding: none — AUTHOR TO CONFIRM.] [Conflicts of interest: none — AUTHOR
TO CONFIRM.]

# References

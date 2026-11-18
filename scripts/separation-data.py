#!/usr/bin/env python3
"""Build the public source-separation validation set and its reference outputs.

Ten synthetic 8.5 s, 44.1 kHz stereo mixtures (syn-mix00..09) of public clips (ODAQ, vocadito,
OpenGameArt; the 48 kHz originals resampled), four sources each: vocals, drums, bass (clips
low-passed at 250 Hz) and other. A -50 dB independent floor per channel keeps mono clips made
stereo from having exactly collinear channels; every third mixture has silent vocals from 3 s
to 5 s. The "estimates" are the sources filtered by a random 6-tap FIR, delayed by 0-39
samples, with 2-30 % of every other source leaked in, noise 10-30 dB down and, in every fourth
mixture, clipping at +-0.15. Seeded, so the set is reproducible.

Writes <out>/syn-mixNN/{ref,est}-{vocals,drums,bass,other,accompaniment}.wav (32-bit float;
accompaniment = sum of the other three) and prints the reference outputs in the format of
tests/data/separation-public.tsv: museval evaluate(win=hop=44100, v4) and pb_bss_eval si_sdr,
both run as black boxes (pip packages museval, pb_bss_eval, soundfile, scipy).

usage: python3 scripts/separation-data.py <clips_dir> <out_dir> > separation-public.tsv
       (<clips_dir> holds <item>.wav for the public items named below)
"""
import math
import os
import sys

import numpy as np
import scipy.signal as ss
import soundfile as sf

FS, T = 44100, int(8.5 * 44100)
VOCALS = ["vocal-solo-english", "vocal-solo-tagalog-a", "vocal-solo-tagalog-b", "vocal-solo-vocalise", "vocal-choral"]
DRUMS = ["attack-castanets", "attack-claves", "preecho-glockenspiel", "electronic-retrowave", "rock-fight-them"]
BASS = ["piano-regret", "rock-metal-boss-battle", "electronic-slipstream", "rock-theme-song", "piano-vampire"]
OTHER = ["acoustic-guitar", "classical-violin", "brass-trumpet", "jazz-ensemble", "strings-violin-solo",
         "electronic-synth-dialogue", "rock-metal-better-future"]
NAMES = ["vocals", "drums", "bass", "other"]


def load(clips, name):
    x, fs = sf.read(f"{clips}/{name}.wav", dtype="float64", always_2d=True)
    if fs != FS:
        x = ss.resample_poly(x, 147, 160, axis=0)  # 48 kHz -> 44.1 kHz
    if x.shape[1] == 1:
        x = np.hstack([x * 0.8, x * 0.6])
    if len(x) < T:
        x = np.vstack([x] * (T // len(x) + 1))
    return x[:T]


def build(clips, out):
    rng = np.random.default_rng(11)
    blp = ss.butter(4, 250 / (FS / 2), output="sos")
    for k in range(10):
        v = load(clips, VOCALS[k % 5])
        d = load(clips, DRUMS[k % 5])
        b = ss.sosfilt(blp, load(clips, BASS[(k + 2) % 5]), axis=0) * 2.0
        o = load(clips, OTHER[k % 7])
        for x in (v, d, b, o):
            x += rng.normal(0, 1, x.shape) * np.sqrt(np.mean(x ** 2, axis=0)) * 10 ** (-50 / 20)
        if k % 3 == 0:
            v[3 * FS:5 * FS] = 0.0
        S = np.stack([v, d, b, o]) * 0.5
        E = np.empty_like(S)
        for j in range(4):
            h = rng.normal(0, 1, 6) * 0.15
            h[0] += 1.0
            delay = int(rng.integers(0, 40))
            y = ss.lfilter(h, [1.0], S[j], axis=0)
            y = np.vstack([np.zeros((delay, 2)), y[:T - delay]])
            leak = sum(rng.uniform(0.02, 0.3) * S[m] for m in range(4) if m != j)
            noise = rng.normal(0, 1, S[j].shape) * np.sqrt(np.mean(S[j] ** 2)) * 10 ** (-rng.uniform(10, 30) / 20)
            e = y + leak + noise
            if k % 4 == 1:
                e = np.clip(e, -0.15, 0.15)
            E[j] = e
        dk = f"{out}/syn-mix{k:02d}"
        os.makedirs(dk, exist_ok=True)
        for j, n in enumerate(NAMES):
            sf.write(f"{dk}/ref-{n}.wav", S[j].astype(np.float32), FS, subtype="FLOAT")
            sf.write(f"{dk}/est-{n}.wav", E[j].astype(np.float32), FS, subtype="FLOAT")
        for kind in ("ref", "est"):
            acc = sum(sf.read(f"{dk}/{kind}-{n}.wav", dtype="float64")[0] for n in NAMES[1:])
            sf.write(f"{dk}/{kind}-accompaniment.wav", acc.astype(np.float32), FS, subtype="FLOAT")


def fmt(v):
    return "nan" if math.isnan(v) else repr(float(v))


def reference_outputs(out):
    import museval
    from pb_bss_eval.evaluation import si_sdr
    for k in range(10):
        dk = f"{out}/syn-mix{k:02d}"
        for setup, srcs in (("four", NAMES), ("voc", ["vocals", "accompaniment"])):
            rd = lambda kind, n: sf.read(f"{dk}/{kind}-{n}.wav", dtype="float64")[0]
            R = np.stack([rd("ref", n) for n in srcs])
            X = np.stack([rd("est", n) for n in srcs])
            sdr, isr, sir, sar = museval.evaluate(R, X, win=44100, hop=44100)
            for j in range(len(srcs)):
                for w in range(sdr.shape[1]):
                    vals = [fmt(m[j, w]) for m in (sdr, isr, sir, sar)]
                    print("\t".join([f"syn-mix{k:02d}", setup, "bss", str(j), str(w)] + vals))
                for c in range(2):
                    print("\t".join([f"syn-mix{k:02d}", setup, "si", str(j), str(c),
                                     fmt(si_sdr(R[j][:, c], X[j][:, c]))]))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2])
    reference_outputs(sys.argv[2])

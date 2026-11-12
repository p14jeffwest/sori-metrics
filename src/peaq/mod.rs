//! ITU-R BS.1387 (PEAQ®) Basic Version: the FFT-based ear model and the two Model Output
//! Variables the 2f model needs, `AvgModDiff1B` and `ADBB`.
//!
//! Every step cites the Recommendation (BS.1387-2, Annex 2) by section and equation number.
//! Written from the Recommendation text only; places where the text is
//! ambiguous are marked `INTERPRETATION` with the reading that matched the reference
//! outputs of the Python 2f tool (outputs only).
//!
//! Scope: only what `AvgModDiff1B` and `ADBB` need. Masking, loudness, the error signal and
//! the other nine MOVs are not computed.
//!
//! PEAQ® is a registered trademark of OPTICOM GmbH. This is an independent implementation of
//! the published Recommendation, not affiliated with or endorsed by OPTICOM or ITU.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // frame/bin counts ≪ 2^52

mod bands;

use crate::fft::{Complex, Fft};
use crate::source::{Source, Window};
use crate::Error;
pub use bands::Z;
use bands::{FC, FL, FU};

/// Sampling rate the model is defined for (Annex 2 §2.1.1).
pub const SAMPLE_RATE: u32 = 48_000;
/// Frame length, samples (§2.1.2).
pub(crate) const N: usize = 2048;
/// Frame advance `StepSize`, samples (§2.1.2, 50 % overlap).
pub(crate) const STEP: usize = 1024;
/// Pitch resolution of the Basic Version, Bark (§2.1.5).
const RES: f64 = 0.25;
/// Assumed playback level of a full-scale sine, dB SPL (§2.1.3, recommended value).
const LP: f64 = 92.0;
/// Delayed averaging (§5.2.4.1): frames of the first 0.5 s are left out of `AvgModDiff1`.
/// INTERPRETATION: counted from the first frame of the file; 0.5 s = 23.4 frames of 1024
/// samples, so frames 0..=23 are dropped.
const DELAY: usize = 24;
/// Input samples are scaled to the 16-bit integer range before analysis: the data-boundary
/// threshold of §5.2.4.4 (200 over five samples) is stated on that scale.
/// INTERPRETATION: full scale = 32768.
const INT16_SCALE: f64 = 32768.0;

/// The two MOVs of one comparison (channel averaged as in §5.3 for `AvgModDiff1`;
/// `ADB` uses the binaural detection probability of §4.7, eqs 79–80).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Movs {
    /// `AvgModDiff1B` (§4.2.3).
    pub avg_mod_diff1: f64,
    /// `ADBB` (§4.7.2).
    pub adb: f64,
}

/// Frequency-dependent constants of the FFT ear model, computed once.
struct Model {
    fft: Fft,
    /// Hann window with the `sqrt(8/3)` gain of eq 2.
    window: Vec<f64>,
    /// Outer/middle ear amplitude weight `10^(W[k]/20)` (eqs 7–9) times the level scaling
    /// `fac` of eqs 5–6 and the 1/N of eq 4, per FFT line `0..=N/2`.
    gain: Vec<f64>,
    /// Pitch mapping (§2.1.5.1): for each band, `(first line, weights)`.
    groups: Vec<(usize, Vec<f64>)>,
    /// Internal noise `PThres[k]` (eq 13).
    p_thres: [f64; Z],
    /// Lower spreading slope as an energy factor per band step, `10^(-res·27/10)` (eq 16).
    low_step: f64,
    /// Upper slope frequency part `-24 - 230 Hz / fc[k]` (eq 15, without the level term).
    up_base: [f64; Z],
    /// `NormSP[k]` (eqs 19–20).
    norm_sp: [f64; Z],
    /// Time-spreading filter coefficient per band (eqs 21, 24).
    a_time: [f64; Z],
    /// Modulation filter coefficient per band (eqs 54–56).
    a_mod: [f64; Z],
    /// `levWt · EThres[k]^0.3` for `TempWt` (eqs 60, 65; levWt = 100, Table 10).
    lev_thres: [f64; Z],
}

impl Model {
    fn new() -> Self {
        let fft = Fft::new(N);
        let window: Vec<f64> = (0..N)
            .map(|k| {
                let c = (2.0 * std::f64::consts::PI * k as f64 / (N - 1) as f64).cos();
                0.5 * (8.0f64 / 3.0).sqrt() * (1.0 - c)
            })
            .collect();
        let fres = f64::from(SAMPLE_RATE) / N as f64; // 23.4375 Hz (eq 8)
        let norm = level_norm(&fft, &window);
        let fac = 10f64.powf(LP / 20.0) / norm;
        let gain = (0..=N / 2)
            .map(|k| {
                if k == 0 {
                    return 0.0; // W → -∞ dB at 0 Hz (eq 7)
                }
                let f = k as f64 * fres / 1000.0; // kHz
                let w_db = -0.6 * 3.64 * f.powf(-0.8) + 6.5 * (-0.6 * (f - 3.3).powi(2)).exp()
                    - 1e-3 * f.powf(3.6);
                fac * 10f64.powf(w_db / 20.0) / N as f64
            })
            .collect();
        // §2.1.5.1 pseudocode, lines 0..1024.
        let groups = (0..Z)
            .map(|i| {
                let (fl, fu) = (FL[i], FU[i]);
                let mut first = None;
                let mut w = Vec::new();
                for k in 0..N / 2 {
                    let lo = (k as f64 - 0.5) * fres;
                    let hi = (k as f64 + 0.5) * fres;
                    let v = if lo >= fl && hi <= fu {
                        1.0
                    } else if lo < fl && hi > fu {
                        (fu - fl) / fres
                    } else if lo < fl && hi > fl {
                        (hi - fl) / fres
                    } else if lo < fu && hi > fu {
                        (fu - lo) / fres
                    } else {
                        0.0
                    };
                    if v > 0.0 {
                        let f = *first.get_or_insert(k);
                        w.resize(k - f, 0.0);
                        w.push(v);
                    }
                }
                (first.unwrap_or(0), w)
            })
            .collect();
        let p_thres =
            core::array::from_fn(|k| 10f64.powf(0.4 * 0.364 * (FC[k] / 1000.0).powf(-0.8)));
        let low_step = 10f64.powf(-RES * 27.0 / 10.0);
        let up_base = core::array::from_fn(|k| -24.0 - 230.0 / FC[k]);
        let mut norm_sp = [0.0; Z];
        {
            // eqs 19–20: the spreading of a unit energy in every band with the level term 0.
            let ones = [1.0; Z];
            let mut out = [0.0; Z];
            spread_raw(&ones, &[0.0; Z], low_step, &up_base, &mut out);
            norm_sp.copy_from_slice(&out);
        }
        let dt = STEP as f64 / f64::from(SAMPLE_RATE);
        let a_time = core::array::from_fn(|k| {
            let tau = 0.008 + 100.0 / FC[k] * (0.030 - 0.008); // eq 21
            (-dt / tau).exp() // eq 24
        });
        let a_mod = core::array::from_fn(|k| {
            let tau = 0.008 + 100.0 / FC[k] * (0.050 - 0.008); // eq 56
            (-dt / tau).exp()
        });
        let lev_thres = core::array::from_fn(|k| {
            let e_thres = 10f64.powf(0.364 * (FC[k] / 1000.0).powf(-0.8)); // eq 60
            100.0 * e_thres.powf(0.3)
        });
        Self {
            fft,
            window,
            gain,
            groups,
            p_thres,
            low_step,
            up_base,
            norm_sp,
            a_time,
            a_mod,
            lev_thres,
        }
    }
}

/// `Norm` of eqs 5–6: the largest |F| (eq 4, including the 1/N) of a 0 dBFS 1019.5 Hz sine
/// over 10 frames.
fn level_norm(fft: &Fft, window: &[f64]) -> f64 {
    let mut best = 0.0f64;
    let mut buf = vec![(0.0, 0.0); N];
    for frame in 0..10 {
        for (t, b) in buf.iter_mut().enumerate() {
            let i = (frame * STEP + t) as f64;
            let x = INT16_SCALE
                * (2.0 * std::f64::consts::PI * 1019.5 * i / f64::from(SAMPLE_RATE)).sin();
            *b = (x * window[t], 0.0);
        }
        fft.forward(&mut buf);
        for &(re, im) in &buf[..=N / 2] {
            best = best.max((re * re + im * im).sqrt() / N as f64);
        }
    }
    best
}

/// Level-dependent spreading (eqs 15–18) without the `NormSP` division: returns
/// `(Σ_j Eline[j,k]^0.4)^(1/0.4)` in `out`. `energy[j]` is `10^(L[j]/10)` (or 1 for eq 20),
/// `level_db[j]` the level `L[j]` that sets the upper slope (0 for eq 20).
fn spread_raw(
    energy: &[f64; Z],
    level_db: &[f64; Z],
    low_step: f64,
    up_base: &[f64; Z],
    out: &mut [f64; Z],
) {
    let mut acc = [0.0f64; Z];
    for j in 0..Z {
        let su = up_base[j] + 0.2 * level_db[j]; // eq 15, dB/Bark (negative)
        let up_step = 10f64.powf(RES * su / 10.0);
        // Normalising sum of eq 18: Σ_{μ<j} low^(j-μ) + Σ_{μ≥j} up^(μ-j).
        let mut denom = 0.0;
        let mut p = low_step;
        for _ in 0..j {
            denom += p;
            p *= low_step;
        }
        let mut p = 1.0;
        for _ in j..Z {
            denom += p;
            p *= up_step;
        }
        let base = (energy[j] / denom).powf(0.4);
        let (lo4, up4) = (low_step.powf(0.4), up_step.powf(0.4));
        // k < j: lower slope; k ≥ j: upper slope.
        let mut p = base * lo4;
        for k in (0..j).rev() {
            acc[k] += p;
            p *= lo4;
        }
        let mut p = base;
        for a in acc.iter_mut().skip(j) {
            *a += p;
            p *= up4;
        }
    }
    for (o, a) in out.iter_mut().zip(acc) {
        *o = a.powf(1.0 / 0.4);
    }
}

/// Number of whole frames in `len` samples (§2.1.2).
/// INTERPRETATION: only complete 2048-sample frames, no padding at the end.
fn frame_count(len: usize) -> usize {
    if len < N {
        0
    } else {
        (len - N) / STEP + 1
    }
}

/// The caller's samples of one signal, read in place (never copied as a whole).
#[derive(Debug, Clone, Copy)]
pub(crate) enum Input<'a> {
    /// Interleaved samples and their channel count (≥ 1).
    Interleaved(&'a [f32], usize),
    /// One slice per channel.
    Planar(&'a [&'a [f32]]),
}

impl Input<'_> {
    fn channels(&self) -> usize {
        match *self {
            Input::Interleaved(_, ch) => ch,
            Input::Planar(c) => c.len(),
        }
    }

    /// Length of the shortest channel.
    fn len(&self) -> usize {
        match *self {
            Input::Interleaved(x, ch) => x.len() / ch,
            Input::Planar(c) => c.iter().map(|c| c.len()).min().unwrap_or(0),
        }
    }

    /// Sample `i` of channel `c` on the 16-bit scale.
    #[inline]
    fn at(&self, c: usize, i: usize) -> f64 {
        let x = match *self {
            Input::Interleaved(x, ch) => x[i * ch + c],
            Input::Planar(cs) => cs[c][i],
        };
        f64::from(x) * INT16_SCALE
    }
}

/// Read access to one signal for the frame loop: in-memory input, or a [`Window`] over a
/// [`Source`] that is filled ahead of use.
pub(crate) trait Signal {
    fn channels(&self) -> usize;
    /// Length in frames (shortest channel).
    fn len(&self) -> usize;
    /// Make samples `from..to` available to [`Signal::at`].
    fn prepare(&mut self, from: usize, to: usize) -> Result<(), Error>;
    /// Sample `i` of channel `c` on the 16-bit scale.
    fn at(&self, c: usize, i: usize) -> f64;
}

impl Signal for Input<'_> {
    fn channels(&self) -> usize {
        Input::channels(self)
    }
    fn len(&self) -> usize {
        Input::len(self)
    }
    fn prepare(&mut self, _: usize, _: usize) -> Result<(), Error> {
        Ok(())
    }
    #[inline]
    fn at(&self, c: usize, i: usize) -> f64 {
        Input::at(self, c, i)
    }
}

impl<S: Source> Signal for Window<S> {
    fn channels(&self) -> usize {
        Window::channels(self)
    }
    fn len(&self) -> usize {
        self.frames()
    }
    fn prepare(&mut self, from: usize, to: usize) -> Result<(), Error> {
        Window::prepare(self, from, to)
    }
    #[inline]
    fn at(&self, c: usize, i: usize) -> f64 {
        f64::from(Window::at(self, c, i)) * INT16_SCALE
    }
}

/// What one frame of one channel contributes from the reference side.
#[derive(Debug, Clone, Copy)]
struct RefFrame {
    /// `Mod[k, n]` (eq 57).
    modulation: [f64; Z],
    /// `10·log10 E[k, n]` (eq 72, reference excitation).
    level: [f64; Z],
    /// `TempWt[n]` (eq 65).
    weight: f64,
}

/// What one frame of one channel contributes from the test side.
#[derive(Debug, Clone, Copy)]
struct TestFrame {
    modulation: [f64; Z],
    level: [f64; Z],
}

/// Ear-model outputs of one frame of one channel.
struct FrameOut {
    excitation: [f64; Z],
    modulation: [f64; Z],
    e_bar: [f64; Z],
}

/// Frame-by-frame ear model of one signal (all channels), §2.1 and §3.2.
///
/// Two real frames share one complex FFT (real/imaginary packing): the two channels of a
/// stereo signal, or for a mono signal two consecutive frames. Packing only within one
/// signal makes every value depend on that signal alone, so a reference analysed once
/// gives the same values as one analysed next to each test signal.
struct Analysis<'m, G: Signal> {
    model: &'m Model,
    input: G,
    /// Frames the signal itself holds (the mono partner frame is read up to here).
    own_frames: usize,
    states: Vec<ChannelState>,
    buf: Vec<Complex>,
    fsp: [Vec<f64>; 2],
    /// Mono: the second frame of the last pair, not yet returned.
    pending: Option<FrameOut>,
    next: usize,
}

impl<'m, G: Signal> Analysis<'m, G> {
    fn new(model: &'m Model, input: G) -> Self {
        let (own_frames, ch) = (frame_count(input.len()), input.channels());
        Self {
            model,
            input,
            own_frames,
            states: (0..ch).map(|_| ChannelState::default()).collect(),
            buf: vec![(0.0, 0.0); N],
            fsp: [vec![0.0; N / 2 + 1], vec![0.0; N / 2 + 1]],
            pending: None,
            next: 0,
        }
    }

    /// FFT of `(a, b)` windowed frames (`None` = zeros) into `fsp[0]`, `fsp[1]`.
    fn transform(&mut self, a: (usize, usize), b: Option<(usize, usize)>) {
        let m = self.model;
        let inp = &self.input;
        for (t, z) in self.buf.iter_mut().enumerate() {
            let w = m.window[t];
            let re = inp.at(a.0, a.1 + t) * w;
            let im = b.map_or(0.0, |(c, s)| inp.at(c, s + t) * w);
            *z = (re, im);
        }
        m.fft.forward(&mut self.buf);
        for k in 0..=N / 2 {
            // Unpack the two real transforms: A = (Z[k] + Z*[N-k]) / 2, B = (Z[k] - Z*[N-k]) / 2i.
            let (zr, zi) = self.buf[k];
            let (cr, ci) = self.buf[(N - k) % N];
            let (ar, ai) = (0.5 * (zr + cr), 0.5 * (zi - ci));
            let (br, bi) = (0.5 * (zi + ci), -0.5 * (zr - cr));
            let g2 = m.gain[k] * m.gain[k];
            self.fsp[0][k] = g2 * (ar * ar + ai * ai); // eqs 6, 9, 11
            self.fsp[1][k] = g2 * (br * br + bi * bi);
        }
    }

    /// Outputs of the next frame, one per channel, in `out`.
    fn next_frame(&mut self, out: &mut Vec<FrameOut>) -> Result<(), Error> {
        out.clear();
        let n = self.next;
        self.next += 1;
        let s = n * STEP;
        let ch = self.input.channels();
        if ch == 1 {
            if let Some(f) = self.pending.take() {
                out.push(f);
                return Ok(());
            }
            let partner = (n + 1 < self.own_frames).then_some((0, s + STEP));
            self.input
                .prepare(s, s + N + if partner.is_some() { STEP } else { 0 })?;
            self.transform((0, s), partner);
            out.push(self.model.frame(&self.fsp[0], &mut self.states[0]));
            if partner.is_some() {
                let f = self.model.frame(&self.fsp[1], &mut self.states[0]);
                self.pending = Some(f);
            }
            return Ok(());
        }
        self.input.prepare(s, s + N)?;
        let mut c = 0;
        while c < ch {
            let pair = (c + 1 < ch).then_some((c + 1, s));
            self.transform((c, s), pair);
            out.push(self.model.frame(&self.fsp[0], &mut self.states[c]));
            if pair.is_some() {
                out.push(self.model.frame(&self.fsp[1], &mut self.states[c + 1]));
            }
            c += 2;
        }
        Ok(())
    }
}

impl Model {
    /// One frame of §2.1.5–§2.1.8 and §3.2 for one signal.
    #[allow(clippy::needless_range_loop)] // k indexes several per-band arrays at once
    fn frame(&self, fsp: &[f64], st: &mut ChannelState) -> FrameOut {
        let mut energy = [0.0; Z];
        let mut level = [0.0; Z];
        for (i, (first, w)) in self.groups.iter().enumerate() {
            let pe: f64 = w.iter().zip(&fsp[*first..]).map(|(a, b)| a * b).sum();
            let pe = pe.max(1e-12); // §2.1.5.1 limit
            let pp = pe + self.p_thres[i]; // eq 14
            energy[i] = pp;
            level[i] = 10.0 * pp.log10();
        }
        let mut e2 = [0.0; Z];
        spread_raw(&energy, &level, self.low_step, &self.up_base, &mut e2);
        for (e, nrm) in e2.iter_mut().zip(&self.norm_sp) {
            *e /= nrm; // eq 17
        }
        let mut out = FrameOut {
            excitation: [0.0; Z],
            modulation: [0.0; Z],
            e_bar: [0.0; Z],
        };
        let rate = f64::from(SAMPLE_RATE) / STEP as f64;
        for k in 0..Z {
            let a = self.a_time[k];
            st.ef[k] = a * st.ef[k] + (1.0 - a) * e2[k]; // eq 22, Ef[k,-1] = 0
            out.excitation[k] = st.ef[k].max(e2[k]); // eq 23
            let a = self.a_mod[k];
            let loud = e2[k].powf(0.3);
            // eqs 54–55. INTERPRETATION: E2[k,-1] = 0 and both filters start at 0.
            st.e_der[k] = a * st.e_der[k] + (1.0 - a) * rate * (loud - st.prev_loud[k]).abs();
            st.e_bar[k] = a * st.e_bar[k] + (1.0 - a) * loud;
            st.prev_loud[k] = loud;
            out.modulation[k] = st.e_der[k] / (1.0 + st.e_bar[k] / 0.3); // eq 57
            out.e_bar[k] = st.e_bar[k];
        }
        out
    }

    fn ref_frame(&self, f: &FrameOut) -> RefFrame {
        RefFrame {
            modulation: f.modulation,
            level: f.excitation.map(|e| 10.0 * e.log10()), // eq 72
            // eq 65 (levWt = 100).
            weight: (0..Z)
                .map(|k| f.e_bar[k] / (f.e_bar[k] + self.lev_thres[k]))
                .sum(),
        }
    }
}

fn test_frame(f: &FrameOut) -> TestFrame {
    TestFrame {
        modulation: f.modulation,
        level: f.excitation.map(|e| 10.0 * e.log10()), // eq 72
    }
}

/// Filter states carried from frame to frame.
struct ChannelState {
    ef: [f64; Z],
    e_der: [f64; Z],
    e_bar: [f64; Z],
    prev_loud: [f64; Z],
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            ef: [0.0; Z],
            e_der: [0.0; Z],
            e_bar: [0.0; Z],
            prev_loud: [0.0; Z],
        }
    }
}

/// `ModDiff1[n]` (eqs 63–64, negWt = 1, offset = 1).
fn mod_diff1(r: &RefFrame, t: &TestFrame) -> f64 {
    let (mr, mt) = (&r.modulation, &t.modulation);
    let diff: f64 = (0..Z).map(|k| (mt[k] - mr[k]).abs() / (1.0 + mr[k])).sum();
    100.0 / Z as f64 * diff
}

/// Detection probability and steps above threshold of one band (eqs 72–78), from the
/// excitation levels `er`, `et` in dB (eq 72). Returns `(p_c[k,n], q_c[k,n])`.
#[allow(clippy::many_single_char_names)] // the symbols of eqs 72–78
fn detection(er: f64, et: f64) -> (f64, f64) {
    let l = 0.3 * er.max(et) + 0.7 * et; // eq 73
    let s = if l > 0.0 {
        // eq 74
        5.950_72 * (6.394_68 / l).powf(1.713_32)
            + 9.010_33e-11 * l.powi(4)
            + 5.056_22e-6 * l.powi(3)
            - 0.001_024_38 * l * l
            + 0.055_019_7 * l
            - 0.198_719
    } else {
        1.0e30
    };
    let e = er - et; // eq 75
    let b = if er > et { 4.0 } else { 6.0 };
    let a = 10f64.powf(2f64.log10().log10() / b) / s; // eq 76
    let p = 1.0 - 10f64.powf(-(a * e).abs().powf(b)); // eq 77 (b even, so |a·e|^b)
                                                      // eq 78. INTERPRETATION: the integer part of the error magnitude.
    let q = e.abs().trunc() / s;
    (p, q)
}

/// First and last sample index of "actual data" in the reference (§5.2.4.4): the first
/// position, scanning from either end, where the sum of |x| over five succeeding samples
/// exceeds 200 in any channel (16-bit scale). `None` when no such position exists.
/// Only the first `len` samples count. Reads the signal in pieces from either end.
fn data_bounds(x: &mut impl Signal, len: usize) -> Result<Option<(usize, usize)>, Error> {
    const PIECE: usize = 1 << 16;
    if len < 5 {
        return Ok(None);
    }
    let hit = |x: &dyn Fn(usize, usize) -> f64, ch: usize, i: usize| {
        (0..ch).any(|c| {
            let w: [f64; 5] = core::array::from_fn(|t| x(c, i + t));
            w.iter().map(|v| v.abs()).sum::<f64>() > 200.0
        })
    };
    let ch = x.channels();
    let last = len - 5; // last start position
    let mut start = None;
    let mut i0 = 0;
    while i0 <= last && start.is_none() {
        let i1 = (i0 + PIECE).min(last + 1);
        x.prepare(i0, i1 + 4)?;
        start = (i0..i1).find(|&i| hit(&|c, j| x.at(c, j), ch, i));
        i0 = i1;
    }
    let Some(start) = start else {
        return Ok(None);
    };
    let mut end = None;
    let mut i1 = last + 1;
    while i1 > start && end.is_none() {
        let i0 = i1.saturating_sub(PIECE).max(start);
        x.prepare(i0, i1 + 4)?;
        end = (i0..i1).rev().find(|&i| hit(&|c, j| x.at(c, j), ch, i));
        i1 = i0;
    }
    // The start position itself is a hit, so `end` exists.
    Ok(end.map(|e| (start, e + 4)))
}

/// Per-block values of one comparison (one 1024-sample block = one 2048-sample frame).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Blocks {
    pub(crate) channels: usize,
    /// Frames compared.
    pub(crate) frames: usize,
    /// Data range of the reference (§5.2.4.4), `None` if it holds no data.
    pub(crate) bounds: Option<(usize, usize)>,
    /// `ModDiff1[n]` per channel, `[n * channels + c]`.
    pub(crate) mod_diff: Vec<f64>,
    /// `TempWt[n]` per channel, `[n * channels + c]`.
    pub(crate) weight: Vec<f64>,
    /// Binaural detection probability `1 − Π_k (1 − Pbin[k, n])`.
    pub(crate) detection: Vec<f64>,
    /// `Qbin[n] = Σ_k max_c q_c[k, n]`.
    pub(crate) steps: Vec<f64>,
}

impl Blocks {
    /// Frame `n` lies in the reference's data range (§5.2.4.4).
    /// INTERPRETATION: a frame counts when its centre sample (start + 1024) lies inside
    /// [start, end] of the data. Counting every frame that merely overlaps the range put the
    /// attack items (onsets after near-silence) up to 3 points off the reference outputs.
    pub(crate) fn valid(&self, n: usize) -> bool {
        self.bounds
            .is_some_and(|(s, e)| n * STEP + N / 2 > s && n * STEP + N / 2 <= e)
    }

    /// The two MOVs over frames `range`, plus (valid frames, distorted frames) there.
    pub(crate) fn movs(&self, range: core::ops::Range<usize>) -> (Movs, usize, usize) {
        let ch = self.channels;
        let (mut num, mut den) = (vec![0.0; ch], vec![0.0; ch]);
        // ADB (§4.7, §4.7.2) on the binaural channel (eqs 79–82).
        // INTERPRETATION: Qsum adds Qbin[n] of the distorted frames only (Pbin[n] > 0.5),
        // the same frames counted in ndistorted, so ADB is the mean steps per distorted
        // frame. Summing over all valid frames instead disagreed with the reference outputs
        // by up to 4 points on items with quiet passages (see the README).
        let (mut n_valid, mut n_distorted, mut q_sum) = (0usize, 0usize, 0.0);
        for n in range.filter(|&n| n < self.frames && self.valid(n)) {
            n_valid += 1;
            // AvgModDiff1: weighted linear average per channel (eq 90), from frame DELAY.
            if n >= DELAY {
                for c in 0..ch {
                    let (x, w) = (self.mod_diff[n * ch + c], self.weight[n * ch + c]);
                    num[c] += w * x;
                    den[c] += w;
                }
            }
            if self.detection[n] > 0.5 {
                n_distorted += 1;
                q_sum += self.steps[n];
            }
        }
        // Channel mean of the per-channel averages (§5.3).
        let mut amd = 0.0;
        for c in 0..ch {
            amd += if den[c] > 0.0 { num[c] / den[c] } else { 0.0 };
        }
        let avg_mod_diff1 = amd / ch.max(1) as f64;
        let adb = if n_distorted == 0 {
            0.0
        } else if q_sum > 0.0 {
            (q_sum / n_distorted as f64).log10()
        } else {
            -0.5
        };
        (Movs { avg_mod_diff1, adb }, n_valid, n_distorted)
    }
}

/// Per-block values from frame-by-frame reference and test contributions.
///
/// `reference(n, out)` writes the reference contributions of frame `n` (one per channel,
/// frames asked in order); `test` is the test signal's frame stream.
fn blocks_from<G: Signal>(
    ch: usize,
    len: usize,
    bounds: Option<(usize, usize)>,
    mut reference: impl FnMut(usize, &mut Vec<RefFrame>) -> Result<(), Error>,
    test: &mut Analysis<G>,
) -> Result<Blocks, Error> {
    let frames = frame_count(len);
    let mut b = Blocks {
        channels: ch,
        frames,
        bounds,
        mod_diff: Vec::with_capacity(frames * ch),
        weight: Vec::with_capacity(frames * ch),
        detection: Vec::with_capacity(frames),
        steps: Vec::with_capacity(frames),
    };
    let (mut rf, mut tf) = (Vec::with_capacity(ch), Vec::with_capacity(ch));
    for n in 0..frames {
        reference(n, &mut rf)?;
        test.next_frame(&mut tf)?;
        let tfs: Vec<TestFrame> = tf.iter().map(test_frame).collect();
        for c in 0..ch {
            b.mod_diff.push(mod_diff1(&rf[c], &tfs[c]));
            b.weight.push(rf[c].weight);
        }
        let mut p_keep = 1.0; // Π (1 - p_bin)
        let mut q_frame = 0.0;
        for k in 0..Z {
            let (mut p_bin, mut q_bin) = (0.0f64, 0.0f64);
            for c in 0..ch {
                let (p, q) = detection(rf[c].level[k], tfs[c].level[k]);
                p_bin = p_bin.max(p);
                q_bin = q_bin.max(q);
            }
            p_keep *= 1.0 - p_bin;
            q_frame += q_bin;
        }
        b.detection.push(1.0 - p_keep);
        b.steps.push(q_frame);
    }
    Ok(b)
}

/// Per-block values of two signals (the shorter length is used), both read frame by frame.
pub(crate) fn blocks_of<R: Signal, T: Signal>(mut reference: R, test: T) -> Result<Blocks, Error> {
    let model = Model::new();
    let ch = reference.channels().min(test.channels());
    let len = reference.len().min(test.len());
    let bounds = data_bounds(&mut reference, len)?;
    let mut r = Analysis::new(&model, reference);
    let mut t = Analysis::new(&model, test);
    let mut buf = Vec::with_capacity(ch);
    blocks_from(
        ch,
        len,
        bounds,
        |_, out| {
            r.next_frame(&mut buf)?;
            out.clear();
            out.extend(buf.iter().map(|f| model.ref_frame(f)));
            Ok(())
        },
        &mut t,
    )
}

/// The MOVs of two in-memory signals.
pub(crate) fn movs_input(reference: Input, test: Input) -> Movs {
    let b = blocks_of(reference, test).expect("in-memory input cannot fail");
    b.movs(0..b.frames).0
}

/// Compute `AvgModDiff1B` and `ADBB` for time-aligned, equal-rate signals.
///
/// `reference` and `test` hold one slice per channel (1 or 2), samples in [-1, 1] at 48 kHz.
/// The shorter length of all slices is used.
pub fn movs(reference: &[&[f32]], test: &[&[f32]]) -> Movs {
    movs_input(Input::Planar(reference), Input::Planar(test))
}

/// Per-block values read from two [`Source`]s, 10 s at a time.
pub(crate) fn blocks_source<R: Source, T: Source>(reference: R, test: T) -> Result<Blocks, Error> {
    blocks_of(Window::new(reference), Window::new(test))
}

/// A reference signal run through the ear model once, for comparing many test signals with
/// it (see [`crate::twof::Reference`]).
pub(crate) struct RefModel<'a> {
    model: Model,
    input: Input<'a>,
    /// `frames[n * channels + c]`.
    frames: Vec<RefFrame>,
}

impl<'a> RefModel<'a> {
    pub(crate) fn new(input: Input<'a>) -> Self {
        let model = Model::new();
        let ch = input.channels();
        let mut a = Analysis::new(&model, input);
        let total = a.own_frames;
        let mut frames = Vec::with_capacity(total * ch);
        let mut buf = Vec::with_capacity(ch);
        for _ in 0..total {
            a.next_frame(&mut buf).expect("in-memory input cannot fail");
            frames.extend(buf.iter().map(|f| model.ref_frame(f)));
        }
        drop(a);
        Self {
            model,
            input,
            frames,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.input.len()
    }

    /// Per-block values of `test` (same channel count) against this reference.
    pub(crate) fn blocks(&self, test: Input) -> Blocks {
        let ch = self.input.channels().min(test.channels());
        let rch = self.input.channels();
        let len = self.input.len().min(test.len());
        let mut input = self.input;
        let bounds = data_bounds(&mut input, len).expect("in-memory input cannot fail");
        let mut t = Analysis::new(&self.model, test);
        blocks_from(
            ch,
            len,
            bounds,
            |n, out| {
                out.clear();
                out.extend_from_slice(&self.frames[n * rch..n * rch + ch]);
                Ok(())
            },
            &mut t,
        )
        .expect("in-memory input cannot fail")
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact table values
mod tests {
    use super::*;

    #[test]
    fn band_table_is_contiguous() {
        for k in 0..Z - 1 {
            assert!((FU[k] - FL[k + 1]).abs() < 1e-9, "band {k}");
            assert!(FL[k] < FC[k] && FC[k] < FU[k], "band {k}");
        }
        assert_eq!(FL[0], 80.0);
        assert_eq!(FU[Z - 1], 18_000.0);
    }

    #[test]
    fn detection_half_at_one_step() {
        // eq 76 is chosen so that p = 0.5 when e equals s.
        let er = 60.0; // 10·log10(10^6)
        let (p, _) = detection(er, er);
        assert!(p.abs() < 1e-12);
    }
}

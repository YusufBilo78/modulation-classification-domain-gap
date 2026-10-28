# The noise in RML2016.10a, defined

*10 dB of what? Who made the noise, and what are its properties?* This page
answers that for RML2016.10a from three sources:

- **The dataset paper.** O'Shea & West, "Radio Machine Learning Dataset
  Generation with GNU Radio", Proc. 6th GNU Radio Conference, 2016. It names
  the channel model but gives no parameters and no definition of SNR.
- **The generator code DeepSig published.** github.com/radioML/dataset,
  `generate_RML2016.10a.py` and `transmitters.py`, commit `4ecf612` (May
  2017). The released pickle is from October 2016, so this is the code as
  published, not a guaranteed copy of what ran. Section 4 checks it against
  the pickle itself.
- **Two measurements.** The generator's chain was rebuilt in GNU Radio 3.10
  (`tools/rml2016_channel_snr.py`) and run to see what each setting
  produces. Then the SNR was measured in the pickle itself
  (`tools/measure_rml2016_snr.py`, `rml2016_measured_snr.json`,
  figure 35). Where the two disagree, the pickle wins.

---

## 1. Who made the noise

DeepSig (O'Shea, then at Virginia Tech, and West). They generated it
synthetically in GNU Radio, not over the air. The paper, §2.3:

> "For channel simulation we use the GNU Radio Dynamic Channel Model
> hierarchical block. This includes a number of desired effects such as
> random processes for center frequency offset, sample rate offset, additive
> white Gaussian noise, multi-path, and fading."

And on the noise itself:

> "Noise Model (AWGN): The basic GNU Radio additive white Gaussian noise
> model which introduces thermal noise simulation at the receiver at a
> specific noise power level corresponding to the desired signal to noise
> ratio."

That is the whole definition in the paper. Nothing there says which power,
over which bandwidth, or how the "desired" ratio is set.

---

## 2. The chain, with its numbers (from the generator code)

```
bits / audio → modulator → dynamic_channel_model → sample 128 → scale → store
```

| stage | setting | what it means |
|---|---|---|
| sample rate | 200 kHz | a frame of 128 samples is 0.64 ms |
| digital modulators | gr-mapper constellation → root-raised-cosine, **8 samples per symbol**, roll-off 0.35 | 25 kbaud; **a frame is 16 symbols**; the signal occupies 33.75 kHz of the 200 |
| constellation scale | today's gr-mapper sets mean *magnitude* to 1 | unit power for PSK; 1.11 for 16-QAM and 1.13 for 64-QAM. **The pickle does not match this; see §4** |
| sample-rate offset | random walk, std 0.01 Hz per sample, clipped at 50 Hz | clock drift; like the carrier walk below, it never gets near its clip in one run |
| carrier-frequency offset | random walk, std 0.01 Hz per sample, clipped at 500 Hz | in practice it reaches only a few Hz in a run (about 4 Hz, measured on the 3.10 rebuild), so a 128-sample frame rotates by about 1°; the carrier *phase* drifts several radians over a run, so frames differ in absolute phase |
| fading | sum of 8 sinusoids, **Rician K = 4**, Doppler 1 Hz | each frame sees one complex gain; over a run it swings 0.78–1.52 in magnitude (5.7 dB) |
| multipath | 3 paths at delays 0, 0.9, 1.7 samples, gains 1, 0.8, 0.3 | delay spread 0.2 symbol: mild but real inter-symbol interference |
| **noise** | `noise_amp = 10**(-snr/10.0)`, seed `0x1337` | see §3 |
| framing | 128 consecutive samples at a random offset | |
| scaling | each frame divided by Σ\|x\| | the code uses the sum of magnitudes, not the "unit energy" the paper states; this repository rescales every frame to unit power anyway (`cnn.normalize_frames`) |

Every effect except the noise is present at every SNR, including +18 dB.

---

## 3. What "10 dB" is the ratio of

*This section is the published code, rebuilt. §4 measures the pickle: the
slope below is confirmed there, but the classes turn out not to share one
SNR per label.*

The label sets one number, `noise_amp = 10**(-label/10)`. In GNU Radio that
parameter is the noise **amplitude**. Measured on the rebuilt channel:

| label | noise_amp | noise power measured |
|---|---|---|
| +10 dB | 0.1 | 0.0098 (−20.1 dB) |
| 0 dB | 1 | 0.98 (−0.1 dB) |
| −10 dB | 10 | 98 (+19.9 dB) |

So the noise power is 10^(−label/5): **the label moves the noise twice as
fast as its name says.** Nothing in the chain normalises the signal either.
It leaves the transmitter at 1.0 (PSK) or 1.11–1.13 (QAM), and leaves the
fading channel at 1.75 (PSK) or 1.95–1.99 (QAM), because the three path gains
add power (1 + 0.64 + 0.09 = 1.73).

**What the ratio actually is**, at a label of 10 dB, measured:

| class | per-sample SNR, full 200 kHz band | in the signal's own band | Es/N0 |
|---|---|---|---|
| BPSK, QPSK | **22.5 dB** | 30.3 dB | 31.6 dB |
| 16-QAM | 23.0 dB | 30.7 dB | 32.0 dB |
| 64-QAM | 23.1 dB | 30.8 dB | 32.1 dB |

In general, for PSK, **true per-sample SNR ≈ 2 × label + 2.4 dB**. The
signal's own band is 7.7 dB higher than that, because the noise is white over
200 kHz and the signal sits in 33.75 kHz. Es/N0 is 9 dB higher still, the
factor of 8 samples per symbol.

| label | per-sample SNR | Es/N0 (PSK) |
|---|---|---|
| −20 | −37.6 dB | −28.6 dB |
| −10 | −17.6 dB | −8.6 dB |
| 0 | 2.4 dB | 11.4 dB |
| +2 | 6.4 dB | 15.4 dB |
| +6 | 14.4 dB | 23.4 dB |
| +10 | 22.5 dB | 31.6 dB |
| +18 | 38.4 dB | 47.4 dB |

**Properties of the noise.** It is complex, circular, white, and Gaussian.
It is added after the fading and multipath, so its level is fixed while the
signal's level varies from frame to frame with the fading gain.

**Is the channel the same in every run? Not in the dataset.** GNU Radio
hands one seed to every part of the dynamic channel model, the generator
passes the fixed `0x1337`, and it builds a fresh channel for every run. In
GNU Radio 3.10 that makes every run bit-identical, noise included
(`tools/rml2016_channel_audit.py`). But the dataset was made with 3.7, and in
3.7.10 the noise, the carrier walk and the clock walk draw their values from
a seeded pool of 8,192 samples **at indices picked by `lrand48()`**, a
process-wide generator the block never re-seeds
(`fastnoise_source_X_impl.cc.t`). So those three differ from run to run.
Only the fading has its own seeded generator (`flat_fader_impl.cc`) and
repeats identically in every run.

Checked on the pickle (`tools/rml2016_noise_reuse.py`,
`rml2016_noise_reuse.json`): among all 11,000 frames at −20 dB, and again at
−16 dB, **no two frames share a noise segment**. At −18 dB, one pair does
(a BPSK and a WBFM frame, 118 samples), 1 in 11,000. The same test finds a
partner for 17% of frames when the generator's framing is run over one
shared noise sequence. So the noise is not reused, and no train/test leakage
comes from it. Two properties of the 3.7 source remain unchecked in the file:
every noise sample is one of 8,192 fixed values (times the amplitude), and the
fading trajectory is the same in every run.

---

## 4. What the pickle itself says

Measured on the file we train on, on 2026-09-29
(`rml2016_measured_snr.json`, figure 35). For the six digital classes the
signal occupies only |f| < 0.084 of the sample rate, so the spectrum above
20/128 is noise; since the noise is white, that level gives the total. Frames
of known SNR checked the estimator to within 0.5 dB from −10 to +30 dB.

**What it cannot read on this file.** The estimate stops rising at about
18–20 dB whatever the label: the file carries some out-of-band content
about 20 dB below the signal that is not noise and does not scale with the
label. It also stops falling at about −13 to −18 dB, where the noise is no
longer exactly white across the band. Readings outside −13 … +18 dB are
limits of the method on this file, not SNRs.

Inside that window, the measured per-sample SNR (1,000 frames per entry):

| label | BPSK | QPSK | 8PSK | PAM4 | QAM16 | QAM64 |
|---|---|---|---|---|---|---|
| −10 | −12.8 | −11.8 | −13.6 | −7.2 | −4.6 | 1.4 |
| −8 | −9.7 | −9.7 | −10.2 | −3.7 | −0.6 | 5.1 |
| −6 | −6.7 | −6.3 | −6.7 | 0.1 | 3.1 | 9.0 |
| −4 | −3.0 | −2.7 | −3.0 | 3.9 | 6.7 | 12.2 |
| −2 | 1.1 | 1.2 | 0.8 | 7.6 | 10.4 | 15.5 |
| 0 | 4.8 | 5.1 | 4.7 | 11.0 | 14.0 | 17.2 |
| +2 | 8.5 | 8.8 | 8.5 | 14.3 | 17.1 | 18.8 |
| +4 | 12.1 | 12.2 | 11.9 | 17.3 | 18.5 | 19.1 |
| +6 | 15.0 | 14.7 | 15.3 | 19.0 | 18.5 | 19.2 |

Three findings.

**1. Confirmed: the label moves the SNR about twice as fast as its name.**
Over −6 … +4 dB, PSK gains 1.87–1.89 dB of measured SNR per dB of label.
That is what a noise *amplitude* of 10^(−label/10) predicts (2.0), and far
from what a noise *power* would give (1.0). For PSK the level is within
about 2.5 dB of the rebuilt code: 4.8 dB at label 0 against 2.4 predicted.
So "10 dB" in RML2016.10a is not a 10 dB ratio of anything. It is the
setting `noise_amp = 0.1`.

**2. New: the same label means a different SNR for different classes.**
At the same label, and relative to PSK, the measured SNR is higher by

| | measured (labels −8, −6, −4) | predicted if the constellation points were never scaled |
|---|---|---|
| PAM4 | +6.6 dB | +7.0 dB (points ±1, ±3: power 5) |
| QAM16 | +9.5 dB | +10.0 dB (odd-integer grid: power 10) |
| QAM64 | +15.2 dB | +16.2 dB (odd-integer grid: power 42) |

and the three PSKs agree with one another to within 0.4 dB. The code as
published would put them all within 0.6 dB of one another. So the pickle
behaves as if the constellations went out **unscaled**, at the integer
coordinates in gr-mapper's tables, while the noise was the same for all.
The one scaling bug gr-mapper did have in 2016 (fixed on 11 October 2016,
commit `15e71bf`) does not fit: it would have put QPSK 6 dB above BPSK, and
the data shows 0. Which code actually ran is not something the file can
say. What it does say: **at label −6 dB, a QAM64 frame has about 9 dB SNR
and a QPSK frame about −6 dB.**

**3. Unreadable at the top.** Everything at labels ≥ +8 (PSK) or ≥ 0
(QAM64) reads 18–22 dB, the ceiling of the method. The true SNR there is at
least that, and by the slope above probably much more. The code predicts
38 dB for PSK at +18. This file cannot confirm it.

---

## 5. What this changes in the results already reported

Every number in every table stands. They are indexed by label, which is what
everyone who uses this dataset reports. What changes is how the SNR axis
may be read:

- **Across classes, the axis is not shared.** At the same label, QAM16 has
  about 10 dB more SNR than QPSK, and QAM64 about 15 dB more. Figure 33
  shows QAM64 recall climbing from −16 dB, well before QPSK's at −6. That
  is not the model finding QAM64 easier. At −12 dB QAM64 frames sit at
  about −2.5 dB, and QPSK frames below −13 dB, too low for this method to
  read.
- **The sink at −20 dB goes to BPSK and QPSK** (98.8% of decisions,
  `SNR_2016.md`). Those are the classes whose frames at −20 dB carry the
  least signal, so "the model sends pure noise to the PSK classes" and "the
  model sends noise to where it has seen the most noise-like frames" are
  the same statement here.
- **"Recall is flat from +2 dB up"** means flat from a measured 8.5 dB
  (PSK) and from at least 17–19 dB (QAM). The QAM16/QAM64 confusion that
  persists there is at SNRs of 17 dB and more. It comes from 16 symbols,
  carrier offset, fading and multipath, not from noise. That strengthens the
  frame-length reading of `SNR_2016.md`.
- **Comparing SNR axes across datasets or papers is not safe.** RadioML
  2018 has its own generator and its own definition. Nothing here transfers
  to it.

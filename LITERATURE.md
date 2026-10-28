# Literature check

**Status: partial, not a systematic review.** What follows is a targeted search
against four specific claims, done with web search plus reading the full text
of the papers that looked closest. It is not exhaustive.

**Known blind spot:** much of the AMC literature sits behind IEEE Xplore, which
was not accessible here. arXiv coverage is good for recent ML-flavoured work
and poor for older or IEEE-only signal-processing work. "Not found" here means
"not found in arXiv and open sources with these search terms", never "does not
exist".

Four questions, one per claim this project makes.

---

## Q1 — Has anyone attributed an AMC domain gap to a specific signal property by controlled intervention?

**Not found.** The pattern across everything read is: measure a gap, propose a
method, report the improvement. Attribution of the gap to one property, by
intervening on that property and holding the rest fixed, did not appear.

Closest: **WhiteNet** (below) attributes its gap to the "channel envelope" and
supports that with a t-SNE separability measure — session separability drops
from 88.9% to 59.4% after whitening. That is correlational evidence about the
learned representation.

This project's phase-preserving magnitude substitution is an intervention on
the signal rather than an observation of the representation: keep the phase
spectrum exactly, swap only the magnitude, and see whether the prediction
follows. 16QAM went 0.191 → 0.972. Combined with the elimination of symbol-rate
offset, constellation density and channel impairments, this is a causal
argument rather than a correlational one.

**Assessment:** this looks like the most distinctive part of the work, with the
caveat that a methodological contribution is harder to place than a numerical
one.

---

## Q2 — Is spectral whitening already used for RF domain robustness?

**Yes. Found, and it is essentially the same operation.**

**WhiteNet: Robust Identification of Overlapping IEEE 802.11 Signals Across
Unseen Channels** — arXiv:2608.06581

Their transform:

    Y~[m] = Y[m] · sqrt(P̄) / sqrt(P̂[m])

with P̂ a circular moving-average smoothed PSD estimate, window W = 256, noise
floor δ = −30 dB. Dividing the spectrum by its own smoothed envelope is exactly
the operation used here, arrived at independently.

**So "spectral whitening for RF domain generalization" is not a novel proposal.
That claim has to be dropped.**

What differs, and the differences are real:

| | WhiteNet | this project |
|---|---|---|
| task | multi-label 802.11 protocol ID | modulation classification |
| domain shift | **channel variability** (frequency-selective fading, different capture sessions) | **transmitter mismatch** (pulse shaping / spectral envelope) |
| data | **real over-the-air captures**, 3 sessions | RadioML (synthetic) + own generator (synthetic) |
| whitening strength | full (exponent 1) | **swept 0→1; 0.75 beats 1.0** (0.907 vs 0.827) |
| attribution | t-SNE session separability | phase-preserving magnitude substitution |
| headline | 87.6% → 47.6% on unseen channels, recovered to 73.6% held-out | gap 0.182 → 0.038 |

Two things worth keeping:

1. **They whiten fully; the sweep here says that is suboptimal.** At alpha = 1
   in-domain accuracy collapses to 0.879 while 0.75 holds 0.964 with better
   cross-domain. That is a concrete, testable disagreement — on a different task,
   so not a contradiction, but worth reporting.
2. **Their domain shift is the channel; this one is the transmitter.** Related
   but not the same cause, and the fix working for both is itself informative.

Their result is stronger in the way that matters most: **their data is real.**

---

## Q3 — Is there work on *which* class a model defaults to under distribution shift?

**Adjacent field identified: open-set recognition (OSR).** That is the
established literature for what a classifier does when it meets a class it was
never trained on. It was not on my radar before this search and it belongs in
any write-up.

But OSR asks a different question. It is about **detecting** that a sample is
unknown — scoring, thresholding, rejection. Prototype methods such as Nearest
Class Mean assign a sample to the nearest class mean in feature space, which is
related in spirit, but the literature treats that assignment as machinery for
detection rather than as an object of study.

Also relevant: **Deep Open Set Identification for RF Devices** (arXiv:2112.02536)
— OSR applied to RF, though to device identification rather than modulation.

**Not found:** work asking whether the landing class is *predictable from the
taxonomic structure of the label set*. The finding here — that an unseen
modulation lands inside its own modulation family 9/10, verified with a control
where family and density predict opposite answers, and with a label-permutation
control ruling out an output-index artifact — did not turn up.

**Confidence: low.** OSR is a large field and a few searches do not cover it.
This needs a proper pass before any novelty claim, and it is the obvious
question to put to Prof. Abdi.

### Q3 follow-up — a closer read of OSR (second pass)

A more targeted pass sharpened the picture. The distinction that matters:

**OSR treats "unknown lands on similar known class" as a problem to prevent;
this project treats it as a phenomenon to measure and predict.**

A survey-level statement found in the search captures the field's stance
exactly: the core OSR challenge is *preventing* unknown classes — especially
those semantically similar — from being misclassified as their nearest known
neighbour. So the field is aware the confusion is similarity-structured; it
works to suppress it, not to characterise it.

Papers read on this pass:

- **OSNN (Open Set Nearest Neighbour).** Assigns a test sample to its nearest
  known class in feature space unless a distance ratio exceeds a threshold, in
  which case it is rejected. This is the closest prior mechanism: it confirms
  that "unknown resembles nearest known" is long established. But OSNN uses that
  as a *detection rule* — the nearest-class assignment is machinery for deciding
  known-vs-unknown, never studied as a structured, predictable phenomenon.

- **Liu et al., KTH 2026 (arXiv:2603.24268), OSR for unknown UAVs via RF
  semantics.** Same field as this project (RF), but the goal is detect / reject
  / cluster-then-learn. Its confusion matrix is used only to show knowns are
  right and unknowns are caught. It does **not** ask which known class an
  unknown resembles, nor predict it from signal structure.

- **Domain Feature Collapse (arXiv:2512.04034).** Name is close, content is not:
  an information-theoretic account of why OOD *detection* fails on single-domain
  models, on images. Not about which class OOD inputs land on.

**Where this leaves the claim.** "Unknown lands on nearest known class" is not
new — OSNN encodes it. What was not found:

1. Treating the landing class as a **structured, predictable** quantity rather
   than as detection machinery.
2. **Predicting it** from an independent notion of class similarity — here shown
   three ways: hand-drawn family (18/24), the model's own embedding
   (19/24 top-3, circularity broken), and stability across four architectures
   including attention (same sinks, same mistakes).
3. In **AMC specifically**, where the taxonomy (ASK/PSK/APSK/QAM families,
   constellation order) is unusually clean and physically grounded, so
   "structurally similar" has a precise meaning.

**Revised confidence: still cautious, but the gap looks narrower and more
specific than "nobody has looked".** The honest framing is not "a new
phenomenon" but "a known confusion, characterised and predicted rather than
suppressed, in a domain with a clean taxonomy." Whether that clears the bar for
novelty is exactly the judgement to put to Abdi — it is a positioning question,
not a measurement one, and he will know the OSR corner far better than these
searches reach.

---

## Q4 — What is the state of the art on cross-domain AMC?

Active, and more crowded than assumed at the start.

- **Adversarial adaptation.** DANN-style domain-adversarial training, e.g.
  arXiv:2508.06829 for AMC under channel variability. Requires unlabelled
  target data at fit time.
- **Metric learning / embeddings.** Henneke & Kurth, arXiv:2510.23186 — train
  on synthetic wireless protocols with an angular-margin loss, validate on
  **real** RF captures. Baselines include a 26-dimensional higher-order
  moment/cumulant vector, close to the classical baseline used here, and the
  spectral correlation function. No whitening, no gap attribution.
- **Augmentation.** arXiv:1912.03026 — rotation, flip, Gaussian noise, the set
  reproduced here as a baseline.
- **Whitening.** WhiteNet, above.
- **Dataset synthesis.** MDM, arXiv:2408.02714 — multi-domain distribution
  matching for AMC dataset synthesis.
- **Uncertainty / OOD.** arXiv:2608.00796 and others; an active 2025–26 line.

Two facts worth internalising:

1. **Synthetic-to-real is a recognised problem with real solutions already
   published.** Henneke & Kurth do the synthetic→real experiment this project
   cannot yet do.
2. **The published roll-off practice** of drawing alpha from a range, noted in
   an earlier search and confirmed in O'Shea's Table I as U(0.1, 0.4), means
   pulse-shaping variation is known to matter. That it *causes* a specific
   confusion between adjacent QAM orders is the part not seen elsewhere.

---

## The dataset paper, read in full (O'Shea, Roy & Clancy 2018)

Read 2026-09-22, because a
colleague suggested that "some signals in 2018 are especially degraded" and
a switch to 2016 was on the table.

**How the synthetic data is generated (Section III, Table I).** Every
example draws its own channel: root-raised-cosine roll-off α ~ U(0.1, 0.4),
timing offset Δt ~ U(0, 16) samples, sample-rate and carrier offsets
~ N(0, σ_clk), carrier phase ~ U(0, 2π), Rayleigh multipath with delay
spread τ. The paper trains on several variants (AWGN, σ_clk = 0.0001,
σ_clk = 0.01, τ = 0.5 … 4) and does **not** say which of them the released
2018.01A file is. Frames are 1,024 samples, SNR −20 to +30 dB Es/N0. The
"difficult" 24-class set deliberately applies impairments "beyond that which
one would expect" for the high-order modes and keeps the window short.

**Which classes the paper itself reports as hard (Figs. 12–15, 17, 19, 21).**
At high SNR the per-class curves plateau below 1 for 128QAM and 256QAM
(~0.85), 64QAM (~0.9), 32PSK (~0.93), and the AM pairs: AM-SSB-SC and
AM-DSB-SC around 0.85, the with-carrier variants higher. The paper's own
explanation: for high-order QAM/PSK "significant error is expected simply
due to lack of information and similar symbol structure using this or any
other known prior method" at 1,024 samples; for AM, with-carrier against
suppressed-carrier confusion, and "we suspect additional voice data set size
might improve performance" — i.e. the analog message content is a small
voice corpus. Every confusion matrix in the paper, synthetic or OTA, shows
the same three blocks: 16/32PSK, 64/128/256QAM, and the AM WC/SC pairs.

**Our own 24-class run reproduces that picture exactly**
(`train_backbone_rml2018_f512_colab.npz`, ≥10 dB, 2,574 decisions per
class): AM-DSB-WC 45.9% (leaks to AM-DSB-SC), 64QAM 56.4% (to 256QAM and
128QAM), 256QAM 61.5%, AM-SSB-WC 61.6%, 128QAM 69.9%, 16APSK 74.7% (to
16QAM). Everything else is above 79%, and BPSK, QPSK, 8PSK, FM, GMSK are at
100.0%. So "some signals are degraded" is true and published, and it is
confined to the high-order QAM/APSK block and the AM carrier pairs.

**None of those classes is in the four-class table.** BPSK, QPSK, 16QAM and
64QAM: 40,654 of 40,656 above 10 dB. 64QAM at 56% in the 24-class run is
confusion with 128QAM and 256QAM, which the four-class task does not
contain; with them absent it is 10,162 of 10,164.

**2016 is not cleaner; it is worse on exactly this table.** Our 11-class
RML2016.10a run (`train_backbone_rml2016_f1000_colab.npz`, ≥10 dB): WBFM
39.6% (leaks to AM-DSB, a known defect of that dataset), AM-SSB 89.5%, and
QAM16/QAM64 confuse each other 8–9% at high SNR because 128-sample frames
carry too few symbols. The four-class rehearsal on 2016 put QAM16 at 87.9%
where 2018 puts it at 100%. The paper's Fig. 18 is the mechanism: about 3%
accuracy per doubling of window length up to 512–1,024 samples.

**OTA.** The paper captured 1.44 M over-the-air examples at ~10 dB with two
USRP B210s; trained directly they reach 95.6%. A model trained on σ_clk =
0.0001 synthetic data and evaluated on OTA without fine-tuning loses ~7
points (94% → 87%); the confusions before fine-tuning are the AM carrier
pairs and the high-order QAM/APSK modes — the same blocks. That 7-point
synthetic→real drop is the literature's counterpart to the gap this project
measures, and it lands on the same classes.

**Decision this supports:** stay on 2018. The degraded classes are known,
explained by the authors, reproduced by us, and outside the four-class
table; the one dataset that avoids them at 1,024 samples is the one already
in use.

---

## The 2016 dataset paper (O'Shea & West, GRCon 2016), read in full 2026-09-28

"Radio Machine Learning Dataset Generation with GNU Radio", Proc. 6th GNU
Radio Conference. Mostly a survey; §2 is the dataset. It names the channel
(GNU Radio Dynamic Channel Model: SRO, CFO, selective fading, AWGN) but gives
**no parameters and no definition of SNR** — AWGN "at a specific noise power
level corresponding to the desired signal to noise ratio" is all. It says
frames are scaled to unit energy; the published generator scales by the sum
of magnitudes instead. Parameters, and what the SNR label turns out to mean,
are in `NOISE_2016.md`, taken from the generator code (github.com/radioML/
dataset, commit 4ecf612) rather than from the paper.

## MSTFFNet (Sensors 26(16):5208, published 17 August 2026), read in full

Wu, Xiang, Dong, Wang & Xiao, Air Force Engineering University, Xi'an. MDPI
Sensors; received 9 July, accepted 15 August. Open access; no code release
(data statement says "email the authors").

**What it is.** A dual-stream model for RML2016.10a/10b at the native 128
samples. Stream 1: the raw I/Q sequence through three parallel 1-D conv
branches (kernels 3, 5, 7) with channel and temporal attention, two stages
(64, 128 channels), giving 32 tokens of 128 dims. Stream 2: an STFT of the
frame (64-point FFT, 64-sample Hann window, hop 2, bins 0–32 kept, so a
4 × 33 × 33 tensor of Re, Im, |Z|, ∠Z) through 2-D convs, a residual block
and a frequency-attention gate, then two paths (frequency-mean and four
adaptively pooled bands) to another 32 × 128 token sequence. The two are
fused per token by a 1 × 1 conv, passed through one BiLSTM (hidden 64) and
two blocks of depthwise conv + additive (linear) attention + FFN, pooled to
a 128-d vector. A side head predicts one of eight 5-dB SNR bins from the
I/Q tokens; the soft bin probabilities weight learnable 16-d embeddings and
the result is *concatenated* to the feature before a 2-layer MLP. 1.98 M
parameters, 124 MMACs. Loss: 0.7 CE (label smoothing 0.1) + 0.3 focal,
plus auxiliary CE, cross-modal InfoNCE, supervised contrastive and the
SNR-bin CE at small weights. AdamW, cosine, 100 epochs, 6:2:2 split per
SNR.

**Numbers.** RML2016.10a overall 67.33 %, RML2016.10b 70.87 %. Highest
single-SNR point 94.50 %. The gain over the six retrained baselines (GIGNet
63.80, FE-SKViT 63.34, AVGNet 62.93, MCLDNN 62.02, GRU 57.47, LSTM 56.40 on
10a) is almost entirely low-SNR: [−20, 0] dB 45.79 vs 39.88 for GIGNet;
[0, 18] dB 93.45 vs 92.92, i.e. tied at high SNR. Ablations: dropping the
SNR conditioning costs 2.98 overall and 4.38 at low SNR; an oracle SNR bin
adds only 0.54 over the self-estimate; STFT-only is 10.5 points below the
dual model, I/Q-only 5.85 below. **Every number is a single training run
with one seed and one split**; the authors say so and call the
differences "descriptive". The 60/20/20 split is stratified per SNR but
there is no early-stopping or convergence report.

**Where it agrees with us.** WBFM at 31.72 % overall and under 50 % at
high SNR, drained into AM-DSB — the same defect as our 39.6 %, and they
say the same thing: 128 samples are too short for wideband FM. And on
page 15: "all misclassifications occur within modulation families that
are physically similar, rather than being randomly distributed across
classes" — the sink finding, observed from the other direction.

**Where it is silent.** Not evaluated on RadioML 2018.01A. No cross-dataset,
cross-transmitter or over-the-air test; the conclusion lists all three as
future work. So it addresses the benchmark question and not ours.

**What it means for this project.**

1. *For the decision table:* nothing. Its advantage is below 0 dB, where
   the four-class table is at the signal's limit for any method; at high
   SNR it ties with a 0.41 M MCLDNN.
2. *For the domain gap:* a testable prediction. The model feeds the STFT
   magnitude directly, i.e. the spectral envelope this project identified
   as the cause of the gap is an explicit input channel. The prediction is
   that MSTFFNet loses **more** than ICRNNA when the transmitter changes,
   and that whitening its input recovers it. That is a clean experiment for
   the 2016 track: same frames, same protocol, `--arch MSTFFNet`. If it
   holds, the attribution generalises to a model built on the opposite
   design philosophy; if it does not, that is the more interesting result.
3. *For the sink thread:* a sixth column in the architecture control, and
   a question the SNR-conditioning head raises — whether an SNR-aware
   classifier still sinks everything into QPSK below −8 dB or spreads it.
4. *Against the ICRNNA numbers on 2016:* our 3-seed converged ICRNNA is
   62.23 % overall and the faithful build 63.21 %; MSTFFNet's 67.33 % is
   about 4 points above, all of it from below 0 dB, single run.

Reimplementing it from the paper is feasible — the architecture is
specified to the layer — but the exact numbers are not reproducible without
their code and seed, and the STFT front end is hardwired to 128-sample
frames (64-window, hop 2 → 33 frames), so a 1,024-sample version is a
re-parameterisation, not a port.

---

## CSA-DiffRF (CMSS 2026, IEEE conference), read in full — not citable as evidence

Sigamani et al., "Diffusion Based RF Signal Generation and Augmentation for
Automatic Modulation Classification in 6G Networks", Proc. 2nd Int. Conf.
on Modern Sustainable Systems, 2026, pp. 329–334, DOI
10.1109/CMSS69636.2026.11689541. Six pages.

A conditional DDPM (U-Net, conditioned on modulation and SNR embeddings)
generates synthetic RML2016.10a frames at 128 samples; they are added at
100% to the training set of a CNN-Transformer. Reported: 92.7%
"classification accuracy" against CNN 84.2, CLDNN 86.8, Transformer 88.1,
wavelet augmentation 89.0, GAN augmentation 89.4.

**Why it is not usable.** The accuracy is not defined. On RML2016.10a's 11
classes over −20..+18 dB the best published overall figures are in the low
to high 60s (MSTFFNet 67.33, the faithful ICRNNA 63.21), and with chance at
9% below −10 dB an all-SNR mean of 92.7 is not possible; the SNR range
behind the number is never stated, and neither is the baseline CNN's 84.2.
Single run, no seeds, no per-class results, no confusion matrix. The
title and introduction promise evaluation under Rayleigh/Rician fading,
Doppler and CFO; no such experiment is reported. Sample quality is argued
from constellation plots and a t-SNE only.

**What is worth taking.** Only the pointer: the method it builds on is
Xu, Huang, Zhang, Qian & Yang, "Diffusion-based radio signal augmentation
for automatic modulation classification", *Electronics* 13(11):2063, 2024.
That is the one to read if generative augmentation becomes relevant —
which it may, because on 2016 the literature augmentation set is what
closes the cross-domain gap (0.920 → 0.968), not whitening.

---

## "Automatic Modulation Recognition and Spectrum Monitoring in Cognitive Radio" (ICISC 2026), read in full — internally inconsistent, not citable

Mudda et al., Proc. 10th Int. Conf. on Inventive Systems and Control, 2026,
pp. 826–831, DOI 10.1109/ICISC69558.2026.11681655.

A plain 1-D ResNet on RML2016.10a (11 classes, 128 samples, 70/15/15,
max-abs normalisation), plus an FFT of the input presented as "spectrum
monitoring". Claims 88.20% overall accuracy.

**Why it cannot be right as reported:**

1. Fig. 5 shows 42% accuracy at −20 dB. Chance on 11 classes is 9%; at
   −20 dB every model on this file is at chance — our four-class run is at
   25.1% there, exactly chance for four.
2. The confusion matrix (Fig. 4) spreads its errors evenly, 1–3 per
   off-diagonal cell. RML2016.10a's errors concentrate in two places,
   QAM16↔QAM64 and WBFM→AM-DSB; the paper's own text names the QAM
   confusion, and its matrix shows 4 QAM64→QAM16 decisions.
3. WBFM is shown at ~98%. It is the file's known defect: 39.6% in our
   11-class run, 31.72% in MSTFFNet, both drained into AM-DSB.
4. The test set is stated as ~33,000 samples; the matrix rows sum to
   ~900, about 9,900 in total.

88.2% overall is also ~20 points above the best published all-SNR figures
on this file. Single run, no seeds, no convergence report. Useful only as
an illustration of why every number in this project carries its
definition and its count.

---

## Where this leaves the project

**Drop:** any claim that spectral whitening is a new idea for RF domain
robustness. It is published, on real data, with better results in its own
setting.

**Keep, with support:**

- The **attribution method** — causal intervention rather than correlational
  evidence. Q1 found nothing comparable.
- **Partial beats full whitening** — a concrete disagreement with WhiteNet's
  choice, measured over four seeds.
- The **family-structured failure** — Q3 found the adjacent field but not this
  question. Lowest confidence, highest potential.
- The **negative results**: symbol-rate offset, constellation density and
  channel impairments each ruled out by controlled test. Literature tends to
  attribute domain gaps to channel effects; this measured that they do not
  account for it here.

**Weakest point, unchanged:** both domains are synthetic. Henneke & Kurth and
WhiteNet both use real captures. That gap is the difference between a study and
a result, and it is what the SDR is for.

---

## What was searched

Terms covering: spectral whitening / spectrum flattening + AMC; domain
generalization + RF signal classification; domain adaptation + modulation
classification; sink class / default class / collapse under distribution shift;
open-set recognition + nearest class; RadioML generation parameters.

Full text read: O'Shea, Roy & Clancy (arXiv:1712.04578; reread in full 2026-09-22, notes above); MSTFFNet (Sensors 26(16):5208, read in full 2026-09-22, notes above); Henneke & Kurth
(arXiv:2510.23186); WhiteNet (arXiv:2608.06581, HTML). Abstract or summary
only: the remainder.

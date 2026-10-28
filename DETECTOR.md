# The detector, opened

This page answers one question: *what is the box?*
It covers who wrote the model and where it comes from, what each layer does, how
it was trained, how a decision is made, and what was done with an AI assistant.
Everything below is written for the RML2016.10a four-class setting (BPSK, QPSK,
QAM16, QAM64; 128-sample frames), which is the setting all new work uses.

---

## 1. In one paragraph

The detector takes 128 complex baseband samples (I and Q, so a 2 × 128 array),
and returns one of four labels. Two convolution layers turn the raw samples into
32 short feature vectors, each describing about one and a half symbols. A
two-layer bidirectional LSTM reads those 32 vectors forwards and backwards so
that each one carries context from the whole frame. An attention layer takes a
weighted average of the 32 vectors, which gives one summary vector for the
frame. A small two-layer classifier maps that to four scores. The label is the
highest score. There are 785,476 trainable numbers, and 84% of them are in the
LSTM.

---

## 2. Where it comes from

| | |
|---|---|
| Architecture name | ICRNNA (Improved Convolutional Recurrent Neural Network with Attention) |
| Paper | El-Haryqy et al., *Results in Engineering* 26 (2025) 104783 |
| Paper's result on RML2016.10a, 11 classes | 63.24% overall |
| Our code | `src/model_zoo.py`, class `ICRNNA`, used through `model_zoo.backbone(n_classes)` |
| How the code was obtained | transcribed from a peer's reproduction of the paper (`train_amc_standalone.py`), **not** written from the paper |

**It is a reproduction of a reproduction, and it differs from the paper in
five places:**

| | paper | our model |
|---|---|---|
| first conv kernel | 3 | 5 |
| max-pools | one | two |
| BatchNorm around the LSTM | one per LSTM layer | one after the stack |
| attention | with dropout and LayerNorm | neither |
| classifier | two dense layers, 128 and 64, dropout 0.3 | one dense layer of 128, dropout 0.5 |

So it is never called "the published architecture".

**Was the paper checked?** Yes, separately. A second build written directly
from the paper (`colab/icrnna_faithful_2016.py`, 794,827 parameters, the paper
says 0.79M) was trained with the paper's own settings:

| | our model (peer version) | faithful build, 58 epochs | faithful build, trained to convergence | paper |
|---|---|---|---|---|
| RML2016.10a, 11 classes, overall | 62.23% | 61.75% | **63.21%** | 63.24% |

The faithful build matches the paper to 0.03 points once it is allowed to train
until early stopping ends it (epoch 107, not the paper's 58). That tells us the
paper's number is real and that our data pipeline reads the dataset correctly.
Our working model is 1 point below the paper, and that is known and stated.

**Why this model, and not the one we used before.** The project first used a
plain CNN (IQNet). On identical data IQNet reached 99.98% on the training set
and 67.50% on the test set by epoch 30. It memorised. ICRNNA on the same data:
71.88% train, 71.00% test. Almost everything this project reports is a
*difference* between two accuracies, so a model that memorises is the wrong
measuring instrument. That comparison is `src/overfit_2x2.py`, and it is the
reason for the switch on 2026-09-10.

---

## 3. Layer by layer

Input: one frame, 128 complex samples as a 2 × 128 array (row 0 = I, row 1 = Q).
On RML2016.10a there are 8 samples per symbol, so a frame is 16 symbols.

Before the network: every frame is scaled to unit average power
(`cnn.normalize_frames`), so the model cannot use the received level to guess
the SNR or the class.

| # | block | what it does | output shape | parameters |
|---|---|---|---|---|
| 0 | input | I and Q, 128 samples | 2 × 128 | — |
| 1 | Conv1d(2→64, kernel 5) + BatchNorm + ReLU | 64 learned filters, each looks at 5 consecutive samples | 64 × 128 | 704 + 128 |
| | MaxPool(2) + Dropout(0.3) | keep the larger of each pair of samples; halves the length | 64 × 64 | 0 |
| 2 | Conv1d(64→128, kernel 3) + BatchNorm + ReLU | 128 filters over the first layer's outputs | 128 × 64 | 24,704 + 256 |
| | MaxPool(2) + Dropout(0.3) | halves the length again | **128 × 32** | 0 |
| 3 | BiLSTM, 2 layers, 128 units per direction, dropout 0.3 | reads the 32 vectors left→right and right→left; each output knows about the whole frame | 32 × 256 | 659,456 |
| | BatchNorm | rescales the 256 features | 32 × 256 | 512 |
| 4 | Additive attention | gives each of the 32 steps a score, turns the scores into weights that sum to 1 (softmax), and takes the weighted average | **256** | 66,048 |
| 5 | Linear(256→128) + BatchNorm + ReLU + Dropout(0.5) | classifier hidden layer | 128 | 32,896 + 256 |
| 6 | Linear(128→4) | one score ("logit") per class | **4** | 516 |
| | **total** | | | **785,476** |

With 11 classes the last layer has 1,419 parameters and the total is 786,379.
Parameter count in general: 784,960 + 129 × (number of classes).

**What one LSTM time step "sees".** After the two conv + pool stages each of the
32 steps is computed from 12 consecutive input samples, 1.5 symbols, and
neighbouring steps are 4 samples (half a symbol) apart. So the conv front end
extracts short local shapes (transitions, amplitude and phase changes between
neighbouring symbols) and the LSTM is what connects them across the 16 symbols
of the frame.

**Attention, in plain words.** For each step t, score_t = v · tanh(W h_t); the
weights are softmax(score) over the 32 steps; the frame summary is
Σ weight_t · h_t. It is a learned weighted average: the network decides which
parts of the frame to listen to. It is not the "attention" of large language
models; it is one layer, applied once.

**Where the 785k numbers are.** LSTM 84.0%, attention 8.4%, classifier 4.3%,
convolutions 3.3%. This model is mostly a recurrent network with a small
convolutional front end.

**Frame length does not change the model.** Every layer works on any length
(convolution, pooling, LSTM over time, attention over time), so the same
785,476 parameters are used for 128, 64 or 32 samples; only the number of LSTM
steps changes (32, 16, 8). That is what makes the frame-length experiment clean.

---

## 4. How a decision is made

1. Frame → normalised to unit power → network → four scores.
2. The decision is the class with the highest score (argmax). There is no
   threshold and no "unknown" option: every frame gets one of the four labels,
   including frames at −20 dB that contain almost no signal.
3. The model is deterministic at test time (dropout off, BatchNorm frozen).
   The same input always gives the same output. Running one frame 10,000 times
   gives the same answer 10,000 times; the meaningful question is 10,000
   *different* frames, which is what the tables count.

That last point is why low-SNR frames all end up in one class: when the input
carries no information, argmax still has to pick something, and it picks
whichever class the network's scores lean towards when there is no evidence.
That is the QPSK "sink" on the 2018 four-class table (84% of −20 dB frames
called QPSK, QPSK precision 25%, i.e. chance).

---

## 5. How it was trained

| | |
|---|---|
| Data | RML2016.10a, the four classes, all 20 SNRs from −20 to +18 dB, 1,000 frames per (class, SNR) cell |
| Split | per (class, SNR) cell: 70% train, 15% validation, 15% test, stratified, different split per seed (`cnn.split`) |
| Loss | cross-entropy |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4, gradient-norm clip 5 |
| Batch | 256 frames |
| Learning-rate schedule | ×0.2 when validation accuracy stops improving (ReduceLROnPlateau, floor 1e-6) |
| Stopping | early stopping after 20 epochs without validation improvement; ceiling 100 epochs |
| Model kept | the epoch with the best **validation** accuracy |
| Test set | touched once, at the end; nothing is chosen on it |
| Repeats | 14 seeds (different split, initialisation and batch order each) |
| Converged? | yes, every seed: best epochs 16–46, every run ended by early stopping, not by the ceiling |
| Hardware | Colab GPU; code in `src/train_backbone.py`, `src/cnn.py` |

Result above 10 dB: 42,000 test decisions (14 seeds × 5 SNRs × 4 classes × 150
frames), 2,421 wrong (94.24%). Nearly all errors are QAM16 ↔ QAM64; the table
is in `README.md`.

---

## 6. What was written with an AI assistant

Three questions: whom was asked, what was asked, what came back.

**The assistant.** Claude Code (Anthropic), a coding assistant that reads the
repository, writes and edits code, runs commands, and writes documentation.
Different Claude model versions were used over time; every commit it produced
or co-wrote is marked in git (`Co-Authored-By: Claude …`).

**What the record shows.** The git history starts on 2026-09-09. Of the commits
since then, the large majority were written by the assistant, either as author
or co-author. That includes most of `src/` as it stands now, the Colab runners,
the table and figure tools, and most of the text in `README.md`
and `CLAUDE.md`.

**Not from the assistant:**

- the dataset (DeepSig's RadioML; its noise is theirs, see `NOISE_2016.md`)
- the architecture (El-Haryqy et al.) and the peer's reproduction the code was
  transcribed from
- the numbers. The assistant writes code; the numbers come from running that
  code on a GPU. Every result file is a measurement that can be re-run, and
  several were re-run and came back bit-identical
- the choice of problem, the direction (e.g. the move to 2016 on 2026-09-22),
  what to report, and running every training job

**What kind of things were asked, and what came back.** Examples from the
record:

| asked (paraphrased) | what came back |
|---|---|
| "Run the four-class table on 2016 in Colab for me" | changes to `train_backbone.py` and a Colab cell to paste; the run was done on Colab, the result file checked and written up |
| "Shorten the frame: 64, then 32 samples" | a `--frame-len` option; three runs (128/64/32) and the error counts 2,421 / 6,443 / 10,431 |
| "Give me the table at every SNR for every modulation" | `tools/snr_tables.py`, a spreadsheet with a confusion matrix per SNR |
| "Read this paper" (PDFs downloaded and supplied by me) | reading notes in `LITERATURE.md`, including two papers judged not citable and why |
| "Why is the faithful build 1.5 points below the paper?" | a convergence check that found the paper's 58 epochs too few (63.21% at convergence vs 61.75%) and a warning now built into every training script |

**How the assistant's work is checked.**

- The faithful build reproduces the paper's number independently (63.21 vs 63.24).
- Results are re-run on different machines; converged runs repeat bit-identically.
- The test set is only read once; early stopping and model choice use validation.
- Claims that did not survive a rerun are withdrawn in the README rather than
  deleted (e.g. "partial whitening beats full whitening").
- I review what comes back and decide what is kept.

**Before 2026-09-09** (to be completed by me): *[who transcribed the peer's
script into `model_zoo.py`, the peer's name, and how the pre-repository work
was done]*.

---

## 7. What this does not claim

- It is not the published architecture (five differences, section 2).
- Both the training data and the "other domain" in the cross-domain work are
  synthetic. No real over-the-air capture has been classified yet.
- 1,000 frames per cell, 128 samples: the QAM16/QAM64 confusion at high SNR is
  a limit of 16 symbols per frame, not of noise (it does not improve from +2 dB
  to +18 dB, and it gets worse at 64 and 32 samples).

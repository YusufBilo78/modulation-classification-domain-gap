# Study notes — the ICRNNA paper talk

**The deck:** `moshe_paper_icrnna.pptx` as sent on 27 October: your son2 with the LSTM slide added as slide 5, 22 slides.

- **Main talk:** slides 1–10.
- **Backup:** slides 11–22, shown only if asked.

**The paper:** El-Haryqy, Kharbouche, Ouamna, Madini, Zouine. *Improved automatic modulation recognition using deep learning with additive attention.* Results in Engineering 26 (2025) 104783. Page numbers below are the paper's own pages.

**How each slide card is laid out:**

- **On the slide:** every piece of text on the slide, so nothing on it is a surprise.
- **What it means:** each term and number, one at a time: what it is, and what it is not.
- **In the paper:** where it comes from.
- **If asked:** likely questions, each with a short answer to say aloud.
- **Say it in 30 seconds:** the speaker note, ready to rehearse.

**Where the numbers come from:**

- **The paper:** cited by table, figure or page.
- **The authors' code:** released on GitHub on 8 August 2026 (`OFFICIAL_CODE.md`).
- **Our runs:** the result files in the repository.

---

## The talk in two minutes

**The story:**

1. **The task.** A neural network looks at 128 samples of a radio signal and names which of 11 modulation types sent it. On the standard dataset, RML2016.10a, the paper reports 63.24% correct, the best of the 12 models it compares.
2. **The model.** It works in four steps: short filters (CNN), reading in order (bidirectional LSTM), a weighted average (attention), and a decision (dense layers). Step 2 holds 83% of the numbers and does most of the work.
3. **The training.** It was trained on 70% of the data, stopped by itself after 58 passes, and tested once on 15% that it never saw.
4. **The result.** Accuracy goes from chance at −20 dB to about 93% from 0 dB up.
5. **Where it fails.** At 0 dB it still confuses WBFM with AM-DSB, QAM16 with QAM64, and sometimes 8PSK with QPSK.
6. **How much better it really is.** Trained the same way as four older models, it is first, but only by about one point.

**Ten numbers to know cold:**

| number | what it is |
|---|---|
| 63.24% | the paper's overall accuracy on RML2016.10a (Table 3) |
| 62.73% | the runner-up in their table, CNN-BiLSTM-DNN: half a point behind |
| 2 × 128 | the input: 128 samples of I and 128 of Q |
| 16 | symbols in one frame (8 samples per symbol) |
| 220,000 | frames in the dataset: 11 types × 20 SNR labels × 1,000 |
| 794,188 | numbers in the model (their code's count); 83% in the LSTM |
| 58 | the epoch where their training stopped by itself |
| 9% / ~93% | accuracy at −20 dB (chance) / from 0 dB up |
| 57.9% | WBFM frames called AM-DSB at 0 dB |
| 63.21% | our rebuild from the paper's text, one run, trained to epoch 107 |

---

## Glossary

Every technical word on the slides, with what it is and, where people get it wrong, what it is not.

**Signals**

- **Modulation** — How information (bits or audio) is put onto a carrier wave: by changing its amplitude, phase or frequency.
- **Modulation classification (AMC) / recognition (AMR)** — Naming which modulation a received signal uses, without being told. *Not* detection (is a signal there at all?) and *not* demodulation (recovering the bits).
- **Classifier** — A system that assigns one label out of a fixed list. This one always answers with one of 11 types; it cannot say "none".
- **Carrier** — The radio wave at a fixed frequency that the modulation changes.
- **Baseband / complex baseband** — The signal after the receiver has removed the carrier frequency. What remains is the modulation itself, as a stream of complex numbers.
- **I and Q (in-phase and quadrature)** — The two outputs of down-conversion, one from mixing with a cosine and one with a sine. Together they form one complex sample, I + jQ.
  - Amplitude = √(I² + Q²).
  - Phase = atan2(Q, I).
- **Sample** — One complex measurement at one instant. In this dataset, 200,000 samples per second.
- **Symbol** — One transmitted point of the constellation, held for a fixed time. Here a symbol lasts 8 samples.
- **Constellation** — The set of points a digital modulation can send, drawn on the I–Q plane.
  - QPSK: 4 points on a circle.
  - QAM16: a 4 × 4 grid.
- **Frame** — One example in the dataset: 128 consecutive samples, i.e. 16 symbols or 0.64 ms.
- **SNR (signal-to-noise ratio)** — Signal power divided by noise power.
- **dB (decibel)** — 10 × log10 of a power ratio.
  - 0 dB = equal power.
  - +10 dB = 10 times.
  - +20 dB = 100 times.
  - −10 dB = one tenth.
  - +3 dB ≈ twice.
- **SNR label** — The number the dataset attaches to each frame (−20 … +18 dB). It is *not* a measured SNR: in this file the real SNR differs by type at the same label.
- **AWGN** — Additive white Gaussian noise: an independent random number added to every sample, with equal power at all frequencies.
- **Fading** — The received strength rising and falling over time because of the channel. In this dataset it swings by about 5.7 dB over a run.
- **Multipath** — Several delayed copies of the signal adding up. Here there are three paths.
- **Frequency offset** — The receiver's frequency not exactly matching the transmitter's, so the constellation slowly rotates.
- **Radio channel vs. CNN channel** — Two different meanings of "channel".
  - A radio *channel* is the path between transmitter and receiver.
  - In a CNN, *channels* are the number of filter outputs at each time step (64, then 128).

**The 11 types**

- **BPSK** — Phase takes 2 values (0°, 180°). 1 bit per symbol.
- **QPSK** — Phase takes 4 values, 90° apart. 2 bits.
- **8PSK** — Phase takes 8 values, 45° apart. 3 bits.
- **PAM4** — Amplitude takes 4 levels on one axis. 2 bits.
- **QAM16** — 16 points on a 4 × 4 grid, amplitude and phase both change. 4 bits.
- **QAM64** — 64 points on an 8 × 8 grid. 6 bits.
- **GFSK** — Frequency shifts driven by bits, smoothed by a Gaussian filter (as in Bluetooth). Digital.
- **CPFSK** — Frequency shifts driven by bits, with no phase jumps. Digital.
- **WBFM** — Wideband FM: audio changes the frequency (FM radio). Analog.
- **AM-DSB** — Audio changes the amplitude; both sidebands are sent (classic AM). Analog.
- **AM-SSB** — The same, but only one sideband is sent, so it uses half the bandwidth. Analog.
- **Digital vs. analog** — What the *message* is: bits (digital) or audio (analog). All 11 are stored as sampled numbers either way.

**The model**

- **Neural network** — A function built from layers. Its behaviour is set by many adjustable numbers (weights), found by training on examples with known answers.
- **Parameters / weights / "numbers learned"** — Those adjustable numbers. The model has 794,188 by Keras's count.
  - 792,396 of them are trained.
  - The other 1,792 are running averages kept by the normalization layers.
- **Layer** — One processing stage, e.g. a convolution or an LSTM.
- **Convolution (1-D)** — A short filter slid along the signal, multiplying and adding at each position. Like an FIR filter whose taps are learned instead of designed.
- **Filter / kernel / taps** — One such sliding filter. "3 taps" means it looks at 3 consecutive samples.
- **ReLU** — After each filter: negative values become 0, positive ones pass. This makes the network non-linear.
- **Max pooling** — Keeps the larger of each pair of neighbouring values, so the sequence becomes half as long (128 → 64).
- **CNN** — Convolutional neural network: layers of convolutions.
- **RNN / recurrent** — A layer that reads a sequence step by step and carries a state from one step to the next.
- **LSTM (long short-term memory)** — A recurrent layer whose state is controlled by three learned gates.
  - Forget: how much of the old state to keep.
  - Input: what to add from the new step.
  - Output: what to pass on.
  - In signal-processing terms: a learned, non-linear recursive filter.
- **Bidirectional (BiLSTM)** — Two LSTMs, one reading forwards and one backwards, with their outputs side by side.
- **Hidden state / memory** — The LSTM's state vector: 128 numbers per direction.
- **Attention (additive)** — Gives each time step a score from a small network, turns the scores into weights that add up to 1, and averages the steps with those weights. "Additive" because the score adds two transformed vectors inside a tanh. *Not* the transformer attention of ChatGPT.
- **Softmax** — Turns a list of scores into positive numbers that add up to 1: exponentiate each, divide by the sum.
- **Dense (fully connected) layer** — Every input connected to every output: a matrix multiplication plus an offset.
- **Confidence score / probability** — The softmax outputs of the last layer, one per type. The largest is the answer. *Not* checked as a calibrated probability.
- **Batch normalization** — Rescales each feature to a standard range during training, which keeps training stable.
- **Layer normalization** — Rescales one vector to zero mean and unit spread. Used on the attention output.
- **Dropout** — During training, a random fraction of the numbers (here 30%) is switched off, so the network cannot rely on any single one. Switched off at test time.
- **L2 regularization** — A penalty on large weights (here 1e-4).

**Training**

- **Training / validation / test set** — Three parts of the data.
  - Training: what the network learns from.
  - Validation: what decides when to stop.
  - Test: kept aside, used once, at the end, for the reported result.
- **Loss / cross-entropy** — The number training makes smaller: minus the log of the probability the network gives to the right answer, averaged over frames.
- **Gradient** — How the loss changes when each weight changes slightly.
- **Gradient descent / optimizer** — The rule for nudging the weights against the gradient to reduce the loss.
- **Adam** — A widely used version of gradient descent that adapts the step size for each weight.
- **Learning rate** — The size of each nudge; here 0.001 to start.
- **Batch** — The frames processed together for one update; here 32.
- **Step** — One update, i.e. one batch. Here 154,000 / 32 ≈ 4,813 steps per epoch.
- **Epoch / pass** — One pass through all the training frames.
- **Validation loss** — The loss measured on the validation set after each epoch.
- **Early stopping** — Stop when the validation loss has not improved for a set number of epochs (here 6), and go back to the best weights.
- **ReduceLROnPlateau** — Cut the learning rate when the validation loss stalls (here × 0.2 after 5 flat epochs).
- **Patience** — How many flat epochs a rule waits before acting.
- **Gradient clipping** — Capping the size of each update so one bad batch cannot throw training off. Their code uses `clipnorm=1.0`.
- **Seed** — The number that fixes the randomness (starting weights, data order). A different seed gives a slightly different result.
- **Split** — How the frames are divided into the three sets.
- **Overfitting** — Learning the training frames themselves rather than the pattern, so new frames do much worse. Not a problem here: training and validation accuracy are within a point.

**Results**

- **Accuracy** — Correct decisions divided by all decisions.
- **Chance level** — The accuracy of guessing: 1 in 11 ≈ 9.1%.
- **Recall (of a type)** — Of the frames that really are X, the fraction called X.
- **Precision (of a type)** — Of the frames called X, the fraction that really are X.
- **F1 score** — The harmonic mean of precision and recall.
- **Confusion matrix** — Rows: the true type. Columns: what the classifier said. The diagonal is correct. Each row's percentages are that type's recall.
- **Ablation** — Removing parts of a model one at a time to see what each contributes.
- **Benchmark** — A shared dataset everyone reports on, so numbers can be compared.
- **State of the art** — The best published result on a benchmark. The paper uses it to mean "best of the twelve we compare".
- **FLOPs** — Floating-point operations needed for one frame: the computing cost.

---

## Slide 1 — Title

**On the slide**

- "PAPER PRESENTATION"
- "ICRNNA"
- "Improved automatic modulation recognition using deep learning with additive attention"
- "El-Haryqy, Kharbouche, Ouamna, Madini, Zouine · Ibn Tofail University, Morocco · Results in Engineering 26 (2025) 104783"
- Four chips: "Look · Read in order · Weigh · Decide"
- "Yusuf Bilal Cetinkaya · Moshe Kam's research group · NJIT · October 2026"

**What it means**

- **ICRNNA** — Improved Convolutional Recurrent Neural Network with Attention. Each letter is a part of the model:
  - C: the convolution step (CNN).
  - R: the recurrent step (LSTM).
  - A: the attention step.
- **"Improved"** — Improved over a plain CNN + RNN with attention. Two kinds of additions:
  - three small additions to the attention (a learned scale, dropout, a normalization);
  - regularization in every layer (dropout, batch normalization, L2).
  - *Not* a new idea: CNN + LSTM + attention existed before; the paper's own reference [17], CNN-BiLSTM-DNN, uses the same ingredients.
- **"Automatic modulation recognition"** — Naming the modulation type from the received samples alone. The same task we call AMC. *Not* detection and *not* demodulation.
- **"Deep learning"** — A many-layer neural network trained on examples. *Not* hand-written rules or hand-designed features.
- **"Additive attention"** — The 2014 attention of Bahdanau et al.: a score from adding two transformed vectors inside a tanh. *Not* transformer attention.
- **Authors and place** — Five authors at Ibn Tofail University, Kenitra, Morocco.
- **"26 (2025) 104783"** — Volume 26, year 2025, article number 104783. The article number is *not* a page number.
- **Results in Engineering** — An open-access Elsevier journal covering all of engineering. *Not* a signal-processing journal. Received December 2024, published online April 2025; licence CC BY-NC 4.0.
- **Look / Read in order / Weigh / Decide** — The four steps of the model in plain words: CNN, LSTM, attention, dense. Explained on slide 4.

**In the paper**

- Title and authors: page 1.
- The acronym and its parts: Section 1, end of page 2.

**If asked**

- *"What does ICRNNA stand for?"* — "Improved Convolutional Recurrent Neural Network with Attention: a convolution, then a recurrent layer, then attention."
- *"Is this detection?"* — "No, classification. A signal is there; the question is which of eleven types it is."
- *"Why this paper?"* — "Our classifier is built on this design, so I should be able to explain it."
- *"Is it a good journal?"* — "It's an open-access Elsevier engineering journal, broad rather than specialised. That's one reason we checked the number ourselves."

**Say it in 30 seconds**

> Today I present the paper our classifier comes from: ICRNNA, from a group in Morocco, published in 2025. I'll keep it simple: what it does, how it works, what it reports, and whether it holds up.

---

## Slide 2 — The paper in one sentence

**On the slide**

- "A neural network looks at 128 samples of a radio signal and says which of 11 modulations sent it."
- "63.24% — right on RML2016.10a: the best of the 12 models they compare"
- "my classifier — is built on this design"

**What it means**

- **"A neural network"** — A function with 794 thousand adjustable numbers: 2 × 128 numbers in, 11 numbers out.
- **"128 samples"** — 128 complex samples, I and Q, taken over 0.64 ms.
  - At 8 samples per symbol, that is **16 symbols**.
  - *Not* a spectrum, an image or hand-made features: only the raw samples.
- **"of a radio signal"** — The complex baseband signal after the receiver has removed the carrier.
- **"says which of 11 modulations"**
  - Eight digital: BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4, GFSK, CPFSK.
  - Three analog: WBFM, AM-DSB, AM-SSB.
  - It always picks one of the eleven; it *cannot* say "none" or "unknown". This is a closed set. So even a frame that is pure noise gets a type name.
- **"63.24%"** — The fraction of test frames it gets right, counted over **all 20 SNR levels together**.
  - *Not* 63% everywhere: about 9% at −20 dB, which is chance, and about 93% from 0 dB up.
- **"RML2016.10a"** — The public dataset of O'Shea and West, generated in GNU Radio in 2016.
- **"the best of the 12 models they compare"** — First in their Table 3.
  - The runner-up, CNN-BiLSTM-DNN, has 62.73: half a point behind, from one run.
  - *Not* "best in the world": newer models they cite, such as IQFormer, are not in their table.
- **"my classifier is built on this design"** — The classifier in my project uses this architecture.
  - *Not* their code and *not* their trained network: it is a re-implementation with five small differences, and their weights were never released.

**In the paper**

- 63.24% and the 12-model comparison: Abstract and Table 3, page 6.
- The 11 types: Table 1, page 3.

**If asked**

- *"What is a sample? A symbol?"* — "A sample is one complex measurement, I plus jQ. A symbol is one transmitted point of the constellation; it lasts 8 samples, so 128 samples are 16 symbols."
- *"Can it say none of these?"* — "No. It always picks one of the eleven, so it names a type even for frames that are only noise."
- *"Is it 63% at every SNR?"* — "No. It's the average over all twenty levels: 9% at −20 dB, which is chance, and about 93% from 0 dB up."
- *"Is your classifier the same as theirs?"* — "Same design, not their code. Ours differs in five small places. A separate rebuild that follows the paper exactly matches their number: 63.21 against 63.24."

**Say it in 30 seconds**

> The paper in one sentence: a neural network looks at 128 samples of a radio signal and says which of 11 modulations sent it. They report 63.24 percent right on this dataset, the best of the twelve models they compare. My own classifier is built on this design.

---

## Slide 3 — The data: what the model sees

**On the slide**

- "11 — modulation types: 8 digital, 3 analog"
- "20 — SNR labels, −20 to 18 dB, in steps of 2 dB"
- "220,000 — frames: 1,000 per type at each SNR"
- A picture: "one frame: I (orange), Q (blue)"
- "2 × 128 — Each frame: 128 samples of I and 128 of Q. That array is the only input."
- "RML2016.10a, the same dataset we use."

**What it means**

- **"8 digital, 3 analog"** — Digital types carry bits: each symbol is one point from a fixed set. Analog types carry audio: WBFM, AM-DSB and AM-SSB, made from voice recordings.
  - GFSK and CPFSK count as digital because their frequency shifts are driven by bits.
  - "Digital/analog" describes the *message*, *not* how the data is stored.

**SNR** — Signal power over noise power, in dB.

| label | signal power / noise power |
|---|---|
| +20 dB | 100 |
| +10 dB | 10 |
| 0 dB | equal |
| −10 dB | one tenth |
| −20 dB | one hundredth |

**"SNR labels"** — The word "labels" matters. The generator sets the noise *amplitude* from the label, so the real SNR rises about 1.9 dB per dB of label. The classes also differ at the same label. Measured in the file at label 0:

| type | measured SNR at label 0 |
|---|---|
| BPSK, QPSK, 8PSK | ~5 dB |
| QAM16 | ~14 dB |
| QAM64 | ~17 dB |

So label 0 is *not* "signal and noise equal".

- **"in steps of 2 dB"** — The labels are −20, −18, …, +16, +18: twenty levels.
- **What else is in the channel** — At every level, including +18 dB:
  - a slow frequency offset;
  - clock drift;
  - Rician fading, swinging about 5.7 dB over a run;
  - three-path multipath;
  - the Gaussian noise.
- **"220,000 frames"** — 11 types × 20 labels × 1,000 frames. A frame is one example: 128 consecutive samples cut from a long stream at a random point. The data is balanced, so chance is exactly 1 in 11.
- **The picture** — I (orange) and Q (blue) over 128 samples. The dots are the 16 symbols, one every 8 samples.
  - It is *not* a frame from the dataset: we drew a clean QAM16 frame with the same pulse shaping, to show the shape. Real frames also carry noise, fading and frequency offset.
- **"2 × 128"** — 2 rows (I, Q) × 128 columns (time): 256 numbers.
  - "The only input" means the network sees *nothing else*: not the SNR label, not a spectrum.
  - Their code first normalizes each of the 128 positions by its mean and variance over the whole dataset. That includes the test frames, a small leak.
- **Sample rate** — 200 kHz. A frame lasts 0.64 ms, and the symbol rate is 25 kbaud.
- **"RML2016.10a"** — Made by O'Shea and West (DeepSig) in GNU Radio, public since 2016. The paper's reference [23].
  - "10b" is a larger version: 10 types, without AM-SSB, and 1.2 million frames.
  - Both are simulated, *not* recorded over the air.

**In the paper**

- Dataset: Section 2.2 and Table 1, page 3.
- Equation 1 says received = signal + noise; Section 4 (page 5) adds the frequency offsets, fading and multipath.
- The paper writes "BFSK" in its list; it means GFSK, which Table 1 gets right.

**If asked**

- *"What is SNR in dB?"* — "Ten times the log of signal power over noise power. 0 dB is equal power, 10 dB is ten times, −20 dB means the noise is a hundred times stronger."
- *"What are I and Q?"* — "The two outputs of down-conversion, the cosine and sine branches. Together they are one complex sample, which gives amplitude and phase."
- *"Is 0 dB really 0 dB?"* — "It's the dataset's label. We measured it: at label 0 the PSK types are near 5 dB and QAM16 near 14."
- *"Why are GFSK and CPFSK digital?"* — "Their frequency shifts are driven by bits. Analog here means an audio source."
- *"Is the picture from the dataset?"* — "No, it's a clean QAM16 frame we drew to show the shape."
- *"Does the network see the SNR?"* — "No, only the 2-by-128 array. The label is used afterwards, to sort the results."

**Say it in 30 seconds**

> The data is the same dataset we use, RML2016.10a: eleven modulation types, eight digital and three analog. Twenty noise levels, from minus 20 to plus 18 dB in steps of 2, and 1,000 frames of each type at each level: 220,000 frames. Each frame is 128 samples of I and 128 of Q. That is the only input; the network sees nothing else.

---

## Slide 4 — How it works: four steps

**On the slide**

1. "Look at short pieces — CNN — short filters slide along the signal"
2. "Read them in order — LSTM, both directions — carries a memory from piece to piece" (highlighted)
3. "Weigh the pieces — attention — a weighted average of all the pieces"
4. "Decide — dense layers — 11 confidence scores, one per type; the largest wins"

Below the steps:

- "In: 128 samples (I and Q). Out: 11 scores. 794,188 numbers learned, 83% of them in step 2."
- "Step 2, reading the pieces in order, is the heart of it."

**What it means**

- **Step 1, "short pieces"**
  - Filters 3 samples long slide along the frame. There are 64 filters, then 128.
  - A pooling step then halves the length: 128 time steps become **64 pieces**.
  - Each piece summarizes about 6 samples, *less than one symbol*.
- **Step 2, "read them in order"**
  - An LSTM reads the 64 pieces one after another, carrying a memory of 128 numbers. A second LSTM reads them backwards.
  - After this step, every piece "knows" the whole frame.
  - This is the only step that connects symbols to each other, and a modulation is a pattern *across* symbols.
- **Step 3, "weigh the pieces"**
  - Each of the 64 pieces gets a weight, and the weights add up to 1.
  - The output is the weighted average: one summary vector of 256 numbers for the whole frame.
- **Step 4, "decide"**
  - Two ordinary layers, then 11 outputs. A softmax turns them into 11 positive numbers that add up to 1: the **confidence scores**. The largest is the answer.
  - *Not* calibrated probabilities: nobody checked that 0.9 means right 90% of the time.
- **"794,188 numbers learned"** — The parameter count printed by their code, matching the paper's 0.79 million.
  - Strictly, 792,396 are learned by training; 1,792 are running averages of the normalization layers.

**"83% of them in step 2"** — 659,456 of the 794,188 are in the two bidirectional LSTM layers.

| step | numbers | share |
|---|---|---|
| CNN | 25,920 | 3% |
| LSTM | 659,456 | 83% |
| attention | 66,177 | 8% |
| dense | 42,635 | 5% |

- **"Heart of it"** — Supported by their own experiment (Table 6): the LSTM alone gets 63.13%, the full model 63.24%.

**In the paper**

- Figure 2 (page 4) and Sections 3.1–3.3 (pages 4–5).
- The sizes come from their code.

**If asked**

- *"What do the filters learn?"* — "We haven't looked inside. They act like a learned FIR filter bank; typically they respond to phase jumps and amplitude steps between neighbouring samples."
- *"Why read in both directions?"* — "The whole frame is available at once, so there's no need to be causal. Reading backwards too means every piece knows what comes before and after."
- *"Why is step 2 the heart?"* — "Telling types apart needs the pattern across symbols, and step 1 sees less than one symbol. Their own ablation agrees: the LSTM alone gets 63.1 of the 63.2."
- *"What do you mean by confidence?"* — "The eleven outputs add up to one; the largest is the answer and its size is how sure it is. It's a score, not a checked probability."
- *"Why 83 percent?"* — "Each LSTM layer has four gate matrices of 128 by its input. Two layers, two directions: 659 thousand numbers."

**Say it in 30 seconds**

> How does it work? Four steps. One: it looks at short pieces of the signal, with small filters that slide along it. Two: it reads those pieces in order, forwards and backwards, carrying a memory from piece to piece. Three: it weighs the pieces and takes a weighted average. Four: it gives eleven confidence scores, one per modulation, and the largest wins. Most of the network, 83 percent, is in step two. That is the heart of it.

---

## Slide 5 — Step 2 up close: what is an LSTM?

**On the slide**

- "It reads the 64 pieces one at a time and carries a memory of 128 numbers from piece to piece. At every piece, three gates act like valves:"
- Three cards:
  1. "Keep — forget gate — how much of the old memory to keep: 0 erases it, 1 keeps all of it"
  2. "Write — input gate — how much of the new piece to add to the memory"
  3. "Show — output gate — how much of the memory to pass on"
- A dark card: "Signal-processing view — memory(t) = keep × memory(t−1) + write × new — A leaky integrator whose pole the data sets at every step: near 1 it holds, near 0 it resets."
- "Each gate is a small learned layer giving numbers between 0 and 1. In ICRNNA: read forwards and backwards, 2 layers, 659,456 numbers (83% of the model)."
- "It is the only step that reads across symbols, which is where a modulation shows."

**What it means**

- **LSTM** — Long Short-Term Memory (Hochreiter and Schmidhuber, 1997). A recurrent layer: it reads a sequence step by step and carries a state, the memory, from one step to the next.
- **"The 64 pieces"** — The CNN's output: 64 time steps, each a summary of about 6 samples, less than one symbol.
- **"A memory of 128 numbers"** — The cell state. It is updated at every piece and is what lets the layer collect evidence across the whole frame.
- **"Three gates act like valves"** — Each gate is a small learned layer followed by a sigmoid, so it outputs numbers between 0 and 1, one per memory slot. It looks at the new piece and the previous output, and it is recomputed at every piece.
  - **Keep (forget gate)** — Multiplies the old memory. 1 keeps it, 0 erases it.
  - **Write (input gate)** — Decides how much of the new information goes into the memory.
  - **Show (output gate)** — Decides how much of the memory is passed on as the output at this piece.
- **"memory(t) = keep × memory(t−1) + write × new"** — The heart of the LSTM. The full form is c(t) = f(t) · c(t−1) + i(t) · g(t), element by element, where f is the forget gate, i the input gate and g the new candidate information.
- **"A leaky integrator whose pole the data sets"** — In filter terms this is a one-pole recursive filter, y(t) = a · y(t−1) + b · x(t), except that a (the keep gate) is not fixed: the data chooses it at every step.
  - Near 1, the memory holds what it has.
  - Near 0, it resets.
  - A plain recurrent network has a fixed "pole", so its memory fades exponentially, and during training so do its gradients. That is why plain RNNs cannot learn long dependencies, and the LSTM can.
- **"Read forwards and backwards"** — Bidirectional: a second LSTM reads the 64 pieces from the end. Outputs are side by side, 128 + 128 = 256 per piece.
- **"2 layers"** — A second bidirectional LSTM reads the first one's outputs.
- **"659,456 numbers (83%)"** — Each gate, and the candidate, has an input matrix, a recurrent matrix and a bias:
  - first layer: 2 directions × 4 × (128×128 + 128×128 + 128) = 263,168;
  - second layer: 2 × 4 × (128×256 + 128×128 + 128) = 394,240;
  - plus batch normalization, 2,048.
- **"The only step that reads across symbols"** — The CNN sees less than one symbol; attention only averages. A modulation shows in the pattern across symbols: how many phases recur, whether the amplitude changes. So this is where the type is recognised. Their ablation agrees: the LSTM alone gets 63.13%, the full model 63.24%.
- **What it is not**
  - Not a perfect memory: it keeps a learned 128-number summary, not the samples.
  - Not attention: attention averages all pieces at once; the LSTM reads them in order.
  - Not a transformer, and not a Kalman filter: there is no signal or noise model, everything is learned.
  - Not linear: the sigmoids and tanh make it a nonlinear filter.
  - We have not looked inside it to see what each memory slot encodes; "collecting evidence across symbols" is the plausible picture, not a measurement.

**In the paper**

- Section 3.2, pages 4–5, and Figure 2 (Part B).
- The gate equations are not in the paper; they are the standard LSTM.

**If asked**

- *"What is an LSTM?"* — "A recurrent layer: it reads the 64 pieces in order and keeps a memory of 128 numbers. Three learned gates decide, at every piece, how much of the memory to keep, how much new input to write, and how much to pass on."
- *"In signal-processing terms?"* — "A learned nonlinear recursive filter. The memory update is a leaky integrator whose pole the data sets at every step: near one it holds, near zero it resets."
- *"Why not a plain recurrent network?"* — "Its memory fades exponentially, and so do its gradients in training, so it can't learn long dependencies. The LSTM's gated, additive update fixes that."
- *"How are the gates set?"* — "Each is a small learned layer with a sigmoid, so it gives values between zero and one. It looks at the new piece and the previous output, so it changes at every step."
- *"Why is it 83% of the model?"* — "Four gate matrices of 128 by its input, in two directions and two layers: 659 thousand numbers."

**Say it in 30 seconds**

> What is an LSTM? It is the layer in step two. It reads the 64 pieces one at a time and carries a memory of 128 numbers from piece to piece. At every piece, three gates act like valves. The forget gate decides how much of the old memory to keep: zero erases it, one keeps all of it. The input gate decides how much of the new piece to write into the memory. The output gate decides how much of the memory to pass on. Each gate is a small learned layer, set fresh at every piece. In signal-processing terms, the memory update is a leaky integrator whose pole the data sets at every step: near one it holds, near zero it resets. In ICRNNA it reads the frame forwards and backwards, in two layers, and holds 83 percent of the model's numbers. It matters because a modulation is a pattern across symbols, and this is the only step that reads across them.

---

## Slide 6 — How they trained and tested it

**On the slide**

- "70% of the dataset: training — 154,000 — frames with the right answer: it learns from these"
- "15%: validation — 33,000 — frames to decide when to stop learning"
- "15%: testing the classifier — 33,000 — frames kept aside: every result is counted on these"
- "It stopped by itself after 58 passes over the data, when it stopped improving. Done once."
- "The test frames are never used for learning."

**What it means**

- **The percentages** — Of the whole dataset of 220,000 frames.
- **Training set (154,000)**
  - The network sees each frame with its correct type. Its numbers are nudged after every batch of 32 to make the right answer more likely.
  - Roughly 700 of the 1,000 frames of each type and label.
- **Validation set (33,000)**
  - After each pass, the error is measured here. It decides when to cut the learning rate and when to stop.
  - It is *not* used to learn the weights.
- **Test set (33,000)** — Never seen during training or stopping. All reported results are counted here, once, at the end.
  - Why separate: if the stopping decision used the test frames, the reported number would be chosen on itself and come out optimistic.
- **"58 passes"** — 58 epochs: 58 times through all 154,000 training frames, about 280,000 updates in all.
  - *Not* a number they chose. Their code allows up to 100 and stops when the validation loss has not improved for 6 epochs; for them, that happened at 58.
- **"Done once"** — One random split, one training run, one number. No spread is reported.
  - Their code draws the split at random, not balanced per type, and the split it draws cannot be reproduced from one run to the next.

**In the paper**

- 70/15/15: Section 4.1, page 5.
- 58 epochs: Table 2, page 6.
- The stopping rule: their code.

**If asked**

- *"How does it learn?"* — "It sees a frame, gives eleven scores, and the scores are compared with the right answer. All its numbers are nudged a little to make the right answer more likely. That's repeated over all training frames, 58 times."
- *"Why keep a validation set?"* — "Stopping and learning-rate decisions need data the network isn't trained on. If you used the test set for that, you'd be choosing the number you report."
- *"How do you know it's not memorizing?"* — "Training accuracy 64.2% and validation 63.6% are almost the same. A memorizing model would be much better on its own training frames."
- *"Is the split balanced?"* — "No, random. Each type at each level gets about 150 test frames, not exactly 150."

**Say it in 30 seconds**

> How was it trained and tested? The dataset is split three ways. 70 percent of it, 154,000 frames with the right answer, to learn from. 15 percent, 33,000 frames, to decide when to stop learning. And the last 15 percent, 33,000 frames, kept aside to test the classifier: they are never used for learning, and every result is counted on them. It stopped by itself after 58 passes over the data, and they did it once.

---

## Slide 7 — Their main table: first by half a point

**On the slide**

- A bar chart of the 12 models' accuracy on RML2016.10a, from their Table 3.
- "63.24 vs 62.73: the runner-up is half a point behind."

| model | accuracy | what it is |
|---|---|---|
| ICRNNA (2025) | 63.24 | this paper |
| CNN-BiLSTM-DNN (2024) | 62.73 | CNN + bidirectional LSTM + attention; the closest relative |
| MCLDNN (2020) | 61.42 | separate convolution branches for I, Q and I/Q together, then an LSTM |
| MAMCA (2024) | 60.79 | based on Mamba, a state-space sequence model |
| PET-CGDNN (2021) | 60.58 | estimates and removes a phase, then CNN + GRU; very small (72 thousand numbers) |
| DAE (2022) | 60.22 | an LSTM auto-encoder |
| ResNet (2017) | 57.10 | a residual CNN |
| CVCNN | 56.37 | a complex-valued CNN |
| IC-AMCNet (2020) | 56.30 | a CNN |
| CGDNet (2021) | 56.21 | a shallow CNN + GRU |
| MCNet | 56.20 | a CNN-only model |
| Transformer | 55.34 | an attention-only model |

**What it means**

- **Accuracy** — Over the whole test set, all SNR levels together.
- **"First by half a point"** — 0.51 points ahead, from one run. Is that a real lead?
  - The test set alone moves a run by about ±0.3 points.
  - Retraining with another seed moves our models by 0.5 to 1.1 points.
  - So first and second *cannot* be told apart. The bigger gaps, 6–8 points to the CNN-only models and the Transformer, are real.
- **The pattern** — Mostly, models that read the sequence in order (LSTM, GRU, Mamba) are near the top, and pure CNNs and the Transformer are at the bottom. CGDNet, a small CNN + GRU, is the exception.
- **Where the other numbers come from** — Not stated. Six of the models also appear with confusion matrices in their Figure 11, so they must have run those. The other five (MAMCA, PET-CGDNN, DAE, CVCNN, Transformer) appear only in this table and are probably taken from other papers.
- **The axis starts at 0** — That is deliberate: it shows how small the differences really are.

**In the paper**

- Table 3, page 6.

**If asked**

- *"Did they retrain the others?"* — "The paper doesn't say. Six they clearly ran; five are probably copied from other papers."
- *"Is half a point significant?"* — "Not from one run. The test set alone gives about ±0.3 points, and seeds move a model by half a point to a point."
- *"Why is the Transformer last?"* — "They guess transformers miss local time features, but they don't test it."
- *"What is MCLDNN?"* — "A 2020 model with separate convolution branches for I, Q and both together, followed by an LSTM. We trained it ourselves: 61.4 percent."

**Say it in 30 seconds**

> Their main result. ICRNNA is first, at 63.24 percent. The next model is half a point behind. The well-known older ones are two to three points behind.

---

## Slide 8 — Where it still fails, at 0 dB (their Figure 11)

**On the slide**

- Their Figure 11 (ICRNNA panel): "SNR = 0 dB, Acc = 92.21%".
- "Test frames labelled 0 dB, about 150 per type (≈1,650). Row: the true type. Column: the classifier's answer. Each row adds up to 100%."
- "57.9% — of WBFM called AM-DSB: about 85 of ~145 frames"
- "~6% — QAM16 ↔ QAM64, each way: about 9 to 10 frames"
- "91.9% — of 8PSK right: 148 of 161 frames. The rest: 7 → QPSK, 3 → QAM16, 2 → AM-SSB, 1 → QAM64"
- "The other seven types are 96 to 100% right at this level."

**What it means**

- **How to read it** — Each row is one true type and shows where its frames went. The diagonal is correct. Row percentages are recall.
- **What went in** — The test frames with label 0 dB: about 150 per type, about 1,650 in all. The split is random, so rows differ a little.
- **Where the frame counts come from** — The paper prints only percentages. 161 is the only 8PSK row size that reproduces all five printed values, which gives 148 / 7 / 3 / 2 / 1. The WBFM and QAM rows fit several sizes, so the slide says "about".
- **WBFM → AM-DSB (57.9%)**
  - The known explanation: both analog types are made from voice recordings. In silences, both become a bare carrier, which nothing can tell apart.
  - The confusion goes one way: AM-DSB itself is 98% right.
  - We have *not* measured this ourselves; it is the explanation given in the literature.
- **QAM16 ↔ QAM64 (5.7% and 6.1%)** — A frame has only 16 symbols, too few to tell a 16-point grid from a 64-point one every time. In our own shortened-frame runs, QAM16 recall at high SNR falls from 87% (128 samples) to 66% (64) to 46% (32).
- **8PSK: 148 of 161 right** — The errors:
  - **7 → QPSK.** QPSK is the nearest neighbour: same constant amplitude, and its 4 phases are every other one of 8PSK's 8.
    - 8PSK points are 45° apart, so at equal power neighbouring points are 0.77 apart against 1.41 for QPSK. 8PSK needs about 5.3 dB more SNR to be separated as well.
    - At this label we measured about 4.7 dB for 8PSK, so the 45° steps blur.
  - **3 → QAM16, 1 → QAM64.** Noise and fading make the amplitude vary, so a single ring can look like several.
  - **2 → AM-SSB.** The type the classifier falls back on when it cannot see a signal; probably frames caught in a fading dip.
  - The reverse is rare: only 0.6% of QPSK is called 8PSK. Sixteen symbols all on 4 of 8 phases is strong evidence for QPSK.
- **"The other seven types"** — AM-DSB 98.0, AM-SSB 96.1, BPSK 99.3, CPFSK 100, GFSK 98.6, PAM4 99.2, QPSK 99.4.

**In the paper**

- Figure 11, page 11; text on pages 9–11.
- The text writes "64-QAM 93.%" (a missing digit; the figure shows 93.3).

**If asked**

- *"How many signals went in?"* — "The test frames labelled 0 dB: about 150 of each type, roughly 1,650. For 8PSK it was 161."
- *"Explain the 8PSK errors."* — "Of 161 8PSK frames, 148 are right. Seven are called QPSK, its nearest neighbour: same amplitude, and QPSK's phases are every other one of 8PSK's. At this noise level the 45-degree steps blur. A few go to the QAM types because fading makes the amplitude vary, and two to AM-SSB, the type it falls back on when it can't see a signal."
- *"Why is 8PSK the weakest digital type?"* — "Its points are closest together: 0.77 apart against 1.41 for QPSK at equal power, so it needs about 5 dB more SNR."
- *"Why is WBFM called AM-DSB and not the reverse?"* — "In the silences of the audio, WBFM becomes a bare carrier, and the classifier calls that AM-DSB. AM-DSB itself is 98 percent right."
- *"How do you know 161?"* — "The paper prints percentages to one decimal; 161 is the only row size that gives all five of them."
- *"Do other models do better on WBFM?"* — "In their figure, yes: MCLDNN 63 percent, ResNet 58, against ICRNNA's 41. No model solves it."

**Say it in 30 seconds**

> Where does it still make mistakes, at 0 dB? This is their figure. About 150 test frames of each type, labelled 0 dB, went in; each row is one true type, and the columns are what the classifier said. Three weak spots. WBFM is called AM-DSB more than half the time: about 85 of some 145 frames. QAM16 and QAM64 are mixed up, about 6 percent each way. And 8PSK: of 161 frames, 148 are right; 7 are called QPSK, its nearest neighbour, and a few go to the QAM types. The other seven types are 96 to 100 percent right.

---

## Slide 9 — How much better is it, really?

**On the slide**

- "We trained five published models the same way: same data, same split, 3 repeats each."

| model | right, all noise levels | right, 10 dB and above |
|---|---|---|
| ICRNNA (2025) | 62.2% | 91.5% |
| LSTM2 (2018) | 61.4% | 91.5% |
| MCLDNN (2020) | 61.4% | 91.1% |
| PET-CGDNN (2021) | 61.0% | 90.6% |
| VT-CNN2 (2016) | 56.1% | 82.0% |

- "First, but only by about one point."

**What it means**

- **"The same way"** — One recipe for all five models:
  - all 220,000 frames;
  - one 70/15/15 split, balanced per type and label;
  - three seeds each;
  - each trained until it stopped improving.

  This is *our* experiment, not the paper's.

**The five models**

| model | what it is | numbers |
|---|---|---|
| VT-CNN2 (O'Shea et al., 2016) | the first deep model on this dataset; convolution only | 1.59 million |
| LSTM2 (Rajendran et al., 2018) | two LSTM layers reading amplitude and phase instead of I/Q | 201 thousand |
| MCLDNN (Xu et al., 2020) | separate convolution branches, then an LSTM | 406 thousand |
| PET-CGDNN (Zhang et al., 2021) | estimates and removes a phase, then CNN + GRU | 72 thousand |
| ICRNNA | our working version | 786 thousand |

The four older models were ported from a published benchmark's code.

- **"Right, all noise levels"** — Overall accuracy, like the paper's 63.24.
- **"Right, 10 dB and above"** — Accuracy on clean signals only.
- **"First, but only by about one point"**
  - ICRNNA leads in each of the three seeds, but the top four are within 1.25 points.
  - Above 10 dB the top four are all near 91%.
  - Its real edge is at −6 to −2 dB.
- **Why ICRNNA shows 62.2 here, not 63.2** — This row is our working version: slightly different from the paper's model, with a different recipe (batches of 256) and our split. It is fair among these five, but it is *not* the reproduction. Our faithful rebuild gave 63.21.
- **Agreement with the papers** — For LSTM2, MCLDNN and PET-CGDNN, our numbers are within one point of the published ones, in both directions. The parameter counts of all four match their sources to the digit.

**In the paper**

- Not in the paper. This is our check (`literature_2016.md`).

**If asked**

- *"Why is ICRNNA 62.2 here and 63.2 elsewhere?"* — "This row is our working version, trained with one recipe shared by all five so they can be compared fairly. The rebuild that follows the paper exactly gives 63.21."
- *"Is one point significant?"* — "ICRNNA was first in each seed, so the order is probably real, but in one seed LSTM2 was only a tenth of a point behind. It's a small gain."
- *"Why is VT-CNN2 so far behind?"* — "It has no step that reads in order. Every model with a recurrent layer closes that six-point gap."
- *"Why are they all near 91% above 10 dB?"* — "That's the ceiling of the data: WBFM against AM-DSB and QAM16 against QAM64. No model fixes those."

**Say it in 30 seconds**

> How much better is it, really? We trained five published models the same way: same data, same split, three repeats each. ICRNNA is first, but only by about one point. At 10 dB and above, the best four are all about 91 percent.

---

## Slide 10 — In short

**On the slide**

1. "A neural network in four steps; step 2, reading in order, does most of the work."
2. "It is about one point better than older models, not a big jump."

"For us: a solid classifier to build on."

**What it means**

- **Point 1** — Slides 4 and 5: the LSTM holds 83% of the numbers, and their ablation shows it alone gets almost all the accuracy.
- **Point 2** — Slide 9: trained the same way as four older models, it is first by about one point.
- **"A solid classifier to build on"** — Its result can be rebuilt and it is at the top of the field, so it is a sound base for my domain-gap work.

**If asked**

- *"Is their 63.24 real?"* — "Yes. I rebuilt the model from the paper's text alone and got 63.21, in one run trained to the end. Their own code, run three times, gives 62.56 on average. That's backup slides 19 and 20."
- *"What's next for you?"* — "Using this classifier for my main question: how much accuracy is lost when the test signals come from a different source than the training ones. The paper never tests that."
- *"Would it work on real signals?"* — "Unknown. The paper only tests on this simulated dataset, and that's exactly my research question."

**Say it in 30 seconds**

> To sum up. It is a neural network in four steps, and step two, reading in order, does most of the work. And it is about one point better than older models, not a big jump. For us, it is a solid classifier to build on. Thank you. I have more detail on each step if you'd like.

---

## Slide 11 — Backup

**On the slide**

- "Details, if asked."
- "Each step of the model, the training recipe, their code, and what we checked in their tables."

**What it means**

Show these slides only if a question leads there:

| slides | topic |
|---|---|
| 12 | the model with its sizes |
| 13–16 | each block |
| 17 | the training recipe |
| 18 | what their code tells us |
| 19 | "58 epochs" |
| 20 | our rebuild |
| 21 | claims their tables don't support |
| 22 | what the paper doesn't say |

**Say it in 30 seconds**

> Backup slides with the details, if you want to go deeper.

---

## Slide 12 — The model at a glance

**On the slide**

- A flow: "frame 2 × 128 → CNN 64 × 128 → BiLSTM 64 × 256 → Attention 256 → Dense 128 → 64 → output 11"
- "size of what comes out"
- "One frame in, eleven probabilities out. The largest is the answer."
- "Redrawn from their Fig. 2; sizes from their code."

**What it means**

- **Reading the sizes** — "time steps × numbers per step".
  - **2 × 128 (frame)** — I and Q over 128 samples. Their code turns it to 128 × 2, time first.
  - **64 × 128 (after the CNN)** — 64 time steps (after pooling), each described by 128 filter outputs.
  - **64 × 256 (after the BiLSTM)** — Still 64 steps, now 128 numbers from each direction.
  - **256 (after attention)** — The 64 steps averaged into one vector.
  - **128 → 64 (dense)** — Two ordinary layers shrinking the summary.
  - **11 (output)** — One probability per type.
- **"Eleven probabilities"** — The softmax outputs: positive and adding up to 1. The same thing as the "confidence scores" of slide 4.

**In the paper**

- Figure 2, page 4. The paper does not print the sizes; we read them from their code.

**If asked**

- *"Why does the length go from 128 to 64?"* — "One pooling step keeps the larger of each pair of neighbours. It halves the work for the LSTM."

**Say it in 30 seconds**

> Here is the whole model. A frame of 2 by 128 goes in. Four blocks process it. Eleven probabilities come out, one per modulation, and the largest is the answer. Below each block I wrote the size of what comes out of it, so you can follow one frame through. I'll now take the blocks one at a time.

---

## Slide 13 — Block 1: short learned filters

**On the slide**

1. "64 filters, 3 taps — 2 × 128 → 128 × 64"
2. "128 filters, 3 taps — → 128 × 128"
3. "pool: larger of each pair — → 64 × 128"

Also on the slide:

- "samples →" and "one output per sample, per filter" (an illustration).
- "Like an FIR filter whose taps are learned instead of designed."
- "Each of the 64 steps sums up about 6 samples: less than one symbol."

**What it means**

- **"64 filters, 3 taps"**
  - Each filter looks at 3 consecutive samples of both I and Q: 3 × 2 = 6 weights plus one offset.
  - Sliding it along gives one output per sample, so 64 filters give 128 × 64.
  - This layer has 448 numbers.
- **"128 filters, 3 taps"** — The same again on the 64 outputs: 3 × 64 × 128 + 128 = 24,704 numbers. The result is 128 × 128.
- **After each filter** — ReLU (negatives set to zero) and batch normalization.
- **"Pool: larger of each pair"** — Max pooling with size 2: 128 steps → 64.
  - It keeps the strongest response and gives up the exact position.
  - Then dropout of 30% during training.
- **"Like an FIR filter"** — Same arithmetic as a finite impulse response filter: multiply and add over a short window.
  - The taps are *learned* from data, *not* designed for a passband.
  - Unlike a single FIR filter, it is followed by a non-linearity.
- **"About 6 samples, less than one symbol"** — Two 3-tap layers see 5 samples; pooling adds one, so 6. A symbol is 8 samples. So this block *cannot* see the pattern between symbols; that is left to the LSTM.
- **Total** — 25,920 numbers, 3% of the model.

**In the paper**

- Section 3.1 and Figure 2, page 4.
- The text never gives the kernel size; only Figure 2 does (3).
- The text's Equation 3 is a 2-D image convolution, not the 1-D one the model uses.

**If asked**

- *"Why 3 taps?"* — "Their design choice; the paper doesn't justify it."
- *"What about the edges of the frame?"* — "Zero-padded, so the length stays 128."
- *"Is there a non-linearity?"* — "Yes, ReLU after each filter. Without it, stacked filters would collapse into one linear filter."

**Say it in 30 seconds**

> Block one: two convolution layers. A convolution here is a short filter, three samples long, slid along the frame, like an FIR filter whose taps are learned instead of designed. The first layer has 64 such filters, the second 128. So every time step is now described by 128 numbers. Then a pooling step keeps the larger of each pair of neighbours, so 128 time steps become 64. Each of the 64 steps summarizes about 6 samples, less than one symbol.

---

## Slide 14 — Block 2: reading the steps in order

**On the slide**

- A row of "step 1 … step 64" with two arrows: "forwards, with a memory" and "backwards, a second LSTM".
- "2 layers, 128 numbers each way → 64 × 256"
- "83% — of the model's numbers are here"
- "After this block, every step knows the whole frame."

**What it means**

- **LSTM** — Reads the 64 steps one at a time and keeps a state of 128 numbers. At each step, three learned gates decide:
  - how much of the old state to keep (forget);
  - what to add from the new step (input);
  - what to pass on (output).

  It is a learned, non-linear recursive filter: like an IIR filter, but with a learned, gated state.
- **"Backwards, a second LSTM"** — Bidirectional: one LSTM reads 1 → 64, another reads 64 → 1. Their outputs sit side by side: 128 + 128 = 256 per step.
- **"2 layers"** — The second bidirectional LSTM reads the first one's outputs.
  - After each layer: batch normalization, then 30% dropout during training.
- **"83%"** — 659,456 numbers. Each layer has four gate matrices of size 128 × (input + 128), per direction.
  - First layer: 263,168.
  - Second layer: 394,240.
  - Plus normalization.
- **"Every step knows the whole frame"** — The forward state has seen everything before a step and the backward state everything after it. How much it actually keeps cannot be seen directly.

**In the paper**

- Section 3.2, pages 4–5.
- The text swaps two words: it says "past (backward) and future (forward)". In fact the forward LSTM carries the past and the backward one the future.

**If asked**

- *"What are the gates?"* — "Three learned switches between 0 and 1: how much of the old state to keep, how much of the new step to add, and how much to pass on."
- *"Why two layers?"* — "The second combines longer patterns from the first. The paper doesn't justify two rather than one or three."
- *"Why does this part matter most?"* — "A modulation is a pattern across symbols. This is the only block that reads across them; alone it reaches 63.1 percent in their ablation."

**Say it in 30 seconds**

> Block two: two bidirectional LSTM layers. An LSTM reads the 64 steps in order, one at a time, and carries a memory from step to step. Bidirectional means a second LSTM reads the same steps backwards. So at every step, the output knows what came before and what comes after: 128 numbers from each direction, 256 in total. This block holds 83 percent of the model's numbers.

---

## Slide 15 — Block 3: a learned weighted average

**On the slide**

- "a weight for each of the 64 steps; they add up to 1" (an illustration).
- The formulas:
  - "score = v · tanh(W₁q + W₂k)"
  - "weights = softmax(score)"
  - "output = Σ weightₜ · stepₜ"
- "'Improved' (from their code): a learned scale on the scores · dropout 0.1 on the weights · normalization of the output"
- "64 steps in, one vector of 256 out: the frame's summary."

**What it means**

- **The formula, piece by piece**
  - **q and k** — In their code both come from the second LSTM's output at the same step.
    - q is the normalized and dropped-out version; k (and the value being averaged) is the raw output.
    - So each step is scored on its own content. Steps are *not* compared with each other.
  - **W₁, W₂** — Learned matrices, 256 → 128 each, without offsets.
  - **tanh** — Squashes values into −1 … +1.
  - **v** — A learned vector, 128 → 1, that turns the result into one score per step.
  - **softmax** — Turns the 64 scores into 64 positive weights that add up to 1.
  - **Σ weightₜ · stepₜ** — The weighted average of the 64 step vectors: one vector of 256.
- **"Improved", three additions**
  1. **A learned scale** — All scores are multiplied by one learned number, starting at 1. This sets how sharp the weights are.
  2. **Dropout 0.1 on the weights** — During training, 10% of the weights are randomly zeroed.
  3. **Normalization of the output** — Layer normalization of the 256-number summary.
- **Size** — 66,177 numbers (8%): W₁ 32,768 + W₂ 32,768 + v 128 + scale 1 + normalization 512.
- **What it is not**
  - *Not* the transformer attention of ChatGPT: that compares every step with every other.
  - *Not* shown to be interpretable: the paper never plots the weights.

**In the paper**

- Section 3.3 and Equations 4–6, page 5. The paper describes query, key and value in general textbook terms; which input is which comes from their code.
- Its contribution in the ablation (Table 6): without attention 62.80, with attention 63.24, i.e. +0.44 points from one run.

**If asked**

- *"What does attention do here?"* — "A learned weighted average over the 64 steps. Each step gets a score, a softmax turns the scores into weights, and the steps are averaged into one vector."
- *"What would you do without it?"* — "Take the last step or a plain average. In their ablation that costs about half a point."
- *"Where does it look?"* — "Nobody has shown it, the paper included. It would be a good check to plot the weights."
- *"Why additive?"* — "The two transformed vectors are added inside a tanh, rather than multiplied as in transformer attention."

**Say it in 30 seconds**

> Block three: attention. Each of the 64 steps gets a score from a small learned network. A softmax turns the scores into weights that add up to one. The output is the weighted average of the 64 steps: one vector of 256 numbers for the whole frame. The paper calls it improved because of three additions. From their code we now know exactly what they are: a learned scale on the scores, dropout on the weights, and a normalization of the result.

---

## Slide 16 — Block 4: the decision

**On the slide**

- A bar illustration with the 11 type names.
- "256 → 128 → 64 → 11 probabilities; the largest wins (illustration)"
- A pie: "BiLSTM 83% · Attention 8% · Dense 5% · CNN 3%"
- "794,188 numbers in total"
- "Most of the model is the LSTM."

**What it means**

- **Dense 256 → 128 → 64** — Two fully connected layers, each followed by ReLU, batch normalization and 30% dropout.
- **→ 11** — The output layer, one number per type; a softmax turns them into probabilities.
- **"The largest wins"** — The answer is the type with the highest probability. Nothing else is used: no threshold, no "unsure" answer.
- **"(illustration)"** — The bars on the slide are drawn to show the idea. They are *not* a real output.
- **The pie** — Where the 794,188 numbers sit. The dense part is 42,635 numbers (5%).
- **Total** — 794,188 (paper: 0.79 M). 792,396 are trained; 1,792 are normalization statistics.

**In the paper**

- Figure 2 (Part D), page 4.
- The paper's Table 7 claims the model is far smaller than MCLDNN. That holds only because Table 7 misprints MCLDNN's size (slide 21).

**If asked**

- *"Are the probabilities calibrated?"* — "Not checked, by them or by us. Only the largest is used."
- *"Is 794 thousand a lot?"* — "For this field it's mid-sized: about twice MCLDNN and eleven times PET-CGDNN, which score within about a point of it in our comparison."

**Say it in 30 seconds**

> Block four: the decision. Two ordinary layers, 128 and 64 numbers, and then 11 outputs. A softmax turns them into probabilities, and the largest one is the answer. In total the model has 794 thousand numbers. The pie shows where they are: almost all in the LSTM.

---

## Slide 17 — How it is trained

**On the slide**

- **DATA:** "70% learn: 154,000 frames · 15% watch: 33,000 · 15% test: 33,000"
- **LOOP:** "batches of 32 frames · 4,813 steps per epoch · loss: cross-entropy · Adam, learning rate 0.001"
- **STOPPING:** "loss flat 5 epochs: rate × 0.2 · loss flat 6 epochs: stop · at most 100 epochs"
- "58 — where it stopped"
- "Trained once: one split, one run."

**What it means**

- **"Watch"** — The validation set.
- **"Batches of 32"** — 32 frames go through the network together and give one update.
- **"4,813 steps per epoch"** — 154,000 / 32 = 4,812.5, rounded up.
- **"Loss: cross-entropy"** — For each frame, minus the log of the probability given to the right type, averaged. Lower is better; a perfect, confident answer gives 0.
- **"Adam, learning rate 0.001"** — The optimizer and its starting step size.
- **"Loss flat 5 epochs: rate × 0.2"** — If the validation loss has not improved for 5 epochs, the step size is cut to one fifth.
  - The paper's text says this happened at epoch 43 (0.001 → 0.0002), and the rate was 0.00004 by epoch 58.
- **"Loss flat 6 epochs: stop"** — Early stopping, then the weights from the best epoch are restored.
- **"At most 100 epochs"** — The ceiling in their code. It was never reached.
- **"58: where it stopped"** — Not a chosen budget, although the paper's Table 2 lists it as "Number of Epochs".
- **Also in their code, not in the paper**
  - Gradient clipping (`clipnorm=1.0`).
  - L2 penalty 1e-4 on the convolution, LSTM and dense weights.

**In the paper**

- Table 2, page 6, and Section 4.1, page 5.
- The stopping rule and ceiling come from their code.

**If asked**

- *"Why stop on the loss and not the accuracy?"* — "It's the usual default. The two can disagree: here the loss had flattened while accuracy was still creeping up. That's backup slide 19."
- *"Why cut the learning rate?"* — "When progress stalls, smaller steps let the weights settle into a better point."
- *"How long does it take?"* — "On a Colab GPU, our runs of their code took 133 to 150 minutes each."

**Say it in 30 seconds**

> How it is trained. Seventy percent of the frames to learn from, fifteen to watch, fifteen to test. Batches of 32 frames; the loss is cross-entropy; the optimizer is Adam with learning rate 0.001. When the validation loss stops improving for 5 epochs, the learning rate is cut by five. When it stops improving for 6 epochs, training stops and the best weights are restored. The paper's table says 58 epochs. From their code we now know that 58 is where this stopping rule fired, not a number they chose. They trained once.

---

## Slide 18 — What their code tells us

**On the slide**

1. "The model is the one in the paper — same layers, 794,188 numbers (paper: 0.79 M)"
2. "'58 epochs' was not a choice — training stops by itself when it stops improving; for them, at 58"
3. "They trained it once — one split of the data, one run, one number"

"One run gives one number: how much would it move?"

**What it means**

- **Their code**
  - Published on GitHub on 8 August 2026, sixteen months after the paper, under the account noureddine395.
  - The README names all five authors; MIT licence; archived on Zenodo.
  - TensorFlow/Keras. *No trained weights.*
- **Point 1** — The layers in the code match Figure 2 exactly.
- **Point 2** — The code allows up to 100 epochs and stops on the validation loss.
- **Point 3** — The script has no loop over seeds, and its split cannot even be repeated: it depends on Python's internal ordering of class names, which changes from one run to the next.
- **"How much would it move?"** — We answered this afterwards: three runs of their code on one fixed split gave **62.71, 62.45 and 62.52%**, i.e. 62.56 ± 0.11. Every run is below their 63.24.

**In the paper**

- Not in the paper. See `OFFICIAL_CODE.md`.

**If asked**

- *"How do you know it's really theirs?"* — "It's on an account named after the first author, the README names all five authors and calls it the official code, and it builds exactly the paper's layers."
- *"Did you run it?"* — "Three times, unchanged, on one fixed split: 62.71, 62.45 and 62.52 percent, against their single 63.24. Training stopped by itself at epochs 57 to 64, at the same validation accuracy as theirs."

**Say it in 30 seconds**

> What does their code tell us? Three things. First, the model in the code is the model in the paper: same layers, 794 thousand numbers. Second, the 58 epochs in the paper were not a choice: the training stops by itself when it stops improving, and for them that happened at 58. Third, they trained it once: one split of the data, one run, one number. And one run leaves a question: how much would that number move if you ran it again?

---

## Slide 19 — What "58 epochs" means

**On the slide**

- "61.75% — our rebuild at a fixed 58 epochs (3 runs)"
- "63.21% — the same, left to run: best at epoch 107"
- Their Figure 7: "accuracy still rising at 58"
- "Their rule stopped on the loss; the accuracy was still climbing."

**What it means**

- **"Our rebuild"** — The model we wrote from the paper's text before their code existed (slide 20).
- **"61.75% at a fixed 58 epochs (3 runs)"** — Trained for exactly 58 epochs, as Table 2 seemed to say: 61.75 ± 0.07.
- **"63.21%, best at epoch 107"** — The same model, allowed up to 150 epochs, peaked at 107. **One run.**
- **Their Figure 7** — Validation accuracy is still rising at epoch 58 (63.59%), while their text says the validation loss had flattened near 1.05 after epoch 40.
- **Why loss and accuracy disagree** — The loss also punishes *confident wrong answers*; accuracy only counts which answer is largest. The model can get more frames right while growing over-confident on the ones it gets wrong, so accuracy rises while the loss stays flat.
- **What it is not**
  - *Not* evidence that they stopped wrongly: their rule watches the loss, and the loss had stopped improving.
  - Our rebuild needed longer partly because of a different recipe: no gradient clipping, a different normalization, and stopping on accuracy.
- **What their own code showed later** — Run with their own rule, it stopped at epochs 57, 61 and 64 and gave 62.56% on average.

**In the paper**

- Figure 7 and Section 4.2.2, page 8.

**If asked**

- *"So did they stop too early?"* — "By their rule, no; it watches the loss. By accuracy, maybe slightly. Our rebuild gained 1.5 points by continuing, but with our recipe."
- *"How many runs at epoch 107?"* — "One."

**Say it in 30 seconds**

> One detail about the 58 epochs. Their stopping rule watches the validation loss, and the loss had flattened. But the validation accuracy, in their own figure on the right, is still rising at 58. When I trained our rebuild for a fixed 58 epochs, I got 61.75 percent. Letting it continue, it peaked at epoch 107 at 63.21. So their number is real, but it sits where the accuracy was still climbing.

---

## Slide 20 — We rebuilt it; their code agrees

**On the slide**

- A timeline: "Apr 2025 — paper, no code · Aug 2026 — their code on GitHub, no weights · Sep 2026 — our rebuild from the paper alone"
- "63.21% — our rebuild, trained to the end (paper: 63.24%)"
- **MATCHED THEIR CODE:** "one pooling step · normalization after each LSTM · dropout on attention weights · size: 794,827 vs 794,188"
- **MISSED:** "exact attention form and its learned scale · input normalization · gradient clipping"
- "Their 63.24% is reproducible from the paper alone."

**What it means**

- **"Our rebuild from the paper alone"** — Written in PyTorch from the paper's text and Figure 2 (`colab/icrnna_faithful_2016.py`).
- **Matched** — Three things the text left unclear, which we read right:
  - one pooling step, not two;
  - batch normalization after each LSTM layer, not once after both;
  - dropout on the attention weights.
- **"794,827 vs 794,188"** — Within 0.1%.
  - The small difference comes from building the attention slightly differently.
  - The two programs also count differently: Keras includes the normalization statistics, PyTorch does not.
- **Missed** — Three details the code settled:
  - **Attention form and learned scale:** the two-input form and the single learned multiplier.
  - **Input normalization:** per-position mean and variance over the whole dataset.
  - **Gradient clipping:** each update capped at length 1.
- **"Reproducible from the paper alone"** — The model and its accuracy can be rebuilt from the text: 63.21 against 63.24.
- **The caveat** — 63.21 is one run, and their own code averages 62.56 on our split. So reproducible means "the same level, within about half a point", *not* the same digits.

**In the paper**

- Not in the paper. See `OFFICIAL_CODE.md`.

**If asked**

- *"Did the misses matter?"* — "Not much: trained to the end, our build lands within 0.03 points of their number."
- *"Why did you rebuild it before the code came out?"* — "The paper said 'available on request' and there was no code. A rebuild was the only way to check the number."

**Say it in 30 seconds**

> Before we knew any code existed, I rebuilt the model from the paper's text alone. Its size matched: 794,827 numbers. In August the authors released their code, and it confirms our reading: one pooling step, normalization after each LSTM, dropout on the attention weights. We missed a few details: the exact form of the attention, the input normalization, and gradient clipping. Trained until it stopped improving, our rebuild reached 63.21 percent, against their 63.24.

---

## Slide 21 — Three claims their own tables don't support

**On the slide**

1. "'Best at low SNR' — Table 5: beaten at −10, −8 and −6 dB"
2. "'Full model has the highest precision' — Table 6: LSTM alone is higher, 0.768 vs 0.754"
3. "'MCLDNN: 406 million parameters' — It has 406,199: 0.41 million"

"ICRNNA is about twice MCLDNN's size, not smaller."

**What it means**

**Claim 1: "best at low SNR"** — Their own Table 5 (page 9):

| label | ICRNNA | higher than ICRNNA |
|---|---|---|
| −10 dB | 25.32 | CNN-BiLSTM-DNN 27.73 |
| −8 dB | 37.45 | CNN-BiLSTM-DNN 39.55, MCLDNN 38.09, MCNet 38.00 |
| −6 dB | 55.26 | MCLDNN 56.32, CNN-BiLSTM-DNN 56.18, MCNet 56.00 |
| +4 dB | 93.02 | CNN-BiLSTM-DNN 93.18 |

- ICRNNA leads only at −4, −2, 0 and +2 dB, and there by 3–4 points.
- On the 10b dataset (Table 4) it does lead at every level, by small margins.

- **Claim 2: "highest precision"** — Their own Table 6 (page 13): precision is 0.7676 for the LSTM-only model against 0.7536 for the full model. On 10b it is 0.7200 against 0.7100. So the claim fails on both datasets.
  - Precision (of type X) means: of the frames called X, the fraction really X.
- **Claim 3: "MCLDNN 406 M"** — Their Table 7 lists MCLDNN at 406 million parameters. We built MCLDNN from its published code and counted **406,199**, i.e. 0.41 million; every source agrees to the digit. The table also lists PET-CGDNN at 79.2 M, while it has 71,871.
- **The takeaway** — 794 thousand is about twice MCLDNN's 406 thousand. The paper's "smaller and more efficient" rests on that misprint.
- **What it is not** — *Not* proof that the model is bad. The model and the headline number hold up; the text oversells its own tables.

**If asked**

- *"Show me the low-SNR claim."* — "Their Table 5: at −10 dB CNN-BiLSTM-DNN has 27.7 against 25.3; at −8 and −6, three models are higher. ICRNNA leads from −4 to +2 dB, by three to four points."
- *"How do you know 406,199?"* — "We built MCLDNN from its published code and counted; every source agrees to the digit."
- *"Any other errors?"* — "Small ones: two different overall accuracies in the text, 63.39 and 63.24; a '−5 dB' level that doesn't exist; GFSK written as BFSK."

**Say it in 30 seconds**

> Reading their tables closely, three statements in the text don't match their own numbers. Best at low SNR: in their Table 5, other models are higher at minus 10, minus 8 and minus 6 dB. Highest precision: in Table 6, the LSTM-only model is higher. And MCLDNN is listed with 406 million parameters; it has 406 thousand. So ICRNNA is about twice the size of MCLDNN, not smaller.

---

## Slide 22 — What the paper doesn't tell us

**On the slide**

1. "One run — no spread; our seeds vary by up to a point"
2. "Baselines — retrained on the same split, or copied from their papers?"
3. "SNR label — taken as the true SNR"

Also on the slide:

- "+9 dB — more signal in QAM16 than in QPSK, both labelled 0 dB (measured in the file)"
- "A half-point lead from one run may not be real."

**What it means**

- **"One run, no spread"**
  - The paper reports one training run per model.
  - In our three-seed runs, the overall accuracy of one model moves by 0.5 points (ICRNNA) to 1.1 points (LSTM2).
  - Their own code, run three times by us, moves by 0.26 points and averages 0.7 below their number.
- **"Baselines"** — They never say whether the other models were retrained on the same split or taken from their papers. Published numbers for the same model already differ by about half a point between papers: MCLDNN is 62.08 in one paper and 61.42 in this one.
- **"SNR label taken as the true SNR"** — The paper treats "0 dB" as "signal and noise equal". In this file, at label 0, measured SNR is about 5 dB for QPSK and 14 dB for QAM16.
- **"+9 dB"** — 14.0 − 5.1 = 8.9 dB. QAM16 frames carry about eight times more signal power relative to noise than QPSK frames with the same label.
  - The cause: the generator did not scale the constellations to equal power, and an integer QAM grid has about 10 dB more average power than QPSK.
  - "Measured in the file": we estimated the noise from the part of the spectrum outside the signal band, over 1,000 frames per type and label. The method was checked to within half a dB on frames of known SNR.
- **"A half-point lead from one run may not be real"** — Back to slide 7.

**If asked**

- *"How did you measure +9 dB?"* — "In the dataset file itself. The digital signals occupy only part of the band, so the spectrum outside it is pure noise; that gives the noise level, and the rest is signal."
- *"So is comparing classes at one label unfair?"* — "Across classes, yes: they don't share an SNR. Between models it's fair, since every model sees the same frames."

**Say it in 30 seconds**

> And three things the paper doesn't tell us. It reports one training run, with no spread; across seeds we see up to a point of variation, so a half-point lead may not be real. It doesn't say whether the other models were retrained or copied from their papers. And it treats the SNR label as a true SNR; in this data, at label 0 dB, QAM16 has about 9 dB more signal than QPSK.

---

## [−] Taken out of the deck: accuracy against SNR (their Figure 8)

This slide was removed from the deck. Moshe asked about exactly this curve, so keep it ready.

**On the slide**

- Their Figure 8: accuracy, F1 score and precision against the SNR label.
- "9% — at −20 dB: chance (1 in 11)"
- "~93% — from 0 dB up"
- "Figure from El-Haryqy et al. 2025 (CC BY-NC 4.0)"

**What it means**

- **The x-axis** — The SNR label, −20 to +18 dB.
- **The three curves**
  - Accuracy.
  - F1 score: precision and recall combined.
  - Precision: of the frames called X, the fraction really X, averaged over types.
  - All three follow the same shape.
- **"9% at −20 dB: chance"** — With 11 balanced types, guessing gives 1/11 = 9.1%. At −20 dB the signal is buried and the network is guessing.
  - In practice it dumps almost all frames into one type there. In our runs that type is AM-SSB.
- **The rise** — Fast between −10 dB (about 25%) and 0 dB (about 92%).
- **"~93% from 0 dB up"** — A plateau. More SNR does not help, because the remaining errors are not caused by noise:
  - WBFM against AM-DSB (silences in the audio);
  - QAM16 against QAM64 (too few symbols).
- **Decisions behind each point** — About 1,650: the 33,000 test frames split over 20 levels.

**In the paper**

- Figure 8 and Section 4.2.2, page 8.
- The text quotes "−5 dB" and "−15 dB", levels that do not exist on the 2-dB grid.

**If asked**

- *"How many decisions behind each point?"* — "About 1,650: the test set split over twenty levels, about 150 per type."
- *"Why does it stop at 93 and not 100?"* — "The remaining errors are WBFM against AM-DSB and QAM16 against QAM64. Neither is fixed by more SNR: one is silence in the audio, the other is too few symbols."
- *"Why does it jump so fast near 0 dB?"* — "That's where the signal starts to stand out. And the label understates the SNR: at label 0 the PSK types are actually near 5 dB."
- *"What's the difference between precision and recall?"* — "Recall: of the real X frames, how many were called X. Precision: of the frames called X, how many really were X."

**Say it in 30 seconds**

> Accuracy against noise. At the noisiest level it is right 9 percent of the time, which is guessing among eleven. It improves quickly between minus 10 and 0 dB, and from 0 dB up it is right about 93 percent of the time.

---

## [!] Traps: things not to say

- **"Detector."** Say **classifier**. Detection means deciding whether a signal is present at all.
- **"0 dB means signal and noise are equal."** In this dataset it is a label; at label 0 the PSK types are near 5 dB.
- **"It is 63% accurate."** Say "63% over all noise levels together; about 93% from 0 dB up".
- **"It is the best at low SNR."** Their own Table 5 contradicts that at −10, −8 and −6 dB.
- **"It is smaller than MCLDNN."** It is about twice as large; the table misprints MCLDNN.
- **"Our classifier is the published architecture."** Ours differs in five small places. The faithful rebuild is a separate model.
- **"We get 63.21 every time."** One run. At a fixed 58 epochs, three runs gave 61.75. Their own code gives 62.56 on average.
- **"The picture on slide 3 is from the dataset."** It is a clean QAM16 frame we drew.
- **"The attention shows where the model looks."** Nobody has plotted it.
- **"58 epochs was their choice."** 58 is where early stopping fired.
- **"We measured the WBFM silence effect."** It is the known explanation from the literature; we have not measured it.
- **When you don't know:** "The paper doesn't say" or "I haven't measured that; I can check." Both are good answers.

---

## [+] Since the talk: their code, three runs

The authors' own code, run unchanged by us on Colab, three seeds on one fixed split:

| | run 1 | run 2 | run 3 | paper |
|---|---|---|---|---|
| test accuracy | 62.71% | 62.45% | 62.52% | 63.24% (one run) |
| stopped at epoch | 61 | 64 | 57 | 58 |
| best epoch (restored) | 55 | 58 | 51 | — |
| validation accuracy there | 63.60% | 63.43% | 63.52% | 63.59% |
| accuracy at label 0 dB | 90.5% | 90.8% | 90.2% | 92.21% |

- **Average:** 62.56% ± 0.11. Every run is 0.5–0.8 points below the paper.
- **Training behaves as they describe:** it stops by itself around epoch 58, at the same validation accuracy.
- **So the paper's 63.24 looks like a slightly favourable single draw** (a lucky split or a lucky start), not a different model. It cannot be checked exactly, because their code cannot reproduce its own split.

**If asked:** "We ran their own code three times, unchanged: 62.71, 62.45 and 62.52 percent. Their 63.24 is a single run, so it looks like a slightly favourable draw; the model is the same."

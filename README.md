# Why modulation classifiers lose accuracy across domains

A study of automatic modulation classification (AMC) on the RadioML
benchmarks. Reproducing benchmark accuracy is solved; this project measures,
**explains** and closes the gap between the domain a classifier is trained on
and an independently generated one.

> A classifier that is near-perfect on its own dataset loses a large part of
> that accuracy on IQ generated with a different transmitter. Quantify the
> collapse, attribute it to a specific signal property by intervention rather
> than correlation, then close it.

Yusuf Bilal Cetinkaya · Moshe Kam's research group · NJIT

---

## At a glance

| | |
|---|---|
| **Datasets** | RML2016.10a: 11 types, 128-sample frames, the current focus. RadioML 2018.01A: 24 types, 1024 samples; its results stand, with no new work for now |
| **Classifier** | ICRNNA (El-Haryqy et al., *Results in Engineering* 26, 2025): convolution, bidirectional LSTM, additive attention. Built through `model_zoo.backbone(n_classes)` |
| **Reproduction** | Rebuilt from the paper's text: 63.21% against the paper's 63.24%. The authors' own code, run three times: 62.56% ± 0.11 |
| **Main question** | How much accuracy is lost on an independently generated domain, why, and what closes it |
| **Before quoting any number** | read [`CLAUDE.md`](CLAUDE.md): it says which result files are current |

## Main results

**Classification on RML2016.10a.** Five published architectures, one protocol: same frames, same 70/15/15 split, three seeds, each trained until it stopped improving.

| model | accuracy, all SNR labels | accuracy, ≥ 10 dB | parameters |
|---|---|---|---|
| ICRNNA (2025) | 62.2% | 91.5% | 786k |
| LSTM2 (2018) | 61.4% | 91.5% | 201k |
| MCLDNN (2020) | 61.4% | 91.1% | 406k |
| PET-CGDNN (2021) | 61.0% | 90.6% | 72k |
| VT-CNN2 (2016) | 56.1% | 82.0% | 1.59M |

- **ICRNNA leads, but by about a point.** Its edge is at −6 to −2 dB; from +2 dB up the top four sit at 90–92%.
- **The plateau is a property of the data, not the model.**
  - WBFM is read as AM-DSB during silences in the source audio.
  - QAM16 and QAM64 cannot be fully separated from 16 symbols. Above 10 dB, cropping the frame to 128, 64 and 32 samples gives 2,421, 6,443 and 10,431 errors in 42,000 decisions.

**The domain gap: trained on RadioML, tested on an independently generated domain.**

| method | 2018: cross-domain accuracy | 2016: in-domain / cross-domain |
|---|---|---|
| none | 0.805 | 0.948 / 0.920 |
| standard augmentation (rotation, flip, noise) | 0.807 | 0.969 / 0.968 |
| spectral whitening | 0.989 | 0.864 / 0.843 |
| whitening + augmentation | 0.995 | 0.931 / 0.913 |

- **On 2018 the gap is large and whitening closes it.** The literature augmentation does not.
- **On 2016 the gap is small and augmentation closes it.** Whitening's cost there was the envelope estimate, not the idea: averaged over enough bins it costs nothing in-domain. It still does not close the 2016 gap.
- The 2018 results use 5 seeds; the 2016 results use 14.

**Other findings**

- **The SNR label in RML2016.10a is not a measured SNR.** Measured in the file, the real SNR rises about 1.9 dB per dB of label, and at the same label QAM16 carries about 9.5 dB more than the PSK types ([`NOISE_2016.md`](NOISE_2016.md)).
- **An unseen modulation lands in its own family.** Held out of training, 16 of 24 types are classified into their own pre-registered family, against 3.5 expected by chance.

## Documents

| file | what it is |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | which results are current, and the conventions every run follows |
| [`DETECTOR.md`](DETECTOR.md) | the classifier end to end: layers, training, provenance, and what was written with an AI assistant |
| [`OFFICIAL_CODE.md`](OFFICIAL_CODE.md) | the ICRNNA authors' code against the paper and our builds, with three runs of it |
| [`NOISE_2016.md`](NOISE_2016.md) | what "10 dB" means in RML2016.10a, measured |
| [`SNR_2016.md`](SNR_2016.md) | the four-class box, one SNR at a time |
| [`literature_2016.md`](literature_2016.md), [`LITERATURE_CHECK.md`](LITERATURE_CHECK.md) | the five-architecture comparison and its check against the papers |
| [`LITERATURE.md`](LITERATURE.md) | reading notes |
| `deck/` | meeting decks and study pages, generated from the result files |

## Running it

- **Locally:** `cd src && python train_backbone.py --data rml2016 --data-path <RML2016.10a_dict.pkl> --arch ICRNNA --seeds 3`
- **On Colab** (GPU runtime, Drive mounted):

  ```
  %env AMC_TASK=<task>
  !python /content/amc/colab/run_experiment.py
  ```

  The tasks are listed at the top of `colab/run_experiment.py`. Results go to `Drive/amc-results/`, and long runs resume where they stopped.
- **Data is not in the repository.** It needs the RML2016.10a pickle (`RML2016.10a_dict.pkl`) and, for 2018, `GOLD_XYZ_OSC.0001_1024.hdf5`.

---

## Detailed record

What follows is the full record, in the order the work was done, with every
caveat. Where it disagrees with the summary above or with `CLAUDE.md`, those
are current.

### The 2018 method table, first pass (60-epoch ceiling)

Superseded by the converged run (`compare_methods_ICRNNA_es_e100.npz`, in the summary above); kept because of the convergence lesson below.

`compare_methods.py --arch ICRNNA --seeds 5 --epochs 60 --patience 20`,
**all 20 cells** (`compare_methods_ICRNNA_es.npz`):

| method | in-domain | cross-domain | gap | 16QAM |
|---|---|---|---|---|
| none | 0.999 ±0.001 | 0.804 ±0.005 | **+0.195** ±0.005 | 0.060 ±0.021 |
| standard augmentation | 1.000 ±0.000 | 0.806 ±0.003 | +0.194 ±0.003 | 0.036 ±0.014 |
| whitening α=0.75 | 1.000 ±0.000 | **0.993** ±0.001 | **+0.007** ±0.001 | **0.991** ±0.006 |
| whitening + standard | 0.999 ±0.001 | 0.994 ±0.003 | +0.005 ±0.003 | 0.998 ±0.003 |

**Read this table with the convergence caveat below.**

Three readings:

1. **The gap is real and large.** A model at 0.999 in-domain drops to 0.804 on
   an independently generated domain, and 16QAM — the class the whole diagnosis
   centres on — collapses to 0.060.
2. **The literature augmentation set does not address it.** Rotation, conjugate
   flip and additive Gaussian noise (arXiv:1912.03026) moved the gap by 0.001.
   Whatever these transforms are good for, this failure mode is not it.
3. **Whitening costs nothing in-domain.** In-domain accuracy stays at 1.000
   while the gap falls to +0.007, so the recovery is not bought by trading away
   benchmark performance.

A cross-domain accuracy of 0.993 sits close enough to the ceiling that it
deserves a sceptical pass of its own before it goes in a report.

Completing the table also settled the fourth row, which stood at 2 of 5 seeds.
Combining whitening with the literature augmentation adds **+0.001 (0.3 s.d.)**
over whitening alone. They do not compose; whitening is doing all of the work
and the rotation/flip/noise set contributes nothing on top of it. That is the
opposite of what this file's own docstring predicted, and the prediction was
written before the measurement — so it stays as a refuted hypothesis rather
than being quietly reworded.

### The convergence caveat on this table

The three cells run last recorded where training stopped: peaks at epochs
**53, 37 and 50**, under a 60-epoch ceiling with patience 20. Early stopping
fires 20 epochs after the peak, so two of the three needed to reach epoch 73
and 70 and instead **ran out of budget at 60**. Those two cells did not
converge and their accuracies are floors.

The other seventeen cells were run under the same ceiling before the stopping
epoch was recorded, so the same is likely true of them and unprovable either
way from the file.

How much this matters depends on which comparison you read:

- **The headline survives.** Whitening moves cross-domain accuracy from 0.804
  to 0.993, a difference of 0.19 against a seed spread of 0.005. Under-training
  depresses cells; it does not manufacture a nineteen-point gap, and the
  2016 and 2018 runs showed the penalty for stopping a few epochs early is
  fractions of a point.
- **The small comparisons do not.** Whitening against whitening-plus-standard
  is +0.001, and none against standard augmentation is +0.002. Differences that
  size, between cells that stopped at an arbitrary budget rather than at
  convergence, are not measurements. Neither is safe to report until the table
  is rerun with a ceiling that lets early stopping decide.

This was nearly missed. The warning added for exactly this case tested
`best_epoch >= ceiling`, which is too weak — a peak at 53 under a ceiling of 60
is not at the ceiling, but the run still stopped because the budget ended
rather than because it had converged. The test is now
`best_epoch + patience <= ceiling` in `compare_methods.py`,
`whitening_seeds.py` and `train_backbone.py`, and it names the epoch a rerun
would need.

#### The rerun at 100 epochs, and what it showed about convergence

`compare_methods_ICRNNA_es_e100.npz` is the same table under a 100-epoch
ceiling, **complete at 20 of 20** (the last five cells took 33 minutes on an
A100):

| method | in-domain | cross-domain | gap | 16QAM |
|---|---|---|---|---|
| none | 0.999 ±0.001 | 0.805 ±0.005 | +0.193 ±0.004 | 0.056 ±0.018 |
| standard augmentation | 1.000 ±0.000 | 0.807 ±0.005 | +0.192 ±0.005 | 0.043 ±0.023 |
| whitening a=0.75 | 0.999 ±0.001 | 0.989 ±0.007 | +0.011 ±0.007 | 0.981 ±0.009 |
| whitening + standard | 1.000 ±0.000 | 0.995 ±0.003 | +0.005 ±0.002 | 0.999 ±0.000 |

Whitening against none: +0.183. Standard augmentation
against none: +0.002. Combining against whitening
alone: **+0.006**, which at a pooled seed spread of
0.005 is not a measurement — the same
verdict the 60-epoch table gave, now from cells that had room to converge.

Every cell recorded its stopping epoch, and those turned out to be the
interesting part:

| method | stopping epochs | mean | gradient updates |
|---|---|---|---|
| none | 51, 41, 48, 36, 38 | 43 | 9.8k – 13.9k |
| standard augmentation | 89, 48, 83, 65, 82 | 73 | 13.1k – 24.3k |
| whitening a=0.75 | 26, 26, 33, 43, 25 | 31 | 6.8k – 11.7k |
| whitening + standard | 66, 37, 53, 37, 50 | 49 | 10.1k – 18.0k |

**Convergence speed is a property of the method, not just of the dataset.**
Whitening converges 2.4× faster than the augmentation row and 1.4× faster than
training with no method at all; add the augmentation set on top of whitening
and the stopping epoch rises again, from 31 to 49. That is what the mechanism predicts: whitening
removes a nuisance dimension, so there is less to fit; augmentation adds
nuisance variation, so there is more. It also corrects the generalisation
drawn from the two classification runs — 23k updates was not a constant of the
optimizer, it was those two problems.

**Three cells ran out of budget at 100**, all of them in the standard
augmentation row: peaks at 89, 83 and 82, which need 109, 103 and 102 to stop.
That row is the comparison baseline for the headline claim, so it was the worst
one to have under-trained. They were rerun under a 150 ceiling
(`--redo-unconverged`, nine minutes on an A100) and came back **bit-identical**:
the same peaks at 89, 83 and 82, the same accuracies to the last digit, this
time with the twenty further epochs that let early stopping fire. The file now
records a ceiling per cell — 150 for those three, 100 for the rest — and
`best_epoch + patience <= ceiling` holds in all twenty. This is the third time
a cell flagged by that test has turned out to be sitting at its true peak (the
four-class run's seeds 0 and 1 were the first two); the flag has yet to catch a
cell that was still climbing, and it is kept anyway, because the one time it
was weakened it missed cells that were.

**The two tables are independent samples, not one refined into the other.**
No cell reproduces its 60-epoch counterpart bit for bit — the `none` row reads
0.8066 against 0.8076 on the first seed, and so on down the table. The 2016 and
2018 convergence checks *were* bit-identical to the runs they re-ran, so this
is not how the code behaves on one machine; the 60-epoch table was measured on
the laptop and this one on Colab, and cuDNN and AMP do not promise the same
arithmetic across two GPUs. It costs nothing for the headline, which is a
nineteen-point difference, but it does mean a 60-vs-100 comparison at the cell
level confounds the ceiling with the hardware and should not be read as an
effect of the ceiling.

The one cell worth naming is whitening seed 2, which lands at 0.9750 here
against 0.9914 at the lower ceiling and drags that row's spread from ±0.0014 to
±0.0070. It peaked at epoch 33 under a ceiling of 100, so it converged with 47
epochs to spare — this is seed-level variation in the whitening row, not an
under-trained cell, and the row's honest spread is the wider one.

Fixing it did not cost another full table. **The ceiling only matters to a
cell that hits it**: a cell that peaked at epoch 36 peaks at 36 whatever
ceiling it ran under, so a table whose cells all converged is sound even
though they ran under different ceilings. `compare_methods.py` stores the
ceiling per cell and takes `--redo-unconverged`, which on resume clears only
the cells that ran out of budget and reruns those — three cells rather than
twenty — and the per-cell ceilings are in the file so the claim can be checked
rather than taken on trust.

### Plain classification on RML2016.10a

`train_backbone.py --data rml2016 --seeds 3 --epochs 60 --patience 20`, all
1000 frames per (class, SNR) cell, 70/15/15, run on Colab
(`train_backbone_rml2016_f1000_colab.npz`, 3/3 seeds):

| | accuracy |
|---|---|
| overall, all SNRs | **0.6223 ± 0.0019** (0.6248, 0.6200, 0.6222) |
| SNR ≥ 10 dB | **0.9148 ± 0.0013** |

The curve is the shape this dataset is supposed to give: chance (0.091, eleven
classes) below −16 dB, rising from −12 dB, past 0.90 at +2 dB, then flat at
0.915 to the top of the range. Seed spread never exceeds 0.014.

**What this establishes, and what it does not.** El-Haryqy et al. report 63.24%
for ICRNNA on this dataset and this lands 1.01 points below it. That is a
sanity check on the pipeline, *not* a reproduction of the paper: this backbone
differs from the published description in five places (see the provenance
caveat below) and the training protocol differs too — batch 256 against the
paper's 32, LR schedule on validation accuracy against validation loss. The
build that can actually answer the reproduction question is
`colab/icrnna_faithful_2016.py`, still unrun.

The high-SNR plateau is a property of the dataset rather than of the model.
WBFM and AM-DSB are both generated from voice recordings, and during the
silences in the source audio both degrade to an unmodulated carrier, which no
architecture can separate; ~0.92 is the practical ceiling this imposes and is
where the plateau sits.

Almost all of the remaining error is one cell. **WBFM is read as AM-DSB 60% of
the time**, and that single cell is 64% of every misclassified frame at SNR ≥ 10
dB; drop WBFM and the other ten classes sit at 0.968. This is the documented
dataset artifact rather than a property of the model — both classes are
generated from voice recordings, and during the silences in the source audio
both collapse to an unmodulated carrier. QAM16 ↔ QAM64 accounts for most of the
rest, at 0.06 and 0.08.

### Plain classification on RadioML 2018.01A

`train_backbone.py --data rml2018 --seeds 3 --epochs 60 --patience 20`, 512 of
the 4096 frames per (class, SNR) cell, native 1024-sample frames, 70/15/15, run
on Colab (`train_backbone_rml2018_f512_colab.npz`, 3/3 seeds):

| | accuracy |
|---|---|
| overall, all 26 SNRs | **0.5699 ± 0.0081** (0.5616, 0.5809, 0.5672) |
| SNR ≥ 10 dB | **0.8716 ± 0.0141** |

Chance is 0.042 with 24 classes, and the curve sits there below −14 dB. It
lifts from −12, passes 0.526 at 0 dB and 0.80 at +6, then flattens at 0.87 from
+8 dB to the top of the range, peaking at 0.880 at +12 dB.

Seed spread reaches 0.021, against 0.014 on 2016. Three seeds on 24 classes is
thin, and a difference of less than about 0.03 between two configurations
measured this way is not yet a difference.

**Against the literature.** Deep residual networks trained on the *whole*
dataset are reported at roughly 95% above 8 dB. This run is about 8 points
under that on 12.5% of the frames with a 786k-parameter recurrent model, and
two things differ at once — data volume and architecture — so the deficit is
not attributable to either. Closing it is not this project's goal; the number
exists so that the cross-domain differences measured elsewhere are known to
start from a competent classifier rather than a broken one.

Wall clock: about 9–13 minutes per seed on a Colab GPU, including a one-off
sequential copy of the 21 GB HDF5 to local disk and the staging of 512 frames
per cell from it.

#### Where the 2018 error goes, and why it is worth a look

Pooling the three seeds at SNR ≥ 10 dB, **84.2% of all error mass lands inside
the same modulation family** — using the taxonomy in `sink_class_24.py`, which
was written before any of this was measured. If misclassifications were spread
uniformly over the other 23 classes the share would be 16.6%, so this is 5.1×
that baseline. Family-level recall is 1.000 for ASK, PSK and analog, 0.988 for
QAM, and 0.896 for APSK.

Two groups carry three quarters of it. The **with-carrier / suppressed-carrier
analog pairs** (AM-DSB-WC ↔ AM-DSB-SC, AM-SSB-WC ↔ AM-SSB-SC) are 40.9%; drop
those four classes and the remaining twenty sit at 0.909. **Confusion inside the
QAM family** — mostly 64QAM ↔ 256QAM at 0.31 and 0.25 — is another 35.4%.

The exceptions are the interesting part. Every cross-family leak above 0.02 is
APSK → QAM *at matched constellation order*: 16APSK → 16QAM at 0.25, 64APSK →
64QAM at 0.10, 128QAM ↔ 128APSK at 0.03 and 0.02. Where the model leaves the
family it keeps the order.

**This is not the sink experiment rerun.** That one holds a class out of
training entirely and asks where an *unseen* modulation lands; this is ordinary
in-distribution error on classes the model was trained on. The two are
consistent, and the second is much weaker evidence than the first — errors
falling between similar classes is what any classifier does. What is worth
recording is the size of the effect and that the order-matched APSK → QAM leaks
survive it. Open work item 3 is still open.

### The decision table

The number a review asks for first is not the average, it is the table the
average is computed over: N frames of a known modulation go in, and how often
did the classifier answer each of the classes available to it? Row =
transmitted, column = decided. `confusion_table.py` prints it from a finished
run's stored counts, so for a run that already exists it costs nothing.

RML2016.10a at SNR ≥ 10 dB, three seeds pooled, 2,250 decisions per row
(percentages of the row):

| transmitted \ decided | 8PSK | AM-DSB | AM-SSB | BPSK | CPFSK | GFSK | PAM4 | QAM16 | QAM64 | QPSK | WBFM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **8PSK** | **98.3** | | 0.8 | | | | | | 0.1 | 0.7 | |
| **AM-DSB** | | **100.0** | | | | | | | | | |
| **AM-SSB** | 1.0 | 2.9 | **89.5** | 0.8 | 0.8 | 1.7 | 1.0 | 0.3 | 0.1 | 0.9 | 1.0 |
| **BPSK** | | 0.1 | 0.8 | **98.8** | | | 0.2 | | | | |
| **CPFSK** | | | | | **100.0** | | | | | | |
| **GFSK** | | | | | | **99.9** | | | | | 0.1 |
| **PAM4** | | | 0.7 | 0.3 | 0.1 | | **98.8** | | | | |
| **QAM16** | 0.5 | | 0.9 | | | | 0.1 | **91.8** | 6.4 | 0.2 | |
| **QAM64** | 0.2 | | 0.8 | | | | 0.2 | 7.8 | **90.8** | 0.1 | |
| **QPSK** | | | 1.0 | | | | | 0.1 | 0.1 | **98.7** | |
| **WBFM** | | 60.3 | | | | | | | | | **39.6** |

22,641 of 24,750 correct, which is the 0.9148 reported above — the same number,
with its structure left in. Two cells carry nearly all of the error: WBFM
decided as AM-DSB 1,357 times out of 2,250, and the QAM16 ↔ QAM64 pair at 143
and 176. Everything else in the table is below 3%.

**A subset of a table is not a smaller experiment.** `confusion_table.py
--classes BPSK,QPSK,16QAM,64QAM` narrows which rows are printed, but the model
still had all of its classes available, so the rows do not sum to 100% across
the four columns and the remainder is reported in an `other` column rather than
renormalised away. On the 24-class 2018 run those four rows read 100.0, 100.0,
98.5 and 56.4 — and that last one is not a four-class result, it is 64QAM
losing 30.7% to 256QAM and 11.5% to 128QAM, neither of which is on the table.

The four-class *experiment* is `train_backbone.py --classes
BPSK,QPSK,16QAM,64QAM`, which builds the model with four outputs and so gives
it four answers to choose between. The two numbers answer different questions
and neither substitutes for the other.

The subset is pushed down into the loader rather than applied afterwards, and
that is what makes the run cheap: only those cells are read, so the same memory
buys four times the frames per class that the 24-class run can afford. At 2,048
frames per cell the test set holds **308 frames per (class, SNR) cell**, which
over the eleven SNR levels at or above 10 dB and three seeds is **about 10,200
decisions per class** — enough that a 1% confusion is a hundred events rather
than three. Staging is class-restricted too, and takes every frame of the four
rather than a subsample of all twenty-four: 3.5 GB against 10.5.

Because both loaders draw each cell from one generator walking the classes in
order, iterating four consumes it differently from iterating twenty-four, so a
subset run is **not** trained on the frames the 24-class run gave those
classes. It is its own experiment in its own file, which is the honest thing
for it to be; what it must not be is quoted as a slice of the other.

### The four-class decision table, measured

`train_backbone.py --data rml2016 --classes BPSK,QPSK,QAM16,QAM64 --seeds 3`,
all 1000 frames per (class, SNR) cell
(`train_backbone_rml2016_f1000_c4_colab.npz`). Overall **0.7086 ± 0.0076**
across all twenty SNRs, **0.9439 ± 0.0098** at SNR ≥ 10 dB, in 6.6 minutes on a
Colab T4. Converged: peaks at epochs 19, 23 and 33 under a 60-epoch ceiling
with patience 20, so the last of them stopped sixteen epochs inside the budget.

Decision counts, three seeds pooled, 2,250 frames per row:

| transmitted \ decided | BPSK | QPSK | QAM16 | QAM64 | recall |
|---|---|---|---|---|---|
| **BPSK** | **2235** | 14 | 0 | 1 | 99.3% |
| **QPSK** | 9 | **2232** | 5 | 4 | 99.2% |
| **QAM16** | 9 | 19 | **1978** | 244 | 87.9% |
| **QAM64** | 13 | 13 | 174 | **2050** | 91.1% |

8,495 of 9,000 correct, 94.39%. Two cells hold 83% of the error: QAM16 decided
as QAM64 244 times and the reverse 174 times. Everything outside the QAM pair
is at or below 0.8%.

#### Fewer classes did not make QAM16 easier

The obvious reading of a four-class table is that it is the easy case. Against
the eleven-class run on the same dataset it is not, and the direction is
consistent seed by seed:

| | four classes | eleven classes |
|---|---|---|
| BPSK | 99.33 ± 0.11 | 98.80 ± 0.19 |
| QPSK | 99.20 ± 0.58 | 98.71 ± 0.27 |
| **QAM16** | **87.91 ± 2.27** | **91.82 ± 0.88** |
| QAM64 | 91.11 ± 2.54 | 90.76 ± 0.44 |
| QAM16 → QAM64 | **10.84** | **6.36** |

BPSK and QPSK improve, which is what removing seven competitors is supposed to
do. QAM16 goes the other way, and it is not a seed picked out of three: its
four-class seeds are 84.8, 88.8 and 90.1 against 92.0, 90.7 and 92.8, so the
ranges do not overlap, and neither do the leak's — 8.9, 9.3, 14.3 against 4.9,
6.5, 7.6.

Part of that is arithmetic. With eleven classes QAM16's error also leaves for
8PSK, AM-SSB, QPSK and PAM4, 1.7 points in total; close those routes and the
mass has nowhere to go but QAM64. But the leak rose by 4.5 points, not 1.7, so
rerouting does not account for it. The seed spread also quadruples, 0.88 to
2.27.

**This is stated as an observation, not a mechanism.** Three seeds is thin for
a difference this size, and the two runs are not trained on the same frames:
both loaders draw each cell from one generator walking the classes in order, so
iterating four consumes it differently from iterating twenty-four. What can be
said is that the four-class table is not a magnified corner of the eleven-class
one, and that whichever way the effect is explained, it argues for reporting
the table rather than the average — an overall figure that went *up*, 91.48% to
94.39%, is hiding a class that went down.

#### And it converged four times faster

Best epochs 19, 23 and 33 at 219 steps per epoch is **4,156 to 7,218 gradient
updates**, against 23,439 for the eleven-class run on the same dataset. That is
a third of the updates for a problem with a third of the classes, and it is the
same lesson as the `compare_methods` rows: the update count is a property of
the problem, not a constant of the optimizer, so a ceiling that was safe for
one configuration says nothing about another.

**Rerun at 14 seeds (2026-09-22), `train_backbone_rml2016_f1000_c4_colab_s14.npz`
— quote this one.** Same four classes, all 1000 frames per cell, ceiling 100
with patience 20; every seed converged, peaks between 16 and 46. Above
10 dB that is 42,000 decisions, 2,421 of them wrong:

| transmitted \ decided | BPSK | QPSK | QAM16 | QAM64 | recall |
|---|---|---|---|---|---|
| BPSK | 10,430 | 55 | 0 | 15 | 99.3% |
| QPSK | 55 | 10,388 | 44 | 13 | 98.9% |
| QAM16 | 55 | 63 | 9,108 | 1,274 | 86.7% |
| QAM64 | 59 | 62 | 726 | 9,653 | 91.9% |

The per-SNR matrices (2,100 decisions per cell) show what the pooled table
cannot: QAM16 recall is 83% at 0 dB and then **flat between 86% and 88%
from +2 dB to +18 dB**; QAM64 is flat between 91% and 93%. More SNR does not
separate the pair. On 2018, at 1,024 samples, the same pair is 100.0% from
+14 dB up. So the 2016 ceiling is set by the frame — 128 samples is 16
symbols at 8 samples per symbol, and sixteen symbols of a 64-point
constellation can look like a 16-point one — not by noise. BPSK and QPSK
never reach 100 either: 99.0–99.8% at every level from 0 dB up, with the
misses spread evenly over the other three classes, which reads as a small
fraction of degenerate frames in the file rather than a classifier limit.

### Frame length on 2016: 128 samples against 64

`train_backbone_rml2016_f1000_c4_L64_colab_s14.npz` — the four-class 2016
run above with every frame cut to its first 64 samples, i.e. 8 symbols
instead of 16. Same 14 seeds, same split, all converged. Above 10 dB, 42,000
decisions:

| transmitted \ decided | BPSK | QPSK | QAM16 | QAM64 | recall (64) | recall (128) |
|---|---|---|---|---|---|---|
| BPSK | 10,433 | 66 | 0 | 1 | 99.4% | 99.3% |
| QPSK | 136 | 10,337 | 17 | 10 | 98.4% | 98.9% |
| QAM16 | 60 | 158 | 6,922 | 3,360 | 65.9% | 86.7% |
| QAM64 | 89 | 103 | 2,443 | 7,865 | 74.9% | 91.9% |

Errors go from 2,421 to 6,443, and 5,803 of them are QAM16 and QAM64 read
as each other. The PSK rows do not move. Per SNR the QAM recall is again
flat from +2 dB up (QAM16 63–69%, QAM64 72–77%), just lower: halving the
number of symbols lowers the ceiling, and adding SNR does not lift it. At
0 dB the pooled accuracy is 82.3% against 92.8%. This is the experiment
Moshe asked for on 22 September and it came out the way he predicted —
shorter frames, more errors — with the added information that the errors
are confined to the one decision that needs many symbols.

### Frame length on 2016: 128, 64 and 32 samples

Adding the 32-sample run (`train_backbone_rml2016_f1000_c4_L32_colab_s14.npz`,
4 symbols per frame, 14 seeds, converged) completes the series. Above
10 dB, 42,000 decisions each:

| samples | symbols | errors | BPSK | QPSK | QAM16 | QAM64 |
|---|---|---|---|---|---|---|
| 128 | 16 | 2,421 | 99.3% | 98.9% | 86.7% | 91.9% |
| 64 | 8 | 6,443 | 99.4% | 98.4% | 65.9% | 74.9% |
| 32 | 4 | 10,431 | 99.4% | 92.0% | 46.3% | 62.9% |

The decisions fail in order of how many symbols they need. BPSK never
moves. The QAM pair degrades first and at 32 samples is a coin flip
between the two (4,864 against 5,029 for QAM16). At 32 samples QPSK starts
to fail too, and it fails into BPSK — 809 frames — which is what counting
symbols predicts: a QPSK frame whose four symbols all fall on one diagonal
is indistinguishable from BPSK, and that happens to at most 2 in 16 random
four-symbol frames (the pulse shaping spreads neighbouring symbols into
the window, so the observed 7.7% is below the bound). At 64 samples the
same bound is 2 in 256 and the observed QPSK→BPSK rate is 1.3%.

The recall in every column is again flat from +2 dB upward. So on this
file there are two separate levers: SNR sets where the curve rises, frame
length sets how high it can go. Moshe's prediction on 22 September —
shorter frames, more errors — holds, and the errors land where the
geometry says they must.

### The method table on 2016, and why it is not the 2018 table

`compare_methods_ICRNNA_es_rml2016_e100.npz` — the same four methods, the
same protocol, RML2016.10a as the training domain and the synthetic domain
generated at 128 samples. Five shared classes, 14 seeds, ceiling 100; the
eleven cells that ran out of budget were rerun at 150, and **all 56 cells
have now converged**.

| method | in-domain | cross-domain | gap | 16QAM cross |
|---|---|---|---|---|
| none | 0.948 ±0.009 | 0.920 ±0.011 | +0.028 | 0.770 ±0.050 |
| standard augmentation | 0.969 ±0.005 | 0.968 ±0.010 | +0.002 | 0.918 ±0.033 |
| whitening α=0.75 | 0.864 ±0.007 | 0.843 ±0.004 | +0.021 | 0.645 ±0.051 |
| whitening + standard | 0.931 ±0.007 | 0.913 ±0.007 | +0.017 | 0.867 ±0.013 |

**Every conclusion of the 2018 table is reversed here.** The gap is small
(0.028 against 0.194). The literature augmentation set — rotation,
conjugate flip, additive noise — closes it, and lifts in-domain accuracy as
well. Whitening does not close it and costs 8.4 points in-domain, where on
2018 it cost nothing. Adding whitening to the augmentation set makes it
worse, not better: −3.9 points in-domain and −5.5 cross-domain, and lower
in all 14 seeds on both.

**The rerun at 150 changed nothing that matters.** Eleven cells had peaked
too close to the 100-epoch ceiling for early stopping to fire (ten of
whitening + standard, one of standard augmentation). Rerun with
`AMC_COMPARE_EPOCHS=150 AMC_REDO_UNCONVERGED=1`, seven came back
**bit-identical** — same peak epoch, same accuracy to every digit; the
ceiling had only denied them the twenty epochs that prove the peak. The
other four found later peaks (100→105, 82→109, 92→104, 97→106) worth at
most half a point. The last row moved from 0.930 / 0.913 to 0.931 / 0.913.
Peaks now run 16–45 for none, 26–35 for whitening, 30–87 for augmentation
and 69–109 for whitening + standard. As on 2018, augmentation converges
slowest and whitening is fast and tight; the combination is slowest of all,
which is why a single 100-epoch ceiling was not enough for it.

Two things this does not yet say. It does not say the 2018 attribution is
wrong: that was shown by intervention on 2018 frames, and a 2016 table
cannot refute it. And it does not say why whitening hurts at 128 samples.
The obvious candidate is the envelope estimate itself: at 128 samples the
envelope is a single 128-point periodogram smoothed over 5 bins, which is
the same fraction of the band as 33 bins of 1,024 but a much noisier
estimate, so dividing by it may inject more noise than it removes. That is
a hypothesis, and the sweep below confirmed it. What the table
does say is that the claim "whitening closes the gap at no in-domain cost"
holds for 1,024-sample RadioML 2018 and must not be stated without that
qualifier.

#### Why whitening hurt: the smoothing width, measured

`whitening_smoothing.py --source rml2016` (`whitening_smoothing_rml2016_e100.npz`,
figure 34): the same protocol, α = 0.75, the envelope smoothed over 3 to 65
bins of the 128-point spectrum, 5 seeds, all converged (peaks 19–48 under
100). The `none` row and the 5-bin row reproduce seeds 0–4 of the method
table **bit-identically**, so this is the same experiment with one knob
turned.

The premise was measured first, with no training: the relative error of a
frame's envelope estimate is 0.277 at 128 samples and 5 bins, against 0.106
at 1,024 samples and 33 bins — 2.6 times noisier for the same share of the
band.

| smoothing | share of band | in-domain | cross-domain | in-domain vs none |
|---|---|---|---|---|
| none | — | 0.948 ±0.006 | 0.917 ±0.011 | — |
| 3 bins | 2% | 0.816 ±0.003 | 0.806 ±0.006 | −13.3, 5 of 5 seeds lower |
| **5 bins** (the method table) | 4% | 0.866 ±0.006 | 0.845 ±0.006 | **−8.3**, 5 of 5 lower |
| 9 bins | 7% | 0.921 ±0.004 | 0.889 ±0.009 | −2.7, 5 of 5 lower |
| 17 bins | 13% | 0.947 ±0.004 | 0.906 ±0.006 | −0.1, 3 of 5 higher |
| 33 bins | 26% | **0.962 ±0.001** | 0.908 ±0.003 | **+1.3, 5 of 5 higher** |
| 65 bins | 51% | 0.956 ±0.006 | 0.923 ±0.009 | +0.8, 5 of 5 higher |

**The eight points were the envelope estimate, not whitening.** The cost
falls steadily as the smoothing widens, is gone by 17 bins, and at 33 bins
whitening is *better* than no whitening in-domain, in every seed. That
last part is what separates the two readings the sweep alone was feared
unable to separate. If wider smoothing only meant "less whitening", accuracy
would climb back towards `none` from below and stop there. It goes past it.

So the rule used to carry whitening from 2018 to 2016 — same *share of the
band*, 33 of 1,024 → 5 of 128 — was the wrong rule. What mattered was how
many bins the estimate averages, and at 128 samples keeping the share
meant averaging too few.

**It still does not close the 2016 gap.** Cross-domain accuracy never gets
clearly above `none` (best: 65 bins, +0.6, 4 of 5 seeds), and at no width does
it approach standard augmentation's 0.968. At 33 bins in-domain rises and
cross-domain does not, so the gap widens from 0.031 to 0.054. On 2018 the
gap was the transmitter's spectral envelope and whitening removed it. On
2016 the gap is small and something else, and augmentation is what closes
it. That is consistent with the method table rather than a reversal of it;
what changes is that "whitening costs 8 points at 128 samples" is withdrawn.
It costs 8 points at 5 bins.

### The four-class decision table on 2018 — the one that was asked for

`train_backbone.py --data rml2018 --classes BPSK,QPSK,16QAM,64QAM --seeds 3`,
2,048 of the 4,096 frames per (class, SNR) cell, native 1024-sample frames
(`train_backbone_rml2018_f2048_c4_colab.npz`). 114 minutes on a Colab T4.

Above 10 dB, three seeds pooled, **10,164 decisions per row**:

| transmitted \ decided | BPSK | QPSK | 16QAM | 64QAM | recall |
|---|---|---|---|---|---|
| **BPSK** | **10164** | 0 | 0 | 0 | 100.00% |
| **QPSK** | 0 | **10164** | 0 | 0 | 100.00% |
| **16QAM** | 0 | 0 | **10164** | 0 | 100.00% |
| **64QAM** | 0 | 0 | 2 | **10162** | 99.98% |

**40,654 of 40,656 correct.** Both errors are 64QAM decided as 16QAM, one
each in seeds 0 and 1; seed 2 is perfect. Overall across all 26 SNR levels,
**0.7465 ± 0.0015**, with the curve at chance (0.25) up to −12 dB, 0.796 at
0 dB, 0.983 at +4, and at 1.000 from +14 dB up.

Set against the 2016 table above, where the same QAM pair still confused at
7–11% above 10 dB: the difference is the frame. 1024 samples against 128 is
eight times the symbols to read a constellation from, and at that length
16QAM and 64QAM stop being confusable at high SNR at all.

#### Two things this run did not do

**It did not converge — and the rerun says that cost nothing.** Peaks at
epochs 46, 48 and 59 under a 60-epoch ceiling with patience 20; the last of them
needed 79, so every seed was flagged. Rerun at a 150-epoch ceiling
(`train_backbone_rml2018_f2048_c4_colab_e150.npz`, 139 minutes on the same
T4), all three converged, peaks at 46, 48 and 66, overall
**0.7467 ± 0.0015** against 0.7465 ± 0.0015.

Seed by seed the comparison is sharper than the averages:

- **Seeds 0 and 1 are bit-identical** to the 60-epoch run — same peak epoch,
  same overall accuracy to six decimals, all 26 per-SNR matrices equal. They
  had reached their peak at 46 and 48 all along; what the 60 ceiling denied
  them was the twenty further epochs that would have *proved* it.
- **Seed 2 moved** from a peak at 59 to one at 66, and its accuracy at every
  SNR moved by a point or less — with the movement confined to the QAM pair,
  16QAM down and 64QAM up by about the same amount at each level, the two
  summing to what they were.

Read that as a note on the convergence test rather than a reason to weaken
it. `best_epoch + patience > ceiling` cannot tell "peaked, and would have
stopped" from "still climbing" — both look the same from inside a ceiling
that ended before patience could fire — so it flags both, and two of these
three were the first kind. The flag means *unproven*, not *wrong*; the cost
of turning it into either is one rerun, and the weaker test that would not
have flagged them is the one that missed real cases in `compare_methods`.

**It did not store the table that matters — at first.** The pooled matrix
above 10 dB is a diagonal, which demonstrates the method and tells nobody
anything about where decisions go. The informative tables are lower down, and
this run recorded only per-SNR *accuracy* there, not the matrix.
`train_backbone.py` now stores a confusion matrix at every SNR level
(`confusions_by_snr`), `confusion_table.py --snr 0` prints one, and
`eval_by_snr.py` rebuilt them for this run from its saved checkpoints — after
proving, count for count, that it had reconstructed the same test set the file
was scored on. All three seeds matched. The file now carries all 26 levels.

#### Where the decisions go, by SNR

Three seeds pooled, **924 decisions per row** at each level. Recall on the
diagonal, and the one cell each class loses most to:

| SNR | BPSK | QPSK | 16QAM | 64QAM | what the errors do |
|---|---|---|---|---|---|
| −8 dB | 70.7 | 70.0 | **2.5** | **8.5** | everything drains into QPSK: 16QAM → QPSK 69%, 64QAM → QPSK 73% |
| −4 dB | 100.0 | 73.3 | **15.7** | 48.4 | 16QAM splits between QPSK (42%) and 64QAM (42%); QPSK leaks to 64QAM 18% |
| 0 dB | 100.0 | 100.0 | 58.3 | 60.1 | the PSK/QAM boundary is closed; the QAM pair is a coin flip against each other |
| +4 dB | 100.0 | 100.0 | 97.6 | 95.5 | 42 + 22 errors, all inside the QAM pair |
| ≥ 10 dB | 100.0 | 100.0 | 100.0 | 99.98 | 2 in 40,656 |

The table at 0 dB, in full:

| transmitted \ decided | BPSK | QPSK | 16QAM | 64QAM | recall |
|---|---|---|---|---|---|
| **BPSK** | **924** | 0 | 0 | 0 | 100.0% |
| **QPSK** | 0 | **924** | 0 | 0 | 100.0% |
| **16QAM** | 0 | 9 | **539** | 376 | 58.3% |
| **64QAM** | 0 | 6 | 363 | **555** | 60.1% |

2,942 of 3,696, 79.6%. Of the 754 errors, 739 stay inside the QAM pair.

**The decision resolves in two stages, at two different SNRs.** *Is this PSK
or QAM* is settled somewhere between −4 and 0 dB: at −4 a 16QAM frame still
goes to QPSK 42% of the time, at 0 dB it never does. *Which QAM* is settled
between 0 and +6: a coin flip at 0, 96–98% at +4, perfect by +8. Those are
different questions with different SNR thresholds, and an accuracy curve
averages them into one number per level.

**Below the first threshold the unreadable classes do not scatter, they
sink.** At −8 dB, 16QAM and 64QAM are read as QPSK 69% and 73% of the time
— far above the 25% a uniform spread would give — and 16QAM's own recall,
2.5%, is well *below* chance. A QAM constellation under that much noise is
being decided as the simplest constellation it resembles, not as "unknown".
That is the same shape as the leave-one-out finding (an unseen modulation
lands inside its family) seen from a different angle: here the class *was*
trained on, and it still drains to the nearest simpler one once the SNR takes
away what distinguishes it.

One caveat travels with these tables: 924 decisions per row is one third of
what the pooled table has, so a 1% cell is nine frames.

The other one is closed. These are the 60-epoch run's tables; the converged
150-epoch run gives, recall per class, 60 → 150:

| SNR | BPSK | QPSK | 16QAM | 64QAM |
|---|---|---|---|---|
| -8 dB | 70.7 → 70.0 | 70.0 → 69.2 | 2.5 → 2.8 | 8.5 → 9.7 |
| -4 dB | 100.0 → 100.0 | 73.3 → 72.5 | 15.7 → 15.4 | 48.4 → 49.1 |
| +0 dB | 100.0 → 100.0 | 100.0 → 100.0 | 58.3 → 56.2 | 60.1 → 61.6 |
| +4 dB | 100.0 → 100.0 | 100.0 → 100.0 | 97.6 → 97.5 | 95.5 → 96.0 |

Nothing moves by more than two points, and every move is inside the QAM pair.
The 0 dB table from the converged run, three seeds pooled:

| transmitted \ decided | BPSK | QPSK | 16QAM | 64QAM | recall |
|---|---|---|---|---|---|
| **BPSK** | **924** | 0 | 0 | 0 | 100.0% |
| **QPSK** | 0 | **924** | 0 | 0 | 100.0% |
| **16QAM** | 0 | 9 | **519** | 396 | 56.2% |
| **64QAM** | 0 | 6 | 349 | **569** | 61.6% |

2,936 of 3,696, 79.4%.

### The faithful build, measured

`colab/icrnna_faithful_2016.py`, built from the paper rather than from a
reproduction, run under the paper's own hyperparameters — batch 32, 58 epochs,
LR on validation *loss* with patience 5 — across three seeds
(`icrnna_faithful_results.json`):

| | |
|---|---|
| test accuracy | **61.75% ± 0.07** (61.79, 61.65, 61.82) |
| paper (Table 3) | 63.24% |
| difference | **−1.49 points** |
| parameters | 794,827 (the paper states 0.79M) |

The parameter count matches the paper's figure, which says the architecture was
read correctly. The accuracy does not, and three things make that worth stating
carefully rather than waving through:

1. **The deficit is not seed noise.** Spread across seeds is 0.07 points; the
   gap to the paper is 21× that. Whatever causes it is systematic.
2. **The file's own verdict — "within ~1.5 points: the reproduction stands" —
   passes by 0.01 points.** A threshold a result squeaks under is not evidence
   that the result is fine. It is written here as a near miss, because that is
   what it is.
3. **Two of the three seeds peaked at the final epoch** (best epoch 58, 58, 55)
   and early stopping at patience 15 never fired. The model was still improving
   when the budget ran out, so the paper's "Number of Epochs 58" is binding
   here.

#### It was the epoch budget

Same build, same everything, ceiling raised to 150 so that early stopping at
patience 15 decides where to stop (`icrnna_faithful_e150_results.json`):

| | 58-epoch ceiling | 150-epoch ceiling |
|---|---|---|
| best epoch | 58, 58, 55 — at the ceiling | **107** — early stopping decided |
| test accuracy | 61.75% ± 0.07 | **63.21%** |
| against the paper's 63.24% | −1.49 | **−0.03** |

**The faithful build reproduces the paper.** It needed nearly twice the
published epoch count to do it, and that is the whole of the 1.49-point
deficit; nothing about the architecture was involved. The paper's "Number of
Epochs 58" is best read as a report of where its own early stopping landed than
as a recipe, and it does not transfer.

The gain is everywhere the signal is usable — +1.4 to +3.6 points at every SNR
from −8 dB up, +1.78 across the plateau — and nothing at all below −10 dB,
where there is nothing to learn. Trained to convergence it also passes the peer
reproduction, 63.21% against 62.23% overall and 92.55% against 91.48% at
SNR ≥ 10 dB.

#### What that implies for this project's own numbers

An under-trained run reports a floor, not a measurement, and this one was under
the ceiling by 49 epochs without anyone noticing. The same question applies
here, and the arithmetic is not reassuring:

| run | batch | epochs | gradient updates |
|---|---|---|---|
| faithful ICRNNA, converged | 32 | 107 | **515k** |
| `train_backbone --data rml2016` | 256 | 60 ceiling | 36k |
| `train_backbone --data rml2018` | 256 | 60 ceiling | 52k |

Fourteen times fewer updates on the same dataset. That is not evidence those
runs were under-trained — early stopping at patience 20 may well have fired
long before 60 — but **nothing in the saved results says which**, because the
stopping epoch was never recorded. It is now: `cnn.train_model` reports where
it stopped, `train_backbone.py` stores `best_epochs` in its `.npz` and says
loudly when a seed peaked at the ceiling.

#### 2016 converged — by one epoch

Rerun at a 300-epoch ceiling with the stopping epoch recorded
(`train_backbone_rml2016_f1000_colab_e300.npz`, one seed): **best epoch 39,
early stop at 59**. The committed 60-epoch run's ceiling was 60. The result is
bit-identical to that run's seed 0 — 0.624758 overall and 0.914909 at
SNR ≥ 10 dB in both — so nothing after epoch 39 improved on it and the
committed number is a measurement, not a floor.

The margin was **one epoch**. Had the trajectory peaked two epochs later, or
patience been 21, the committed run would have been cut off at the ceiling and
nothing in the file would have said so. Sixty epochs is not a comfortable
default for this configuration; it happened to be enough.

#### 2018 converged too, with room to spare

Same check at a 200-epoch ceiling
(`train_backbone_rml2018_f512_colab_e200.npz`, one seed): **best epoch 26,
early stop at 46**, against the committed run's ceiling of 60 — fourteen epochs
of margin. Bit-identical to that run's seed 0 again, 0.561637 overall and
0.856546 at SNR ≥ 10 dB in both. **Both committed classification results are
measurements, not floors.**

2018 converged *sooner* in epochs than 2016 despite being the harder problem,
which is the wrong way round until the unit is fixed:

| | train frames | steps/epoch | best epoch | gradient updates |
|---|---|---|---|---|
| 2016, 11 classes, 128 samples | 154,000 | 601 | 39 | **23,439** |
| 2018, 24 classes, 1024 samples | 223,392 | 872 | 26 | **22,672** |

Two datasets differing in class count and in sequence length by a factor of
eight, converging within 3% of each other in *updates*. Under this optimizer
what the model needs is about 23k steps, and the epoch count is just that
number divided by however many batches the dataset happens to make.

#### Which makes the 60-epoch ceiling a problem for the sweeps

`compare_methods.py` and `whitening_seeds.py` train on the five shared classes:
99,840 frames, 69,888 after the split, **273 steps per epoch**. At 23k updates
that is **84 epochs** — above the 60 both are usually run with.

That is an extrapolation, not a measurement. Five classes is a much easier
problem than eleven or twenty-four and may well converge in far fewer updates.
But the consequence if it does not is worse here than for a single accuracy:
every number those files report is a *difference between two cells*, so one
cell stopped at the ceiling does not make the comparison low, it makes it
meaningless.

Both now record `best_epochs` and refuse to report quietly: a cell that peaked
at the ceiling is named as it happens and again in a warning above the summary
table. The three cells still missing from `compare_methods_ICRNNA_es.npz` will
therefore answer the question for the seventeen that are already in it — those
predate the recording and are stored as NaN.

**Being faithful did not move it toward the paper.** Against the peer
reproduction in `model_zoo.ICRNNA` on the same dataset: 61.75% versus 62.23%
overall, and 90.76% versus 91.48% at SNR ≥ 10 dB. Correcting the five
differences toward the published description moved the number half a point
*away* from the published number.

That comparison is suggestive and nothing more, because the two runs differ in
protocol as well as architecture — batch 32 against 256, LR scheduled on
validation loss against validation accuracy, 58 fixed epochs against 60 with
patience 20, no mixed precision against AMP. It is not an architecture
comparison and must not be reported as one. What it does rule out is the
comfortable assumption that the five differences were holding the number back.

The two curves are indistinguishable in the noise-limited regime — 33.19%
against 33.29% averaged below 0 dB — and separate only across the plateau.
Whatever differs between them acts where the signal is clean.

---

## Setup

Python 3.12, virtual environment at `C:\Users\yusuf\.venvs\amc` (outside
OneDrive — packages are multi-GB and syncing them is painful).

```bash
cd src && "C:\Users\yusuf\.venvs\amc\Scripts\python.exe" compare_methods.py --arch ICRNNA --seeds 5 --epochs 60 --patience 20
```

Datasets live in `C:\Users\yusuf\amc-data` (also outside OneDrive, ~40 GB,
untracked). GPU: RTX 3070 Laptop, 8 GB. **Keep the machine on mains power** —
on battery the GPU is capped near 35 W and runs about 5× slower, measured.

`h5py` is blocked by this machine's application-control policy, which refuses
unsigned DLLs in a user-writable venv. `radioml.py` reads `.npy` memmaps
converted once by `convert_radioml.py`, falling back to pure-Python pyfive.

Training can also run on Colab: `colab/run_training.py` is a single cell that
clones this branch, finds the dataset in Drive, stages it to local disk and runs
`train_backbone.py` with `--out-dir` pointing back at Drive, so results survive
the runtime being reclaimed and a rerun resumes rather than restarts. The 21 GB
file does not have to be in Drive, and does not fit in a free one: no run reads
all of it, so `export_subset.py` carves out exactly the frames a given
`--frames-per-cell` would use — 2.6 GB at 512 — and both sides select those
frames with the same function at the same seed, so the two subsets hold the same
frames rather than merely the same number of them.

---

## Layout

```
src/modem.py            baseband IQ generator, 11 modulation classes
src/radioml.py          RadioML 2018.01A loader
src/rml2016.py          RML2016.10a loader, same interface as radioml.py
src/export_subset.py    carve an uploadable subset out of 2018
src/domains.py          domain abstraction: RadioML, synthetic, capture
src/features.py         cumulants and instantaneous features
src/augment.py          augmentation transforms, including whitening
src/model_zoo.py        ICRNNA, ResNet1D, GRU, Transformer, backbone()
src/cnn.py              training loop, splits, evaluation
src/train_backbone.py   plain classification on 2018 or 2016, current backbone
src/plots.py            the run's two figures, rebuildable from a saved .npz
src/crossdomain.py      the central train-on-A / test-on-B experiment
src/whitening_*.py      whitening and its alpha sweep
src/compare_methods.py  whitening against the literature baseline
src/sink_*.py           where an unseen modulation lands
src/literature_models.py  VT-CNN2, LSTM2, MCLDNN, PET-CGDNN, for the comparison
src/eval_by_snr.py      per-SNR confusion matrices rebuilt from checkpoints
src/vote_fusion.py      do the models disagree? majority vote on the same frames
src/verify_dataset.py   check the data assumptions before training on them
tools/                  tables, figures and the RML2016.10a SNR measurement
colab/run_training.py   run train_backbone.py on Colab
colab/run_experiment.py run the open-work sweeps on Colab (AMC_TASK=...)
colab/icrnna_faithful_2016.py  paper-faithful ICRNNA, for calibration
colab/official_icrnna_2016.py  the ICRNNA authors' own code, run as published
deck/                   meeting decks and study pages, built from the result files
```

The progress deck for the semester meetings lives in `deck/`. It is
generated, not drawn: `extract_data.py` reads every quoted number out of the
committed result files into `deck_data.json`, `crops.py` cuts the panels it
shows out of `figures/`, and `build.js` (pptxgenjs) writes
`amc_progress.pptx` from those two inputs alone. Rebuild with

    python deck/extract_data.py && python deck/crops.py
    cd deck && npm install && node build.js

New results are appended as slides; a withdrawn claim keeps its slide.

---

## Rules that keep this honest

- **Predict before measuring.** A hypothesis written down after seeing the
  number is not a hypothesis.
- **Negative results are kept.** Four explanations for the gap were tested and
  refuted; they are part of the argument, not failed drafts.
- **Controls that can come out the other way.** A control which confirms
  whatever happens is decoration.
- **Three-way splits wherever anything is selected.** `cnn.split(...,
  val_fraction=0.15)` returns `(train, val, test)`; early stopping, the LR
  schedule and checkpoint choice read validation only. Passing the test set
  there means selecting on the number being reported.
- **Long runs checkpoint and resume.** Every sweep writes partial results after
  each cell and skips finished ones on restart. This is a laptop, and runs have
  been lost to a flat battery.

---

## The diagnosis — attribution by intervention

The gap concentrates almost entirely in one cell: 16QAM read as 64QAM. Four
explanations were tested and each refuted by controlled experiment:

| # | hypothesis | test | outcome |
|---|---|---|---|
| 1 | occupied bandwidth differs | `match_radioml_params.py` | refuted — matching bandwidth did not recover 16QAM |
| 2 | constellation density | `diagnose_16qam.py` | refuted — density is not what the model reads |
| 3 | symbol-timing regularity | `test_timing_hypothesis.py` | refuted — RadioML has **no** symbol-rate spectral line, and driving the line down did not move the prediction |
| 4 | channel impairments (CFO, multipath, IQ imbalance) | `ablate_impairments.py` | refuted — removing them one at a time changed little |

What did work is the piece of this project with the least precedent in the
literature: **phase-preserving magnitude substitution** (`narrow_the_search.py`).
Take a frame from the failing domain, keep its phase spectrum exactly, and
substitute only the magnitude spectrum of the training domain. Nothing else
changes — not the symbols, not the timing, not the channel.

The prediction follows the magnitude spectrum. That is an intervention on the
signal rather than an observation of the representation, so it supports a causal
claim rather than a correlational one. Combined with the four refutations, the
cause is the **transmitter's spectral envelope**, chiefly its pulse-shaping
roll-off.

The interventions themselves are signal-level and do not depend on the
classifier; the accuracies attached to them are pending re-measurement.

---

## The fix

**Attempt 1 — randomise the envelope during training. Failed.** Documented with
its ablation rather than dropped, because the failure is informative: teaching
invariance by sampling nuisance parameters did not transfer.

**Attempt 2 — remove the envelope. Works.** Spectral whitening: divide each
frame's spectrum by a smoothed estimate of its own magnitude envelope,
`X / smooth(|X|)^α`, with a circular moving average so the wrap-around is not a
discontinuity.

α is swept from 0 to 1, five seeds per point, 100-epoch ceiling
(`whitening_seeds_e100.npz`, 125 minutes on an A100, every cell converged):

| α | in-domain | cross-domain | gap | 16QAM | stopping epochs |
|---|---|---|---|---|---|
| 0.00 | 0.999 ±0.001 | 0.805 ±0.005 | +0.193 | 0.056 ±0.018 | 51, 41, 48, 36, 38 |
| 0.25 | 0.999 ±0.000 | 0.855 ±0.009 | +0.144 | 0.305 ±0.044 | 35, 47, 40, 31, 25 |
| 0.50 | 1.000 ±0.000 | 0.940 ±0.019 | +0.060 | 0.739 ±0.096 | 62, 52, 36, 39, 31 |
| 0.75 | 0.999 ±0.001 | 0.988 ±0.007 | +0.011 | 0.981 ±0.009 | 26, 26, 33, 59, 35 |
| 1.00 | 0.998 ±0.001 | 0.993 ±0.001 | +0.005 | 0.985 ±0.006 | 26, 22, 35, 30, 36 |

**The claim that partial whitening beats full whitening does not survive.** It
came from the superseded backbone at one seed per point; on the current one,
across five seeds, α=1.0 is at least as good as α=0.75 (0.993
against 0.988, a difference of +0.005 at a pooled spread of
0.005) and seven times more stable (±0.001 against
±0.007). The curve is monotone in α and saturates by 0.75. That was
the one concrete disagreement this project had with WhiteNet's choice of full
whitening, and it is withdrawn: on this task the two agree.

Two things the sweep does say. The recovery is carried by 16QAM at every step
— 0.056 → 0.305 → 0.739 → 0.981 → 0.985 — so α is turning one confusion off,
not lifting everything a little. And the middle of the sweep is where the seeds
disagree: α=0.5 has a spread of ±0.019 on cross-domain accuracy and
±0.096 on 16QAM, against ±0.001–0.007 at either end. Half-removing the
envelope leaves a model that sometimes learns to read through it and sometimes
does not; removing it fully makes that a non-question.

The existing tables keep α=0.75 because that is what they measured, and 0.005
is not a reason to rerun them. New work should use α=1.0.

The α=0 row reproduces `compare_methods`' "none" row bit for bit, all five
seeds, and the α=0.75 row reproduces its whitening row on seeds 0–2 — the two
seeds that differ were the two run in a different session on a different GPU.
That is the same pipeline-determinism-within-a-machine seen throughout, and
it doubles as a check that the two scripts are measuring the same thing.

### What the mechanism is not

The working account through most of the project was shortcut learning: the model
reads modulation order off the spectral envelope. Two experiments were designed
to confirm it and both killed it — most directly the envelope transplant, where
handing 16QAM frames a 64QAM envelope left the prediction at 16QAM. The envelope
does not *carry* the class; removing it still helps. That distinction is
unresolved and is stated as unresolved.

---

## Where an unseen modulation lands

Train on 23 of RadioML's 24 classes, probe with the held-out one at 10 dB and
above, and record which known class absorbs it. The claim under test: **an
unseen modulation lands inside its own modulation family** — QAM into QAM,
PSK into PSK — rather than scattering or draining into one dense class.

On the earlier backbone this was checked five independent ways (hand-drawn
taxonomy, the model's own embedding, four architectures, a discriminating
control, label permutation). **It has now been re-measured on the current
backbone**, with the taxonomy fixed before the run as before:

| family | members |
|---|---|
| ASK | OOK, 4ASK, 8ASK |
| PSK | BPSK, QPSK, 8PSK, 16PSK, 32PSK |
| APSK | 16APSK, 32APSK, 64APSK, 128APSK |
| QAM | 16QAM, 32QAM, 64QAM, 128QAM, 256QAM |
| analog | AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM |
| other | GMSK, OQPSK |

### The 24-class leave-one-out, measured

`sink_class_24_e60.npz` / `sink_class_24_partial_e60.json` — ICRNNA, 256
frames per (class, SNR) cell, 70/15/15 split, early stopping on validation
with patience 10 under a 60-epoch ceiling. **All 24 runs converged**: best
epochs 7 to 43, so the latest stop was epoch 53, seven inside the ceiling.

**16 of 24 held-out classes sink into their own family. Chance is 3.5.**
Ten of the sixteen land on the order-adjacent member (4ASK ↔ 8ASK, 16PSK ↔
32PSK, 128APSK → 64APSK, 128QAM → 256QAM, and all four AM variants pairwise);
six land elsewhere in the family (OOK → 8ASK, QPSK → 16PSK, 8PSK → 32PSK,
32QAM → 128QAM, 64QAM → 256QAM, 256QAM → 64QAM). The sink is usually
decisive — 15 of the 24 top shares are above 75%, ten of them above 95%.

The eight misses, in full, because they are the informative part:

| held out | sink | share | second |
|---|---|---|---|
| BPSK | OQPSK | 62% | QPSK 33% |
| 16APSK | 16QAM | 100% | — |
| 32APSK | 32QAM | 82% | 128APSK 13% |
| 64APSK | 32QAM | 44% | 128APSK 37% |
| 16QAM | 64APSK | 42% | 128APSK 31% |
| FM | GMSK | 100% | — |
| GMSK | 8PSK | 74% | FM 18% |
| OQPSK | 16QAM | 75% | 64QAM 10% |

Four of the eight are APSK ↔ QAM. Both are amplitude-and-phase constellations
— APSK is QAM on rings — and the model treats them as one family: 16APSK goes
to 16QAM with 100% of its probes, and 16QAM's top three sinks are all APSK
or QAM. Two more are FM ↔ GMSK, both constant-envelope frequency
modulations, and BPSK → OQPSK is a phase-shift keying landing on an offset
phase-shift keying that the taxonomy filed under "other". So the misses are
not scatter; they are a disagreement between the hand-drawn families and the
signal-level ones. **That reading is post hoc.** Merging APSK with QAM would
give 20 of 24, but the taxonomy was fixed in advance precisely so that this
kind of rescoring is not available, and the number quoted is 16.

Figure: `figures/22_sink_class_24_e60.png`.

### The architecture control, measured

`sink_across_arch_e60.json` — six held-out classes fixed in advance (the
highest order of each digital family plus FM and OQPSK), five architectures:
IQNet (1-D CNN), ResNet1D, GRU, Transformer and the backbone itself. 30 runs,
same protocol, 29 converged; `Transformer | 32PSK` peaked at epoch 56 under
the 60 ceiling with patience 10, so its sink stands and its in-distribution
accuracy is a floor.

| held out | IQNet | ResNet1D | GRU | Transformer | ICRNNA |
|---|---|---|---|---|---|
| 8ASK | 4ASK 100% | 4ASK 100% | 4ASK 100% | 4ASK 100% | 4ASK 100% |
| 32PSK | 16PSK 92% | 16PSK 100% | 16PSK 82% | 16PSK 70% | 16PSK 97% |
| 128APSK | 64APSK 54% | 64APSK 61% | 64APSK 38% | 64APSK 86% | 64APSK 81% |
| 256QAM | 64QAM 70% | 64QAM 63% | 64QAM 50% | 64APSK 21% | 64QAM 77% |
| FM | AM-DSB-WC 62% | GMSK 100% | GMSK 100% | 32PSK 85% | GMSK 100% |
| OQPSK | 16QAM 26% | 16PSK 39% | 16APSK 55% | 16QAM 38% | 16QAM 75% |
| **same family** | **5/6** | **4/6** | **4/6** | **3/6** | **4/6** |

Chance is 0.78 of 6. Three of the six held-out classes have the same sink on
all five architectures; 256QAM has the same sink on four. The two hard cases
are the two the rule put in on purpose — FM and OQPSK — and they miss on
every architecture except IQNet's FM, which lands on an AM class. The
finding is not a property of one inductive bias.

The ICRNNA column doubles as a reproducibility check. The two scripts draw the
same training frames from the same seed, and the six ICRNNA rows have the
same best epoch and the same in-distribution accuracy to every digit as the
corresponding rows of the 24-class run — the trainings are bit-identical.
The probe frames are an independent draw, and moving them shifts the sink
shares by at most 0.3 points and changes no sink.

Figure: `figures/26_sink_across_architectures_e60.png`.

### What is still on the old backbone

The embedding-based check, the discriminating control, the label permutation
and family recovery (`family_recovery.py`) have not been rerun. The two
results above are the load-bearing ones; the rest are corroboration and are
queued behind them.

---

## Caveats, stated rather than buried

- **Both domains are synthetic.** RadioML is simulated; so is the generator here.
  Until SDR capture is possible this is a study, not a result. It is the single
  largest weakness of the work.
- **Spectral whitening is not novel.** WhiteNet (arXiv:2608.06581) arrived at
  essentially the same operation, on *real* over-the-air captures, and the claim
  of novelty was dropped on finding it. See `LITERATURE.md`. What survives is the
  attribution method and the sink finding; the partial-versus-full disagreement
  did not survive its own rerun (see the α sweep).
- **`model_zoo.ICRNNA` is not the published architecture.** It is transcribed
  from a peer's reproduction. Checked against El-Haryqy et al., *Results in
  Engineering* 26 (2025) 104783, it differs in five places: conv1 kernel 5 vs 3,
  two max-pools vs one, one BatchNorm after the LSTM stack vs one per layer, no
  attention dropout or LayerNorm, and one dense layer of 128 at dropout 0.5 vs
  two of 128 and 64 at 0.3. **Do not call the current model "the published
  architecture".** The faithful build has now been run and, trained to
  convergence, reproduces the paper (63.21% against 63.24%).
- **`dann.py` has not been ported.** It builds on a legacy feature extractor
  with no equivalent in the current backbone, so its numbers are not comparable
  to the rest until it is rewritten.
- **Open-set recognition is the adjacent field** for the sink work, and a few
  searches do not cover it. The novelty question there is genuinely open.

---

## Open work

**The focus is RML2016.10a** (since 2026-09-22), because the people around the
project work on it and results have to be comparable with theirs. Nothing
measured on 2018 is retracted, but no new 2018 work is planned for now.

Next, in order:

1. **Do the classifiers disagree enough to combine?** Moshe's exercise: run the
   top three models on the same test frames and count how often they agree,
   with a majority vote where two of three do (`vote_fusion.py`, Colab task
   `vote_2016`). Then all five, and a binary version, where decision fusion
   with each classifier's detection and false-alarm rates is well understood.
2. **Accuracy at each SNR, method against method**, rather than one average
   over all twenty labels.
3. **Smaller sets of modulations.** In practice only a few types are present at
   once; how does that change the picture? Not started yet, by agreement.
4. Port `dann.py` to the current backbone, or drop the comparison.
5. Real SDR capture when hardware and lab access allow.
6. The remaining sink corroborations on the current backbone: embedding
   similarity, the discriminating control, label permutation, family recovery.

**Closed.** The sink thread on the current backbone — 16 of 24 held-out classes sink into their own family against a chance of 3.5, all 24 runs converged, and the six-class control holds on all five architectures. The α sweep — 25 cells, all converged; full whitening is at least as good as partial and the WhiteNet disagreement is withdrawn. The method table at a ceiling every cell had room under — `compare_methods_ICRNNA_es_e100.npz`, 20 of 20 converged, the three augmentation cells rerun at 150 and bit-identical. The four-class decision table on 2018 — measured at every SNR,
converged at a 150 ceiling (peaks 46, 48, 66), two of three seeds
bit-identical to the 60-epoch run and the third within a point at every
level. Validate `colab/icrnna_faithful_2016.py` against 63.24% — at the
paper's 58-epoch ceiling it lands 1.49 points under; trained to convergence it
reaches 63.21%. Check whether the two classification runs converged — both did,
best epoch 39 on 2016 and 26 on 2018.

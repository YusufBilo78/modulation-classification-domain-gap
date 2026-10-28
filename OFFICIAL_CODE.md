# The ICRNNA paper's own code

El-Haryqy et al., *Results in Engineering* 26 (2025) 104783, published April
2025 with "data will be made available on request" and no code. The authors
released their training code sixteen months later:

- **Repository:** https://github.com/noureddine395/mproved-automatic-modulation-recognition-using-deep-learning-with-additive-attention
- **Contents:** four commits, all on **2026-08-08** (UTC+1), by the account `noureddine395`. Read here at commit `ac000a0`.
- **Licence and archive:** MIT; archived as Zenodo 10.5281/zenodo.21854712 (v1.0.0).
- **Who uploaded it:** the account shows no name. README and `CITATION.cff` name the paper's five authors and call it the official code. The same account holds two other AMR repositories:
  - `MCSBA`, a CNN-BiLSTM-attention model on 2016.10a/b and RML22, with a placeholder paper title;
  - `utcn-amr-fpga`, a small temporal CNN on an Intel Cyclone IV FPGA, 62.55 % on 2016.10a.
- **Files:** `rml2016_10a.py` and `rml2016_10b.py`, one script per dataset. TensorFlow/Keras.
- **No trained weights.** The scripts save `best_model.keras` and `final_model.keras` locally; neither is in the repository.

## What the code settles that the paper left open

Read line by line against the paper and against our two PyTorch versions:
- the faithful build, `colab/icrnna_faithful_2016.py`, written from the paper alone before the code existed;
- Esmer's re-implementation, `model_zoo.ICRNNA`, which all our other results use.

| | paper text | official code (`rml2016_10a.py`) | our faithful build | Esmer's version |
|---|---|---|---|---|
| conv layers | 64, 128 filters, kernel 3 | 64, 128, kernel 3 | same | conv1 kernel 5 |
| max-pools | ambiguous | **one**, after conv2 | one | two |
| BatchNorm in RNN | after each LSTM | after each LSTM | same | one after the stack |
| attention inputs | not stated | query = LSTM-2 output after BN and dropout; key/value = raw LSTM-2 output | one input, the BN'd output | — |
| attention size | not stated | W1, W2: 128 units, no bias; V: 1 | W: 256 × 256 with bias | — |
| "scaling" | "if scaling is enabled" | a **learnable scalar**, initialised to 1 | none | none |
| attention dropout / LayerNorm | 0.1 / yes | 0.1 on the weights / LayerNorm on the context | same | neither |
| dense head | 128, 64, dropout 0.3 | same | same | one layer of 128, dropout 0.5 |
| normalization | "normalization" | Keras `Normalization(axis=-1)`: mean and variance **per time index**, over all frames and both I and Q, fitted on the **whole dataset before the split** | each frame to unit RMS | each frame to unit power |
| L2 1e-4 | yes | `l2(1e-4)` on conv, LSTM input kernels and dense kernels | Adam `weight_decay=1e-4` on every parameter | AdamW 1e-4 |
| optimizer | not named | Adam, lr 1e-3, **gradient clipping `clipnorm=1.0`** | Adam, no clipping | AdamW, clip 5 |
| LR schedule | ×0.2 after 5 flat epochs (val loss) | same | same | ×0.2 on val accuracy |
| epochs | Table 2: "58" | **up to 100, early stopping on val loss, patience 6, best weights restored** | fixed 58 (or 150 ceiling, patience 15 on val accuracy) | ceiling 100, patience 20 |
| batch | 32 | 32 | 32 | 256 |
| split | 70/15/15 | 70/15/15, random (not stratified), seed 2016, **one split, one run** | stratified per (class, SNR), per seed | stratified per cell |
| parameters | 0.79 M | **794,188** (Keras count, BN statistics included) | 794,827 | 786,379 |

Our faithful build got the structural guesses right:
- one pool;
- BatchNorm after each LSTM;
- dropout on the attention weights.

It missed four details:
- the attention's two-input form and its learnable scale;
- the normalization;
- gradient clipping;
- the stopping rule.

## Three things the code changes in our reading of the paper

1. **"58 epochs" was where early stopping fired, not a chosen budget.**
   - The code trains up to 100 epochs and stops when the validation loss has not improved for 6 epochs, restoring the best-loss weights.
   - The paper's text says the validation loss "stabilizes around 1.05 after epoch 40", which fits a stop at 58.
   - Our earlier point "their own curve is still rising at 58" stands for the validation *accuracy* (Fig. 7). But their stopping rule watches the *loss*, which had flattened.
   - So "it needs more than 58 epochs" is true of our faithful build under a fixed 58-epoch budget (61.75 %). It is not a criticism of their procedure. The paper deck must be reworded once their code has been run.
2. **The split is not reproducible, despite the README.**
   - `preprocess_data` builds the class list from a Python `set` of strings. That order changes from one Python process to the next (hash randomisation) unless `PYTHONHASHSEED` is fixed.
   - The data are concatenated in that order and then shuffled with seed 2016, so each run of their script draws a different train/validation/test split.
   - Checked here: the same fake dataset gives split fingerprint `54c27115e056` with `PYTHONHASHSEED=0` and `1bf5e5673d36` with `PYTHONHASHSEED=1`.
   - The label ids themselves are stable (`LabelEncoder` sorts the names).
3. **The normalization uses the test set.**
   - The scaling statistics are computed on all 220,000 frames before splitting.
   - The effect on accuracy is probably small (one mean and variance per time index), but it is a leak.

What the code confirms:
- One training run (the script has no loop over seeds).
- The test set is used once, on the weights with the lowest validation loss.
- The per-SNR figures in the paper match what the plotting functions draw. F1 is `average='weighted'`.

## Running it

`colab/official_icrnna_2016.py` imports their module and calls their own functions:
- `load_data`, `preprocess_data` and `build_improved_model`;
- their split, callbacks and `fit` arguments, copied from their `__main__`.

It adds:
- three runs, each with its own weight-initialisation seed;
- a fixed `PYTHONHASHSEED`, so all runs share one split, recorded as a fingerprint;
- a results file with the per-epoch history, the stopping epoch, the test accuracy and the test confusion matrix at every SNR;
- resume.

Colab: `AMC_TASK=official_2016` in `colab/run_experiment.py`. The task clones their repository at `ac000a0`. Results go to `Drive/amc-results/official_icrnna/official_icrnna_2016_results.json`.

Smoke-tested here on a synthetic pickle (TensorFlow 2.21, CPU):
- training;
- resume;
- the split-fingerprint check;
- the figures.

## Runs on the real data (2026-10-13, complete)

`official_icrnna_2016_results.json`, Colab GPU, one split for all three runs: fingerprint `b86f4de2f128` (`PYTHONHASHSEED=0`).

| | run 1 (seed 0) | run 2 (seed 1) | run 3 (seed 2) | paper |
|---|---|---|---|---|
| test accuracy | **62.71 %** (20,694 of 33,000) | **62.45 %** (20,609) | **62.52 %** (20,633) | 63.24 % |
| stopped by early stopping at | epoch 61 | epoch 64 | epoch 57 | "58 epochs" |
| restored (best validation loss) | epoch 55 | epoch 58 | epoch 51 | — |
| validation accuracy there | 63.60 % | 63.43 % | 63.52 % | 63.59 % at epoch 58 (Fig. 7) |
| learning-rate cuts after epoch | 50 and 60 | 49 and 63 | 47 and 56 | 43 (text) |
| accuracy at label 0 dB | 90.5 % | 90.8 % | 90.2 % | 92.21 % (Fig. 11) |
| wall time | 141.6 min | 149.9 min | 133.0 min | — |

**Three runs: 62.56 % ± 0.11** (population s.d.), all three below the paper's 63.24 by 0.5–0.8 points.

- **Training behaves as the paper describes.** Early stopping on the loss fires at epochs 57–64, around their 58, at the same validation accuracy.
- **The test number is lower in every run.** The paper's 63.24 is one run on a different random split; their code cannot reproduce its own split (see above). The test set alone gives a standard error of about 0.27 points. So the paper's run looks like a slightly favourable draw, not a different model.
- **At label 0 dB**, the three runs pooled give 90.5 % over about 5,000 decisions. The paper's Fig. 11 gives 92.21 % from one run of about 1,650.

The results file is copied from Drive by re-serializing its content (the Drive connector returns text, not the file). Each copy is checked before it is committed:

- the confusion matrices reproduce every recorded test accuracy exactly;
- the earlier runs are unchanged;
- the size is byte-identical to the Drive file (109,250 bytes for three runs).

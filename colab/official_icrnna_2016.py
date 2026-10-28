"""
The paper's own code, run as published, with its results written down.

El-Haryqy et al. released their code on 2026-08-08, sixteen months after the
paper (Results in Engineering 26 (2025) 104783):

    https://github.com/noureddine395/mproved-automatic-modulation-recognition-using-deep-learning-with-additive-attention
    commit ac000a0, MIT licence, Zenodo 10.5281/zenodo.21854712

Their `rml2016_10a.py` trains once and shows its results as plots and printed
tables; nothing is saved but the model. This wrapper imports their module and
calls their own functions -- load_data, preprocess_data, build_improved_model
-- with their split (random_state 2016), their callbacks and their fit
arguments, copied from their `__main__` block line for line. It adds only:

  - several runs, each with its own weight initialisation seed. Their script
    sets no TensorFlow seed, so every run of it already starts from different
    weights; the seed here only makes our runs repeatable. The data split is
    theirs and identical in every run.
  - a results file: the per-epoch history, where training stopped, the test
    accuracy, and the test confusion matrix at every SNR, in the same layout
    as icrnna_faithful_2016.py so the same tools read both.
  - resume: a run already in the results file, with its weights saved, is
    skipped.
  - a fixed PYTHONHASHSEED. Their preprocess_data builds the class list from
    a Python set of strings, whose order changes from one process to the next
    unless the hash seed is fixed; the data are concatenated in that order and
    then shuffled with seed 2016, so each run of their script draws a
    different train/validation/test split despite the fixed shuffle seed. The
    wrapper refuses to run without PYTHONHASHSEED and records a fingerprint of
    the split, so every run here -- including resumed ones -- uses the same.

Environment:
  AMC_OFFICIAL_DIR   clone of their repository (required)
  AMC_PKL            path to RML2016.10a_dict.pkl (required)
  AMC_OUT            output directory (required)
  AMC_OFFICIAL_RUNS  number of runs, default 3
  AMC_OFFICIAL_EPOCHS  their ceiling is 100; only for a smoke test
"""
import hashlib
import json
import os
import sys
import time

if not os.environ.get("PYTHONHASHSEED"):
    raise SystemExit("Set PYTHONHASHSEED (e.g. 0) before starting Python: without it, the "
                     "official code's split changes from one process to the next.")

os.environ.setdefault("MPLBACKEND", "Agg")      # their module imports pyplot

REPO = os.environ["AMC_OFFICIAL_DIR"]
PKL = os.environ["AMC_PKL"]
OUT = os.environ["AMC_OUT"]
RUNS = int(os.environ.get("AMC_OFFICIAL_RUNS", "3"))
EPOCHS = int(os.environ.get("AMC_OFFICIAL_EPOCHS", "100"))   # theirs: epochs=100
TAG = "" if EPOCHS == 100 else f"_e{EPOCHS}"

sys.path.insert(0, REPO)
import numpy as np                                               # noqa: E402
import tensorflow as tf                                          # noqa: E402
from sklearn.model_selection import train_test_split             # noqa: E402
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,  # noqa: E402
                                        ReduceLROnPlateau)

import rml2016_10a as official                                   # noqa: E402

os.makedirs(OUT, exist_ok=True)
RESULTS = os.path.join(OUT, f"official_icrnna_2016{TAG}_results.json")


def weights_path(seed):
    return os.path.join(OUT, f"official_icrnna_2016{TAG}_seed{seed}.weights.h5")


print(f"their code : {REPO}")
print(f"data       : {PKL}")
print(f"tensorflow : {tf.__version__}, GPU: {tf.config.list_physical_devices('GPU')}")

# ---- their __main__, data part, verbatim apart from the path ---------------
dataset = official.load_data(PKL)
X, Y, classes, snrs = official.preprocess_data(dataset)
del dataset
if X.shape[1] != 2 or X.shape[2] != 128:
    X = X.transpose(0, 2, 1)
X_train, X_temp, Y_train, Y_temp, snrs_train, snrs_temp = train_test_split(
    X, Y, snrs, test_size=0.3, random_state=official.RANDOM_STATE)
X_val, X_test, Y_val, Y_test, snrs_val, snrs_test = train_test_split(
    X_temp, Y_temp, snrs_temp, test_size=0.5, random_state=official.RANDOM_STATE)
del X, X_temp, Y_temp
classes = [str(c) for c in classes]
SPLIT_ID = hashlib.sha1(np.argmax(Y_test, axis=1).astype(np.int8).tobytes()
                        + np.asarray(snrs_test, np.int16).tobytes()).hexdigest()[:12]
print(f"{len(classes)} classes {classes}")
print(f"split id   : {SPLIT_ID} (PYTHONHASHSEED={os.environ['PYTHONHASHSEED']})")
print(f"split      : {len(X_train):,} / {len(X_val):,} / {len(X_test):,} (theirs, random_state 2016)")


def confusions_by_snr(model):
    pred = np.argmax(model.predict(X_test, batch_size=1024, verbose=0), axis=1)
    true = np.argmax(Y_test, axis=1)
    levels = sorted(int(v) for v in np.unique(snrs_test))
    k = len(classes)
    mats = np.zeros((len(levels), k, k), dtype=np.int64)
    for j, v in enumerate(levels):
        at = snrs_test == v
        np.add.at(mats[j], (true[at], pred[at]), 1)
    return 100.0 * float(np.mean(pred == true)), levels, mats.tolist()


def run(seed):
    print(f"\n{'=' * 66}\n RUN seed {seed}\n{'=' * 66}")
    # Weight initialisation and dropout only; the split above is fixed.
    tf.keras.utils.set_random_seed(seed)
    model = official.build_improved_model(len(classes), input_shape=(2, 128))
    n_params = int(model.count_params())
    print(f"  {n_params:,} parameters (paper: 0.79 M)")

    # ---- their callbacks and fit, verbatim apart from the checkpoint path --
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, mode='min', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint(os.path.join(OUT, f"official_best_val_acc_seed{seed}.keras"),
                        monitor='val_accuracy', mode='max', save_best_only=True, verbose=0),
    ]
    t0 = time.time()
    history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=32,
                        validation_data=(X_val, Y_val), callbacks=callbacks, verbose=2)
    minutes = (time.time() - t0) / 60

    # Their evaluation runs on `model` as fit() leaves it: with
    # restore_best_weights, the epoch with the lowest validation loss.
    h = history.history
    stopped = len(h["loss"])
    best = int(np.argmin(h["val_loss"])) + 1
    acc, levels, mats = confusions_by_snr(model)
    model.save_weights(weights_path(seed))
    print(f"  stopped after {stopped} epochs, weights from epoch {best} (lowest val loss)")
    print(f"  TEST {acc:.2f}%   (paper reports 63.24%)   {minutes:.0f} min")
    lrs = h.get("learning_rate", h.get("lr", [None] * stopped))
    return {"seed": seed, "test_acc": round(acc, 4), "best_epoch": best, "stopped_epoch": stopped,
            "minutes": round(minutes, 1), "n_params": n_params,
            "history": [{"epoch": i + 1, "train_loss": round(float(h["loss"][i]), 5),
                         "train_acc": round(100 * float(h["accuracy"][i]), 3),
                         "val_loss": round(float(h["val_loss"][i]), 5),
                         "val_acc": round(100 * float(h["val_accuracy"][i]), 3),
                         "lr": None if lrs[i] is None else float(lrs[i])} for i in range(stopped)],
            "snrs": levels, "confusions_by_snr": mats}


def write(results):
    accs = [r["test_acc"] for r in results]
    with open(RESULTS, "w") as f:
        json.dump({"source": "official code, github.com/noureddine395/mproved-automatic-modulation-"
                             "recognition-using-deep-learning-with-additive-attention @ ac000a0",
                   "config": {"epochs": EPOCHS, "batch_size": 32, "split_random_state": 2016,
                              "early_stopping": "val_loss, patience 6, restore best weights",
                              "lr_schedule": "ReduceLROnPlateau val_loss x0.2 patience 5, min 1e-6",
                              "optimizer": "Adam 1e-3, clipnorm 1.0"},
                   "tag": f"official{TAG}", "classes": classes, "paper_target": 63.24,
                   "split_id": SPLIT_ID, "pythonhashseed": os.environ["PYTHONHASHSEED"],
                   "per_seed": results,
                   "summary": {"mean_test_acc": round(float(np.mean(accs)), 4),
                               "std_test_acc": round(float(np.std(accs)), 4), "n_seeds": len(accs)}},
                  f, indent=1)


results = []
if os.path.isfile(RESULTS):
    saved = json.load(open(RESULTS))
    if saved.get("split_id") != SPLIT_ID:
        raise SystemExit(f"{RESULTS} was made on split {saved.get('split_id')}, this process has "
                         f"{SPLIT_ID}. Use the same PYTHONHASHSEED, or a new AMC_OUT.")
    results = [r for r in saved["per_seed"] if os.path.isfile(weights_path(r["seed"]))]
    if results:
        print(f"resuming: {len(results)} run(s) already done")
done = {r["seed"] for r in results}
for s in range(RUNS):
    if s in done:
        continue
    results.append(run(s))
    write(results)
write(results)
accs = [r["test_acc"] for r in results]
print(f"\n{len(accs)} runs of the official code: {np.mean(accs):.2f}% +- {np.std(accs):.2f}%   |   paper: 63.24%")
print(f"results: {RESULTS}")

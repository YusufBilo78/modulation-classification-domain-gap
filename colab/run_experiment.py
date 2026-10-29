"""
run_experiment.py -- run the open-work sweeps on Colab.

Usage:
  CELL 1:  from google.colab import drive; drive.mount('/content/drive')
  CELL 2:  paste this file, set TASK below, run. Pick a GPU runtime.

Three tasks, matching the numbered list in README.md "Open work":

  "compare_methods"   `compare_methods --arch ICRNNA`. The committed table is
                      complete at 20/20 but two of the three cells with a
                      recorded stopping epoch ran out of budget at the
                      60-epoch ceiling rather than converging, and the other
                      seventeen predate the recording. COMPARE_EPOCHS below
                      raises the ceiling and COMPARE_TAG sends the result to
                      its own file, because a table run to convergence is a
                      different experiment from one that stopped at 60.
  "whitening_seeds"   item 2 -- rerun the alpha sweep with error bars, because
                      alpha=0.75 is unverified for the current backbone
  "compare_methods_2016"  the method table with RML2016.10a as the training
                      domain: 128-sample frames, SNR -20..18, synthetic
                      domain generated at the same length. The 2016 track,
                      kept apart from the 2018 results by file name
  "whitening_smoothing_2016"  why whitening costs 8 points on 2016: the
                      envelope smoothing width swept at 128 samples
  "whitening_crop_2018"  the same question from the other side: 2018 frames
                      cropped to 128 samples, none against whitening
  "literature_2016"   VT-CNN2, LSTM2, MCLDNN and PET-CGDNN (literature_models.py)
                      trained under train_backbone's protocol on 11-class 2016,
                      next to ICRNNA's committed run
  "vote_2016"         decision fusion: the top three
                      models of literature_2016 (and then all five) run on the
                      same test frames from their checkpoints; how often they
                      agree, and what a majority vote does. No training.
                      vote_fusion.py; results to Drive/amc-results/
  "wbfm_silence_2016" are the WBFM frames every model calls AM-DSB silent?
                      Frequency and envelope activity of every WBFM and AM-DSB
                      frame, lined up with the decisions vote_2016 saved.
                      wbfm_silence.py; no training, no GPU needed
  "rebuild_snr_2018"  per-SNR confusion matrices for the 24-class 2018 run,
                      rebuilt from its checkpoints (eval_by_snr.py), for the
                      class x SNR recall table; no training
  "faithful_2016"     item 5 -- run the paper-faithful ICRNNA against 63.24%
  "converged_2018"    the open question: the 2018 classification run again,
                      with a ceiling high enough that early stopping decides.
                      2016 turned out to have converged, but with one epoch of
                      margin against the 60-epoch ceiling, and 2018 is the
                      harder problem -- 24 classes and 1024-sample frames
                      against 11 and 128. If it ran to the ceiling its 0.5699
                      is a floor, and so is anything else measured under that
                      ceiling.
  "converged_2016"    done -- the 2016 classification run again, with a
                      ceiling high enough that early stopping decides and the
                      stopping epoch recorded. The existing result used a
                      60-epoch ceiling and never recorded where it stopped, so
                      whether it converged is unknown and its accuracy may be a
                      floor. Writes to its own file; the existing one is left
                      alone either way.
  "official_2016"     the paper's own code (released 2026-08-08, TensorFlow),
                      run as published on RML2016.10a through
                      colab/official_icrnna_2016.py: their split, their
                      callbacks, three runs, results and per-SNR confusion
                      matrices written to Drive/amc-results/official_icrnna/
  "faithful_paper"    the paper-faithful ICRNNA, checked against the paper:
                      (1) the 58-epoch run's three seeds get their test
                      confusion matrices, rebuilt from their checkpoints, no
                      training; (2) the converged run (ceiling 150) goes from
                      one seed to three -- seed 42 is reused and backfilled,
                      43 and 44 are trained -- with the per-epoch training
                      curve recorded, so the paper's Figs. 7, 8, 11 and 12
                      can be put next to ours
  "faithful_budget"   the question item 5 left open: two of its three seeds
                      peaked at the final epoch and early stopping never
                      fired, so the paper's 58-epoch ceiling was binding.
                      Same build, same everything else, a ceiling high enough
                      for early stopping to decide. One seed, because the
                      faithful run measured a seed spread of 0.07 points
                      against a 1.49-point deficit -- one run is already 21x
                      the noise, and a second buys nothing until the first
                      says the gap moved.

Item 5 needs only the 2016 pickle. The other two read RadioML 2018 and share
the arrangements `run_training.py` already makes: repository on local disk,
dataset staged to local disk, results written to Drive so that a reclaimed
runtime costs nothing but the cell that was in flight.

One thing here is not just plumbing. `compare_methods` is being *resumed*: 17
of its 20 cells were computed on another machine against the complete 21 GB
file. For the 3 that remain to belong in the same table they have to see the
same frames, and `RadioML.load` picks frames by position inside each
(class, SNR) cell -- so the staging keeps **every** frame of the five shared
classes rather than a subsample of them, and the existing .npz is copied into
the output directory first so the finished cells are recognised as done rather
than recomputed. Get either wrong and the run completes, reports numbers, and
quietly puts two different experiments in one file.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

# ==========================================================================
# Config
# ==========================================================================
# Every setting below can be overridden from the environment, so this file can
# be run straight out of a clone instead of being pasted and edited in place:
#
#     %env AMC_TASK=compare_methods
#     %env AMC_COMPARE_EPOCHS=150
#     %env AMC_REDO_UNCONVERGED=1
#     !python /content/amc/colab/run_experiment.py
#
# Unset means the default in each case, so a bare run behaves exactly as
# before. Same helper as run_training.py; duplicated rather than imported
# because the repository is not on disk yet when this block runs.


def _env(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not a valid "
                         f"{cast.__name__}") from None


def _flag(raw):
    """"1", "true", "yes" -> True; "0", "false", "no" -> False."""
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(raw)


TASK = _env("AMC_TASK", "compare_methods")
                           # "compare_methods" | "whitening_seeds"
                           # "faithful_2016" | "faithful_budget" | "faithful_paper" | "official_2016"
                           # "converged_2016" | "converged_2018"

# converged_2016 only. 300 is a ceiling, not a run length. One seed: this asks
# where training stops, not what the accuracy is to three decimals -- if the
# answer is "well short of 60" the existing result stands and nothing more is
# needed, and if it is "at the ceiling" one seed has already settled it.
CONVERGE_EPOCHS = 300
CONVERGE_SEEDS = 1

# converged_2018 only. Lower ceiling than 2016's because an epoch there costs
# about 0.18 min against 0.07, so 200 is already 36 minutes of worst case. 2016
# peaked at 39; if 2018 needs more than 200 that is itself the answer.
CONVERGE_EPOCHS_2018 = 200
FRAMES_PER_CELL_2018_CHECK = 512   # must match the committed run to compare

# faithful_budget only. 150 is a ceiling, not a run length: early stopping at
# patience 15 decides where it actually stops, which is the whole point.
BUDGET_EPOCHS = 150
BUDGET_SEEDS = 1

# faithful_paper: the same 150 ceiling, now with three seeds, so the converged
# number has an error bar like the 58-epoch one. Seed 42 already finished.
PAPER_SEEDS = _env("AMC_PAPER_SEEDS", 3, int)

# official_2016: the authors' code, pinned to the commit we read. Each run is
# one full training of their script (early stopping on validation loss).
OFFICIAL_REPO = ("https://github.com/noureddine395/mproved-automatic-modulation-"
                 "recognition-using-deep-learning-with-additive-attention.git")
OFFICIAL_COMMIT = "ac000a0"
OFFICIAL_RUNS = _env("AMC_OFFICIAL_RUNS", 3, int)

SEEDS = _env("AMC_SEEDS", 5, int)   # compare_methods: 4 methods x 5 seeds
EPOCHS = _env("AMC_EPOCHS", 60, int)
PATIENCE = _env("AMC_PATIENCE", 20, int)

# compare_methods and whitening_seeds. The worst peak seen under the old
# ceiling was epoch 53, which needs 73 to stop; 100 leaves room above that
# without spending the budget on cells that were going to stop at 57 anyway.
# The tag keeps this out of the committed 60-epoch file -- with it empty the
# run would resume into that file, find 20 of 20 cells done, and do nothing.
COMPARE_EPOCHS = _env("AMC_COMPARE_EPOCHS", 100, int)
COMPARE_TAG = _env("AMC_COMPARE_TAG", "e100")

# The second pass. Three augmentation cells peaked at 89, 83 and 82 under the
# 100 ceiling, so they needed 109, 103 and 102 to stop and ran out of budget
# instead. Setting this clears exactly those cells and reruns them; every cell
# that converged is left alone, because the ceiling only matters to a cell that
# hits it. Raise COMPARE_EPOCHS to 150 in the same edit -- rerunning them at
# 100 would reproduce the same problem.
REDO_UNCONVERGED = _env("AMC_REDO_UNCONVERGED", False, _flag)

# The sink thread: sink_24 is leave-one-class-out on all 24 classes with the
# backbone; sink_arch is the six-class architecture control, now with ICRNNA
# in the list. Both stage all 24 classes at SINK_FRAMES per cell (the probe
# draws 300 per cell, so 512 is the floor that leaves it room) into the same
# directory the 2018 convergence check used, and both take a ceiling plus
# patience rather than the fixed fifteen epochs they were written with.
SINK_FRAMES = _env("AMC_SINK_FRAMES", 512, int)
SINK_EPOCHS = _env("AMC_SINK_EPOCHS", 60, int)
SINK_PATIENCE = _env("AMC_SINK_PATIENCE", 10, int)
SINK_TAG = _env("AMC_SINK_TAG", "e60")

# whitening_smoothing_2016 / whitening_crop_2018: why whitening hurts on 2016.
# Own settings on purpose -- a Colab kernel keeps %env values between cells,
# and the 14 seeds and 150-epoch ceiling set for the method table would turn
# a 70-minute sweep into four hours. Seeds 0..4 are the method table's first
# five, so the none and 5-bin rows can be checked against it cell for cell.
SMOOTH_SEEDS = _env("AMC_SMOOTH_SEEDS", 5, int)
SMOOTH_EPOCHS = _env("AMC_SMOOTH_EPOCHS", 100, int)
SMOOTH_BINS = _env("AMC_SMOOTH_BINS", "3,5,9,17,33,65")

# literature_2016: four published architectures, trained under train_backbone's
# protocol on the 11-class RML2016.10a, for the comparison against ICRNNA.
# 3 seeds to match the committed ICRNNA run; a 150 ceiling because LSTM-type
# models can converge more slowly than ICRNNA did (best epoch 39).
LIT_ARCHS = _env("AMC_LIT_ARCHS", "VTCNN2,PETCGDNN,LSTM2,MCLDNN")
LIT_SEEDS = _env("AMC_LIT_SEEDS", 3, int)
LIT_EPOCHS = _env("AMC_LIT_EPOCHS", 150, int)

# vote_2016: which models to combine. Each set is one run of vote_fusion.py;
# the first model of a set is the fallback when there is no majority. The top
# three by overall accuracy first, then all five.
VOTE_SETS = _env("AMC_VOTE_SETS", "ICRNNA,LSTM2,MCLDNN;ICRNNA,LSTM2,MCLDNN,PETCGDNN,VTCNN2")

# Tasks that measure rather than train, and so run on a CPU runtime too.
CPU_TASKS = ("wbfm_silence_2016",)

_TASKS = ("literature_2016", "vote_2016", "wbfm_silence_2016", "rebuild_snr_2018", "compare_methods", "compare_methods_2016", "whitening_seeds",
          "whitening_smoothing_2016", "whitening_crop_2018",
          "faithful_2016", "faithful_budget", "faithful_paper", "official_2016", "converged_2016",
          "converged_2018", "sink_24", "sink_arch")
if TASK not in _TASKS:
    raise SystemExit(f"AMC_TASK={TASK!r}: expected one of {', '.join(_TASKS)}")

BRANCH = "claude/iacs-modulation-classification-gwwv4m"
REPO = "https://github.com/YusufBilo78/modulation-classification-domain-gap.git"
DRIVE = pathlib.Path("/content/drive/MyDrive")
REPO_DIR = pathlib.Path("/content/amc")
OUT_DIR = DRIVE / "amc-results"
STAGE_DIR = pathlib.Path("/content/amc-shared5")

PKL_HINTS_2016 = ("RadioML/RML2016.10a_dict.pkl",
                  "RML2016.10a_dict.pkl",
                  "RadioML2016/RML2016.10a_dict.pkl")
H5_HINTS = ("RadioML/GOLD_XYZ_OSC.0001_1024.hdf5",
            "RadioML2018/GOLD_XYZ_OSC.0001_1024.hdf5",
            "GOLD_XYZ_OSC.0001_1024.hdf5")

# The five classes that mean the same thing in every domain, as ids in
# RadioML 2018's own 24-class numbering. Both sweeps use exactly these.
SHARED_IDS = [3, 4, 5, 12, 14]     # BPSK, QPSK, 8PSK, 16QAM, 64QAM


def hr(title):
    print(f"\n{'=' * 68}\n {title}\n{'=' * 68}")


def sh(cmd, cwd=None):
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def sh_env(cmd, cwd, env):
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main():
    hr("1. Environment")
    if not DRIVE.exists():
        raise SystemExit("Drive is not mounted. Run CELL 1 first:\n"
                         "  from google.colab import drive; "
                         "drive.mount('/content/drive')")
    import torch

    print(f"torch {torch.__version__}   cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"gpu  : {gpu.name}, {gpu.total_memory / 1e9:.1f} GB")
    elif TASK in CPU_TASKS:
        print("no GPU -- fine for this task, which trains nothing")
    else:
        raise SystemExit("No GPU. Runtime > Change runtime type > GPU.")
    st = os.statvfs("/content")
    free_gb = st.f_bavail * st.f_frsize / 1e9
    print(f"local disk free: {free_gb:.0f} GB")

    hr("2. Repository")
    if (REPO_DIR / ".git").exists():
        sh(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR)
        sh(["git", "checkout", BRANCH], cwd=REPO_DIR)
        sh(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=REPO_DIR)
    else:
        sh(["git", "clone", "--branch", BRANCH, "--single-branch", REPO,
            str(REPO_DIR)])
    sh(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    SRC = REPO_DIR / "src"
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(REPO_DIR / "colab"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    if TASK == "converged_2016":
        hr("3. Did the 2016 run converge, or did the ceiling stop it?")
        print("Same protocol as the committed result -- 11 classes, all 1000")
        print(f"frames per cell, 70/15/15, patience 20 -- with the ceiling at")
        print(f"{CONVERGE_EPOCHS} instead of 60 and the stopping epoch recorded.")
        print("\nIt writes to train_backbone_rml2016_f1000_colab_e"
              f"{CONVERGE_EPOCHS}.npz, so the")
        print("existing result is untouched whichever way this comes out.")
        print("\nAbout 4 s an epoch on 2016, so the full ceiling is ~20 min.\n")

        pkl = None
        for hint in PKL_HINTS_2016:
            if (DRIVE / hint).is_file():
                pkl = DRIVE / hint
                break
        if pkl is None:
            found = list(DRIVE.rglob("RML2016.10a_dict.pkl"))
            if not found:
                raise SystemExit("RML2016.10a_dict.pkl not found in Drive.")
            pkl = found[0]
        print(f"data: {pkl}\n")

        cmd = [sys.executable, "train_backbone.py", "--data", "rml2016",
               "--data-path", str(pkl.parent), "--frames-per-cell", "1000",
               "--epochs", str(CONVERGE_EPOCHS), "--patience", "20",
               "--seeds", str(CONVERGE_SEEDS),
               "--tag", f"colab_e{CONVERGE_EPOCHS}", "--out-dir", str(OUT_DIR)]
        print(" ".join(cmd) + "\n")
        t0 = time.time()
        sh(cmd, cwd=SRC)
        hr(f"Done in {(time.time() - t0) / 60:.1f} min")
        print("Read the per-seed line: 'converged' means the existing 60-epoch")
        print("result stands. 'NOT converged' means it was a floor.")
        return

    if TASK == "faithful_budget":
        hr("3. Is the 58-epoch ceiling what costs the 1.49 points?")
        print(f"Same faithful build, ceiling raised to {BUDGET_EPOCHS}, "
              f"{BUDGET_SEEDS} seed.")
        print("A different budget is no longer the paper's protocol, so this")
        print(f"writes to *_e{BUDGET_EPOCHS}.* and leaves the faithful run's")
        print("results untouched.\n")
        print("Worst case is the full ceiling at about 0.74 min an epoch, so")
        print(f"budget up to {BUDGET_EPOCHS * 0.74 / 60:.1f} h; early stopping")
        print("should land well short of that.\n")
        env = dict(os.environ, AMC_FAITHFUL_EPOCHS=str(BUDGET_EPOCHS),
                   AMC_FAITHFUL_SEEDS=str(BUDGET_SEEDS))
        proc = subprocess.Popen(
            [sys.executable, str(REPO_DIR / "colab" / "icrnna_faithful_2016.py")],
            cwd=REPO_DIR / "colab", env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        if proc.returncode != 0:
            raise SystemExit(f"failed ({proc.returncode})")
        return

    if TASK == "official_2016":
        hr("3. The paper's own code, as published")
        off = pathlib.Path("/content/icrnna_official")
        if not (off / ".git").exists():
            sh(["git", "clone", "-q", OFFICIAL_REPO, str(off)])
        sh(["git", "checkout", "-q", OFFICIAL_COMMIT], cwd=off)
        sh(["git", "log", "--oneline", "-1"], cwd=off)
        pkl = None
        for hint in PKL_HINTS_2016:
            if (DRIVE / hint).is_file():
                pkl = DRIVE / hint
                break
        if pkl is None:
            found = list(DRIVE.rglob("RML2016.10a_dict.pkl"))
            if not found:
                raise SystemExit("RML2016.10a_dict.pkl not found in Drive.")
            pkl = found[0]
        print(f"data: {pkl}")
        print(f"{OFFICIAL_RUNS} runs of their training, batch 32, up to 100 epochs with")
        print("early stopping on validation loss. Budget about an hour a run; a")
        print("rerun of this cell skips runs that already finished.\n")
        env = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg",
                   AMC_OFFICIAL_DIR=str(off), AMC_PKL=str(pkl),
                   AMC_OUT=str(OUT_DIR / "official_icrnna"),
                   AMC_OFFICIAL_RUNS=str(OFFICIAL_RUNS))
        sh_env([sys.executable, str(REPO_DIR / "colab" / "official_icrnna_2016.py")], REPO_DIR / "colab", env)
        hr("Done")
        print(f"Results: {OUT_DIR / 'official_icrnna' / 'official_icrnna_2016_results.json'}")
        return

    if TASK == "faithful_paper":
        hr("3. The paper's model, checked against the paper")
        print("Step 1: the paper's 58-epoch protocol, three seeds already done.")
        print("Only their test confusion matrices are added, from the saved")
        print("checkpoints. No training; a few minutes.\n")
        script = str(REPO_DIR / "colab" / "icrnna_faithful_2016.py")
        env = dict(os.environ)
        env.pop("AMC_FAITHFUL_EPOCHS", None)
        env.pop("AMC_FAITHFUL_SEEDS", None)
        sh_env([sys.executable, script], REPO_DIR / "colab", env)

        print(f"\nStep 2: ceiling 150, {PAPER_SEEDS} seeds. Seed 42 finished")
        print("earlier and only gets its matrices; the others train. About")
        print("0.74 min an epoch and ~120 epochs, so budget ~1.5 h a seed.")
        print("A rerun of this cell skips every seed that already finished.\n")
        env = dict(os.environ, AMC_FAITHFUL_EPOCHS="150",
                   AMC_FAITHFUL_SEEDS=str(PAPER_SEEDS))
        sh_env([sys.executable, script], REPO_DIR / "colab", env)
        hr("Done")
        print("Results: Drive/RadioML/ICRNNA_faithful/icrnna_faithful_results.json")
        print("     and Drive/RadioML/ICRNNA_faithful/icrnna_faithful_e150_results.json")
        return

    if TASK == "faithful_2016":
        hr("3. Open work item 5 -- the paper-faithful ICRNNA on RML2016.10a")
        print("Needs only the 2016 pickle; it finds it in Drive itself.")
        print("Three seeds, 58 epochs each at batch 32 -- budget about an hour")
        print("a seed. Results and checkpoints go to Drive, and a rerun of this")
        print("cell skips seeds that already finished.\n")
        sh([sys.executable, str(REPO_DIR / "colab" / "icrnna_faithful_2016.py")],
           cwd=REPO_DIR / "colab")
        return

    if TASK == "compare_methods_2016":
        hr("3. The method table on RML2016.10a -- 128-sample frames")
        print("Same four methods, same protocol, the 2016 file as the training")
        print("domain and a synthetic domain generated at 128 samples. Writes")
        print(f"compare_methods_ICRNNA_es_rml2016_{COMPARE_TAG}.npz; the 2018")
        print("table is untouched. The pickle is 225 MB and is read straight")
        print("from Drive, nothing to stage.\n")
        pkl = None
        for hint in PKL_HINTS_2016:
            if (DRIVE / hint).is_file():
                pkl = DRIVE / hint
                break
        if pkl is None:
            found = list(DRIVE.rglob("RML2016.10a_dict.pkl"))
            if not found:
                raise SystemExit("RML2016.10a_dict.pkl not found in Drive.")
            pkl = found[0]
        print(f"data: {pkl}\n")
        cmd = [sys.executable, "compare_methods.py", "--arch", "ICRNNA",
               "--source", "rml2016", "--data-path", str(pkl),
               "--seeds", str(SEEDS), "--epochs", str(COMPARE_EPOCHS),
               "--patience", str(PATIENCE), "--tag", COMPARE_TAG,
               "--out-dir", str(OUT_DIR)]
        if REDO_UNCONVERGED:
            cmd.append("--redo-unconverged")
        print(f"results -> {OUT_DIR}  (Drive: survives the runtime)")
        print("Written after every cell; a rerun resumes.\n")
        print(" ".join(cmd) + "\n")
        t0 = time.time()
        sh(cmd, cwd=SRC)
        hr(f"Done in {(time.time() - t0) / 60:.1f} min")
        return

    if TASK == "whitening_smoothing_2016":
        hr("3. Why whitening hurts on 2016 -- the smoothing-width sweep")
        print("Whitening at alpha 0.75 with the envelope smoothed over")
        print(f"{SMOOTH_BINS} bins of the 128-point spectrum, plus a none row,")
        print(f"{SMOOTH_SEEDS} seeds, ceiling {SMOOTH_EPOCHS}. If a noisy envelope")
        print("estimate is the cost, it should shrink as the smoothing widens.")
        print("The none and 5-bin rows must match seeds 0-4 of the method table.\n")
        pkl = None
        for hint in PKL_HINTS_2016:
            if (DRIVE / hint).is_file():
                pkl = DRIVE / hint
                break
        if pkl is None:
            found = list(DRIVE.rglob("RML2016.10a_dict.pkl"))
            if not found:
                raise SystemExit("RML2016.10a_dict.pkl not found in Drive.")
            pkl = found[0]
        print(f"data: {pkl}\n")
        cmd = [sys.executable, "whitening_smoothing.py", "--source", "rml2016",
               "--data-path", str(pkl), "--bins", SMOOTH_BINS,
               "--seeds", str(SMOOTH_SEEDS), "--epochs", str(SMOOTH_EPOCHS),
               "--patience", str(PATIENCE), "--out-dir", str(OUT_DIR)]
        print(f"results -> {OUT_DIR}  (Drive: survives the runtime)")
        print("Written after every cell; a rerun resumes.\n")
        print(" ".join(cmd) + "\n")
        t0 = time.time()
        sh(cmd, cwd=SRC)
        hr(f"Done in {(time.time() - t0) / 60:.1f} min")
        return

    if TASK == "literature_2016":
        hr("3. Literature models on RML2016.10a, 11 classes, ICRNNA's protocol")
        print(f"architectures: {LIT_ARCHS}; {LIT_SEEDS} seeds; ceiling {LIT_EPOCHS},")
        print("patience 20; same frames, split and seeds as the ICRNNA run")
        print("(train_backbone_rml2016_f1000_colab.npz). One file per model;")
        print("rerun to resume -- finished seeds are skipped.\n")
        pkl = None
        for hint in PKL_HINTS_2016:
            if (DRIVE / hint).is_file():
                pkl = DRIVE / hint
                break
        if pkl is None:
            found = list(DRIVE.rglob("RML2016.10a_dict.pkl"))
            if not found:
                raise SystemExit("RML2016.10a_dict.pkl not found in Drive.")
            pkl = found[0]
        t0 = time.time()
        for arch in LIT_ARCHS.split(","):
            hr(f"{arch}")
            cmd = [sys.executable, "train_backbone.py", "--data", "rml2016",
                   "--data-path", str(pkl), "--frames-per-cell", "1000",
                   "--arch", arch, "--seeds", str(LIT_SEEDS),
                   "--epochs", str(LIT_EPOCHS), "--patience", "20",
                   "--tag", "colab", "--out-dir", str(OUT_DIR)]
            print(" ".join(cmd) + "\n")
            sh(cmd, cwd=SRC)
        hr(f"Done in {(time.time() - t0) / 60:.1f} min")
        return

    if TASK == "wbfm_silence_2016":
        hr("3. Are the WBFM frames every model calls AM-DSB silent?")
        print("Measures how much the frequency and the envelope move in every WBFM")
        print("and AM-DSB frame, and lines the test frames up with what ICRNNA, LSTM2")
        print("and MCLDNN said about them (saved by vote_2016). No training; about")
        print("a minute, most of it loading the pickle.\n")
        if not (OUT_DIR / "vote_rml2016_ICRNNA_LSTM2_MCLDNN_frames.npz").is_file():
            raise SystemExit("vote_rml2016_ICRNNA_LSTM2_MCLDNN_frames.npz is not in "
                             f"{OUT_DIR}. Run AMC_TASK=vote_2016 first.")
        pkl = None
        for hint in PKL_HINTS_2016:
            if (DRIVE / hint).is_file():
                pkl = DRIVE / hint
                break
        if pkl is None:
            found = list(DRIVE.rglob("RML2016.10a_dict.pkl"))
            if not found:
                raise SystemExit("RML2016.10a_dict.pkl not found in Drive.")
            pkl = found[0]
        cmd = [sys.executable, "wbfm_silence.py", "--results-dir", str(OUT_DIR), "--data-path", str(pkl)]
        print(" ".join(cmd) + "\n")
        sh(cmd, cwd=SRC)
        print(f"\nfigure: {OUT_DIR / 'wbfm_silence_rml2016.png'}")
        return

    if TASK == "vote_2016":
        hr("3. Do the models disagree? Majority vote on the same test frames")
        print("Every model of literature_2016 was scored on the same 33,000 test")
        print("frames per seed. vote_fusion.py reruns their checkpoints on those")
        print("frames, checks each against its stored matrices, and counts how")
        print("often they agree. No training; a few minutes.\n")
        pkl = None
        for hint in PKL_HINTS_2016:
            if (DRIVE / hint).is_file():
                pkl = DRIVE / hint
                break
        if pkl is None:
            found = list(DRIVE.rglob("RML2016.10a_dict.pkl"))
            if not found:
                raise SystemExit("RML2016.10a_dict.pkl not found in Drive.")
            pkl = found[0]
        t0 = time.time()
        for models in VOTE_SETS.split(";"):
            hr(models)
            cmd = [sys.executable, "vote_fusion.py", "--results-dir", str(OUT_DIR),
                   "--data-path", str(pkl), "--models", models]
            print(" ".join(cmd) + "\n")
            sh(cmd, cwd=SRC)
        hr(f"Done in {(time.time() - t0) / 60:.1f} min")
        return

    if TASK == "rebuild_snr_2018":
        hr("3. Per-SNR matrices for the 24-class 2018 run, from its checkpoints")
        print("train_backbone_rml2018_f512_colab.npz stored only the pooled matrix")
        print("above 10 dB. eval_by_snr.py rebuilds a matrix at every SNR from the")
        print("three checkpoints next to it, and writes nothing unless the rebuilt")
        print("pooled matrix matches the stored one count for count. No training.\n")
        npz = OUT_DIR / "train_backbone_rml2018_f512_colab.npz"
        if not npz.exists():
            raise SystemExit(f"{npz} not found")
        h5 = None
        for hint in H5_HINTS:
            if (DRIVE / hint).is_file():
                h5 = DRIVE / hint
                break
        if h5 is None:
            raise SystemExit("GOLD_XYZ_OSC.0001_1024.hdf5 not found in Drive.")
        import run_training

        stage = pathlib.Path("/content/amc-data")
        run_training.stage_from_hdf5(h5, stage, 512, free_gb)
        os.environ["AMC_DATA_DIR"] = str(stage)
        cmd = [sys.executable, "eval_by_snr.py", str(npz), "--data-path", str(stage)]
        print(" ".join(cmd) + "\n")
        t0 = time.time()
        sh(cmd, cwd=SRC)
        hr(f"Done in {(time.time() - t0) / 60:.1f} min")
        return

    if TASK == "converged_2018":
        hr("3. Did the 2018 run converge, or did the ceiling stop it?")
        print("Same protocol as the committed result -- 24 classes, "
              f"{FRAMES_PER_CELL_2018_CHECK} frames per cell,")
        print(f"native 1024 samples, 70/15/15, patience 20 -- with the ceiling "
              f"at\n{CONVERGE_EPOCHS_2018} instead of 60 and the stopping "
              f"epoch recorded.\n")
        print("2016 converged at epoch 39, one epoch inside its ceiling. This")
        print("is the harder problem, so it is the one that might not have.\n")

        h5 = None
        for hint in H5_HINTS:
            if (DRIVE / hint).is_file():
                h5 = DRIVE / hint
                break
        if h5 is None:
            raise SystemExit(
                "GOLD_XYZ_OSC.0001_1024.hdf5 not found in Drive. Looked at:\n  "
                + "\n  ".join(str(DRIVE / h) for h in H5_HINTS))
        print(f"found: {h5}  ({h5.stat().st_size / 1e9:.0f} GB)\n")

        import run_training

        stage = pathlib.Path("/content/amc-data")
        run_training.stage_from_hdf5(h5, stage,
                                     FRAMES_PER_CELL_2018_CHECK, free_gb)
        os.environ["AMC_DATA_DIR"] = str(stage)

        cmd = [sys.executable, "train_backbone.py", "--data", "rml2018",
               "--data-path", str(stage),
               "--frames-per-cell", str(FRAMES_PER_CELL_2018_CHECK),
               "--epochs", str(CONVERGE_EPOCHS_2018), "--patience", "20",
               "--seeds", str(CONVERGE_SEEDS),
               "--tag", f"colab_e{CONVERGE_EPOCHS_2018}",
               "--out-dir", str(OUT_DIR)]
        print(" ".join(cmd) + "\n")
        t0 = time.time()
        sh(cmd, cwd=SRC)
        hr(f"Done in {(time.time() - t0) / 60:.1f} min")
        print("Read the per-seed line. 'converged' and the committed 0.5699")
        print("stands; 'NOT converged' and it is a floor, as is anything else")
        print("measured under the same ceiling.")
        return

    if TASK in ("sink_24", "sink_arch"):
        script = ("sink_class_24.py" if TASK == "sink_24"
                  else "sink_across_architectures.py")
        hr(f"3. The sink thread -- {script}")
        print("Where does a modulation the model has never seen land? Train on")
        print("23 classes, probe with the 24th drawn from RadioML itself. The")
        print("finding on the superseded backbone was 'inside its own family';")
        print("this is the rerun on the current one.\n")

        h5 = None
        for hint in H5_HINTS:
            if (DRIVE / hint).is_file():
                h5 = DRIVE / hint
                break
        if h5 is None:
            raise SystemExit(
                "GOLD_XYZ_OSC.0001_1024.hdf5 not found in Drive. Looked at:\n  "
                + "\n  ".join(str(DRIVE / h) for h in H5_HINTS))
        print(f"found: {h5}  ({h5.stat().st_size / 1e9:.0f} GB)\n")

        import run_training

        stage = pathlib.Path("/content/amc-data")
        run_training.stage_from_hdf5(h5, stage, SINK_FRAMES, free_gb)
        os.environ["AMC_DATA_DIR"] = str(stage)

        cmd = [sys.executable, script,
               "--epochs", str(SINK_EPOCHS), "--patience", str(SINK_PATIENCE),
               "--tag", SINK_TAG, "--out-dir", str(OUT_DIR)]
        print(f"results -> {OUT_DIR}  (Drive: survives the runtime)")
        print("Written after every run. If this runtime dies, rerun this same")
        print("cell -- finished runs are skipped, not repeated.\n")
        print(" ".join(cmd) + "\n")
        t0 = time.time()
        sh(cmd, cwd=SRC)
        hr(f"Done in {(time.time() - t0) / 60:.1f} min")
        return

    # ------------------------------------------------- 2018 for both sweeps
    hr("3. Dataset -- every frame of the five shared classes")
    h5 = None
    for hint in H5_HINTS:
        if (DRIVE / hint).is_file():
            h5 = DRIVE / hint
            break
    if h5 is None:
        raise SystemExit(
            "GOLD_XYZ_OSC.0001_1024.hdf5 not found in Drive. Looked at:\n  "
            + "\n  ".join(str(DRIVE / h) for h in H5_HINTS))
    print(f"found: {h5}  ({h5.stat().st_size / 1e9:.0f} GB)")
    print(f"staging to {STAGE_DIR} -- all 4096 frames per cell for "
          f"{len(SHARED_IDS)} classes, about 4.4 GB.")
    print("Not a subsample: these sweeps must select the same frames a run")
    print("against the complete file would.\n")

    import run_training

    run_training.stage_from_hdf5(h5, STAGE_DIR, None, free_gb,
                                 class_ids=SHARED_IDS)

    # The sweeps run as subprocesses, so an in-process edit of
    # radioml.SEARCH_DIRS would not reach them. AMC_DATA_DIR does, and is
    # inherited by anything launched from here.
    os.environ["AMC_DATA_DIR"] = str(STAGE_DIR)
    print(f"AMC_DATA_DIR={STAGE_DIR}")

    # ----------------------------------------------------------------------
    if TASK == "compare_methods":
        hr("4. compare_methods, rerun to convergence")
        name = f"compare_methods_ICRNNA_es_{COMPARE_TAG}.npz"
        print(f"Ceiling {COMPARE_EPOCHS} instead of 60, writing to {name}.")
        print("The committed 60-epoch table is left alone: it is a different")
        print("experiment, and two of its measured cells stopped because the")
        print("budget ended rather than because they had converged.\n")
        print("Twenty cells from cold, but this resumes: as of the last run"
              " 15 were")
        print("done and the whitening + standard row was what remained. Expect"
              " roughly")
        print("3 hours for a full table -- early stopping ends many cells well")
        print("before the ceiling.\n")
        if (OUT_DIR / name).exists():
            import numpy as np

            done = int(np.count_nonzero(~np.isnan(
                np.load(OUT_DIR / name)["in_domain"])))
            print(f"resuming: {done}/20 cells already in Drive\n")

        cmd = [sys.executable, "compare_methods.py", "--arch", "ICRNNA",
               "--seeds", str(SEEDS), "--epochs", str(COMPARE_EPOCHS),
               "--patience", str(PATIENCE), "--tag", COMPARE_TAG,
               "--out-dir", str(OUT_DIR)]
        if REDO_UNCONVERGED:
            cmd.append("--redo-unconverged")
            print("REDO_UNCONVERGED: cells whose early stopping never fired")
            print("will be cleared and rerun; converged cells are kept.\n")

    elif TASK == "whitening_seeds":
        hr("4. Open work item 2 -- the alpha sweep, with error bars")
        print("alpha=0.75 was measured on the superseded backbone and one seed")
        print("per point. This reruns all five alphas across several seeds on")
        print("the current backbone, which is what makes the claim checkable.\n")
        cmd = [sys.executable, "whitening_seeds.py",
               "--seeds", str(SEEDS), "--epochs", str(COMPARE_EPOCHS),
               "--patience", str(PATIENCE), "--tag", COMPARE_TAG,
               "--out-dir", str(OUT_DIR)]
    elif TASK == "whitening_crop_2018":
        hr("4. Why whitening hurts on 2016 -- the crop test on 2018")
        print("RadioML 2018 frames cut to their first 128 samples, where the")
        print("envelope estimate is as noisy as on 2016. none against whitening")
        print("at 5 and 33 bins. On 2016 the 5-bin cost was the estimate; if")
        print("2018 at 128 samples shows the same cost at 5 bins and none at 33,")
        print("the explanation holds on a second dataset.\n")
        cmd = [sys.executable, "whitening_smoothing.py", "--source", "rml2018",
               "--frame-len", "128", "--bins", "5,33",
               "--seeds", str(SMOOTH_SEEDS), "--epochs", str(SMOOTH_EPOCHS),
               "--patience", str(PATIENCE), "--out-dir", str(OUT_DIR)]
    else:
        raise SystemExit(f"unknown TASK {TASK!r}")

    print(f"results -> {OUT_DIR}  (Drive: survives the runtime)")
    print("Written after every cell. If this runtime dies, rerun this same")
    print("cell -- finished cells are skipped, not repeated.\n")
    print(" ".join(cmd) + "\n")
    t0 = time.time()
    sh(cmd, cwd=SRC)
    hr(f"Done in {(time.time() - t0) / 60:.1f} min")
    print(f"results : {OUT_DIR}")
    print("The .npz is the record worth committing to the repository.")


if __name__ == "__main__":
    main()

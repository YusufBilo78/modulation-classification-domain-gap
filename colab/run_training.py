"""
run_training.py -- run train_backbone.py on Colab.

Usage:
  CELL 1:  from google.colab import drive; drive.mount('/content/drive')
  CELL 2:  paste this file, set DATASET below, run. Pick a GPU runtime.

What it does, and why each piece is the way it is
-------------------------------------------------
**Code on local disk, results on Drive.** A Colab runtime is reclaimed after 12
hours, on idle, and sometimes for no reason at all. `train_backbone.py` writes
its .npz after every seed and skips finished seeds on restart, but that only
helps if the .npz outlives the runtime -- so `--out-dir` points at Drive and
every result, checkpoint and figure lands there as it is produced. A dropped
session is then resumed by rerunning this same cell. The repository itself is
cloned to /content instead, because git against a Drive FUSE mount is slow and
prone to failing on file operations Drive does not fully support.

**The dataset is staged onto local disk too.** The training loop reads the whole
array once at startup, and the Drive mount is slow enough that 2.6 GB of it is
worth copying to /content first.

**If the 21 GB file is not in Drive**, it does not have to be. No run reads all
of it: `src/export_subset.py` carves out exactly the frames a given
--frames-per-cell would use -- 2.6 GB at 512 -- as a `radioml_X/y/z.npy` triple
that this cell then finds directly. Both sides call the same selection function
at the same seed, so the two subsets hold the same frames, not merely the same
number of them.

**2018 from HDF5 is subsampled during staging.** The full file is 21 GB and
Colab has neither the local disk headroom nor the patience for it over FUSE.
Only the frames the run will actually use are extracted, cell by cell, so the
(class, SNR) balance is preserved exactly.

Nothing here trains differently from the local machine. The protocol lives in
train_backbone.py and this file only arranges for it to be runnable.
"""

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
# be run straight out of a clone instead of being pasted into a cell and edited
# in place. Nothing about the run changes; it is the same constants, read from
# somewhere a %env line can reach:
#
#     %env AMC_CLASSES=BPSK,QPSK,16QAM,64QAM
#     %env AMC_FRAMES_2018=2048
#     !python /content/amc/colab/run_training.py
#
# Unset is the default in each case, so a bare run behaves exactly as before.


def _env(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not a valid "
                         f"{cast.__name__}") from None


DATASET = _env("AMC_DATASET", "rml2018")   # "rml2016" (11 classes, fast) or
                                           # "rml2018" (24)
SEEDS = _env("AMC_SEEDS", 3, int)
EPOCHS = _env("AMC_EPOCHS", 60, int)
PATIENCE = _env("AMC_PATIENCE", 20, int)
# Frames drawn per (class, SNR) cell, per dataset -- they are not comparable
# quantities. 2018 has 4096 available and each frame costs 4.2x a 2016 one, so
# 512 is the affordable setting there. 2016 has 1000 and is cheap enough to use
# all of them, which is also what the paper this backbone comes from does.
FRAMES_PER_CELL_2018 = _env("AMC_FRAMES_2018", 512, int)
FRAMES_PER_CELL_2016 = _env("AMC_FRAMES_2016", 1000, int)
TAG = _env("AMC_TAG", "colab")
# The frame-length experiment: keep only the first N samples of every frame.
# Unset means the native length (128 on 2016, 1024 on 2018).
FRAME_LEN = _env("AMC_FRAME_LEN", None, int)

# Train on a subset of the classes, as a genuinely smaller problem: the model
# is built with this many outputs, so its confusion matrix rows sum over these
# columns and nothing else. None trains on all of them.
#
#   rml2018: "BPSK,QPSK,16QAM,64QAM"
#   rml2016: "BPSK,QPSK,QAM16,QAM64"   (2016 spells the QAM classes the other
#                                       way round; confusion_table.py prints
#                                       the available names on a mismatch)
#
# The selection is pushed down into the loader, so only those cells are read
# and the same memory buys far more frames per class than a 24-class run can
# afford. On 2018 that is the point of the exercise: FRAMES_PER_CELL_2018 can
# go to 2048 for four classes, which is four times the frames of the full run
# and puts the test set past ten thousand decisions per class once the seeds
# are pooled -- enough that a 1% confusion is a hundred events rather than
# three.
#
# A subset run is its own experiment, written to its own file. Both loaders
# draw each cell from one generator walking the classes in order, so iterating
# four consumes it differently from iterating twenty-four and these are not the
# same frames the 24-class run gave those classes. Nothing here depends on
# them being the same; what is not allowed is quoting one as a slice of the
# other.
CLASSES = _env("AMC_CLASSES", None)

if DATASET not in ("rml2016", "rml2018"):
    raise SystemExit(f"AMC_DATASET={DATASET!r}: expected rml2016 or rml2018")

BRANCH = "claude/iacs-modulation-classification-gwwv4m"
REPO = "https://github.com/YusufBilo78/modulation-classification-domain-gap.git"
DRIVE = pathlib.Path("/content/drive/MyDrive")
REPO_DIR = pathlib.Path("/content/amc")         # local disk: git on a Drive
                                                # mount is slow and flaky
OUT_DIR = DRIVE / "amc-results"                 # Drive: outputs must outlive
                                                # the runtime
STAGE_DIR = pathlib.Path("/content/amc-data")   # local disk, fast

# Where the datasets might be sitting in Drive. Add yours if it is elsewhere.
PKL_HINTS = ("RML2016.10a_dict.pkl", "RadioML/RML2016.10a_dict.pkl",
             "RadioML2016/RML2016.10a_dict.pkl")
H5_HINTS = ("RadioML2018/GOLD_XYZ_OSC.0001_1024.hdf5",
            "GOLD_XYZ_OSC.0001_1024.hdf5",
            "RadioML/GOLD_XYZ_OSC.0001_1024.hdf5")
NPY_HINTS = ("amc-data", "RadioML2018", "RadioML", ".")


def staged_data_matches(stage_dir, marker, expected_ids):
    """
    True when the staging on disk is the one this run wants.

    The marker says what was staged last; the labels say what is actually
    there. Both are checked, because the marker alone was once enough to let
    a five-class run reuse a four-class staging: the previous run's marker had
    not been removed, and `marker.exists() and x_out.exists()` was satisfied
    by a file with no 8PSK in it. The label check makes the data the
    authority and the marker merely a hint that skips reading it.
    """
    import numpy as np

    x_out = stage_dir / "radioml_X.npy"
    y_out = stage_dir / "radioml_y.npy"
    if not (marker.exists() and x_out.exists() and y_out.exists()):
        return False
    present = sorted(int(c) for c in np.unique(np.load(y_out, mmap_mode="r")))
    if present != list(expected_ids):
        print(f"  staged data holds classes {present}, this run needs "
              f"{list(expected_ids)}: restaging")
        return False
    return True


def stage_from_hdf5(h5_path, stage_dir, frames_per_cell, free_gb,
                    class_ids=None):
    """
    Pull the frames a run needs out of the 21 GB HDF5 onto local disk.

    `class_ids` restricts the export to those classes and, when given, keeps
    **every** frame of each rather than subsampling. That is not a convenience:
    `RadioML.load` chooses its frames with `rng.choice` over the positions
    inside each (class, SNR) cell, so a staged cell that still holds all 4096
    in the file's order yields the same selection here as a run against the
    complete file, and a cell short of that silently yields different frames.
    An experiment whose other cells were computed elsewhere needs the former.
    """
    import numpy as np

    stage_dir.mkdir(parents=True, exist_ok=True)
    x_out = stage_dir / "radioml_X.npy"

    # The marker names the staging, so switching between a subsample and a
    # class-restricted full export restages rather than silently reusing the
    # wrong one -- which is the whole failure this function exists to avoid.
    if class_ids is None:
        tag = f"f{frames_per_cell}"
        what = f"{frames_per_cell} frames per cell, all 24 classes"
        need_bytes = need_estimate(frames_per_cell)
    else:
        tag = "all-" + "-".join(str(c) for c in sorted(class_ids))
        what = (f"every frame of {len(class_ids)} classes "
                f"(ids {sorted(class_ids)})")
        need_bytes = len(class_ids) * 26 * 4096 * 1024 * 2 * 4
    marker = stage_dir / f"staged_{tag}.txt"
    expected_ids = sorted(class_ids) if class_ids is not None else list(range(24))

    if staged_data_matches(stage_dir, marker, expected_ids):
        print(f"already staged ({what}), reusing")
    else:
        import h5py

        # Whatever was staged here before is about to be overwritten, so its
        # marker must go with it. Leaving it would let a later run with that
        # older configuration find its marker still present and reuse this
        # run's data as if it were its own.
        for stale in stage_dir.glob("staged_*.txt"):
            stale.unlink()

        need_gb = need_bytes / 1e9
        print(f"staging {what} (~{need_gb:.1f} GB) to {stage_dir}")
        if free_gb < need_gb * 1.3:
            raise SystemExit(
                f"only {free_gb:.0f} GB free locally, need ~{need_gb:.1f}. "
                "Lower frames_per_cell.")

        # Staging reads about 12% of the rows, scattered across
        # 21 GB. Over the Drive mount that is hundreds of thousands of
        # small reads and it crawls. One bulk sequential copy to local
        # disk first is far faster where there is room: the copy
        # streams at the mount's full rate and every scattered read
        # afterwards is local. Placed here, after the already-staged
        # check, so a rerun that only needs the existing subsample does
        # not copy 21 GB for nothing.
        h5_size = h5_path.stat().st_size
        local_h5 = pathlib.Path("/content") / h5_path.name
        room = (h5_size + need_bytes) / 1e9 * 1.15
        if local_h5.exists() and local_h5.stat().st_size == h5_size:
            print(f"  HDF5 already on local disk, reusing it")
            h5_path = local_h5
        elif free_gb > room:
            print(f"  copying the HDF5 to local disk first "
                  f"({h5_size / 1e9:.0f} GB, sequential -- much faster "
                  f"than scattered reads over the mount)")
            t_copy = time.time()
            shutil.copyfile(h5_path, local_h5)
            el = max(time.time() - t_copy, 1e-9)
            print(f"  copied in {el / 60:.1f} min "
                  f"({h5_size / 1e6 / el:.0f} MB/s)")
            h5_path = local_h5
        else:
            print(f"  only {free_gb:.0f} GB free locally, need "
                  f"{room:.0f} to copy first. Reading over the mount "
                  f"instead: slower, but it works.")
            local_h5 = None

        t0 = time.time()
        with h5py.File(h5_path, "r") as f:
            print("  reading labels and SNR ...")
            y_all = np.asarray(f["Y"][:]).argmax(axis=1).astype(np.int8)
            z_all = np.asarray(f["Z"][:]).ravel().astype(np.int16)
            print(f"    {len(y_all):,} rows, {y_all.max() + 1} classes, "
                  f"SNR {z_all.min()}..{z_all.max()}")

            if class_ids is None:
                # radioml.select_cell_balanced is the single definition of this
                # selection, shared with src/export_subset.py, so a subset
                # staged here and one exported on the machine that holds the
                # data hold the same frames.
                import radioml

                sel = radioml.select_cell_balanced(
                    y_all, z_all, frames_per_cell)
            else:
                sel = np.flatnonzero(np.isin(y_all, class_ids))
            print(f"    selected {len(sel):,} of {len(y_all):,} frames")

            out = np.lib.format.open_memmap(
                x_out, mode="w+", dtype=np.float32,
                shape=(len(sel), 1024, 2))
            chunk = 4096
            for start in range(0, len(sel), chunk):
                stop = min(start + chunk, len(sel))
                out[start:stop] = f["X"][sel[start:stop]]
                done = stop / len(sel)
                el = time.time() - t0
                print(f"\r    {done:6.1%}  {stop:>8,}/{len(sel):,}  "
                      f"{el:5.0f}s elapsed, "
                      f"~{el / max(done, 1e-9) - el:5.0f}s left", end="")
            out.flush()
            del out
            print()

        np.save(stage_dir / "radioml_y.npy", y_all[sel])
        np.save(stage_dir / "radioml_z.npy", z_all[sel])
        marker.write_text(f"{what}\n")
        print(f"  staged in {(time.time() - t0) / 60:.1f} min")

        # The 21 GB copy has served its purpose -- training reads the
        # staged subsample. Reclaim the space.
        if local_h5 is not None and h5_path == local_h5 and local_h5.exists():
            local_h5.unlink()
            print(f"  removed the local copy, {h5_size / 1e9:.0f} GB freed")


def need_estimate(frames_per_cell):
    """Bytes the staged 2018 subsample will occupy: 24 classes x 26 SNRs."""
    return 24 * 26 * frames_per_cell * 1024 * 2 * 4


def first_existing(hints, predicate):
    """First path under DRIVE matching one of `hints`, or None."""
    for hint in hints:
        candidate = DRIVE / hint
        if predicate(candidate):
            return candidate
    return None


def hr(title):
    print(f"\n{'=' * 68}\n {title}\n{'=' * 68}")


def sh(cmd, cwd=None):
    """Run a command, streaming its output, and fail loudly."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main():
    # ==========================================================================
    # 1. Environment
    # ==========================================================================
    hr("1. Environment")
    if not DRIVE.exists():
        raise SystemExit("Drive is not mounted. Run CELL 1 first:\n"
                         "  from google.colab import drive; "
                         "drive.mount('/content/drive')")

    import torch

    print(f"torch {torch.__version__}   cuda: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Runtime > Change runtime type > GPU, then rerun.")
    gpu = torch.cuda.get_device_properties(0)
    print(f"gpu  : {gpu.name}, {gpu.total_memory / 1e9:.1f} GB")

    # cnn.py calls torch.amp.autocast("cuda", ...) and torch.amp.GradScaler("cuda",
    # ...), which are the 2.4+ spellings. Colab is normally well past this; check
    # rather than fail three hours in.
    major, minor = (int(v) for v in torch.__version__.split(".")[:2])
    if (major, minor) < (2, 4):
        raise SystemExit(f"torch {torch.__version__} is too old for the "
                         "torch.amp API cnn.py uses. Needs 2.4+.")

    free_gb = os.statvfs("/content").f_bavail * os.statvfs("/content").f_frsize / 1e9
    print(f"local disk free: {free_gb:.0f} GB")


    # ==========================================================================
    # 2. Repository, in Drive so results survive the runtime
    # ==========================================================================
    hr("2. Repository")
    if (REPO_DIR / ".git").exists():
        print(f"{REPO_DIR} exists, updating to the tip of {BRANCH}")
        sh(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR)
        sh(["git", "checkout", BRANCH], cwd=REPO_DIR)
        sh(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=REPO_DIR)
    else:
        print(f"cloning into {REPO_DIR}")
        sh(["git", "clone", "--branch", BRANCH, "--single-branch", REPO,
            str(REPO_DIR)])
    sh(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)

    SRC = REPO_DIR / "src"
    sys.path.insert(0, str(SRC))


    # ==========================================================================
    # 3. Find the dataset in Drive
    # ==========================================================================
    hr("3. Dataset")

    # Names are what a person writes; the loader and the staging step both
    # want dataset ids. Resolve once, here, so a typo fails before anything
    # has been copied.
    CLASS_IDS = None
    if CLASSES and DATASET == "rml2018":
        import radioml          # src is already on sys.path, just above

        names = [c.strip() for c in CLASSES.split(",")]
        unknown = [c for c in names if c not in radioml.CLASSES]
        if unknown:
            raise SystemExit(
                f"CLASSES: not in rml2018: {', '.join(unknown)}\n"
                f"available: {', '.join(radioml.CLASSES)}")
        CLASS_IDS = [list(radioml.CLASSES).index(c) for c in names]
        print(f"classes: {names} -> ids {CLASS_IDS}\n")


    if DATASET == "rml2016":
        pkl = first_existing(PKL_HINTS, lambda p: p.is_file())
        if pkl is None:
            print("RML2016.10a_dict.pkl not found at any of:")
            for h in PKL_HINTS:
                print(f"  {DRIVE / h}")
            print("\nSearching the whole Drive (this can take a minute) ...")
            found = list(DRIVE.rglob("RML2016.10a_dict.pkl"))
            if not found:
                raise SystemExit(
                    "Not in Drive. Upload it, or add its path to PKL_HINTS.")
            pkl = found[0]
        print(f"found: {pkl}  ({pkl.stat().st_size / 1e6:.0f} MB)")
        data_path = str(pkl.parent)
        frames_per_cell = FRAMES_PER_CELL_2016

    else:
        # Case A: the .npy triple is already in Drive -- read it once, directly.
        npy_dir = first_existing(
            NPY_HINTS, lambda p: (p / "radioml_X.npy").is_file()
                                 and (p / "radioml_y.npy").is_file()
                                 and (p / "radioml_z.npy").is_file())
        if npy_dir is not None:
            print(f"found the .npy triple in {npy_dir}")
            data_path = str(npy_dir)
            frames_per_cell = FRAMES_PER_CELL_2018
            if CLASSES:
                print("  note: this triple was staged at some frames-per-cell "
                      "already. If it holds\n  fewer than "
                      f"{FRAMES_PER_CELL_2018} per cell the loader says so and "
                      "takes those cells whole,\n  which is a smaller run than "
                      "this one is asking for -- stage from the HDF5\n"
                      "  instead if the warning appears.")
        else:
            # Case B: only the HDF5. Extract the frames this run needs onto local
            # disk, keeping every (class, SNR) cell equally represented.
            h5 = first_existing(H5_HINTS, lambda p: p.is_file())
            if h5 is None:
                raise SystemExit(
                    "Neither radioml_X/y/z.npy nor GOLD_XYZ_OSC.0001_1024.hdf5 "
                    f"found in Drive.\nLooked under {DRIVE} at:\n  "
                    + "\n  ".join(list(NPY_HINTS) + list(H5_HINTS))
                    + "\n\nThe 21 GB file does not fit in a free Drive and "
                      "does not need to. On the\nmachine that holds the data, "
                      "run:\n    cd src && python export_subset.py "
                      f"--frames-per-cell {FRAMES_PER_CELL_2018}\n"
                      "and upload the resulting directory to MyDrive/amc-data "
                      "(2.6 GB), which is\nalready searched -- or add wherever "
                      "you put it to NPY_HINTS above.")
            print(f"found HDF5: {h5}  ({h5.stat().st_size / 1e9:.0f} GB)")

            import numpy as np

            # A class subset stages every frame of those classes instead of a
            # subsample of all of them: four classes whole is 3.5 GB against
            # 10.5 for a 2048-per-cell subsample of twenty-four, and it leaves
            # the cells full, so frames_per_cell is free to be raised without
            # restaging.
            if CLASS_IDS:
                stage_from_hdf5(h5, STAGE_DIR, FRAMES_PER_CELL_2018, free_gb,
                                class_ids=CLASS_IDS)
            else:
                stage_from_hdf5(h5, STAGE_DIR, FRAMES_PER_CELL_2018, free_gb)

            data_path = str(STAGE_DIR)
            frames_per_cell = FRAMES_PER_CELL_2018


    # ==========================================================================
    # 4. Train
    # ==========================================================================
    hr("4. Training")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"results -> {OUT_DIR}  (Drive: survives the runtime)")
    print("Written after every seed. If this runtime dies, rerun this same "
          "cell --\nfinished seeds are skipped, not repeated.\n")

    cmd = [sys.executable, "train_backbone.py",
           "--data", DATASET,
           "--data-path", data_path,
           "--frames-per-cell", str(frames_per_cell),
           "--epochs", str(EPOCHS),
           "--patience", str(PATIENCE),
           "--seeds", str(SEEDS),
           "--tag", TAG,
           "--out-dir", str(OUT_DIR)]
    if CLASSES:
        cmd += ["--classes", CLASSES]
    if FRAME_LEN:
        cmd += ["--frame-len", str(FRAME_LEN)]
    print(" ".join(cmd) + "\n")
    t0 = time.time()
    sh(cmd, cwd=SRC)

    hr(f"Done in {(time.time() - t0) / 60:.1f} min")
    subset = f"_c{len(CLASSES.split(','))}" if CLASSES else ""
    crop = f"_L{FRAME_LEN}" if FRAME_LEN else ""
    stem = f"train_backbone_{DATASET}_f{frames_per_cell}{subset}{crop}_{TAG}"
    print(f"results  : {OUT_DIR / (stem + '.npz')}")
    print(f"figures  : {OUT_DIR / 'figures'}/29_{stem}_accuracy.png")
    print(f"           {OUT_DIR / 'figures'}/30_{stem}_confusion.png")
    print(f"\nAll of these are in Drive and survive this runtime. The .npz is "
          f"the record\nworth committing to the repository -- it is small, and "
          f"a result nobody can look\nup later is a result nobody can check.")


if __name__ == "__main__":
    main()

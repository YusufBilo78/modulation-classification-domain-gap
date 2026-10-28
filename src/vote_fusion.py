"""
vote_fusion.py -- do the 2016 classifiers disagree enough to be worth combining?

Decision fusion, the simplest form. Take the best three models of the
literature comparison, give each exactly the same test frames, and for every
frame look at the three decisions together:

  - all three agree;
  - two agree (a majority exists, take it);
  - all three differ (no majority: count how often this happens).

If the models were identical, every frame would be a unanimous one and
combining them could not help. Fusion gains only where they are right on
different frames, so the numbers that matter are how often the majority
corrects a wrong single model, how often it overrules a right one, and the
oracle (at least one model right), which bounds any way of combining their
decisions.

No training. The five models were trained by `train_backbone.py --arch` on the
same frames, split and seeds (`train_backbone_rml2016_f1000[_ARCH]_colab.npz`
with `_seed{n}.pt` next to each), so for a given seed every model was scored
on the same 33,000 test frames. This rebuilds that test set, runs every
checkpoint on it, and -- as `eval_by_snr.py` does -- refuses to go on unless
each model's rebuilt per-SNR confusion matrices match the stored ones count
for count.

    cd src && python vote_fusion.py --results-dir /content/drive/MyDrive/amc-results \\
        --data-path /content/drive/MyDrive/RadioML/RML2016.10a_dict.pkl \\
        --models ICRNNA,LSTM2,MCLDNN

Writes `vote_rml2016_<models>.json` (counts and rates, pooled over seeds, per
SNR and overall) and, next to it, `.npz` with every frame's decisions and
softmax outputs, for other fusion rules later.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

import cnn
import train_backbone as tb

HIGH_SNR = 10


def npz_name(arch: str, fpc: int = 1000, tag: str = "colab") -> str:
    return f"train_backbone_rml2016_f{fpc}" + ("" if arch == "ICRNNA" else f"_{arch}") + f"_{tag}"


def scores(model, X, batch_size=512) -> np.ndarray:
    """Softmax outputs, computed exactly as cnn.predict computes its argmax."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i: i + batch_size]).to(cnn.DEVICE)
            with torch.amp.autocast("cuda", enabled=cnn.DEVICE.type == "cuda"):
                logits = model(xb)
            out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out)


def tally(P: np.ndarray, S: np.ndarray, y: np.ndarray) -> dict:
    """
    P: (models, frames) decisions, S: (models, frames, classes) softmax, y: truth.
    Counts, so that seeds and SNR levels can be added before rates are taken.
    """
    k, n = P.shape
    right = P == y                                        # (k, n)
    votes = np.zeros((n, S.shape[2]), dtype=np.int64)
    for m in range(k):
        np.add.at(votes, (np.arange(n), P[m]), 1)
    top = votes.max(1)                                    # size of the largest agreeing group
    label = votes.argmax(1)
    majority = top * 2 > k
    vote = np.where(majority, label, P[0])                # no majority: fall back to the first model
    soft = S.mean(0).argmax(1)
    t = {
        "frames": int(n),
        "correct": {f"model{m}": int(right[m].sum()) for m in range(k)},
        "unanimous": int((top == k).sum()),
        "unanimous_right": int(((top == k) & right[0]).sum()),
        "majority": int(majority.sum()),
        "majority_right": int((majority & (label == y)).sum()),
        "no_majority": int((~majority).sum()),
        "no_majority_one_right": int((~majority & right.any(0)).sum()),
        "vote_right": int((vote == y).sum()),
        "soft_vote_right": int((soft == y).sum()),
        "oracle_right": int(right.any(0).sum()),
        "all_wrong": int((~right.any(0)).sum()),
        # what the vote does to a single model: fixes its errors, or breaks its right answers
        "vote_fixes": {f"model{m}": int((~right[m] & (vote == y)).sum()) for m in range(k)},
        "vote_breaks": {f"model{m}": int((right[m] & (vote != y)).sum()) for m in range(k)},
        "largest_group": {str(g): int((top == g).sum()) for g in range(1, k + 1)},
    }
    if k == 3:
        split = majority & (top == 2)
        lone = np.where(P[0] == label, np.where(P[1] == label, 2, 1), 0)   # index of the dissenter
        lone_right = right[lone, np.arange(n)]
        t["two_one"] = int(split.sum())
        t["two_one_majority_right"] = int((split & (label == y)).sum())
        t["two_one_dissenter_right"] = int((split & lone_right).sum())
    return t


def add(a: dict, b: dict) -> dict:
    if not a:
        return json.loads(json.dumps(b))
    for key, v in b.items():
        a[key] = add(a[key], v) if isinstance(v, dict) else a[key] + v
    return a


def rates(t: dict, names: list[str]) -> dict:
    n = t["frames"]
    r = {"frames": n}
    r.update({f"acc_{names[m]}": t["correct"][f"model{m}"] / n for m in range(len(names))})
    r.update({
        "unanimous": t["unanimous"] / n,
        "majority": t["majority"] / n,
        "no_majority": t["no_majority"] / n,
        "acc_majority_frames": t["majority_right"] / max(t["majority"], 1),
        "acc_vote": t["vote_right"] / n,
        "acc_soft_vote": t["soft_vote_right"] / n,
        "acc_oracle": t["oracle_right"] / n,
    })
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="where the train_backbone .npz files and their .pt checkpoints are")
    p.add_argument("--data-path", default=None, help="the RML2016.10a pickle")
    p.add_argument("--models", default="ICRNNA,LSTM2,MCLDNN",
                   help="comma-separated; the first is the fallback when there is no majority")
    p.add_argument("--frames-per-cell", type=int, default=1000, help="as in the result file names")
    p.add_argument("--tag", default="colab", help="as in the result file names")
    p.add_argument("--out-dir", default=None, help="default: --results-dir")
    args = p.parse_args()

    names = args.models.split(",")
    rdir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else rdir
    stem_of = {a: npz_name(a, args.frames_per_cell, args.tag) for a in names}
    stored = {a: dict(np.load(rdir / f"{stem_of[a]}.npz", allow_pickle=True)) for a in names}
    first = stored[names[0]]
    for a in names[1:]:
        for key in ("class_names", "snrs", "frames_per_cell"):
            assert np.array_equal(stored[a][key], first[key]), (a, key)
        assert len(stored[a]["overall"]) == len(first["overall"]), a
    class_names = [str(c) for c in first["class_names"]]
    n_seeds = len(first["overall"])
    fpc = int(first["frames_per_cell"])

    print(f"models: {', '.join(names)}; {n_seeds} seeds; {fpc} frames per cell")
    X, y, z, loaded = tb.load_dataset("rml2016", fpc, args.data_path)
    assert loaded == class_names, (loaded, class_names)
    snrs = np.unique(z)
    assert np.array_equal(snrs, first["snrs"]), "SNR levels differ from the files"
    n = len(class_names)

    total, by_snr, per_seed = {}, {int(s): {} for s in snrs}, []
    dec, prob, truth, level = [], [], [], []
    for seed in range(n_seeds):
        # Same calls, same order, as train_backbone.run_seed: the split depends
        # on the seed only, so every model below sees the same test frames.
        torch.manual_seed(seed)
        np.random.seed(seed)
        _, _, te = cnn.split(X, y, z, test_fraction=0.15, val_fraction=0.15, seed=seed)
        yt, zt = y[te], z[te]

        P, S = [], []
        for a in names:
            model = tb.build(a, n, X.shape[-1]).to(cnn.DEVICE)
            ckpt = rdir / f"{stem_of[a]}_seed{seed}.pt"
            model.load_state_dict(torch.load(ckpt, map_location=cnn.DEVICE))
            s = scores(model, X[te])
            pred = s.argmax(1)
            rebuilt = np.zeros((len(snrs), n, n), dtype=np.int64)
            for k, lvl in enumerate(snrs):
                at = zt == lvl
                np.add.at(rebuilt[k], (yt[at], pred[at]), 1)
            if not np.array_equal(rebuilt, stored[a]["confusions_by_snr"][seed]):
                bad = int(np.abs(rebuilt - stored[a]["confusions_by_snr"][seed]).sum())
                raise SystemExit(f"{a} seed {seed}: rebuilt matrices differ from the stored ones "
                                 f"in {bad} counts; refusing to combine decisions from a "
                                 "test set or checkpoint that is not the one scored")
            print(f"  seed {seed} {a:9s}: matches its file, accuracy {(pred == yt).mean():.4f}")
            P.append(pred)
            S.append(s)
        P, S = np.stack(P), np.stack(S)

        t = tally(P, S, yt)
        total = add(total, t)
        per_seed.append(rates(t, names))
        for lvl in snrs:
            at = zt == lvl
            by_snr[int(lvl)] = add(by_snr[int(lvl)], tally(P[:, at], S[:, at], yt[at]))
        dec.append(P.astype(np.int8)); prob.append(S.astype(np.float16)); truth.append(yt.astype(np.int8)); level.append(zt)

    high = {}
    for lvl in snrs:
        if lvl >= HIGH_SNR:
            high = add(high, by_snr[int(lvl)])

    out = {
        "models": names, "seeds": n_seeds, "classes": class_names, "snrs": [int(s) for s in snrs],
        "fallback_without_majority": names[0],
        "overall": {"counts": total, "rates": rates(total, names)},
        f"snr_ge_{HIGH_SNR}": {"counts": high, "rates": rates(high, names)},
        "by_snr": {str(k): {"counts": v, "rates": rates(v, names)} for k, v in by_snr.items()},
        "per_seed": per_seed,
    }
    stem = "vote_rml2016_" + "_".join(names)
    (out_dir / f"{stem}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(out_dir / f"{stem}_frames.npz", decisions=np.stack(dec), softmax=np.stack(prob),
                        truth=np.stack(truth), snr=np.stack(level), models=np.array(names))

    r = out["overall"]["rates"]
    print(f"\n{'SNR':>5} " + " ".join(f"{a[:8]:>8}" for a in names)
          + f" {'vote':>7} {'soft':>7} {'oracle':>7} {'unanim':>7} {'no maj':>7}")
    for k in [str(s) for s in snrs] + ["all"]:
        rr = r if k == "all" else out["by_snr"][k]["rates"]
        print(f"{k:>5} " + " ".join(f"{100 * rr['acc_' + a]:8.1f}" for a in names)
              + f" {100 * rr['acc_vote']:7.1f} {100 * rr['acc_soft_vote']:7.1f} {100 * rr['acc_oracle']:7.1f}"
              f" {100 * rr['unanimous']:7.1f} {100 * rr['no_majority']:7.1f}")
    print(f"\nwrote {out_dir / (stem + '.json')} and {stem}_frames.npz")


if __name__ == "__main__":
    main()

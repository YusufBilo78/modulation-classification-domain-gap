"""Do the 2016 classifiers make the same mistakes? The numbers in VOTE_2016.md.

Reads what src/vote_fusion.py wrote: the counts in the committed
vote_rml2016_*.json files and, for the frame-level part, the decisions in the
matching _frames.npz (on Drive, in amc-results, not committed -- 4 to 7 MB).
No model is run.

    python tools/vote_analysis.py                                  # counts only
    python tools/vote_analysis.py --frames vote_rml2016_ICRNNA_LSTM2_MCLDNN_frames.npz
"""
import argparse
import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETS = ["vote_rml2016_ICRNNA_LSTM2_MCLDNN.json", "vote_rml2016_ICRNNA_LSTM2_MCLDNN_PETCGDNN_VTCNN2.json"]
HIGH = 10


def counts(path):
    d = json.loads((ROOT / path).read_text())
    names = d["models"]
    print(f"\n== {' + '.join(names)}: {d['seeds']} seeds ==")
    for part in ("overall", f"snr_ge_{HIGH}"):
        c, r = d[part]["counts"], d[part]["rates"]
        n = c["frames"]
        print(f"[{part}] {n} frames")
        print("  single: " + ", ".join(f"{a} {100 * r['acc_' + a]:.2f}" for a in names))
        print(f"  all agree {c['unanimous']} ({100 * c['unanimous'] / n:.1f}%), "
              f"all agree and wrong {c['unanimous'] - c['unanimous_right']}")
        if "two_one" in c:
            print(f"  two against one {c['two_one']} ({100 * c['two_one'] / n:.1f}%): the two right "
                  f"{c['two_one_majority_right']}, the one right {c['two_one_dissenter_right']}")
        print(f"  no majority {c['no_majority']} ({100 * c['no_majority'] / n:.1f}%), "
              f"one of them right {c['no_majority_one_right']}")
        print(f"  majority vote {100 * r['acc_vote']:.2f}, averaged softmax {100 * r['acc_soft_vote']:.2f}, "
              f"at least one right {100 * r['acc_oracle']:.2f}")
        # what "all wrong" would be if each model erred independently of the others
        p = [c["correct"][f"model{m}"] / n for m in range(len(names))]
        print(f"  all wrong {100 * c['all_wrong'] / n:.2f}%; independent errors would give "
              f"{100 * math.prod(1 - q for q in p):.2f}%")
        if len(names) == 3:
            a, b, g = p
            maj = a * b * g + a * b * (1 - g) + a * (1 - b) * g + (1 - a) * b * g
            print(f"  two of three right, if independent: {100 * maj:.1f}%")
    print("  per seed (first model, vote, soft): "
          + ", ".join(f"{100 * s['acc_' + names[0]]:.2f}/{100 * s['acc_vote']:.2f}/{100 * s['acc_soft_vote']:.2f}"
                      for s in d["per_seed"]))
    print("  SNR " + "".join(f"{a[:7]:>8}" for a in names) + "    vote    soft  oracle  agree  nomaj")
    for s in d["snrs"]:
        r = d["by_snr"][str(s)]["rates"]
        print(f"  {s:>3} " + "".join(f"{100 * r['acc_' + a]:8.1f}" for a in names)
              + f"{100 * r['acc_vote']:8.1f}{100 * r['acc_soft_vote']:8.1f}{100 * r['acc_oracle']:8.1f}"
              f"{100 * r['unanimous']:7.1f}{100 * r['no_majority']:7.1f}")
    return d


def frames(path, meta):
    f = np.load(path, allow_pickle=False)
    names = [str(a) for a in f["models"]]
    assert names == meta["models"], (names, meta["models"])
    cls = meta["classes"]
    k = len(names)
    P = f["decisions"].astype(int).transpose(1, 0, 2).reshape(k, -1)   # (models, seeds * frames)
    T = f["truth"].astype(int).reshape(-1)
    Z = f["snr"].reshape(-1)
    right = P == T
    stored = [meta["overall"]["counts"]["correct"][f"model{m}"] for m in range(k)]
    assert right.sum(1).tolist() == stored, "frames file does not match the counts file"
    same = (P == P[0]).all(0)
    votes = np.zeros((P.shape[1], len(cls)), int)
    for m in range(k):
        np.add.at(votes, (np.arange(P.shape[1]), P[m]), 1)
    vote = np.where(votes.max(1) * 2 > k, votes.argmax(1), P[0])
    hi = Z >= HIGH

    print(f"\n== frame by frame, {' + '.join(names)} ==")
    print("error correlation (phi of the 'wrong' indicators):")
    for lab, mask in (("all SNR", np.ones_like(hi)), (f">= {HIGH} dB", hi)):
        w = (~right[:, mask]).astype(float)
        print(f"  {lab}: " + ", ".join(f"{names[i]}-{names[j]} {np.corrcoef(w[i], w[j])[0, 1]:.2f}"
                                     for i in range(k) for j in range(i + 1, k)))
    shared = hi & same & ~right[0]
    print(f">= {HIGH} dB, all {k} give the same wrong answer on {int(shared.sum())} frames:")
    pairs = {}
    for t, p in zip(T[shared], P[0][shared]):
        pairs[(cls[t], cls[p])] = pairs.get((cls[t], cls[p]), 0) + 1
    for (t, p), n in sorted(pairs.items(), key=lambda x: -x[1])[:5]:
        print(f"  {t:7s} -> {p:7s} {n:5d} ({100 * n / shared.sum():.1f}%)")
    for m, a in enumerate(names):
        e = hi & ~right[m]
        print(f"  {a}: {int(e.sum())} errors, {int((e & same).sum())} of them made by all {k} "
              f"({100 * (e & same).sum() / e.sum():.1f}%)")
    print(f">= {HIGH} dB by class: frames, {names[0]} wrong, vote wrong, fixed, broken")
    for c, name in enumerate(cls):
        m = hi & (T == c)
        iw, vw = int((P[0][m] != c).sum()), int((vote[m] != c).sum())
        fx, br = int(((P[0][m] != c) & (vote[m] == c)).sum()), int(((P[0][m] == c) & (vote[m] != c)).sum())
        if iw or vw:
            print(f"  {name:7s} {int(m.sum()):6d} {iw:6d} {vw:6d} {fx:6d} {br:6d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default=None, help="a vote_rml2016_*_frames.npz written by vote_fusion.py")
    args = p.parse_args()
    meta = {s: counts(s) for s in SETS}
    if args.frames:
        stem = pathlib.Path(args.frames).name.replace("_frames.npz", ".json")
        frames(args.frames, meta[stem])


if __name__ == "__main__":
    main()

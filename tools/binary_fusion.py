"""Binary decision fusion of the 2016 classifiers: Chair-Varshney against Kam-Zhu-Gray.

The multi-class vote (VOTE_2016.md) showed the three models err together. This
asks the binary version of the question, in the setting of distributed
detection: three fixed local detectors, one fusion centre that sees only their
bits.

  - Hypotheses: a pair of classes, H0 and H1, e.g. QAM16 against QAM64. Only
    frames whose true class is one of the pair are used.
  - Local decision of model i: u_i = 1 when its softmax output for H1 exceeds
    its output for H0.
  - Fusion rules, all with equal priors (the test set is balanced):
      majority          two of three
      Chair-Varshney    log-likelihood ratio assuming the u_i independent given
                        the hypothesis (Chair & Varshney, IEEE AES 1986)
      Kam-Zhu-Gray, 2   Bahadur-Lazarsfeld expansion with the pairwise
                        conditional correlations (Kam, Zhu & Gray, IEEE AES
                        1992)
      Kam-Zhu-Gray, all the expansion to all orders, which for three detectors
                        is the joint pmf of (u_1, u_2, u_3): the optimal rule
                        for these bits
      softmax average   not a bit-fusion rule; what the centre could do if it
                        received the scores instead of the bits
  - Each model's P_D, P_F and the correlations are estimated per seed (each seed
    is a different set of trained models) and per SNR label, on one half of the
    frames, and the rules are scored on the other half; then the halves swap.
    So every frame is scored once, by a rule that did not see it. Above 10 dB,
    where the detectors no longer change with SNR, a second pass estimates on
    the five labels pooled: five times the frames per estimate.
  - The floor: the fewest errors any rule on these bits can make, counted on
    the frames themselves (for each of the 8 bit patterns, the smaller of its
    two class counts). It is optimistic, being fitted to the frames it scores,
    so no cross-fitted rule should beat it.

No model is run. Reads the per-frame file vote_fusion.py wrote (on Drive).

    python tools/binary_fusion.py --frames vote_rml2016_ICRNNA_LSTM2_MCLDNN_frames.npz
        -> fusion_binary_rml2016_ICRNNA_LSTM2_MCLDNN.json, figures/39_binary_fusion_2016.png
"""
import argparse
import itertools
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAIRS = [("QAM16", "QAM64"), ("AM-DSB", "WBFM"), ("QPSK", "8PSK")]   # (H0, H1)
RULES = ["majority", "chair_varshney", "kzg_pairs", "kzg_all", "softmax_avg"]
HIGH = 10
SMOOTH = 0.5                       # add-half to every count: no probability is estimated as 0 or 1
PATTERNS = np.array(list(itertools.product([0, 1], repeat=3)))   # the 8 values of (u1, u2, u3)


def fit(U, y):
    """P_D, P_F, conditional correlations and the 8-cell pmf, from training frames.
    U: (frames, 3) bits, y: (frames,) 0 or 1."""
    est = {}
    for h in (0, 1):
        Uh = U[y == h]
        n = len(Uh)
        p = (Uh.sum(0) + SMOOTH) / (n + 2 * SMOOTH)                 # P(u_i = 1 | H_h)
        z = (Uh - p) / np.sqrt(p * (1 - p))
        g2 = {(i, j): float((z[:, i] * z[:, j]).mean()) if n else 0.0 for i, j in itertools.combinations(range(3), 2)}
        g3 = float((z[:, 0] * z[:, 1] * z[:, 2]).mean()) if n else 0.0
        cells = np.array([(Uh == pat).all(1).sum() for pat in PATTERNS], float)
        pmf = (cells + SMOOTH) / (n + 8 * SMOOTH)
        est[h] = {"p": p, "g2": g2, "g3": g3, "pmf": pmf}
    return est


def log_lik(est_h, P, order):
    """log P(u | H) for every row of P under a model of the given order:
    0 = independent (Chair-Varshney), 2 = pairwise Bahadur-Lazarsfeld, 'all' = joint pmf."""
    if order == "all":
        idx = P[:, 0] * 4 + P[:, 1] * 2 + P[:, 2]
        return np.log(est_h["pmf"][idx])
    p = est_h["p"]
    base = (P * np.log(p) + (1 - P) * np.log(1 - p)).sum(1)
    if order == 0:
        return base
    z = (P - p) / np.sqrt(p * (1 - p))
    bracket = 1 + sum(g * z[:, i] * z[:, j] for (i, j), g in est_h["g2"].items())
    return base + np.log(np.clip(bracket, 1e-9, None))     # a truncated expansion can go negative


def decide(est, U, S, tie):
    """Every rule's decision for frames U (bits) and S (scores, log p1 - log p0 per model)."""
    out = {"majority": (U.sum(1) >= 2).astype(int)}
    for name, order in (("chair_varshney", 0), ("kzg_pairs", 2), ("kzg_all", "all")):
        llr = log_lik(est[1], U, order) - log_lik(est[0], U, order)
        out[name] = np.where(llr > 0, 1, np.where(llr < 0, 0, tie))
    out["softmax_avg"] = (S > 0).astype(int)
    return out


def check_expansion():
    """The Bahadur-Lazarsfeld expansion to all orders reproduces the joint pmf of
    three binary variables exactly -- why 'Kam-Zhu-Gray, all' is the 8-cell pmf."""
    rng = np.random.default_rng(0)
    q = rng.dirichlet(np.ones(8))
    p = (PATTERNS * q[:, None]).sum(0)
    z = (PATTERNS - p) / np.sqrt(p * (1 - p))
    g2 = {(i, j): (q * z[:, i] * z[:, j]).sum() for i, j in itertools.combinations(range(3), 2)}
    g3 = (q * z[:, 0] * z[:, 1] * z[:, 2]).sum()
    ind = np.prod(np.where(PATTERNS == 1, p, 1 - p), 1)
    full = ind * (1 + sum(g * z[:, i] * z[:, j] for (i, j), g in g2.items()) + g3 * z[:, 0] * z[:, 1] * z[:, 2])
    assert np.allclose(full, q), (full, q)


def predicted_error(est):
    """The error Chair-Varshney expects of itself: right if the bits were independent."""
    lik = {h: np.exp(log_lik(est[h], PATTERNS, 0)) for h in (0, 1)}
    return float(0.5 * np.minimum(lik[0], lik[1]).sum())


def floor_errors(U, y):
    """Fewest errors of any rule on these bits, in sample: per pattern, the minority class."""
    pat = U[:, 0] * 4 + U[:, 1] * 2 + U[:, 2]
    return int(np.minimum(np.bincount(pat[y == 0], minlength=8), np.bincount(pat[y == 1], minlength=8)).sum())


def crossfit(U, y, S, group, seed):
    """Every rule's decision on every frame: fitted within the frame's group, on the other fold."""
    rng = np.random.default_rng(1000 + seed)
    fold = np.zeros(len(y), int)
    for g in np.unique(group):                       # two folds, balanced within every (group, class)
        for h in (0, 1):
            idx = np.flatnonzero((group == g) & (y == h))
            fold[rng.permutation(idx)[1::2]] = 1
    dec = {r: np.zeros(len(y), int) for r in RULES}
    for g in np.unique(group):
        for k in (0, 1):
            tr, te = (group == g) & (fold != k), (group == g) & (fold == k)
            d = decide(fit(U[tr], y[tr]), U[te], S[te], tie=U[te, 0])
            for r in RULES:
                dec[r][te] = d[r]
    return dec


def figure(result, path, pairs=("QAM16_vs_QAM64", "AM-DSB_vs_WBFM"), lo=-10):
    """Error rate against SNR label for the two pairs that tell the story."""
    ink, muted, grid = "#52514e", "#8a8984", "#e6e5e0"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    names = result["models"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, key in zip(axes, pairs):
        pr = result["pairs"][key]
        levels = [s for s in map(int, pr["by_snr"]) if s >= lo]
        r = [pr["by_snr"][str(s)] for s in levels]
        best = [100 * min(x["error_rate"][a] for a in names) for x in r]
        series = [
            (best, "best single model at each label (chosen afterwards)", aqua, "-", "o"),
            ([100 * x["error_rate"]["chair_varshney"] for x in r], "Chair–Varshney, measured", blue, "-", "o"),
            ([100 * x["cv_predicted_error"] for x in r], "Chair–Varshney, its own prediction (independent bits)", blue, ":", None),
            ([100 * x["error_rate"]["kzg_all"] for x in r], "Kam–Zhu–Gray, all orders, measured", orange, "-", "o"),
            ([100 * x["floor"] for x in r], "floor: fewest errors any rule on the bits can make", ink, "-", None),
        ]
        for ys, label, color, ls, mk in series:
            ax.plot(levels, ys, ls, color=color, lw=1.2 if color == ink else 2, label=label, marker=mk,
                    ms=5, mec="white", mew=1, solid_capstyle="round", zorder=3 if mk else 2)
        n = r[0]["frames"]
        ax.set_title(f"{pr['h0']} (H0) against {pr['h1']} (H1)\n{n:,} decisions behind every point",
                     fontsize=10.5, color=ink, loc="left")
        ax.set_xlabel("SNR label (dB)", color=ink, fontsize=9.5)
        ax.set_xticks(levels[::2])
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9, length=0)
    axes[0].set_ylabel("error rate (%)", color=ink, fontsize=9.5)
    axes[0].set_ylim(0, 55)
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.legend(h, lab, loc="upper left", ncol=2, frameon=False, fontsize=8.8, labelcolor=ink,
               bbox_to_anchor=(0.055, 0.95))
    fig.suptitle("Binary fusion of ICRNNA, LSTM2 and MCLDNN on RML2016.10a, 3 seeds", x=0.06, y=1.0,
                 ha="left", fontsize=12, color=ink)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="vote_rml2016_*_frames.npz written by src/vote_fusion.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    check_expansion()

    f = np.load(args.frames, allow_pickle=False)
    names = [str(a) for a in f["models"]]
    assert len(names) == 3, "the rules here are written for three detectors"
    meta = json.loads((ROOT / f"vote_rml2016_{'_'.join(names)}.json").read_text())
    cls = meta["classes"]
    D, SM, T, Z = f["decisions"], f["softmax"].astype(np.float64), f["truth"].astype(int), f["snr"]
    n_seeds = D.shape[0]
    snrs = sorted(int(s) for s in np.unique(Z))

    result = {"models": names, "seeds": n_seeds, "smoothing": SMOOTH, "pairs": {}}
    for h0, h1 in PAIRS:
        c0, c1 = cls.index(h0), cls.index(h1)
        cells = {s: {"frames": 0, "errors": {m: 0 for m in names + RULES}, "floor": 0,
                     "pd": np.zeros(3), "pf": np.zeros(3), "g_h0": [], "g_h1": [], "pred_cv": []} for s in snrs}
        band = {"frames": 0, "errors": {r: 0 for r in RULES}, "floor": 0, "pred_cv": []}
        for seed in range(n_seeds):
            keep = (T[seed] == c0) | (T[seed] == c1)
            y = (T[seed][keep] == c1).astype(int)
            z = Z[seed][keep]
            sm = SM[seed][:, keep]                                  # (models, frames, classes)
            U = (sm[:, :, c1] > sm[:, :, c0]).astype(int).T         # (frames, 3)
            S = np.log(sm[:, :, c1].mean(0) + 1e-12) - np.log(sm[:, :, c0].mean(0) + 1e-12)
            dec = crossfit(U, y, S, z, seed)                        # estimated per SNR label
            for s in snrs:
                at = z == s
                c = cells[s]
                c["frames"] += int(at.sum())
                for m, a in enumerate(names):
                    c["errors"][a] += int((U[at, m] != y[at]).sum())
                for r in RULES:
                    c["errors"][r] += int((dec[r][at] != y[at]).sum())
                c["floor"] += floor_errors(U[at], y[at])
                full = fit(U[at], y[at])                            # descriptive, on all frames of the label
                c["pd"] += full[1]["p"] / n_seeds
                c["pf"] += full[0]["p"] / n_seeds
                c["g_h0"].append(list(full[0]["g2"].values()))
                c["g_h1"].append(list(full[1]["g2"].values()))
                c["pred_cv"].append(predicted_error(full))
            hi = z >= HIGH                                          # estimated on the high band pooled
            dec_b = crossfit(U[hi], y[hi], S[hi], np.zeros(int(hi.sum()), int), seed)
            band["frames"] += int(hi.sum())
            for r in RULES:
                band["errors"][r] += int((dec_b[r] != y[hi]).sum())
            band["floor"] += floor_errors(U[hi], y[hi])
            band["pred_cv"].append(predicted_error(fit(U[hi], y[hi])))

        def summary(levels):
            frames = sum(cells[s]["frames"] for s in levels)
            err = {k: sum(cells[s]["errors"][k] for s in levels) for k in names + RULES}
            fl = sum(cells[s]["floor"] for s in levels)
            g0 = np.mean([g for s in levels for g in cells[s]["g_h0"]], 0)
            g1 = np.mean([g for s in levels for g in cells[s]["g_h1"]], 0)
            return {"frames": frames, "errors": err, "floor_errors": fl,
                    "error_rate": {k: v / frames for k, v in err.items()}, "floor": fl / frames,
                    "pd": np.mean([cells[s]["pd"] for s in levels], 0).tolist(),
                    "pf": np.mean([cells[s]["pf"] for s in levels], 0).tolist(),
                    "corr_h0": dict(zip(["12", "13", "23"], g0.round(4).tolist())),
                    "corr_h1": dict(zip(["12", "13", "23"], g1.round(4).tolist())),
                    "cv_predicted_error": float(np.mean([e for s in levels for e in cells[s]["pred_cv"]]))}

        hb = summary([s for s in snrs if s >= HIGH])
        hb_band = {"frames": band["frames"], "errors": {**hb["errors"], **band["errors"]},
                   "floor_errors": band["floor"], "floor": band["floor"] / band["frames"],
                   "cv_predicted_error": float(np.mean(band["pred_cv"]))}
        hb_band["error_rate"] = {k: v / band["frames"] for k, v in hb_band["errors"].items()}
        result["pairs"][f"{h0}_vs_{h1}"] = {
            "h0": h0, "h1": h1,
            "overall": summary(snrs),
            f"snr_ge_{HIGH}": hb,
            f"snr_ge_{HIGH}_band_estimated": hb_band,
            "by_snr": {str(s): summary([s]) for s in snrs},
        }

    out = pathlib.Path(args.out) if args.out else ROOT / f"fusion_binary_rml2016_{'_'.join(names)}.json"
    out.write_text(json.dumps(result, indent=1))
    if args.out is None:
        figure(result, ROOT / "figures" / "39_binary_fusion_2016.png")

    for key, pr in result["pairs"].items():
        print(f"\n== {pr['h0']} (H0) vs {pr['h1']} (H1): error rate, %  ({pr['by_snr'][str(snrs[0])]['frames']} decisions per label)")
        cols = names + RULES
        print("  label " + "".join(f"{c[:9]:>10}" for c in cols) + "     floor  CV-pred.  corr H0 / H1")
        for k in [str(s) for s in snrs] + [f"snr_ge_{HIGH}", f"snr_ge_{HIGH}_band_estimated", "overall"]:
            r = pr[k] if not k.lstrip("-").isdigit() else pr["by_snr"][k]
            e = r["error_rate"]
            corr = (f"  {np.mean(list(r['corr_h0'].values())):.2f} / {np.mean(list(r['corr_h1'].values())):.2f}"
                    if "corr_h0" in r else "")
            label = {f"snr_ge_{HIGH}": f">={HIGH}", f"snr_ge_{HIGH}_band_estimated": f">={HIGH}b"}.get(k, k)
            print(f"  {label:>6}" + "".join(f"{100 * e[c]:10.1f}" for c in cols)
                  + f"{100 * r['floor']:10.1f}{100 * r['cv_predicted_error']:10.1f}" + corr)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

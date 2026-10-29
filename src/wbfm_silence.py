"""
wbfm_silence.py -- are the WBFM frames that every model misses silent?

Above 10 dB, 1,331 of the 1,508 frames that ICRNNA, LSTM2 and MCLDNN all get
wrong in the same way are WBFM read as AM-DSB (VOTE_2016.md), and in the binary
study no fusion rule gets below 29.6% on that pair (BINARY_FUSION_2016.md).
The usual explanation is silence in the source audio: with no audio, WBFM is
an unmodulated carrier, and so is AM-DSB with its carrier left in. This
measures it rather than citing it.

Two numbers per frame, both blind to the class:

  freq_dev  standard deviation of the instantaneous frequency (cycles/sample),
            from the phase step between consecutive samples. FM puts the
            audio here; a silent WBFM frame has only noise.
  env_cv    standard deviation of the envelope |x| over its mean. AM puts the
            audio here; WBFM and a silent AM-DSB frame have only noise.

A frame low on both is a bare carrier, whatever its label.

The test frames are rebuilt exactly as train_backbone.run_seed and
vote_fusion.py build them and lined up with the decisions vote_fusion.py
saved (vote_rml2016_ICRNNA_LSTM2_MCLDNN_frames.npz): the script stops unless
the class and SNR of every test frame match the saved ones. No model is run.

    cd src && python wbfm_silence.py --results-dir /content/drive/MyDrive/amc-results \\
        --data-path /content/drive/MyDrive/RadioML/RML2016.10a_dict.pkl

Writes, next to the frames file:
  wbfm_silence_rml2016.json         the summary below, per label and above 10 dB
  wbfm_silence_rml2016_frames.npz   both measures for every WBFM and AM-DSB
                                    frame of the dataset, the test frames and
                                    decisions of each seed, and the IQ of the
                                    frames at labels >= 0 dB (float16)
  wbfm_silence_rml2016.png          the two measures against each other
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

import cnn
import train_backbone as tb

HIGH = 10
FRAMES = "vote_rml2016_ICRNNA_LSTM2_MCLDNN_frames.npz"


def measures(X: np.ndarray) -> dict:
    """X (n, 2, L) -> per-frame frequency and envelope activity."""
    x = X[:, 0].astype(np.float64) + 1j * X[:, 1].astype(np.float64)
    f = np.angle(x[:, 1:] * np.conj(x[:, :-1])) / (2 * np.pi)       # instantaneous frequency
    env = np.abs(x)
    med = np.median(f, 1, keepdims=True)
    return {"freq_dev": f.std(1),
            "freq_dev_mad": 1.4826 * np.median(np.abs(f - med), 1),   # robust to phase jumps at envelope nulls
            "freq_mean": f.mean(1),
            "env_cv": env.std(1) / env.mean(1)}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random pos is above a random neg), ties counted half (Mann-Whitney)."""
    neg = np.sort(neg)
    below = np.searchsorted(neg, pos, "left")
    tied = np.searchsorted(neg, pos, "right") - below
    return float((below + 0.5 * tied).sum() / (len(pos) * len(neg)))


def pct(v: np.ndarray) -> dict:
    return {f"p{q}": float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95)} if len(v) else {}


def figure(path, m, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#52514e", "#8a8984", "#e6e5e0"
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for (label, idx, color, size, alpha) in groups:
        ax.scatter(m["freq_dev"][idx], m["env_cv"][idx], s=size, color=color, alpha=alpha,
                   edgecolors="none", label=f"{label} ({len(idx):,})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency activity: s.d. of instantaneous frequency (cycles/sample)", color=ink, fontsize=9.5)
    ax.set_ylabel("envelope activity: s.d. of |x| / mean of |x|", color=ink, fontsize=9.5)
    ax.set_title(f"RML2016.10a test frames at labels >= {HIGH} dB, seed 0\n"
                 "each dot one frame; colour = true class and what all three models said",
                 fontsize=10.5, color=ink, loc="left")
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(which="both", colors=muted, labelsize=9)
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, fontsize=8.8, labelcolor=ink, loc="upper left", markerscale=2,
              bbox_to_anchor=(0, -0.13), ncol=3, handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, help=f"where {FRAMES} is")
    p.add_argument("--data-path", default=None, help="the RML2016.10a pickle")
    p.add_argument("--frames-per-cell", type=int, default=1000)
    p.add_argument("--out-dir", default=None, help="default: --results-dir")
    args = p.parse_args()
    rdir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else rdir

    saved = np.load(rdir / FRAMES, allow_pickle=False)
    names = [str(a) for a in saved["models"]]
    D, T, Z = saved["decisions"].astype(int), saved["truth"].astype(int), saved["snr"]
    n_seeds = D.shape[0]

    X, y, z, class_names = tb.load_dataset("rml2016", args.frames_per_cell, args.data_path)
    wb, dsb = class_names.index("WBFM"), class_names.index("AM-DSB")
    snrs = sorted(int(s) for s in np.unique(z))
    print(f"{len(y):,} frames; WBFM = {wb}, AM-DSB = {dsb}; models {', '.join(names)}")

    # every WBFM and AM-DSB frame of the dataset, measured once
    sel = np.flatnonzero((y == wb) | (y == dsb))
    m = measures(X[sel])
    cls = (y[sel] == wb).astype(np.int8)                          # 1 = WBFM, 0 = AM-DSB
    lvl = z[sel]
    pos_of = np.full(len(y), -1)
    pos_of[sel] = np.arange(len(sel))

    # the test frames of each seed, checked against the saved decisions
    test_pos, test_dec = [], []
    for seed in range(n_seeds):
        _, _, te = cnn.split(X, y, z, test_fraction=0.15, val_fraction=0.15, seed=seed)
        if not (np.array_equal(y[te], T[seed]) and np.array_equal(z[te], Z[seed])):
            raise SystemExit(f"seed {seed}: rebuilt test set does not match {FRAMES}; refusing to "
                             "pair measurements with decisions made on other frames")
        keep = (y[te] == wb) | (y[te] == dsb)
        test_pos.append(pos_of[te[keep]])
        test_dec.append(D[seed][:, keep])
        print(f"  seed {seed}: test set matches the saved decisions frame for frame "
              f"({int(keep.sum())} WBFM/AM-DSB frames)")

    # WBFM test frames above 10 dB: what all three models said
    groups = {"wbfm_all_right": [], "wbfm_all_amdsb": [], "wbfm_mixed": [], "amdsb": []}
    for tp, dec in zip(test_pos, test_dec):
        hi = lvl[tp] >= HIGH
        is_wb = cls[tp] == 1
        all_wb, all_dsb = (dec == wb).all(0), (dec == dsb).all(0)
        groups["wbfm_all_right"].append(tp[hi & is_wb & all_wb])
        groups["wbfm_all_amdsb"].append(tp[hi & is_wb & all_dsb])
        groups["wbfm_mixed"].append(tp[hi & is_wb & ~all_wb & ~all_dsb])
        groups["amdsb"].append(tp[hi & ~is_wb])
    pooled = {k: np.concatenate(v) for k, v in groups.items()}

    summary = {"models": names, "seeds": n_seeds, "high_snr": HIGH, "groups_ge_high": {}}
    print(f"\nWBFM and AM-DSB test frames at labels >= {HIGH} dB, pooled over {n_seeds} seeds "
          "(a frame can be in more than one seed's test set):")
    print(f"  {'group':34s} {'frames':>7} {'freq_dev p5/p50/p95':>28} {'env_cv p5/p50/p95':>26}")
    for k, idx in pooled.items():
        fd, ec = m["freq_dev"][idx], m["env_cv"][idx]
        summary["groups_ge_high"][k] = {"frames": int(len(idx)), "freq_dev": pct(fd), "env_cv": pct(ec)}
        if len(idx):
            print(f"  {k:34s} {len(idx):7d} "
                  f"{np.percentile(fd, 5):9.4f}{np.percentile(fd, 50):9.4f}{np.percentile(fd, 95):9.4f}  "
                  f"{np.percentile(ec, 5):8.3f}{np.percentile(ec, 50):8.3f}{np.percentile(ec, 95):8.3f}")

    # does frequency activity alone tell the missed WBFM frames from the recognised ones?
    right, missed = pooled["wbfm_all_right"], pooled["wbfm_all_amdsb"]
    if not len(right) or not len(missed):
        raise SystemExit("no WBFM frames that all three models get right, or none they all call "
                         "AM-DSB: nothing to compare")
    a = auc(m["freq_dev"][right], m["freq_dev"][missed])
    cand = np.unique(np.quantile(m["freq_dev"][np.concatenate([right, missed])], np.linspace(0.01, 0.99, 197)))
    bal = [0.5 * ((m["freq_dev"][missed] < t).mean() + (m["freq_dev"][right] >= t).mean()) for t in cand]
    t_f = float(cand[int(np.argmax(bal))])
    summary.update({"auc_freq_dev_right_vs_missed": a, "freq_threshold": t_f,
                    "freq_threshold_balanced_accuracy": float(max(bal))})
    print(f"\nfrequency activity, recognised WBFM against WBFM all three call AM-DSB: AUC {a:.3f}")
    print(f"best single threshold {t_f:.4f} cycles/sample: balanced accuracy {max(bal):.3f}")

    # every frame of the dataset: how many are bare carriers, by class and label
    print("\nWhole dataset, 1,000 frames per class and label. 'flat' = frequency activity below the")
    print("threshold above AND envelope activity below the 95th percentile of WBFM's at that label")
    print("(WBFM's envelope carries no audio, so that percentile is the noise level of a flat envelope).")
    print(f"  {'label':>5} {'WBFM low-freq':>14} {'WBFM flat':>10} {'AM-DSB flat':>12} {'pair floor if flat = unreadable':>32}")
    summary["by_snr"] = {}
    for s in snrs:
        w, d = (lvl == s) & (cls == 1), (lvl == s) & (cls == 0)
        t_e = float(np.percentile(m["env_cv"][w], 95))
        flat = (m["freq_dev"] < t_f) & (m["env_cv"] < t_e)
        lowf_w = float((m["freq_dev"][w] < t_f).mean())
        fw, fd = int(flat[w].sum()), int(flat[d].sum())
        floor = min(fw, fd) / (w.sum() + d.sum())
        summary["by_snr"][str(s)] = {"wbfm_frames": int(w.sum()), "amdsb_frames": int(d.sum()),
                                     "wbfm_low_freq": lowf_w, "wbfm_flat": fw, "amdsb_flat": fd,
                                     "env_threshold": t_e, "pair_floor_if_flat_unreadable": floor,
                                     "wbfm_freq_dev": pct(m["freq_dev"][w]), "amdsb_env_cv": pct(m["env_cv"][d])}
        print(f"  {s:5d} {100 * lowf_w:13.1f}% {fw:10d} {fd:12d} {100 * floor:31.1f}%")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wbfm_silence_rml2016.json").write_text(json.dumps(summary, indent=1))
    keep_iq = lvl >= 0
    np.savez_compressed(
        out_dir / "wbfm_silence_rml2016_frames.npz",
        dataset_index=sel, is_wbfm=cls, snr=lvl, class_names=np.array(class_names),
        **{k: v.astype(np.float32) for k, v in m.items()},
        iq_index=np.flatnonzero(keep_iq), iq=X[sel[keep_iq]].astype(np.float16),
        **{f"test_pos_seed{s}": tp for s, tp in enumerate(test_pos)},
        **{f"test_dec_seed{s}": dec.astype(np.int8) for s, dec in enumerate(test_dec)},
        models=np.array(names))
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    figure(out_dir / "wbfm_silence_rml2016.png", m,
           [("AM-DSB", groups["amdsb"][0], aqua, 7, 0.45),
            ("WBFM, all three say WBFM", groups["wbfm_all_right"][0], blue, 7, 0.6),
            ("WBFM, all three say AM-DSB", groups["wbfm_all_amdsb"][0], orange, 7, 0.6)])
    print(f"\nwrote wbfm_silence_rml2016.json, wbfm_silence_rml2016_frames.npz and "
          f"wbfm_silence_rml2016.png to {out_dir}")


if __name__ == "__main__":
    main()

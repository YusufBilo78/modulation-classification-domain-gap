"""
wbfm_silence.py -- are the WBFM frames that every model misses bare carriers?

Above 10 dB, 1,331 of the 1,508 frames that ICRNNA, LSTM2 and MCLDNN all get
wrong in the same way are WBFM read as AM-DSB (VOTE_2016.md), and in the binary
study no fusion rule gets below 29.6% on that pair (BINARY_FUSION_2016.md).
The usual explanation is silence in the source audio. With no audio, WBFM is an
unmodulated carrier, and so is AM-DSB with its carrier left in. This script
measures that instead of citing it.

Three numbers per frame, all blind to the class and to the SNR:

  fm_ratio   how far the phase wanders from a straight line (a constant
             carrier offset), divided by what the noise alone makes it wander.
             FM puts its audio here.
  am_ratio   how far the envelope moves, divided by the same noise level. AM
             puts its audio here.
  offset     the straight line's slope: the frame's mean frequency, in cycles
             per sample. FM with slow audio would show up here as a shift.

The noise level comes from the frame itself. White noise on a carrier moves
the phase (in radians) and the relative envelope by the same amount,
sqrt(N/2)/A. First differences remove slow modulation and keep that noise, so
the noise level is the average of s.d.(diff(phase))/sqrt(2) and
s.d.(diff(envelope / mean envelope))/sqrt(2). A bare carrier therefore gives
both ratios close to 1 at any SNR. Two thresholds decide when a frame has
modulation beyond noise, both taken from the class that cannot have that
modulation:
  - FM is visible when fm_ratio exceeds the 99th percentile of AM-DSB's
    fm_ratio at the same label (AM carries no FM).
  - AM is visible when am_ratio exceeds the 99th percentile of WBFM's
    am_ratio at the same label (FM has a constant envelope).
A frame with neither is a bare carrier.

The test frames are rebuilt exactly as train_backbone.run_seed and
vote_fusion.py build them and lined up with the decisions vote_fusion.py
saved (vote_rml2016_ICRNNA_LSTM2_MCLDNN_frames.npz). The script stops unless
the class and SNR of every test frame match the saved ones. No model is run.

    cd src && python wbfm_silence.py --results-dir /content/drive/MyDrive/amc-results \\
        --data-path /content/drive/MyDrive/RadioML/RML2016.10a_dict.pkl

Writes, next to the frames file:
  wbfm_carrier_rml2016.json         every table below
  wbfm_carrier_rml2016_frames.npz   the three numbers for every WBFM and AM-DSB
                                    frame, and each seed's test frames and
                                    decisions
  wbfm_carrier_rml2016.png          fm_ratio against am_ratio above 10 dB
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
NULL_PCT = 99
FS = 200e3                      # the generator's sample rate (NOISE_2016.md); only used to print Hz
FRAMES = "vote_rml2016_ICRNNA_LSTM2_MCLDNN_frames.npz"
STEM = "wbfm_carrier_rml2016"


def measures(X: np.ndarray) -> dict:
    """X (n, 2, L) -> fm_ratio, am_ratio, offset and the noise level, per frame."""
    x = X[:, 0].astype(np.float64) + 1j * X[:, 1].astype(np.float64)
    L = x.shape[1]
    phase = np.unwrap(np.angle(x), axis=1)
    V = np.vstack([np.ones(L), np.arange(L)]).T
    coef, *_ = np.linalg.lstsq(V, phase.T, rcond=None)
    wander = phase - (V @ coef).T
    env = np.abs(x)
    env = env / env.mean(1, keepdims=True)
    noise = 0.5 * (np.diff(wander, axis=1).std(1) + np.diff(env, axis=1).std(1)) / np.sqrt(2)
    return {"fm_ratio": wander.std(1) / noise, "am_ratio": env.std(1) / noise,
            "offset": coef[1] / (2 * np.pi), "noise": noise}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random pos is above a random neg), ties counted half (Mann-Whitney)."""
    neg = np.sort(neg)
    below = np.searchsorted(neg, pos, "left")
    tied = np.searchsorted(neg, pos, "right") - below
    return float((below + 0.5 * tied).sum() / (len(pos) * len(neg)))


def pct(v: np.ndarray) -> dict:
    return {f"p{q}": float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95)} if len(v) else {}


def figure(path, m, groups, t_fm, t_am):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ink, muted, grid = "#52514e", "#8a8984", "#e6e5e0"
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for label, idx, color in groups:
        ax.scatter(m["fm_ratio"][idx], m["am_ratio"][idx], s=7, color=color, alpha=0.55,
                   edgecolors="none", label=f"{label} ({len(idx):,})")
    ax.axvline(t_fm, color=ink, lw=0.9)
    ax.axhline(t_am, color=ink, lw=0.9)
    ax.text(t_fm * 0.97, t_am * 0.97, "bare carrier:\nno FM, no AM\nbeyond noise", ha="right", va="top",
            fontsize=8.5, color=ink)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FM: phase wander / noise  (1 = none beyond noise)", color=ink, fontsize=9.5)
    ax.set_ylabel("AM: envelope motion / noise  (1 = none beyond noise)", color=ink, fontsize=9.5)
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
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, fontsize=8.8, labelcolor=ink, loc="upper left", markerscale=2,
              bbox_to_anchor=(0, -0.13), ncol=2, handletextpad=0.2, columnspacing=1.2)
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

    sel = np.flatnonzero((y == wb) | (y == dsb))       # every WBFM and AM-DSB frame, measured once
    m = measures(X[sel])
    is_wb = y[sel] == wb
    lvl = z[sel]
    pos_of = np.full(len(y), -1)
    pos_of[sel] = np.arange(len(sel))

    # thresholds per label, from the class that cannot carry that modulation
    t_fm = {s: float(np.percentile(m["fm_ratio"][~is_wb & (lvl == s)], NULL_PCT)) for s in snrs}
    t_am = {s: float(np.percentile(m["am_ratio"][is_wb & (lvl == s)], NULL_PCT)) for s in snrs}
    vis_fm = m["fm_ratio"] > np.array([t_fm[s] for s in lvl])
    vis_am = m["am_ratio"] > np.array([t_am[s] for s in lvl])
    bare = ~vis_fm & ~vis_am

    test_pos, test_dec, n_test_high = [], [], 0
    for seed in range(n_seeds):
        _, _, te = cnn.split(X, y, z, test_fraction=0.15, val_fraction=0.15, seed=seed)
        if not (np.array_equal(y[te], T[seed]) and np.array_equal(z[te], Z[seed])):
            raise SystemExit(f"seed {seed}: rebuilt test set does not match {FRAMES}; refusing to "
                             "pair measurements with decisions made on other frames")
        keep = (y[te] == wb) | (y[te] == dsb)
        test_pos.append(pos_of[te[keep]])
        test_dec.append(D[seed][:, keep])
        n_test_high += int((z[te] >= HIGH).sum())
        print(f"  seed {seed}: test set matches the saved decisions frame for frame "
              f"({int(keep.sum())} WBFM/AM-DSB frames)")

    summary = {"models": names, "seeds": n_seeds, "high_snr": HIGH, "null_percentile": NULL_PCT,
               "fs_hz_for_printing": FS}

    # 1. above 10 dB, by what all three models said
    groups = {"wbfm_all_wbfm": [], "wbfm_all_amdsb": [], "wbfm_mixed": [], "amdsb": []}
    cross = {"wbfm": {}, "amdsb": {}}
    for tp, dec in zip(test_pos, test_dec):
        hi = lvl[tp] >= HIGH
        all_wb, all_dsb = (dec == wb).all(0), (dec == dsb).all(0)
        w = hi & is_wb[tp]
        groups["wbfm_all_wbfm"].append(tp[w & all_wb])
        groups["wbfm_all_amdsb"].append(tp[w & all_dsb])
        groups["wbfm_mixed"].append(tp[w & ~all_wb & ~all_dsb])
        groups["amdsb"].append(tp[hi & ~is_wb[tp]])
        for k in np.flatnonzero(w):
            key = ("FM visible" if vis_fm[tp[k]] else "no FM") + " | " + (
                "all WBFM" if all_wb[k] else "all AM-DSB" if all_dsb[k] else "mixed")
            cross["wbfm"][key] = cross["wbfm"].get(key, 0) + 1
        for k in np.flatnonzero(hi & ~is_wb[tp]):
            key = ("AM visible" if vis_am[tp[k]] else "no AM") + " | " + (
                "all AM-DSB" if all_dsb[k] else "some WBFM" if (dec[:, k] == wb).any() else "other")
            cross["amdsb"][key] = cross["amdsb"].get(key, 0) + 1
    pooled = {k: np.concatenate(v) for k, v in groups.items()}
    print(f"\nTest frames at labels >= {HIGH} dB, pooled over {n_seeds} seeds "
          "(a frame can be in more than one seed's test set):")
    print(f"  {'group':16s} {'frames':>7}   {'fm_ratio p5/p50/p95':>22}   {'am_ratio p5/p50/p95':>22}   |offset| Hz p50")
    summary["groups_ge_high"] = {}
    for k, idx in pooled.items():
        fm, am, off = m["fm_ratio"][idx], m["am_ratio"][idx], np.abs(m["offset"][idx]) * FS
        summary["groups_ge_high"][k] = {"frames": int(len(idx)), "fm_ratio": pct(fm), "am_ratio": pct(am),
                                        "abs_offset_hz": pct(off)}
        if len(idx):
            print(f"  {k:16s} {len(idx):7d}   " + "".join(f"{np.percentile(fm, q):7.2f}" for q in (5, 50, 95))
                  + "   " + "".join(f"{np.percentile(am, q):7.2f}" for q in (5, 50, 95)) + f"   {np.median(off):8.2f}")

    r, mi = pooled["wbfm_all_wbfm"], pooled["wbfm_all_amdsb"]
    hi_all = lvl >= HIGH
    bw, bd = np.flatnonzero(hi_all & is_wb & bare), np.flatnonzero(hi_all & ~is_wb & bare)
    summary["auc"] = {
        "recognised_vs_missed_wbfm": {"fm_ratio": auc(m["fm_ratio"][r], m["fm_ratio"][mi]),
                                      "abs_offset": auc(np.abs(m["offset"][r]), np.abs(m["offset"][mi]))},
        "bare_wbfm_vs_bare_amdsb": {k: auc(v[bw], v[bd]) for k, v in
                                    (("fm_ratio", m["fm_ratio"]), ("am_ratio", m["am_ratio"]),
                                     ("abs_offset", np.abs(m["offset"])))}}
    a = summary["auc"]
    print(f"\nAUC, WBFM all three recognise against WBFM all three call AM-DSB: "
          f"fm_ratio {a['recognised_vs_missed_wbfm']['fm_ratio']:.3f}, "
          f"|offset| {a['recognised_vs_missed_wbfm']['abs_offset']:.3f}")
    print(f"AUC, bare-carrier WBFM against bare-carrier AM-DSB (whole dataset, >= {HIGH} dB; 0.5 = "
          "indistinguishable): " + ", ".join(f"{k} {v:.3f}" for k, v in a["bare_wbfm_vs_bare_amdsb"].items()))

    summary["cross_ge_high"] = cross
    print(f"\nWBFM test frames >= {HIGH} dB: visible FM against what the three models said")
    for k in sorted(cross["wbfm"]):
        print(f"  {k:28s} {cross['wbfm'][k]:6d}")
    print(f"AM-DSB test frames >= {HIGH} dB: visible AM against what the three models said")
    for k in sorted(cross["amdsb"]):
        print(f"  {k:28s} {cross['amdsb'][k]:6d}")

    # 2. every frame of the dataset, by label
    print(f"\nWhole dataset, 1,000 frames per class and label. Null medians should sit near 1 where the "
          "measure works.")
    print(f"  {'label':>5} {'null FM':>8} {'null AM':>8} {'WBFM bare':>10} {'AM-DSB bare':>12} {'pair floor':>11}")
    summary["by_snr"] = {}
    for s in snrs:
        w, d = is_wb & (lvl == s), ~is_wb & (lvl == s)
        fw, fd = float(bare[w].mean()), float(bare[d].mean())
        floor = min(int(bare[w].sum()), int(bare[d].sum())) / int(w.sum() + d.sum())
        summary["by_snr"][str(s)] = {
            "null_fm_median": float(np.median(m["fm_ratio"][d])), "null_am_median": float(np.median(m["am_ratio"][w])),
            "fm_threshold": t_fm[s], "am_threshold": t_am[s],
            "wbfm_fm_visible": float(vis_fm[w].mean()), "amdsb_am_visible": float(vis_am[d].mean()),
            "wbfm_bare": fw, "amdsb_bare": fd, "pair_floor": floor}
        print(f"  {s:5d} {np.median(m['fm_ratio'][d]):8.2f} {np.median(m['am_ratio'][w]):8.2f} "
              f"{100 * fw:9.1f}% {100 * fd:11.1f}% {100 * floor:10.1f}%")

    # 3. sensitivity to the null percentile, and what it costs an 11-class classifier
    summary["sensitivity_ge_high"] = {}
    print(f"\nAbove {HIGH} dB, pooled, against the percentile that sets the thresholds:")
    for q in (95, 99, 99.9):
        tf = np.array([np.percentile(m["fm_ratio"][~is_wb & (lvl == s)], q) for s in snrs])
        ta = np.array([np.percentile(m["am_ratio"][is_wb & (lvl == s)], q) for s in snrs])
        k = np.searchsorted(snrs, lvl)
        b = (m["fm_ratio"] <= tf[k]) & (m["am_ratio"] <= ta[k])
        fw, fd = float(b[hi_all & is_wb].mean()), float(b[hi_all & ~is_wb].mean())
        floor = min(int(b[hi_all & is_wb].sum()), int(b[hi_all & ~is_wb].sum())) / int(hi_all.sum())
        summary["sensitivity_ge_high"][str(q)] = {"wbfm_bare": fw, "amdsb_bare": fd, "pair_floor": floor}
        print(f"  {q:>5}th: WBFM bare {100 * fw:5.1f}%, AM-DSB bare {100 * fd:5.1f}%, pair floor {100 * floor:5.1f}%")
    bare_test = sum(int((bare[tp] & is_wb[tp] & (lvl[tp] >= HIGH)).sum()) for tp in test_pos)
    summary["bare_wbfm_test_frames_ge_high"] = bare_test
    summary["test_frames_ge_high"] = n_test_high
    print(f"\nBare-carrier WBFM frames among all {n_test_high:,} test frames at >= {HIGH} dB: {bare_test:,} "
          f"({100 * bare_test / n_test_high:.1f}% of all decisions there, unreadable as WBFM by any classifier)")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{STEM}.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(
        out_dir / f"{STEM}_frames.npz",
        dataset_index=sel, is_wbfm=is_wb.astype(np.int8), snr=lvl, class_names=np.array(class_names),
        **{k: v.astype(np.float32) for k, v in m.items()},
        fm_visible=vis_fm, am_visible=vis_am,
        **{f"test_pos_seed{s}": tp for s, tp in enumerate(test_pos)},
        **{f"test_dec_seed{s}": dec.astype(np.int8) for s, dec in enumerate(test_dec)},
        models=np.array(names))
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    hi_thr = (np.mean([t_fm[s] for s in snrs if s >= HIGH]), np.mean([t_am[s] for s in snrs if s >= HIGH]))
    figure(out_dir / f"{STEM}.png", m,
           [("AM-DSB", groups["amdsb"][0], aqua),
            ("WBFM, all three say WBFM", groups["wbfm_all_wbfm"][0], blue),
            ("WBFM, all three say AM-DSB", groups["wbfm_all_amdsb"][0], orange)], *hi_thr)
    print(f"\nwrote {STEM}.json, {STEM}_frames.npz and {STEM}.png to {out_dir}")


if __name__ == "__main__":
    main()

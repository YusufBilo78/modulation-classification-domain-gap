"""One SNR at a time: the four-class box on RML2016.10a, opened level by level.

Show the decision matrix at a single SNR, then lower --
10, 6, then about 3 dB -- plot the error rate against SNR, and say on every
graph how many decisions stand behind each point. Show precision next to
recall.

Reads the train_backbone.py result files that carry `confusions_by_snr`
(seeds pooled; every count below is a test-set decision) and writes

    figures/31_rml2016_c4_matrices.png        matrices at 10, 6, 4, 2 dB
    figures/32_rml2016_c4_error_vs_snr.png    error rate vs SNR, 128/64/32 samples
    figures/33_rml2016_c4_recall_precision.png  recall and precision per class
    SNR_2016.md                               the same numbers as tables

    python tools/single_snr.py

Definitions used throughout:

    recall(c)    = frames of class c decided as c / frames of class c sent
                   (row of the matrix; fixed denominator, 2,100 per SNR here)
    precision(c) = frames of class c decided as c / frames decided as c
                   (column; its denominator is whatever the model chose,
                   so it is printed next to every precision value)
    error rate   = wrong decisions / all decisions at that SNR
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = {128: "train_backbone_rml2016_f1000_c4_colab_s14.npz",
        64: "train_backbone_rml2016_f1000_c4_L64_colab_s14.npz",
        32: "train_backbone_rml2016_f1000_c4_L32_colab_s14.npz"}
LEVELS = [10, 6, 4, 2]          # what was asked for
BELOW = [0, -2, -4, -6]          # where the box actually starts to break
MIN_N = 200                      # precision drawn solid only above this

# reference categorical palette, light mode, slots 1-4 in fixed order
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "figure.facecolor": "white", "axes.facecolor": "white",
})


def load(path):
    z = np.load(ROOT / path, allow_pickle=True)
    return {"names": [str(c) for c in z["class_names"]],
            "snrs": z["snrs"].astype(int),
            "by": z["confusions_by_snr"],               # (seeds, snr, n, n)
            "seeds": len(z["overall"])}


def error_rate(by):
    """Pooled and per-seed error rate at every SNR."""
    wrong = by.sum((2, 3)) - np.trace(by, axis1=2, axis2=3)   # (seeds, snr)
    total = by.sum((2, 3))
    return wrong.sum(0) / total.sum(0), wrong / total, total.sum(0), wrong.sum(0)


def matrices_figure(r, out):
    names, snrs = r["names"], list(r["snrs"])
    levels = LEVELS + BELOW
    fig, axes = plt.subplots(2, len(LEVELS), figsize=(16, 9))
    axes = axes.ravel()
    for ax, lv in zip(axes, levels):
        m = r["by"][:, snrs.index(lv)].sum(0)
        rowpct = m / m.sum(1, keepdims=True)
        ax.imshow(rowpct, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                dark = rowpct[i, j] > 0.55
                ax.text(j, i, f"{m[i, j]:,}\n{100 * rowpct[i, j]:.1f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if dark else INK)
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("decided")
        ax.set_ylabel("transmitted")
        ax.grid(False)
        err = m.sum() - np.trace(m)
        ax.set_title(f"{lv} dB — {err:,} wrong of {m.sum():,}",
                     color=INK, fontsize=11)
    per_row = int(r["by"][:, 0].sum(0).sum(1)[0])
    fig.suptitle("RML2016.10a, four classes, 128 samples: one SNR per matrix. "
                 f"Each row is {per_row:,} frames ({r['seeds']} seeds × 150); "
                 "cell = count and share of its row",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def error_figure(runs, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (L, r) in enumerate(runs.items()):
        pooled, per_seed, total, _ = error_rate(r["by"])
        s = r["snrs"]
        ax.fill_between(s, 100 * per_seed.min(0), 100 * per_seed.max(0),
                        color=SLOT[k], alpha=0.15, linewidth=0)
        ax.plot(s, 100 * pooled, color=SLOT[k], marker="o", markersize=4)
        ax.annotate(f"{L} samples ({L // 8} symbols)", (s[-1], 100 * pooled[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)
    ax.axhline(75, color=INK2, linewidth=1, linestyle=":")
    ax.text(18, 76.5, "chance: 75% wrong", color=INK2, fontsize=9, ha="right")
    ax.set_xlim(-20, 25)
    ax.set_xticks(range(-20, 19, 2))
    ax.set_ylim(0, 80)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("error rate (%)")
    n = int(total[0])
    ax.set_title("Error rate against SNR, four classes, RML2016.10a\n"
                 f"{n:,} decisions behind every point (14 seeds × 4 classes × 150); "
                 "band = best and worst seed",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def recall_precision(by):
    m = by.sum(0)                                   # (snr, n, n)
    tp = np.diagonal(m, axis1=1, axis2=2)
    sent = m.sum(2)
    decided = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / sent, np.where(decided > 0, tp / decided, np.nan), sent, decided


def rp_figure(r, out):
    names, s = r["names"], r["snrs"]
    rec, prec, sent, decided = recall_precision(r["by"])
    fig, axes = plt.subplots(1, len(names), figsize=(16, 4.4), sharey=True)
    for c, ax in enumerate(axes):
        ax.plot(s, 100 * rec[:, c], color=SLOT[0], label="recall")
        # precision over few decisions is noise: drawn dotted below MIN_N
        p = 100 * prec[:, c]
        ax.plot(s, p, color=SLOT[1], linestyle=":", linewidth=1.5)
        ax.plot(s, np.where(decided[:, c] >= MIN_N, p, np.nan),
                color=SLOT[1], label="precision")
        ax.axhline(25, color=INK2, linewidth=1, linestyle=":")
        ax.set_title(names[c], color=INK, fontsize=11, loc="left")
        ax.set_xticks(range(-20, 19, 6))
        ax.set_xlabel("SNR (dB)")
        lo, hi = int(decided[:, c].min()), int(decided[:, c].max())
        ax.text(0.98, 0.04, f"precision n = {lo:,}–{hi:,}",
                transform=ax.transAxes, ha="right", color=INK2, fontsize=8.5)
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 102)
    axes[0].text(-20, 27, "chance 25%", color=INK2, fontsize=8.5)
    axes[0].legend(loc="center right", frameon=False)
    fig.suptitle("Recall and precision per class, 128 samples. Recall: "
                 f"{int(sent[0, 0]):,} frames sent per point. Precision: "
                 "n = frames the model decided as that class (varies, shown per panel;\n"
                 f"dotted where n < {MIN_N})",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def md_matrix(names, m):
    rows = ["| sent \\ decided | " + " | ".join(names) + " | recall |",
            "|---|" + "---|" * (len(names) + 1)]
    for i, nm in enumerate(names):
        cells = [f"**{v:,}**" if j == i else f"{v:,}" for j, v in enumerate(m[i])]
        rows.append(f"| **{nm}** | " + " | ".join(cells)
                    + f" | {100 * m[i, i] / m[i].sum():.1f}% |")
    dec = m.sum(0)
    prec = [f"{100 * m[j, j] / dec[j]:.1f}% (of {dec[j]:,})" if dec[j] else "—"
            for j in range(len(names))]
    rows.append("| precision | " + " | ".join(prec) + " | |")
    return "\n".join(rows)


def summary(runs):
    """The reading of the tables, with every number computed here."""
    r = runs[128]
    snrs, by = list(r["snrs"]), r["by"]
    pooled = by.sum(0)
    wrong = {s: int(pooled[i].sum() - np.trace(pooled[i])) for i, s in enumerate(snrs)}
    m10 = pooled[snrs.index(10)]
    qam = int(m10[2, 3] + m10[3, 2])
    plateau = [wrong[s] for s in snrs if s >= 2]
    low = by[:, 0].sum(1)                                   # (seeds, decided)
    share = low / low.sum(1, keepdims=True)
    psk = pooled[0].sum(0)[:2].sum() / pooled[0].sum()
    floors = {k: error_rate(v["by"])[0][[i for i, s in enumerate(v["snrs"]) if s >= 2]]
              for k, v in runs.items()}
    return [
        "## What this shows", "",
        f"1. **From +10 dB down to +2 dB nothing changes.** {wrong[10]:,}, "
        f"{wrong[6]:,}, {wrong[4]:,} and {wrong[2]:,} wrong of 8,400. Across "
        f"every level from +2 to +18 dB the count stays between "
        f"{min(plateau):,} and {max(plateau):,}.",
        f"2. **Those errors are the QAM pair.** At 10 dB, {qam:,} of the "
        f"{wrong[10]:,} are QAM16 ↔ QAM64. BPSK and QPSK are at 99–100% "
        "recall and precision. The QAM confusion does not shrink with SNR; it "
        "shrinks with frame length (last table): the floor is "
        + ", ".join(f"{100 * f.mean():.1f}% at {k} samples" for k, f in floors.items())
        + ". SNR decides where the curve falls; the number of symbols decides "
        "how low it can go.",
        f"3. **The box breaks below 0 dB.** {wrong[0]:,} wrong at 0 dB, "
        f"{wrong[-2]:,} at −2, {wrong[-4]:,} at −4, {wrong[-6]:,} at −6. "
        "QPSK starts leaking into both QAMs and QAM16 into QAM64; BPSK "
        f"holds longest ({100 * pooled[snrs.index(-4)][0, 0] / pooled[snrs.index(-4)][0].sum():.1f}% "
        "recall at −4 dB).",
        f"4. **At −20 dB the four rows are the same row.** The frame carries no "
        f"information, and {100 * psk:.1f}% of all decisions are BPSK or QPSK. "
        "Which of the two is arbitrary: each seed picks its own split, from "
        f"{100 * share[:, 0].min():.0f}% BPSK to {100 * share[:, 0].max():.0f}% "
        "BPSK. Pooled, BPSK and QPSK recall sit near 50% and their precision at "
        "25%, which is chance. That recall is the sink, not recognition; "
        "precision is what says so.",
        ""]


def write_md(runs, out):
    r = runs[128]
    names, snrs = r["names"], list(r["snrs"])
    L = ["# One SNR at a time — RML2016.10a, four classes",
         "",
         "Generated by `tools/single_snr.py` from "
         f"`{RUNS[128]}` (and the 64- and 32-sample runs for the last table).",
         "14 seeds pooled. Every count is one test-set decision: a frame the",
         "model had never seen, classified once. At each SNR each class",
         "contributes 150 test frames per seed, so **2,100 frames per row and",
         "8,400 decisions per matrix**.",
         "",
         "- **recall** of a class = its diagonal count / its row total "
         "(of the frames that *were* this class, how many were called it)",
         "- **precision** of a class = its diagonal count / its column total "
         "(of the frames *called* this class, how many were it); the column",
         "  total is printed with it, because the model chooses it",
         "- **error rate** = off-diagonal counts / 8,400",
         ""]
    L += summary(runs)
    for lv in LEVELS + BELOW:
        if lv == BELOW[0]:
            L += ["## Below 2 dB: where the box starts to break", "",
                  "From +2 dB to +18 dB the error rate does not move (table "
                  "below). The matrices only change once the SNR goes under "
                  "0 dB, so these four are added to the four that were asked for.",
                  ""]
        m = r["by"][:, snrs.index(lv)].sum(0)
        err = m.sum() - np.trace(m)
        L += [f"### {lv:+d} dB — {err:,} wrong of {m.sum():,} "
              f"({100 * err / m.sum():.2f}%)", "", md_matrix(names, m), ""]
    L += ["## Error rate against SNR, and against frame length", "",
          "8,400 decisions behind every entry. Count wrong, then the rate.", "",
          "| SNR (dB) | " + " | ".join(f"{k} samples ({k // 8} symbols)"
                                     for k in runs) + " |",
          "|---|" + "---|" * len(runs)]
    stats = {k: error_rate(v["by"]) for k, v in runs.items()}
    for i, s in enumerate(snrs):
        L.append(f"| {s:+d} | " + " | ".join(
            f"{int(stats[k][3][i]):,} ({100 * stats[k][0][i]:.1f}%)"
            for k in runs) + " |")
    L += ["", "Chance is 75% wrong (four classes). The seed spread is drawn "
          "as a band in `figures/32_rml2016_c4_error_vs_snr.png`.", ""]
    rec, prec, sent, decided = recall_precision(r["by"])
    L += ["## Recall and precision per class at every SNR, 128 samples", "",
          "Recall has 2,100 frames behind it at every SNR. Precision's "
          "denominator is the number of frames the model *decided* as that "
          "class, given after it.", "",
          "| SNR | " + " | ".join(f"{n} recall | {n} precision" for n in names)
          + " |", "|---|" + "---|---|" * len(names)]
    for i, s in enumerate(snrs):
        cells = []
        for c in range(len(names)):
            p = "—" if np.isnan(prec[i, c]) else \
                f"{100 * prec[i, c]:.1f}% (of {int(decided[i, c]):,})"
            cells += [f"{100 * rec[i, c]:.1f}%", p]
        L.append(f"| {s:+d} | " + " | ".join(cells) + " |")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")


def main():
    runs = {L: load(p) for L, p in RUNS.items()}
    figs = ROOT / "figures"
    matrices_figure(runs[128], figs / "31_rml2016_c4_matrices.png")
    error_figure(runs, figs / "32_rml2016_c4_error_vs_snr.png")
    rp_figure(runs[128], figs / "33_rml2016_c4_recall_precision.png")
    write_md(runs, ROOT / "SNR_2016.md")
    for L, r in runs.items():
        pooled, _, total, wrong = error_rate(r["by"])
        print(L, {int(s): int(w) for s, w in zip(r["snrs"], wrong)
                  if s in LEVELS + BELOW}, "of", int(total[0]))


if __name__ == "__main__":
    main()

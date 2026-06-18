#!/usr/bin/env python
"""Regenerate the v2 figures (6,7,9) in the SAME house style as fig1-5
(make_paper_figures.py): Wong palette, no top/right spines, subtle grid, dpi 300.
Reads the recorded CSVs so numbers stay identical to the tables.

    python scripts/make_v2_figures.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
RES = Path(__file__).resolve().parents[1] / "results"
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200, "savefig.dpi": 300,
})
C_EXEC, C_HMM, C_RAW, C_WEAK, C_GOOD = "#0072B2", "#D55E00", "#888888", "#E69F00", "#009E73"


def _save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        if "cal_reps" in r and int(r["cal_reps"]) != 5:
            continue
        d[int(r["subject"])] = r
    return d


# ---- fig6: wideband paired per-subject scatter --------------------------------
def fig_wideband():
    wb = _load(RES / "exp_wideband_full.csv"); cur = _load(RES / "exp_cur_full.csv")
    subs = sorted(set(wb) & set(cur))
    n = np.array([float(cur[s]["hmm_bal_acc"]) for s in subs]) * 100
    w = np.array([float(wb[s]["hmm_bal_acc"]) for s in subs]) * 100
    up = w > n
    fig, ax = plt.subplots(figsize=(4.4, 4.3))
    lo, hi = 55, 95
    ax.plot([lo, hi], [lo, hi], ls="--", color=C_RAW, lw=1.3, zorder=1)
    ax.scatter(n[up], w[up], s=46, color=C_EXEC, zorder=3, edgecolors="white", linewidths=0.6,
               label=f"wider band better ({up.sum()}/{len(subs)})")
    ax.scatter(n[~up], w[~up], s=46, color=C_WEAK, zorder=3, edgecolors="white", linewidths=0.6,
               label=f"narrow better ({(~up).sum()}/{len(subs)})")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("per-window accuracy, 20–120 Hz (%)")
    ax.set_ylabel("per-window accuracy, 20–450 Hz (%)")
    ax.set_title("Wider acquisition band\n(+2.6 pts, $p$=0.005)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    _save(fig, "fig6_wideband")


# ---- fig7: amputee per-subject per-execution (sorted) -------------------------
def fig_amputee():
    amp = _load(RES / "db3_xfer12.csv"); subs = sorted(amp)
    acc = np.sort(np.array([float(amp[s]["seg_bal_acc"]) for s in subs]) * 100)[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    ax.grid(axis="x", alpha=0)
    # reported pooled range across decoders (Table: ours 63.8 -- classical 74.2)
    ax.axhspan(63.8, 74.2, color=C_GOOD, alpha=0.13, zorder=0)
    ax.bar(range(len(acc)), acc, color=C_EXEC, width=0.72, zorder=3)
    ax.axhline(94.0, color=C_RAW, ls="--", lw=1.3, zorder=2)
    ax.text(len(acc) - 0.4, 95.5, "able-bodied 94%", color="#555", fontsize=8.5, ha="right", va="bottom")
    ax.text(len(acc) - 0.4, 69.0, "pooled mean 63.8–74.2%", color="#2e7d52", fontsize=8.5,
            ha="right", va="center")
    ax.set_xticks(range(len(acc))); ax.set_xticklabels([f"A{i+1}" for i in range(len(acc))], fontsize=8)
    ax.set_ylim(0, 108); ax.set_xlabel("amputee subject (sorted)")
    ax.set_ylabel("per-execution accuracy (%)")
    ax.set_title("Within-amputee accuracy ($n$=12): high between-subject variance", fontsize=10.5)
    _save(fig, "fig7_amputee")


# ---- fig9: DB6 day-separation curve + recalibration-budget curve --------------
def fig_daycurve():
    rows = list(csv.DictReader(open(RES / "db6_daycurve.csv")))
    days = [1, 2, 3, 4, 5]
    def col(pre):
        m = [np.nanmean([float(r[f"{pre}_d{K}"]) for r in rows]) for K in days]
        s = [np.nanstd([float(r[f"{pre}_d{K}"]) for r in rows]) for K in days]
        return np.array(m), np.array(s)
    na, nas = col("noadapt"); rc, rcs = col("recal")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.9))
    # (a) day-separation collapse and recovery
    ax.axhline(na[0], color=C_RAW, ls=":", lw=1.3, zorder=1)
    ax.text(1.35, 101.5, "within-session ceiling", color="#555", fontsize=8.5, va="bottom", ha="left")
    ax.errorbar(days, rc, yerr=rcs, marker="s", ms=7, lw=2.4, color=C_GOOD, capsize=3, zorder=3,
                label="per-don recalibration")
    ax.errorbar(days, na, yerr=nas, marker="o", ms=7, lw=2.4, color=C_HMM, capsize=3, zorder=3,
                label="day-1 decoder, no adaptation")
    ax.set_xticks(days); ax.set_ylim(0, 108); ax.set_xlim(0.7, 5.7)
    ax.set_xlabel("test day (re-donning separation from day-1 calibration)")
    ax.set_ylabel("per-execution accuracy (%)")
    ax.set_title("(a) Cross-session collapse and recovery ($n$=10)", fontsize=10.5)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, bbox_to_anchor=(0.0, 0.02))
    # (b) recalibration-budget curve at the worst (day-5) separation
    b = list(csv.DictReader(open(RES / "db6_recal_budget.csv")))
    g = lambda k: np.array([float(r[k]) for r in b])
    reps = [1, 2, 4, 8]
    bm = np.array([g(f"recal{K}_pe").mean() for K in reps])
    bs = np.array([g(f"recal{K}_pe").std() for K in reps])
    ceil = g("within_pe").mean(); noad = g("noad_pe").mean()
    ax2.axhline(ceil, color=C_RAW, ls=":", lw=1.3, zorder=1)
    ax2.text(0.95, 101.5, "within-session ceiling", color="#555", fontsize=8.5, va="bottom", ha="left")
    ax2.axhline(noad, color=C_HMM, ls="--", lw=1.2, zorder=1)
    ax2.text(7.9, noad + 2.0, "no recalibration", color=C_HMM, fontsize=8.5, va="bottom", ha="right")
    ax2.errorbar(reps, bm, yerr=bs, marker="o", ms=7, lw=2.4, color=C_GOOD, capsize=3, zorder=3)
    ax2.set_xscale("log", base=2); ax2.set_xticks(reps); ax2.set_xticklabels([str(r) for r in reps])
    ax2.set_ylim(0, 108); ax2.set_xlim(0.85, 9.4)
    ax2.set_xlabel("recalibration repetitions per don (day 5)")
    ax2.set_ylabel("per-execution accuracy (%)")
    ax2.set_title("(b) Recalibration-budget curve", fontsize=10.5)
    _save(fig, "fig9_crossday")


# ---- fig8: online streaming (legible compressed timeline + latency hist) ------
def fig_streaming(subj=3):
    d = np.load(RES / f"streaming_arrays_S{subj}.npz", allow_pickle=True)
    y = d["y_true"]; dec = d["decoded"]; grp = d["group"]; lat = d["latency_ms"]
    stride = float(d["stride_ms"]); names = list(d["names"]); nC = len(names)
    # build a legible strip: every movement segment + a short rest pad, concatenated
    # (the long inter-grasp rests are compressed so the decode tracking is visible).
    keep = []
    pad = int(round(384 / stride))                      # ~0.4 s rest context each side
    for g in np.unique(grp):
        idx = np.flatnonzero(grp == g)
        if y[idx][0] == 0:                              # rest segment: keep only a short pad
            keep.append(idx[:pad]); continue
        keep.append(idx)
    sel = np.concatenate(keep)
    # cap to a readable length (first ~14 grip episodes)
    move_segs = [g for g in np.unique(grp) if y[np.flatnonzero(grp == g)][0] != 0]
    if len(move_segs) > 14:
        cut = np.flatnonzero(grp == move_segs[14])[0]
        sel = sel[sel < cut]
    yt, dd = y[sel], dec[sel]
    t = np.arange(len(yt)) * stride / 1000

    from matplotlib.patches import Patch
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 4.6), gridspec_kw=dict(height_ratios=[1.15, 1]))

    # ---- top: temporal-segmentation ribbon (standard in action-segmentation / HAR):
    # ground-truth strip above the decode strip, colour = grip. Agreement = colours
    # align vertically; a decode error shows as a colour break against the strip above.
    dt = stride / 1000.0; nT = len(yt)
    CLASS_COLORS = ["#cfcfcf", "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]

    def runs(sig, val):
        m = sig == val; out = []; i = 0
        while i < nT:
            if m[i]:
                j = i
                while j < nT and m[j]:
                    j += 1
                out.append((i * dt, (j - i) * dt)); i = j
            else:
                i += 1
        return out

    ax1.grid(False)
    for gC in range(nC):
        col = CLASS_COLORS[gC % len(CLASS_COLORS)]
        ax1.broken_barh(runs(yt, gC), (1.10, 0.86), facecolors=col, zorder=2)   # true intent (top)
        ax1.broken_barh(runs(dd, gC), (0.04, 0.86), facecolors=col, zorder=2)   # streamed decode (bottom)
    ax1.set_yticks([0.47, 1.53]); ax1.set_yticklabels(["streamed\ndecode", "true\nintent"], fontsize=9)
    ax1.set_ylim(-0.04, 2.04); ax1.set_xlim(0, nT * dt)
    for s in ("top", "right", "left"):
        ax1.spines[s].set_visible(False)
    ax1.tick_params(length=0)
    ax1.set_xlabel("time (s; inter-grasp rest compressed for legibility)")
    ax1.set_title("Online streaming decode tracks intent, causally (representative user)",
                  fontsize=10.5, pad=24)
    ax1.legend(handles=[Patch(facecolor=CLASS_COLORS[i], label=names[i]) for i in range(nC)],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=nC, fontsize=7.8, frameon=False,
               handlelength=1.1, columnspacing=0.9, handletextpad=0.4)

    # ---- bottom: per-decision latency on a LOG x-axis so the distribution and the
    # 64 ms budget both fit in frame (linear wasted ~90% of the panel as whitespace).
    pooled_path = RES / "streaming_pooled_latency.npz"
    if pooled_path.exists():
        pdz = np.load(pooled_path); lat_h = pdz["pooled_ms"]; n_users = int(pdz["n_users"])
        med_h = float(np.median(pdz["per_subj_median"])); who = f"{n_users} users"
    else:
        lat_h = lat; med_h = float(np.median(lat)); who = "representative user"
    # zoom to where the data actually is; the 64 ms budget (17x larger) is stated,
    # not drawn, so the panel isn't 90% empty headroom.
    hi = 8.0
    ax2.grid(axis="x", alpha=0.3); ax2.grid(axis="y", alpha=0)
    ax2.hist(np.clip(lat_h, 0, hi), bins=np.linspace(0, hi, 41), color=C_EXEC, zorder=3)
    ax2.axvline(med_h, color=C_HMM, ls="--", lw=1.6, zorder=4)
    ax2.annotate(f"median {med_h:.1f} ms", xy=(med_h, ax2.get_ylim()[1]*0.96), xytext=(med_h + 0.4, ax2.get_ylim()[1]*0.96),
                 ha="left", va="top", color=C_HMM, fontsize=8.5)
    ax2.set_xlim(2, hi)
    ax2.set_xlabel("per-decision compute latency (ms)"); ax2.set_ylabel("windows")
    ax2.set_title(f"Per-decision compute, {who}: median {med_h:.1f} ms ($17\\times$ under the 64 ms budget)", fontsize=10)
    _save(fig, "fig8_streaming")


if __name__ == "__main__":
    fig_wideband(); fig_amputee(); fig_daycurve(); fig_streaming()
    print("wrote fig6_wideband, fig7_amputee, fig8_streaming, fig9_crossday (house style)")

#!/usr/bin/env python
"""Generate the within-user paper figures from the FINAL recorded results.

Numbers are hard-coded from the canonical run (db2_reable_hand, 20 users, rig:
RTX 4080 SUPER; SSL+SupCon base, per-user calibration, two-stage CNN + HMM logic).
Writes publication-ready PDF + PNG to paper/figures/.

    python scripts/make_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200, "savefig.dpi": 300,
})

# ---- FINAL within-user results (5-rep headline + calibration curve) ------------
CAL = [1, 2, 3, 4, 5]
RAW_WIN = [54.9, 61.8, 67.6, 70.0, 72.6]       # per-window, no logic (per-subject mean)
HMM_WIN = [56.9, 64.3, 70.3, 72.4, 75.2]       # per-window, + HMM sequence logic
PER_EXEC = [69.8, 82.6, 90.4, 90.5, 94.0]      # per-grasp-execution (clinical)
FALSE_ACT = [1.2, 1.9, 2.0, 1.7, 2.2]          # rest false-activation %

# per-subject SD over the 20 test users (same canonical run) — for 95% CIs
RAW_SD = [9.7, 10.2, 9.0, 8.7, 7.8]
HMM_SD = [11.0, 11.1, 9.2, 8.9, 7.8]
EXEC_SD = [20.7, 15.4, 12.2, 12.1, 11.4]
N_SUBJ = 20
T95 = 2.093                                    # t_{0.975, df=19}
def _ci(sd):                                   # 95% CI half-width from per-subject SD
    return [T95 * s / np.sqrt(N_SUBJ) for s in sd]

# colour-blind-safe palette (Wong 2011): blue / vermillion / orange / grey
C_EXEC, C_HMM, C_RAW, C_WEAK = "#0072B2", "#D55E00", "#888888", "#E69F00"

GRIPS = ["Rest", "Power", "Pinch", "Tripod", "Lateral", "Point", "Hook"]
RECALL = [98, 95, 95, 80, 90, 100, 100]        # per-execution recall, 5 reps

# hysteresis-decoded per-window confusion (row-normalised %, 5 reps)
CM = np.array([
    [94, 1, 1, 1, 1, 1, 1],
    [17, 70, 4, 1, 2, 0, 6],
    [22, 0, 71, 3, 4, 0, 0],
    [25, 4, 2, 62, 4, 1, 1],
    [25, 2, 3, 1, 67, 0, 2],
    [9, 4, 0, 1, 3, 79, 4],
    [21, 8, 3, 1, 2, 2, 64],
], dtype=float)

# two-stage operating points (gate sweep, 5 reps): move coverage, grasp acc, false-act
GATE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GRASP_ACC = [82.9, 84.1, 84.7, 85.2, 85.7, 86.0, 86.7]
GATE_FALSE = [17.7, 11.2, 7.7, 5.8, 4.4, 3.2, 2.2]


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _band(ax, x, y, sd, color):
    """shaded 95% CI band around a curve."""
    y = np.asarray(y); h = np.asarray(_ci(sd))
    ax.fill_between(x, y - h, y + h, color=color, alpha=0.15, linewidth=0)


def fig_calibration_curve():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    _band(ax, CAL, PER_EXEC, EXEC_SD, C_EXEC)
    _band(ax, CAL, HMM_WIN, HMM_SD, C_HMM)
    _band(ax, CAL, RAW_WIN, RAW_SD, C_RAW)
    ax.plot(CAL, PER_EXEC, "o-", lw=2.2, color=C_EXEC, label="Per-execution (clinical)")
    ax.plot(CAL, HMM_WIN, "s-", lw=2, color=C_HMM, label="Per-window + sequence logic")
    ax.plot(CAL, RAW_WIN, "^--", lw=1.6, color=C_RAW, label="Per-window (raw)")
    ax.axhline(90, color="#c33", ls=":", lw=1, alpha=0.7)
    ax.text(1.05, 90.6, "90% line", color="#c33", fontsize=9)
    ax.set_xlabel("Calibration repetitions per grip")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_xticks(CAL); ax.set_ylim(50, 100)
    ax.legend(loc="lower right", fontsize=9, frameon=False,
              title="shaded = 95% CI (n=20)", title_fontsize=8)
    ax.set_title("Within-user calibration curve", fontsize=11)
    _save(fig, "fig1_calibration_curve")


def fig_per_grip():
    order = np.argsort(RECALL)
    g = [GRIPS[i] for i in order]; r = [RECALL[i] for i in order]
    # colour-blind-safe: orange marks the weakest grip(s) (<90%), blue otherwise
    colors = [C_WEAK if v < 90 else C_EXEC for v in r]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.barh(g, r, color=colors)
    for i, v in enumerate(r):
        ax.text(v - 4, i, f"{v}%", va="center", ha="right", color="white", fontsize=9)
    ax.set_xlim(0, 100); ax.set_xlabel("Per-execution recall (%)")
    ax.set_title("Per-grip recall (5 calibration reps, n=20)", fontsize=11)
    _save(fig, "fig2_per_grip_recall")


def fig_confusion():
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(CM, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(GRIPS, rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=9)
    ax.set_yticklabels(GRIPS, fontsize=9)
    ax.tick_params(axis="both", length=0)
    for i in range(7):
        for j in range(7):
            v = CM[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > 45 else "#222",
                        fontsize=11, fontweight="bold" if i == j else "normal")
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title("Per-window confusion (row-normalised %)", fontsize=11)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("%", fontsize=10)
    _save(fig, "fig3_confusion_matrix")


def fig_operating_point():
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(GATE_FALSE, GRASP_ACC, "o-", lw=2, color="#36c")
    for x, y, g in zip(GATE_FALSE, GRASP_ACC, GATE):
        ax.annotate(f"{g}", (x, y), fontsize=8, xytext=(3, -8),
                    textcoords="offset points", color="#666")
    ax.axvline(2, color="#c33", ls=":", lw=1)
    ax.text(2.2, 82.4, "≈2% clinical bar", color="#c33", fontsize=9, rotation=90, va="bottom")
    ax.set_xlabel("Rest false-activation (%)")
    ax.set_ylabel("Grasp accuracy on acted windows (%)")
    ax.set_title("Operating points (gate threshold labelled)", fontsize=11)
    _save(fig, "fig4_operating_point")


def fig_logic_gain():
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    x = np.arange(len(CAL)); w = 0.38
    ekw = dict(capsize=2.5, ecolor="#444", elinewidth=0.9)
    ax.bar(x - w/2, RAW_WIN, w, yerr=_ci(RAW_SD), label="Raw evidence",
           color=C_RAW, error_kw=ekw)
    ax.bar(x + w/2, HMM_WIN, w, yerr=_ci(HMM_SD), label="+ sequence logic (HMM)",
           color=C_HMM, error_kw=ekw)
    for i in range(len(CAL)):
        ax.text(x[i] + w/2, HMM_WIN[i] + _ci(HMM_SD)[i] + 0.8,
                f"+{HMM_WIN[i]-RAW_WIN[i]:.1f}", ha="center", fontsize=8, color=C_HMM)
    ax.set_xticks(x); ax.set_xticklabels(CAL)
    ax.set_xlabel("Calibration reps"); ax.set_ylabel("Per-window balanced acc (%)")
    ax.set_ylim(50, 85)
    ax.legend(fontsize=9, frameon=False, loc="upper left",
              title="error bars = 95% CI (n=20)", title_fontsize=8)
    ax.set_title("Sequence-logic gain over raw argmax (per-subject mean)", fontsize=11)
    _save(fig, "fig5_logic_gain")


if __name__ == "__main__":
    print(f"writing figures -> {OUT}")
    fig_calibration_curve()
    fig_per_grip()
    fig_confusion()
    fig_operating_point()
    fig_logic_gain()
    print("done.")

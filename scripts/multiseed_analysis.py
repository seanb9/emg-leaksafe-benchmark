#!/usr/bin/env python
"""Multi-seed robustness of the wideband per-window gain.
For each seed, pair wideband vs narrowband per-window CNN+HMM balanced accuracy at
5 calibration reps across the 20 test subjects; report mean gain + paired Wilcoxon.
seed 1337 uses the original exp_*_full.csv; seeds 7/2024 use the ms_* re-runs.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parents[1] / "results"
SEEDS = {
    1337: ("exp_wideband_full.csv", "exp_cur_full.csv"),
    7:    ("ms_exp_wb_s7.csv",       "ms_exp_nb_s7.csv"),
    2024: ("ms_exp_wb_s2024.csv",   "ms_exp_nb_s2024.csv"),
}


def load(p):
    d = {}
    for r in csv.DictReader(open(RES / p)):
        if int(r["cal_reps"]) != 5:
            continue
        d[int(r["subject"])] = float(r["hmm_bal_acc"]) * 100
    return d


print(f"{'seed':>6} {'narrow':>8} {'wide':>8} {'gain':>7} {'p(Wilcoxon)':>12} {'better':>8}")
gains = []
for s, (wbf, nbf) in SEEDS.items():
    if not (RES / wbf).exists() and (RES / nbf).exists():
        print(f"{s:>6}  (missing csv)"); continue
    try:
        wb, nb = load(wbf), load(nbf)
    except FileNotFoundError as e:
        print(f"{s:>6}  missing: {e.filename}"); continue
    subs = sorted(set(wb) & set(nb))
    w = np.array([wb[x] for x in subs]); n = np.array([nb[x] for x in subs])
    diff = w - n
    try:
        _, p = wilcoxon(w, n)
    except ValueError:
        p = float("nan")
    gains.append(diff.mean())
    print(f"{s:>6} {n.mean():8.1f} {w.mean():8.1f} {diff.mean():+7.2f} {p:12.3f} {f'{(diff>0).sum()}/{len(subs)}':>8}")

if gains:
    g = np.array(gains)
    print(f"\nper-window wideband gain across {len(g)} seeds: mean {g.mean():+.2f}, "
          f"range [{g.min():+.2f}, {g.max():+.2f}], all positive: {bool((g>0).all())}")

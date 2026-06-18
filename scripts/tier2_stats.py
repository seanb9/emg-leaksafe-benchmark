#!/usr/bin/env python
"""Tier-2 reviewer-response statistics, all from already-saved per-subject CSVs:
 (1) multi-seed headline within-user (3 seeds)
 (2) effect sizes (Cohen's d, rank-biserial r) + 95% CIs on the primary paired gains
 (3) subject-level win-counts + practical-significance for the decoder gain
 (4) bootstrap 95% CIs on the DB6 collapse-and-recovery
RUN: python3 scripts/tier2_stats.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, t as tdist

RES = Path(__file__).resolve().parents[1] / "results"
np.random.seed(1337)


def load(p, cal=5):
    return [r for r in csv.DictReader(open(RES / p)) if int(r.get("cal_reps", 5)) == cal]


def col(rows, c):
    return np.array([float(r[c]) for r in rows]) * 100


def ci95_mean(x):
    n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    h = tdist.ppf(0.975, n - 1) * se
    return m, m - h, m + h


def paired_stats(diff, label):
    n = len(diff); m = diff.mean()
    _, lo, hi = ci95_mean(diff)
    d = m / diff.std(ddof=1)                       # Cohen's d (paired)
    try:
        W, p = wilcoxon(diff)
    except Exception:
        p = float("nan")
    pos = int((diff > 0).sum()); n_nonzero = int((diff != 0).sum())
    # rank-biserial = (sum positive ranks - sum negative ranks)/total
    ranks = np.argsort(np.argsort(np.abs(diff))) + 1
    rpos = ranks[diff > 0].sum(); rneg = ranks[diff < 0].sum()
    rb = (rpos - rneg) / ranks.sum()
    print(f"  {label}: +{m:.2f} pts  95%CI [{lo:.2f}, {hi:.2f}]  Cohen d={d:.2f}  "
          f"rank-biserial r={rb:.2f}  Wilcoxon p={p:.4f}  wins {pos}/{n}")
    return m, lo, hi, d, rb, p, pos, n


SEEDS = {1337: ("exp_cur_full.csv", "exp_wideband_full.csv"),
         7: ("ms_exp_nb_s7.csv", "ms_exp_wb_s7.csv"),
         2024: ("ms_exp_nb_s2024.csv", "ms_exp_wb_s2024.csv")}

print("=" * 70)
print("(1) MULTI-SEED HEADLINE within-user (narrowband), 3 seeds")
pw_raw, pw_logic, pe = [], [], []
for s, (nb, _) in SEEDS.items():
    r = load(nb)
    a, b, c = col(r, "bal_acc").mean(), col(r, "hmm_bal_acc").mean(), col(r, "seg_bal_acc").mean()
    pw_raw.append(a); pw_logic.append(b); pe.append(c)
    print(f"  seed {s:>4}: per-window raw {a:.1f} | +logic {b:.1f} | per-exec {c:.1f}")
for name, v in [("per-window raw", pw_raw), ("per-window +logic", pw_logic), ("per-execution", pe)]:
    v = np.array(v); print(f"  -> {name:18}: {v.mean():.1f} +/- {v.std(ddof=1):.1f} across seeds (range {v.min():.1f}-{v.max():.1f})")

print("=" * 70)
print("(2,3) DECODER GAIN (per-window +logic over raw argmax), pooled over 3 seeds")
gains = []
for s, (nb, _) in SEEDS.items():
    r = load(nb)
    diff = col(r, "hmm_bal_acc") - col(r, "bal_acc")
    paired_stats(diff, f"seed {s}")
    gains.append(diff)
allg = np.concatenate(gains)
paired_stats(allg, "ALL 3 seeds pooled (n=60)")
med = np.median(allg)
print(f"  practical significance: median per-subject gain {med:.2f} pts, "
      f"IQR [{np.percentile(allg,25):.2f}, {np.percentile(allg,75):.2f}], "
      f"positive in {int((allg>0).sum())}/{len(allg)} (subject,seed) cases")

print("=" * 70)
print("(2) WIDEBAND GAIN (per-window +logic, wider band over narrow), per seed + effect size")
wbg = []
for s, (nb, wb) in SEEDS.items():
    rn, rw = load(nb), load(wb)
    subs = sorted(set(int(r["subject"]) for r in rn) & set(int(r["subject"]) for r in rw))
    dn = {int(r["subject"]): float(r["hmm_bal_acc"]) * 100 for r in rn}
    dw = {int(r["subject"]): float(r["hmm_bal_acc"]) * 100 for r in rw}
    diff = np.array([dw[x] - dn[x] for x in subs])
    paired_stats(diff, f"seed {s} (n={len(subs)})")
    wbg.append(diff.mean())
wbg = np.array(wbg); print(f"  -> across seeds: +{wbg.mean():.2f} +/- {wbg.std(ddof=1):.2f} pts (range {wbg.min():.1f}-{wbg.max():.1f})")

print("=" * 70)
print("(4) BOOTSTRAP 95% CIs on DB6 collapse-and-recovery (resample subjects, 10000x)")
dc = list(csv.DictReader(open(RES / "db6_daycurve.csv")))
g = lambda k: np.array([float(r[k]) for r in dc])


def boot(x, n=10000):
    idx = np.random.randint(0, len(x), (n, len(x)))
    bs = x[idx].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


for lab, key in [("day-1 within-session", "noadapt_d1"), ("day-5 no adaptation", "noadapt_d5"),
                 ("day-5 + recalibration", "recal_d5")]:
    m, lo, hi = boot(g(key)); print(f"  {lab:24}: {m:.1f} [{lo:.1f}, {hi:.1f}]")
rb = list(csv.DictReader(open(RES / "db6_recal_budget.csv")))
gb = lambda k: np.array([float(r[k]) for r in rb])
print("  recalibration-budget curve (day-5):")
for K in (1, 2, 4, 8):
    m, lo, hi = boot(gb(f"recal{K}_pe")); print(f"    {K} rep(s): {m:.1f} [{lo:.1f}, {hi:.1f}]")
print("=" * 70)

#!/usr/bin/env python
"""Decide whether our in-harness EMGBench baseline is faithful, by comparing it to
EMGBench's actual repo on the SAME LOSO folds. No tuning-to-target — the repo is
ground truth; we test agreement.

Inputs (both per leftout_subject x proportion, Macro_Acc):
  --repo     CSV from emgbench_repo_crosscheck.py (their actual repo)
  --inhouse  CSV from our in-harness EMGBench baseline LOSO runner

Verdict per proportion:
  * |mean(repo) - mean(inhouse)| within noise (<= --tol pp, default 3.0) AND
    paired differences not systematically biased -> FAITHFUL: use in-harness as
    the baseline for the full sweep.
  * otherwise -> NOT FAITHFUL: report the gap; THEIR repo numbers are the baseline.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--inhouse", required=True)
    ap.add_argument("--tol", type=float, default=3.0, help="agreement tolerance in pp")
    args = ap.parse_args()

    repo = pd.read_csv(args.repo)
    ours = pd.read_csv(args.inhouse)
    # normalise column names: expect leftout_subject, proportion, macro_acc (0..1)
    key = ["leftout_subject", "proportion"]
    m = repo.merge(ours, on=key, suffixes=("_repo", "_ours"))
    if m.empty:
        print("No overlapping (subject, proportion) folds between the two CSVs.")
        return 1

    print(f"{'prop':>5} {'n':>3} {'repo%':>7} {'ours%':>7} {'Δmean':>7} {'maxΔ':>6}  verdict")
    overall_faithful = True
    for prop, g in m.groupby("proportion"):
        r = g["macro_acc_repo"].to_numpy() * 100
        o = g["macro_acc_ours"].to_numpy() * 100
        dmean = float(np.mean(o) - np.mean(r))
        dmax = float(np.max(np.abs(o - r)))
        faithful = abs(dmean) <= args.tol and dmax <= 2 * args.tol
        overall_faithful &= faithful
        print(f"{prop:>5} {len(g):>3} {np.mean(r):>6.1f} {np.mean(o):>6.1f} "
              f"{dmean:>+6.1f} {dmax:>6.1f}  {'FAITHFUL' if faithful else 'GAP — report'}")

    print()
    if overall_faithful:
        print(f"VERDICT: in-harness baseline agrees with EMGBench repo within {args.tol}pp.")
        print("        -> use the in-harness baseline for the full sweep.")
    else:
        print("VERDICT: in-harness baseline does NOT match the repo within tolerance.")
        print("        -> EMGBench REPO numbers are the baseline; report the gap; fix re-impl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

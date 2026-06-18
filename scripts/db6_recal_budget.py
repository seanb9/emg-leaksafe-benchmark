#!/usr/bin/env python
"""DB6 per-don recalibration BUDGET curve (full cohort, n=10) + label-free renorm
at n=10 -- closes the two open reviewer items on the cross-session result.

For every test subject we measure, at the WORST separation (day 5 re-donning):
  - within-session day-1 accuracy (the ceiling)
  - day-5 no-adaptation (the collapse)
  - day-5 + unsupervised renormalisation (label-free fallback)  <- now n=10
  - day-5 + full recalibration on K reps, K in {1,2,4,8}        <- budget curve
so the deployment recipe gets a price tag: accuracy recovered vs reps the user
must sit through at each don.

Leak-safe two-fold base (subjects 1-5 <-> 6-10); reuses the saved base for the
6-10 base (test 1-5) fold and trains the complementary base once. Reuses the
db6_s*_d1d5.npz window caches, so no re-extraction.

RUN: python -u scripts/db6_recal_budget.py
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from emgv2.data.splits import cap_rest_indices
from emgv2.models import build_v2net
from emgv2.models.presets import preset
from emgv2.train import engine
from emgv2.eval.metrics import balanced_accuracy, majority_vote_by_group
from emgv2.utils.device import pick_device, machine_label
from db6_crosssession import load_db6_subject, DB6_CLASSES

REP_BUDGETS = [1, 2, 4, 8]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/sean/Desktop/ReAble Matlab/reable_emg_classifier/data/ninapro_db6")
    ap.add_argument("--base-epochs", type=int, default=20); ap.add_argument("--cal-epochs", type=int, default=25)
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default="results/db6_recal_budget.csv")
    args = ap.parse_args()
    device = pick_device(args.device); mlabel = machine_label(device)
    cache = Path(__file__).resolve().parents[1] / "data_cache" / "db6"
    res = Path(__file__).resolve().parents[1] / "results"
    nC = len(DB6_CLASSES); rest_idx = 0; mc = preset("S", supcon=False, adversarial=False)
    print(f"=== DB6 RECAL-BUDGET + n=10 RENORM === {mlabel} | classes={nC}")

    def znorm(X, fit): m = X[fit].mean((0, 1)); sd = X[fit].std((0, 1)) + 1e-6; return ((X - m) / sd).astype(np.float32)

    raw = {s: load_db6_subject(args.root, s, cache) for s in range(1, 11)}
    c_in = raw[1]["X"].shape[2]
    # (test, base, saved-base-file-or-None)
    folds = [([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], res / "base_db6.pt"),
             ([6, 7, 8, 9, 10], [1, 2, 3, 4, 5], res / "base_db6_foldB.pt")]

    def train_base(base_subs, path):
        if path.exists():
            print(f"  reusing base {path.name}"); return torch.load(path, map_location=device)
        Xtr, ytr = [], []
        for s in base_subs:
            d = raw[s]; Xtr.append(znorm(d["X"], np.arange(len(d["y"])))); ytr.append(d["y"])
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
        keep = cap_rest_indices(ytr, rest_class_idx=rest_idx, mult=1.5, seed=1337); Xtr, ytr = Xtr[keep], ytr[keep]
        base = build_v2net(c_in, nC, mc, n_subjects=0).to(device); nval = max(1, len(Xtr) // 10)
        print(f"  training base on {base_subs}: {len(Xtr)} win, {args.base_epochs} ep")
        t0 = time.time()
        engine.train_supervised(base, Xtr[:-nval], ytr[:-nval], Xtr[-nval:], ytr[-nval:], device,
                                n_classes=nC, epochs=args.base_epochs, aug_kw=dict(channel_shift=2),
                                seed=1337, log=lambda *_: None)
        st = {k: v.clone() for k, v in base.state_dict().items()}; torch.save(st, path)
        print(f"  base trained in {time.time()-t0:.0f}s (saved {path.name})")
        return st

    rows = []
    for test_subs, base_subs, bpath in folds:
        bstate = train_base(base_subs, bpath)

        def fit(Xn, ci, y):
            m = build_v2net(c_in, nC, mc, n_subjects=0).to(device); m.load_state_dict(bstate, strict=False)
            return engine.fewshot_adapt(m, Xn[ci], y[ci], device, n_classes=nC, epochs=args.cal_epochs,
                                        lr=3e-4, head_only=False, seed=1337)

        for s in test_subs:
            d = raw[s]; X, y, grp, rep, day = d["X"], d["y"], d["group"], d["rep"], d["day"]
            d1 = day == 1; d5 = day == 5
            d1cal = d1 & np.isin(rep, [1, 2, 3, 4, 5, 6, 7, 8]); d1te = d1 & np.isin(rep, [9, 10, 11, 12])
            d5te = d5 & np.isin(rep, [9, 10, 11, 12])
            if min(d1cal.sum(), d1te.sum(), d5te.sum()) == 0:
                print(f"  S{s}: missing data, skip"); continue

            def cap(mask):
                ci = np.flatnonzero(mask); return ci[cap_rest_indices(y[ci], rest_class_idx=rest_idx, mult=1.5, seed=1337)]

            def pe(model, Xn, mask):
                idx = np.flatnonzero(mask); idx = idx[np.argsort(grp[idx], kind="stable")]
                preds, _ = engine.predict(model, Xn[idx], device)
                st, sp = majority_vote_by_group(y[idx], preds, grp[idx], nC)
                return balanced_accuracy(st, sp, nC) * 100

            Xn_d1 = znorm(X, cap(d1cal)); Xn_d5 = znorm(X, np.flatnonzero(d5))
            m_d1 = fit(Xn_d1, cap(d1cal), y)
            rec = dict(subject=s)
            rec["within_pe"] = pe(m_d1, Xn_d1, d1te)            # ceiling
            rec["noad_pe"]   = pe(m_d1, Xn_d1, d5te)            # collapse
            rec["unsup_pe"]  = pe(m_d1, Xn_d5, d5te)            # label-free renorm (now n=10)
            for K in REP_BUDGETS:                               # budget curve: K recal reps on day 5
                ci = cap(d5 & np.isin(rep, list(range(1, K + 1))))
                mk = fit(Xn_d5, ci, y)
                rec[f"recal{K}_pe"] = pe(mk, Xn_d5, d5te)
            rows.append(rec)
            print(f"  S{s}: within {rec['within_pe']:.0f} | d5 no-adapt {rec['noad_pe']:.0f} | unsup {rec['unsup_pe']:.0f}"
                  + " | recal " + " ".join(f"{K}r={rec[f'recal{K}_pe']:.0f}" for K in REP_BUDGETS))

    import csv
    out = Path(args.out); out = out if out.is_absolute() else res.parent / args.out
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    a = lambda k: np.array([r[k] for r in rows])
    print(f"\n=== RECAL-BUDGET CURVE (n={len(rows)}), day-5 per-exec balanced accuracy ===")
    print(f"  within-session ceiling : {a('within_pe').mean():.1f} +/- {a('within_pe').std():.1f}")
    print(f"  day-5 no adaptation    : {a('noad_pe').mean():.1f} +/- {a('noad_pe').std():.1f}")
    print(f"  + unsup renorm (n=10)  : {a('unsup_pe').mean():.1f} +/- {a('unsup_pe').std():.1f}  ({a('unsup_pe').mean()-a('noad_pe').mean():+.1f})")
    for K in REP_BUDGETS:
        v = a(f"recal{K}_pe"); print(f"  + recal {K:>2} rep(s)      : {v.mean():.1f} +/- {v.std():.1f}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

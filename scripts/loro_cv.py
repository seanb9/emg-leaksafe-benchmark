#!/usr/bin/env python
"""Leave-one-repetition-out cross-validation (robustness of the headline number).

The headline per-execution accuracy rests on a single held-out repetition (cal
reps 1-5, test rep 6), so each subject contributes one execution per class. This
script instead holds out EACH of the 6 repetitions in turn (calibrating on the
other 5) and reports the per-execution and per-window balanced accuracy as a
mean +/- SD over the 6 folds, giving the metric a real distribution rather than
one repetition's outcome. It relaxes the strict earlier->later extrapolation to
interpolation; we report it purely as a robustness check on the headline.

RUN: python -u scripts/loro_cv.py configs/db2_reable_hand.yaml --reuse-base --cal-epochs 20
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from emgv2.config import load_config, seed_everything
from emgv2.data import ninapro_db2 as db2
from emgv2.data.splits import per_subject_zscore_fit_apply, cap_rest_indices
from emgv2.utils.device import pick_device, machine_label
from emgv2.eval.sequence import learn_transition_structure, build_transition, causal_forward_decode
from emgv2.eval.metrics import balanced_accuracy, majority_vote_by_group
from run_within_user import load_subject, in_channels
from streaming_demo import calibrate, batch_emissions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--reuse-base", action="store_true")
    ap.add_argument("--cal-epochs", type=int, default=20)
    ap.add_argument("--subjects", default=",".join(str(i) for i in range(1, 21)))
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default="results/loro_cv.csv")
    args = ap.parse_args()

    cfg = load_config(args.config); seed_everything(int(cfg.seed))
    device = pick_device(args.device); mlabel = machine_label(device)
    n_classes = len(cfg.dataset.classes); rest_idx = int(cfg.get_path("control_loop.rest_class_index", 0))
    gate0 = float(cfg.get_path("control_loop.gate_threshold", 0.5))
    cal_lr = float(cfg.get_path("within_user.cal_lr", 3e-4))
    cal_aug = dict(channel_shift=0, mag_warp_std=0.2, noise_std=0.05, time_mask_frac=0.1,
                   n_time_masks=1, chan_scale=(0.9, 1.1)) if bool(cfg.get_path("within_user.cal_augment", False)) else None
    base_state = torch.load(Path(__file__).resolve().parents[1] / "results" / f"base_{cfg.name}.pt", map_location=device)
    subjects = [int(x) for x in args.subjects.split(",") if x.strip()]

    pre = cfg.get_path("within_user.pretrain_subjects") or list(range(21, 41))
    seqs = []
    for s in pre[:8]:
        wsp = load_subject(cfg, s)
        if wsp is None or len(wsp) == 0: continue
        for g in np.unique(wsp.group): seqs.append(wsp.y[wsp.group == g])
    O, pi = learn_transition_structure(seqs, n_classes); A = build_transition(O, float(cfg.get_path("within_user.hmm_hold", 0.97)))

    print(f"=== LEAVE-ONE-REP-OUT CV === {mlabel} | {cfg.name} | {len(subjects)} users x up to 6 folds")
    rows = []
    t0 = time.time()
    for s in subjects:
        ws = load_subject(cfg, s)
        if ws is None or len(ws) == 0: print(f"  S{s}: no data"); continue
        reps = sorted(set(int(r) for r in np.unique(ws.rep) if r > 0))
        fold_pw_raw, fold_pw_hmm, fold_pe = [], [], []
        for t_rep in reps:
            test_idx = np.flatnonzero(ws.rep == t_rep)
            cal_idx = np.flatnonzero((ws.rep != t_rep) & (ws.rep > 0))
            if len(test_idx) == 0 or len(cal_idx) == 0: continue
            rc = cfg.get_path("balance.rest_cap_mult")
            if rc is not None:
                cal_idx = cal_idx[cap_rest_indices(ws.y[cal_idx], rest_class_idx=rest_idx, mult=float(rc), seed=int(cfg.seed))]
            mean, std = per_subject_zscore_fit_apply(ws.X, cal_idx); Xn = ((ws.X - mean)/std).astype(np.float32)
            cal = calibrate(cfg, base_state, in_channels(cfg), n_classes, rest_idx, Xn, cal_idx, ws.y,
                            device, args.cal_epochs, cal_lr, int(cfg.seed), cal_aug)
            order = np.argsort(ws.group[test_idx], kind="stable"); sidx = test_idx[order]
            yte, gte = ws.y[sidx], ws.group[sidx]
            pA, gp, emis = batch_emissions(cal, Xn, sidx, device, rest_idx, n_classes)
            preds = np.where(pA >= gate0, gp, rest_idx)
            preds_hmm = causal_forward_decode(emis, gte, A, pi)
            seg_t, seg_p = majority_vote_by_group(yte, preds, gte, n_classes)
            fold_pw_raw.append(balanced_accuracy(yte, preds, n_classes))
            fold_pw_hmm.append(balanced_accuracy(yte, preds_hmm, n_classes))
            fold_pe.append(balanced_accuracy(seg_t, seg_p, n_classes))
        if not fold_pe: continue
        r = dict(subject=s, n_folds=len(fold_pe),
                 pw_raw=float(np.mean(fold_pw_raw)), pw_hmm=float(np.mean(fold_pw_hmm)),
                 pe=float(np.mean(fold_pe)), pe_sd=float(np.std(fold_pe)))
        rows.append(r)
        print(f"  S{s}: {len(fold_pe)} folds | per-win raw {r['pw_raw']*100:.0f}% +logic {r['pw_hmm']*100:.0f}% | "
              f"per-exec {r['pe']*100:.0f}% (within-subj fold SD {r['pe_sd']*100:.0f})")

    pw_raw = np.array([r['pw_raw'] for r in rows])*100
    pw_hmm = np.array([r['pw_hmm'] for r in rows])*100
    pe = np.array([r['pe'] for r in rows])*100
    print(f"\n=== LORO-CV AGGREGATE ({len(rows)} subjects, {time.time()-t0:.0f}s) ===")
    print(f"  per-window  (raw)   : {pw_raw.mean():.1f} +/- {pw_raw.std():.1f}")
    print(f"  per-window  (+logic): {pw_hmm.mean():.1f} +/- {pw_hmm.std():.1f}")
    print(f"  per-execution       : {pe.mean():.1f} +/- {pe.std():.1f}   <- distribution over 6 folds, not 1 rep")
    print(f"  (compare headline cal1-5/test6: per-window +logic 75.2, per-exec gate-free ~93)")
    import csv
    out = Path(args.out); out = out if out.is_absolute() else Path(__file__).resolve().parents[1]/out
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

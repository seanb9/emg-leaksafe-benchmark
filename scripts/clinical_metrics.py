#!/usr/bin/env python
"""Pseudo-online clinical control metrics (Hargrove/Kuiken-style) for the deployed
within-user decoder, computed from the causal streaming decode on DB2.

For each held-out movement segment (one grasp attempt) we report the metrics a
rehabilitation reviewer expects from a controller, not just classification accuracy:
  - completion rate : fraction of grasp attempts the controller settles on correctly
                      (majority of the segment's second half is the target grasp)
  - selection time  : onset-to-first-correct-output latency (ms), over completed attempts
  - rest stability  : fraction of rest time held correctly (1 - false-activation)
These are computed on the same causal hysteresis decode used for the paper's numbers.

RUN: python -u scripts/clinical_metrics.py configs/db2_reable_hand.yaml --reuse-base --cal-epochs 25
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from emgv2.config import load_config, seed_everything
from emgv2.data import ninapro_db2 as db2
from emgv2.data.splits import within_user_split, per_subject_zscore_fit_apply, cap_rest_indices
from emgv2.utils.device import pick_device, machine_label
from emgv2.eval.control_loop import hysteresis_decode
from run_within_user import load_subject, in_channels
from streaming_demo import calibrate, batch_emissions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--reuse-base", action="store_true")
    ap.add_argument("--cal-epochs", type=int, default=25)
    ap.add_argument("--subjects", default=",".join(str(i) for i in range(1, 21)))
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default="results/clinical_metrics.csv")
    args = ap.parse_args()
    cfg = load_config(args.config); seed_everything(int(cfg.seed))
    device = pick_device(args.device); mlabel = machine_label(device)
    nC = len(cfg.dataset.classes); rest_idx = int(cfg.get_path("control_loop.rest_class_index", 0))
    stride_ms = float(cfg.get_path("window.stride_ms", 64))
    cal_lr = float(cfg.get_path("within_user.cal_lr", 3e-4))
    cal_aug = dict(channel_shift=0, mag_warp_std=0.2, noise_std=0.05, time_mask_frac=0.1, n_time_masks=1,
                   chan_scale=(0.9,1.1)) if bool(cfg.get_path("within_user.cal_augment", False)) else None
    dec = cfg.get_path("decoder", {}) or {}
    base_state = torch.load(Path(__file__).resolve().parents[1]/"results"/f"base_{cfg.name}.pt", map_location=device)
    subjects = [int(x) for x in args.subjects.split(",") if x.strip()]
    print(f"=== PSEUDO-ONLINE CLINICAL METRICS === {mlabel} | {cfg.name} | {len(subjects)} users")

    rows = []
    for s in subjects:
        ws = load_subject(cfg, s)
        if ws is None or len(ws)==0: continue
        cal_idx, test_idx = within_user_split(ws.y, ws.group, ws.rep, cal_reps=[1,2,3,4,5],
                                              test_reps=cfg.get_path("within_user.test_reps",[6]), rest_class_idx=rest_idx)
        rc = cfg.get_path("balance.rest_cap_mult")
        if rc is not None: cal_idx = cal_idx[cap_rest_indices(ws.y[cal_idx], rest_class_idx=rest_idx, mult=float(rc), seed=int(cfg.seed))]
        mean, std = per_subject_zscore_fit_apply(ws.X, cal_idx); Xn = ((ws.X-mean)/std).astype(np.float32)
        cal = calibrate(cfg, base_state, in_channels(cfg), nC, rest_idx, Xn, cal_idx, ws.y, device, args.cal_epochs, cal_lr, int(cfg.seed), cal_aug)
        order = np.argsort(ws.group[test_idx], kind="stable"); sidx = test_idx[order]
        yte, gte = ws.y[sidx], ws.group[sidx]
        pA, gp, _ = batch_emissions(cal, Xn, sidx, device, rest_idx, nC)
        dec_d = hysteresis_decode(pA, gp, gte, rest_idx, enter_gate=float(dec.get("enter_gate",0.6)),
                                  exit_gate=float(dec.get("exit_gate",0.35)), on_frames=int(dec.get("on_frames",3)),
                                  off_frames=int(dec.get("off_frames",4)), switch_frames=int(dec.get("switch_frames",3)))
        comp, seltimes, rest_held, rest_tot = 0, [], 0, 0
        move_segs = 0
        for gid in np.unique(gte):
            m = gte==gid; target = int(yte[m][0]); d = dec_d[m]
            if target == rest_idx:
                rest_tot += m.sum(); rest_held += int((d==rest_idx).sum()); continue
            move_segs += 1
            half = len(d)//2
            settled = (np.bincount(d[half:], minlength=nC).argmax() == target)   # completion: 2nd-half majority correct
            if settled:
                comp += 1
                fc = np.flatnonzero(d==target)
                if fc.size: seltimes.append(fc[0]*stride_ms)
        if move_segs==0: continue
        rows.append(dict(subject=s, completion_rate=comp/move_segs*100,
                         median_selection_ms=float(np.median(seltimes)) if seltimes else float("nan"),
                         rest_stability=rest_held/max(1,rest_tot)*100, n_move=move_segs))
        print(f"  S{s}: completion {comp/move_segs*100:.0f}% | sel-time {np.median(seltimes) if seltimes else float('nan'):.0f}ms | rest-stability {rest_held/max(1,rest_tot)*100:.0f}%")

    import csv
    a=lambda k:np.array([r[k] for r in rows if not np.isnan(r[k])])
    print(f"\n=== CLINICAL CONTROL METRICS ({len(rows)} subjects) ===")
    print(f"  completion rate     : {a('completion_rate').mean():.1f} +/- {a('completion_rate').std():.1f} %")
    print(f"  selection time      : {a('median_selection_ms').mean():.0f} +/- {a('median_selection_ms').std():.0f} ms (median per subject)")
    print(f"  rest stability      : {a('rest_stability').mean():.1f} +/- {a('rest_stability').std():.1f} %  (= 100 - false-activation)")
    out=Path(args.out); out=out if out.is_absolute() else Path(__file__).resolve().parents[1]/out
    with open(out,"w",newline="") as fh: w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

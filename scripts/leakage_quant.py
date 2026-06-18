#!/usr/bin/env python
"""Quantify the accuracy INFLATION of a leaky window-level split vs our leak-safe
segment split -- the paper's central methodological claim, turned into a number.

Identical pipeline, identical calibration FRACTION, same base model; the ONLY
difference is the split rule:
  * leak-safe  : calibrate on reps 1-5 (contiguous segments), test on held-out rep 6
  * leaky      : the same #windows assigned at RANDOM across all windows, so
                 75%-overlap neighbours straddle the split (the common bad practice)
Per-window balanced accuracy (raw CNN evidence) is reported for both; the gap is the
leakage inflation.

RUN: python3 scripts/leakage_quant.py configs/db2_reable_hand.yaml --reuse-base --cal-epochs 25
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scipy.stats import wilcoxon
from emgv2.config import load_config, seed_everything
from emgv2.data.splits import within_user_split, per_subject_zscore_fit_apply, cap_rest_indices
from emgv2.utils.device import pick_device, machine_label
from emgv2.eval.metrics import balanced_accuracy
from run_within_user import load_subject, in_channels
from streaming_demo import calibrate, batch_emissions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--reuse-base", action="store_true")
    ap.add_argument("--cal-epochs", type=int, default=25)
    ap.add_argument("--subjects", default=",".join(str(i) for i in range(1, 21)))
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default="results/leakage_quant.csv")
    args = ap.parse_args()
    cfg = load_config(args.config); seed_everything(int(cfg.seed))
    device = pick_device(args.device); mlabel = machine_label(device)
    nC = len(cfg.dataset.classes); rest_idx = int(cfg.get_path("control_loop.rest_class_index", 0))
    cal_lr = float(cfg.get_path("within_user.cal_lr", 3e-4)); seed = int(cfg.seed)
    cal_aug = dict(channel_shift=0, mag_warp_std=0.2, noise_std=0.05, time_mask_frac=0.1, n_time_masks=1,
                   chan_scale=(0.9, 1.1)) if bool(cfg.get_path("within_user.cal_augment", False)) else None
    rc = cfg.get_path("balance.rest_cap_mult")
    base_state = torch.load(Path(__file__).resolve().parents[1] / "results" / f"base_{cfg.name}.pt", map_location=device)
    subjects = [int(x) for x in args.subjects.split(",") if x.strip()]
    rng = np.random.default_rng(seed)
    print(f"=== LEAKAGE QUANTIFICATION === {mlabel} | {cfg.name} | {len(subjects)} users")

    def per_window_raw(cal_i, test_i, ws):
        if rc is not None:
            cal_i = cal_i[cap_rest_indices(ws.y[cal_i], rest_class_idx=rest_idx, mult=float(rc), seed=seed)]
        mean, std = per_subject_zscore_fit_apply(ws.X, cal_i); Xn = ((ws.X - mean) / std).astype(np.float32)
        cal = calibrate(cfg, base_state, in_channels(cfg), nC, rest_idx, Xn, cal_i, ws.y, device,
                        args.cal_epochs, cal_lr, seed, cal_aug)
        order = np.argsort(ws.group[test_i], kind="stable"); sidx = test_i[order]
        _, _, emis = batch_emissions(cal, Xn, sidx, device, rest_idx, nC)   # [N, K] per-window posterior
        return balanced_accuracy(ws.y[sidx], emis.argmax(1), nC) * 100

    rows = []
    for s in subjects:
        ws = load_subject(cfg, s)
        if ws is None or len(ws) == 0: continue
        cal_idx, test_idx = within_user_split(ws.y, ws.group, ws.rep, cal_reps=[1, 2, 3, 4, 5],
                                              test_reps=cfg.get_path("within_user.test_reps", [6]), rest_class_idx=rest_idx)
        n = len(ws.y); n_test = len(test_idx)
        perm = rng.permutation(n)                       # leaky: random window-level split, same #test windows
        leak_test = np.sort(perm[:n_test]); leak_cal = np.sort(perm[n_test:])
        safe = per_window_raw(cal_idx, test_idx, ws)
        leaky = per_window_raw(leak_cal, leak_test, ws)
        rows.append(dict(subject=s, leaksafe_pw=safe, leaky_pw=leaky, inflation=leaky - safe))
        print(f"  S{s}: leak-safe {safe:.1f} | leaky {leaky:.1f} | inflation +{leaky-safe:.1f}")

    import csv
    out = Path(args.out); out = out if out.is_absolute() else Path(__file__).resolve().parents[1] / args.out
    with open(out, "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    a = lambda k: np.array([r[k] for r in rows])
    W, p = wilcoxon(a("leaky_pw"), a("leaksafe_pw"))
    print(f"\n=== LEAKAGE INFLATION (per-window raw balanced acc, n={len(rows)}) ===")
    print(f"  leak-safe (segment split) : {a('leaksafe_pw').mean():.1f} +/- {a('leaksafe_pw').std():.1f}")
    print(f"  leaky (random-window split): {a('leaky_pw').mean():.1f} +/- {a('leaky_pw').std():.1f}")
    print(f"  INFLATION                  : +{a('inflation').mean():.1f} +/- {a('inflation').std():.1f} pts "
          f"(Wilcoxon p={p:.2e}, inflated for {int((a('inflation')>0).sum())}/{len(rows)})")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

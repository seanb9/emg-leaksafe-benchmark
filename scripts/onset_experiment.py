#!/usr/bin/env python
"""Onset-latency experiment: break the latency / false-activation tradeoff.

The deployed hysteresis decoder commits a grasp only after a FIXED debounce
(``on_frames`` consecutive active windows above a fixed gate). That forces a
clear, high-evidence onset to wait exactly as long as an ambiguous one, which is
why onset latency sits at ~600-900 ms — far over the <300 ms clinical target.

This script tests a principled alternative: a one-sided CUSUM sequential
change-detector (eval.control_loop.cusum_onset_decode), which is minimum-latency
for a given false-alarm rate (Page/Lorden). It accumulates the activation
log-likelihood-ratio and fires the instant the evidence crosses a threshold — so
a strong onset commits in ~1 window while weak ones still accumulate safely.

For a set of held-out users it calibrates the DEPLOYED decoder once (same
two-stage path as the paper), pools the test-stream emissions, then sweeps BOTH
decoders over their operating points and reports, for each, the median onset
latency, rest false-activation, per-window balanced accuracy, and per-execution
grasp accuracy. The headline is the LATENCY-vs-FALSE-ACTIVATION frontier: at a
matched false-activation budget, how much sooner does CUSUM commit, and does
per-execution accuracy hold? A figure of both frontiers is written.

No retraining — reuses the existing base checkpoint and runs on CPU/MPS.

RUN:
  python scripts/onset_experiment.py configs/db2_reable_hand.yaml --reuse-base \
         --subjects 1,2,3,4,5,6 --cal-epochs 25
  (quick plumbing check: --subjects 3 --cal-epochs 4)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from emgv2.config import load_config, seed_everything
from emgv2.data import ninapro_db2 as db2
from emgv2.data.splits import within_user_split, per_subject_zscore_fit_apply, cap_rest_indices
from emgv2.utils.device import pick_device, machine_label
from emgv2.eval.metrics import balanced_accuracy, majority_vote_by_group
from emgv2.eval.control_loop import (
    hysteresis_decode, cusum_onset_decode, decode_clinical_metrics, mean_onset_latency,
)

from run_within_user import load_subject, in_channels
from streaming_demo import calibrate, batch_emissions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--reuse-base", action="store_true")
    ap.add_argument("--subjects", default="1,2,3,4,5,6", help="comma-separated test subject ids")
    ap.add_argument("--cal-epochs", type=int, default=25)
    ap.add_argument("--fa-budget", type=float, default=2.0, help="false-activation budget (%) for the headline")
    ap.add_argument("--min-seg-acc", type=float, default=95.0,
                    help="per-execution accuracy floor (%) an operating point must meet to count")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/onset_experiment")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    device = pick_device(args.device)
    mlabel = machine_label(device)
    n_classes = len(cfg.dataset.classes)
    rest_idx = int(cfg.get_path("control_loop.rest_class_index", 0))
    stride_ms = float(cfg.get_path("window.stride_ms", 64))
    cal_lr = float(cfg.get_path("within_user.cal_lr", 3e-4))
    cal_aug = dict(channel_shift=0, mag_warp_std=0.2, noise_std=0.05,
                   time_mask_frac=0.1, n_time_masks=1, chan_scale=(0.9, 1.1)) \
        if bool(cfg.get_path("within_user.cal_augment", False)) else None
    subjects = [int(s) for s in args.subjects.split(",") if s.strip()]

    base_path = Path(__file__).resolve().parents[1] / "results" / f"base_{cfg.name}.pt"
    if not (args.reuse_base and base_path.exists()):
        print(f"  ERROR: need saved base at {base_path} (run run_within_user.py once).")
        return 1
    base_state = torch.load(base_path, map_location=device)

    print(f"=== ONSET-LATENCY EXPERIMENT === {mlabel} | {cfg.name}")
    print(f"    CUSUM change-detector vs fixed-debounce hysteresis | "
          f"{len(subjects)} users | stride {stride_ms:.0f} ms | FA budget {args.fa_budget:.1f}%\n")

    # ---- calibrate each user, pool the held-out test-stream emissions ---------
    P, G, Y, GRP = [], [], [], []          # p_active, grasp_pred, y_true, group(offset)
    gbase = 0
    t0 = time.time()
    for s in subjects:
        ws = load_subject(cfg, s)
        if ws is None or len(ws) == 0:
            print(f"  S{s}: no data, skipping"); continue
        cal_idx, test_idx = within_user_split(ws.y, ws.group, ws.rep, cal_reps=[1, 2, 3, 4, 5],
                                              test_reps=cfg.get_path("within_user.test_reps", [6]),
                                              rest_class_idx=rest_idx)
        rest_cap = cfg.get_path("balance.rest_cap_mult")
        if rest_cap is not None:
            keep = cap_rest_indices(ws.y[cal_idx], rest_class_idx=rest_idx,
                                    mult=float(rest_cap), seed=int(cfg.seed))
            cal_idx = cal_idx[keep]
        mean, std = per_subject_zscore_fit_apply(ws.X, cal_idx)
        Xn = ((ws.X - mean) / std).astype(np.float32)
        cal = calibrate(cfg, base_state, in_channels(cfg), n_classes, rest_idx, Xn, cal_idx,
                        ws.y, device, args.cal_epochs, cal_lr, int(cfg.seed), cal_aug)
        order = np.argsort(ws.group[test_idx], kind="stable")
        sidx = test_idx[order]
        pA, gp, _ = batch_emissions(cal, Xn, sidx, device, rest_idx, n_classes)
        P.append(pA); G.append(gp); Y.append(ws.y[sidx]); GRP.append(ws.group[sidx] + gbase)
        gbase += int(ws.group[sidx].max()) + 1
        print(f"  S{s}: calibrated, {len(sidx)} test windows")
    P = np.concatenate(P); G = np.concatenate(G); Y = np.concatenate(Y); GRP = np.concatenate(GRP)
    print(f"  pooled {len(P)} windows / {len(np.unique(GRP))} grasp segments in {time.time()-t0:.0f}s\n")

    def score(decoded):
        dm = decode_clinical_metrics(decoded, Y, GRP, rest_idx, n_classes)
        seg_t, seg_p = majority_vote_by_group(Y, decoded, GRP, n_classes)
        mv = seg_t != rest_idx
        seg_acc = float(np.mean(seg_p[mv] == seg_t[mv])) if mv.any() else float("nan")
        return dict(onset_ms=dm["latency"] * stride_ms, false_act=dm["false_act"] * 100,
                    win_bal=balanced_accuracy(Y, decoded, n_classes) * 100, seg_acc=seg_acc * 100)

    # ---- sweep the DEPLOYED hysteresis decoder (fixed debounce) ---------------
    hys_pts = []
    for enter in (0.5, 0.6, 0.7):
        for on in (1, 2, 3, 4):
            for ex in (0.35,):
                d = hysteresis_decode(P, G, GRP, rest_idx, enter_gate=enter, exit_gate=ex,
                                      on_frames=on, off_frames=4, switch_frames=3)
                r = score(d); r.update(method="hysteresis", knob=f"enter{enter}/on{on}")
                hys_pts.append(r)

    # ---- sweep the CUSUM change-detector --------------------------------------
    cus_pts = []
    for drift in (0.0, 0.25, 0.5, 0.75):
        for h in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            d = cusum_onset_decode(P, G, GRP, rest_idx, drift=drift, h=h,
                                   exit_gate=0.35, off_frames=4)
            r = score(d); r.update(method="cusum", knob=f"drift{drift}/h{h}")
            cus_pts.append(r)

    # ---- Pareto frontier per method (min latency at each FA level, accuracy-valid) ----
    fb, msa = args.fa_budget, args.min_seg_acc
    def valid(pts):
        return [p for p in pts if not np.isnan(p["onset_ms"]) and p["seg_acc"] >= msa]
    def frontier(pts):
        v = sorted(valid(pts), key=lambda p: p["false_act"])
        out, best = [], 1e9
        for p in v:                                   # keep points that lower latency
            if p["onset_ms"] < best - 1e-9:
                out.append(p); best = p["onset_ms"]
        return out
    print(f"=== PARETO FRONTIER (per-exec acc >= {msa:.0f}%; lower latency as FA rises) ===")
    for tag, pts in (("hysteresis", hys_pts), ("CUSUM", cus_pts)):
        print(f"  {tag}:")
        for p in frontier(pts):
            print(f"    FA {p['false_act']:>4.1f}%  ->  onset {p['onset_ms']:>4.0f} ms   "
                  f"(win-bal {p['win_bal']:.1f}%, seg-acc {p['seg_acc']:.1f}%, {p['knob']})")
    print()

    # ---- headline: min onset latency at <= FA budget, accuracy-valid ----
    def best_under_fa(pts):
        ok = [p for p in valid(pts) if p["false_act"] <= fb]
        return min(ok, key=lambda p: p["onset_ms"]) if ok else None
    bh, bc = best_under_fa(hys_pts), best_under_fa(cus_pts)
    print(f"=== HEADLINE — lowest onset latency at FA <= {fb:.1f}% AND per-exec acc >= {msa:.0f}% ===")
    hdr = f"  {'method':<11} {'onset_ms':>9} {'false_act':>10} {'win_bal':>8} {'seg_acc':>8}  knob"
    print(hdr)
    for tag, b in (("hysteresis", bh), ("CUSUM", bc)):
        if b:
            print(f"  {tag:<11} {b['onset_ms']:>8.0f} {b['false_act']:>9.1f}% "
                  f"{b['win_bal']:>7.1f}% {b['seg_acc']:>7.1f}%  {b['knob']}")
        else:
            print(f"  {tag:<11}   (no operating point under {fb:.1f}% false-act)")
    if bh and bc:
        d_ms = bh["onset_ms"] - bc["onset_ms"]
        d_acc = bc["seg_acc"] - bh["seg_acc"]
        print(f"\n  => CUSUM commits {d_ms:+.0f} ms {'SOONER' if d_ms>0 else 'LATER'} "
              f"at the same false-activation budget, with per-execution accuracy "
              f"{d_acc:+.1f} pts ({'no accuracy loss' if d_acc>=-1 else 'accuracy cost'}).")
        print(f"     hysteresis {bh['onset_ms']:.0f} ms -> CUSUM {bc['onset_ms']:.0f} ms "
              f"(clinical target < 300 ms).")

    # ---- figure: latency-vs-false-activation frontier -------------------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for pts, c, lab, mk in ((hys_pts, "#888", "hysteresis (fixed debounce)", "o"),
                                (cus_pts, "#0072B2", "CUSUM (sequential)", "s")):
            fa = [p["false_act"] for p in pts]; lat = [p["onset_ms"] for p in pts]
            ax.scatter(fa, lat, c=c, marker=mk, s=34, alpha=0.8, label=lab, edgecolors="none")
        ax.axvline(fb, color="#c33", ls=":", lw=1.3)
        ax.text(fb, ax.get_ylim()[1]*0.96, f" {fb:.0f}% FA budget", color="#c33", fontsize=8.5, va="top")
        ax.axhline(300, color="#2a2", ls="--", lw=1.2)
        ax.text(ax.get_xlim()[1]*0.98, 300, "300 ms clinical target", color="#2a2",
                fontsize=8.5, ha="right", va="bottom")
        ax.set_xlabel("rest false-activation (%)"); ax.set_ylabel("median onset latency (ms)")
        ax.set_title("Onset latency vs false-activation (lower-left = better)", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        fig.tight_layout()
        out = Path(args.out)
        if not out.is_absolute():
            out = Path(__file__).resolve().parents[1] / out
        out.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
        print(f"\n  wrote frontier figure -> {out}.png / .pdf")
    except Exception as e:
        print(f"  (figure skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

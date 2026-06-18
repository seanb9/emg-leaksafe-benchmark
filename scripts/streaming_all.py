#!/usr/bin/env python
"""All-subjects online streaming validation (for the paper's real-time claim).

For every test user: calibrate the deployed decoder, then replay the held-out
repetition window-by-window through the causal pipeline, timing each decision and
confirming the streamed decode equals the batched offline decode. Aggregates
per-decision latency (mean +/- SD across users) and the causal-match rate, and
writes a paper figure (representative live decode + pooled latency histogram).

RUN: python -u scripts/streaming_all.py configs/db2_reable_hand.yaml --reuse-base --cal-epochs 20
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from emgv2.config import load_config, seed_everything
from emgv2.data import ninapro_db2 as db2
from emgv2.data.splits import within_user_split, per_subject_zscore_fit_apply, cap_rest_indices
from emgv2.utils.device import pick_device, machine_label
from emgv2.eval.sequence import learn_transition_structure, build_transition, causal_forward_decode
from run_within_user import load_subject, in_channels
from streaming_demo import calibrate, emit_window, batch_emissions, OnlineHMM, OnlineHysteresis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--reuse-base", action="store_true")
    ap.add_argument("--cal-epochs", type=int, default=20)
    ap.add_argument("--subjects", default=",".join(str(i) for i in range(1, 21)))
    ap.add_argument("--rep-subject", type=int, default=3, help="subject whose timeline is plotted")
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default="results/fig8_streaming")
    args = ap.parse_args()

    cfg = load_config(args.config); seed_everything(int(cfg.seed))
    device = pick_device(args.device); mlabel = machine_label(device)
    n_classes = len(cfg.dataset.classes); rest_idx = int(cfg.get_path("control_loop.rest_class_index", 0))
    stride_ms = float(cfg.get_path("window.stride_ms", 64)); names = list(cfg.dataset.class_names)
    cal_lr = float(cfg.get_path("within_user.cal_lr", 3e-4))
    cal_aug = dict(channel_shift=0, mag_warp_std=0.2, noise_std=0.05, time_mask_frac=0.1,
                   n_time_masks=1, chan_scale=(0.9, 1.1)) if bool(cfg.get_path("within_user.cal_augment", False)) else None
    dec = cfg.get_path("decoder", {}) or {}
    base_path = Path(__file__).resolve().parents[1] / "results" / f"base_{cfg.name}.pt"
    base_state = torch.load(base_path, map_location=device)
    subjects = [int(x) for x in args.subjects.split(",") if x.strip()]
    print(f"=== ALL-SUBJECTS STREAMING === {mlabel} | {cfg.name} | {len(subjects)} users | budget {stride_ms:.0f} ms")

    # grammar from a held-out slice of pretrain subjects
    pre = cfg.get_path("within_user.pretrain_subjects") or list(range(21, 41))
    seqs = []
    for s in pre[:8]:
        wsp = load_subject(cfg, s)
        if wsp is None or len(wsp) == 0: continue
        for g in np.unique(wsp.group): seqs.append(wsp.y[wsp.group == g])
    O, pi = learn_transition_structure(seqs, n_classes)
    A = build_transition(O, float(cfg.get_path("within_user.hmm_hold", 0.97)))

    per_subj_median, per_subj_p95, match_rates = [], [], []
    pooled_lat = []
    rep = None
    for s in subjects:
        ws = load_subject(cfg, s)
        if ws is None or len(ws) == 0: print(f"  S{s}: no data"); continue
        cal_idx, test_idx = within_user_split(ws.y, ws.group, ws.rep, cal_reps=[1,2,3,4,5],
                                              test_reps=cfg.get_path("within_user.test_reps",[6]), rest_class_idx=rest_idx)
        rc = cfg.get_path("balance.rest_cap_mult")
        if rc is not None:
            cal_idx = cal_idx[cap_rest_indices(ws.y[cal_idx], rest_class_idx=rest_idx, mult=float(rc), seed=int(cfg.seed))]
        mean, std = per_subject_zscore_fit_apply(ws.X, cal_idx); Xn = ((ws.X - mean)/std).astype(np.float32)
        cal = calibrate(cfg, base_state, in_channels(cfg), n_classes, rest_idx, Xn, cal_idx, ws.y,
                        device, args.cal_epochs, cal_lr, int(cfg.seed), cal_aug)
        order = np.argsort(ws.group[test_idx], kind="stable"); sidx = test_idx[order]
        yte, gte = ws.y[sidx], ws.group[sidx]
        hmm = OnlineHMM(A, pi); hys = OnlineHysteresis(rest_idx, enter_gate=float(dec.get("enter_gate",0.6)),
            exit_gate=float(dec.get("exit_gate",0.35)), on_frames=int(dec.get("on_frames",3)),
            off_frames=int(dec.get("off_frames",4)), switch_frames=int(dec.get("switch_frames",3)))
        for _ in range(3): emit_window(cal, Xn[sidx[0]:sidx[0]+1], n_classes, rest_idx, device)  # warm
        dec_hmm = np.zeros(len(sidx), np.int64); dec_hys = np.zeros(len(sidx), np.int64); lat = np.zeros(len(sidx))
        prev = None
        for i, gi in enumerate(gte):
            if gi != prev: hmm.reset(); hys.reset(); prev = gi
            t = time.perf_counter()
            emis, pa, gp = emit_window(cal, Xn[sidx[i]:sidx[i]+1], n_classes, rest_idx, device)
            dec_hmm[i] = hmm.step(emis); dec_hys[i] = hys.step(pa, gp); lat[i] = (time.perf_counter()-t)*1e3
        # causal-match vs batched
        pA, gpb, emis_all = batch_emissions(cal, Xn, sidx, device, rest_idx, n_classes)
        batched = causal_forward_decode(emis_all, gte, A, pi)
        match = float(np.mean(batched == dec_hmm))*100
        med, p95 = np.percentile(lat, [50, 95])
        per_subj_median.append(med); per_subj_p95.append(p95); match_rates.append(match); pooled_lat.append(lat)
        print(f"  S{s}: {len(sidx)} win | match {match:5.1f}% | median {med:4.2f} ms | p95 {p95:4.2f} ms")
        if s == args.rep_subject: rep = (yte, dec_hys)
    pooled = np.concatenate(pooled_lat)
    np.savez_compressed(Path(__file__).resolve().parents[1] / "results" / "streaming_pooled_latency.npz",
                        pooled_ms=pooled, per_subj_median=np.array(per_subj_median),
                        per_subj_p95=np.array(per_subj_p95), n_users=len(per_subj_median))
    med = np.array(per_subj_median); p95 = np.array(per_subj_p95); mr = np.array(match_rates)
    print("\n=== AGGREGATE (across users) ===")
    print(f"  causal match: mean {mr.mean():.1f}% | min {mr.min():.1f}% | {int((mr>=99.99).sum())}/{len(mr)} users at 100%")
    print(f"  per-decision latency: median {med.mean():.2f}+/-{med.std():.2f} ms | p95 {p95.mean():.2f}+/-{p95.std():.2f} ms")
    print(f"  budget {stride_ms:.0f} ms -> using {med.mean()/stride_ms*100:.0f}% (median), {p95.mean()/stride_ms*100:.0f}% (p95)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7.0,5.0),gridspec_kw=dict(height_ratios=[2,1]))
        if rep is not None:
            yte,dh=rep; t=np.arange(len(yte))*stride_ms/1000
            ax1.step(t,yte,where="post",lw=2.4,color="#888",label="true intent")
            ax1.step(t,dh,where="post",lw=1.5,color="#0072B2",label="streamed decode")
            ax1.set_yticks(range(n_classes)); ax1.set_yticklabels(names,fontsize=8)
            ax1.set_xlabel("time (s, streamed at 64 ms/window)"); ax1.set_ylabel("grip")
            ax1.set_title(f"Online streaming decode (representative user), causal",fontsize=10.5)
            ax1.legend(loc="upper right",fontsize=8,frameon=False)
        ax2.hist(pooled,bins=50,color="#0072B2",alpha=0.85)
        ax2.axvline(stride_ms,color="#c33",ls="--",lw=1.5)
        ax2.text(stride_ms,ax2.get_ylim()[1]*0.9,f" {stride_ms:.0f} ms budget",color="#c33",fontsize=8.5,va="top")
        ax2.set_xlim(0,stride_ms*1.05)
        ax2.set_xlabel("per-decision compute latency (ms)"); ax2.set_ylabel("windows")
        ax2.set_title(f"Per-decision compute, {len(mr)} users: median {med.mean():.1f} ms (<< {stride_ms:.0f} ms)",fontsize=10)
        fig.tight_layout()
        out=Path(args.out);  out=out if out.is_absolute() else Path(__file__).resolve().parents[1]/out
        for e in("pdf","png"): fig.savefig(f"{out}.{e}",dpi=150,bbox_inches="tight")
        print(f"  wrote {out}.png / .pdf")
    except Exception as e:
        print(f"  (figure skipped: {e})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

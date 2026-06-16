#!/usr/bin/env python
"""Synthetic positive-control: inject simulated EMG, calibrate, decode.

Generates synthetic 'users' (known grasp structure), runs the SAME within-user
calibrate->decode flow (cal on reps 1-5, test on held-out rep 6, per-window +
HMM + per-execution metrics), and sweeps the signal-quality knobs that limit a
real device (SNR, channel count, electrode drift). This is a CONTROL: it checks
the pipeline recovers known structure and shows how accuracy scales with signal
quality. NOT real or clinical data.

    python scripts/run_synthetic.py                 # default + sweeps
    python scripts/run_synthetic.py --subjects 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emgv2.data.synthetic import make_synthetic_subject
from emgv2.data.features import add_envelope_features
from emgv2.data.splits import within_user_split, per_subject_zscore_fit_apply
from emgv2.models import build_v2net
from emgv2.models.presets import preset
from emgv2.train import engine
from emgv2.eval.metrics import balanced_accuracy, accuracy, majority_vote_by_group
from emgv2.eval.sequence import learn_transition_structure, build_transition, causal_forward_decode
from emgv2.eval.control_loop import false_activation_rate
from emgv2.utils.device import pick_device


def evaluate_subject(device, *, n_channels, snr_db, separability, drift, cal_epochs, seed):
    ws = make_synthetic_subject(n_channels=n_channels, snr_db=snr_db,
                                separability=separability, drift=drift, seed=seed)
    n_classes = 7
    Xf = add_envelope_features(ws.X, win=16, feats=("rms", "wl"))      # match the pipeline
    cal_idx, test_idx = within_user_split(ws.y, ws.group, ws.rep,
                                          cal_reps=[1, 2, 3, 4, 5], test_reps=[6])
    mean, std = per_subject_zscore_fit_apply(Xf, cal_idx)
    Xn = ((Xf - mean) / std).astype(np.float32)
    c_in = Xn.shape[-1]
    mc = preset("S", supcon=False, adversarial=False)
    model = build_v2net(c_in, n_classes, mc).to(device)
    # within-user training from scratch on the user's calibration windows
    model = engine.fewshot_adapt(model, Xn[cal_idx], ws.y[cal_idx], device,
                                 n_classes=n_classes, epochs=cal_epochs, lr=1e-3,
                                 head_only=False, seed=seed,
                                 aug_kw=dict(channel_shift=0, noise_std=0.05))
    preds, logits = engine.predict(model, Xn[test_idx], device)
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    yte, gte = ws.y[test_idx], ws.group[test_idx]
    # HMM transition logic learned from the user's calibration label sequences
    seqs = [ws.y[cal_idx][ws.group[cal_idx] == g] for g in np.unique(ws.group[cal_idx])]
    O, pi = learn_transition_structure(seqs, n_classes)
    A = build_transition(O, hold=0.97)
    preds_hmm = causal_forward_decode(probs, gte, A, pi)
    gt, gp = majority_vote_by_group(yte, preds_hmm, gte, n_classes)
    return dict(
        raw=balanced_accuracy(yte, preds, n_classes),
        hmm=balanced_accuracy(yte, preds_hmm, n_classes),
        seg=balanced_accuracy(gt, gp, n_classes),
        false_act=false_activation_rate(yte, preds_hmm, 0, None),
    )


def run_cell(device, *, label, subjects, cal_epochs, **kw):
    rows = [evaluate_subject(device, cal_epochs=cal_epochs, seed=1000 + s, **kw)
            for s in range(subjects)]
    def ms(k):
        v = np.array([r[k] for r in rows]) * 100
        return v.mean(), v.std()
    rw, rws = ms("raw"); hm, hms = ms("hmm"); sg, sgs = ms("seg"); fa, _ = ms("false_act")
    print(f"  {label:<30} raw {rw:4.1f}+/-{rws:3.1f} | HMM {hm:4.1f}+/-{hms:3.1f} | "
          f"SEG {sg:4.1f}+/-{sgs:3.1f} | false-act {fa:4.1f}%")
    return sg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=6)
    ap.add_argument("--cal-epochs", type=int, default=30)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"=== SYNTHETIC POSITIVE-CONTROL (calibrate->decode) | {device} | "
          f"{args.subjects} synthetic users | metrics = balanced accuracy %, mean+/-SD ===")
    print("  (control, not real data: validates the pipeline + shows scaling with signal quality)\n")

    print("DEFAULT (12ch, SNR 6 dB, separable, no drift):")
    run_cell(device, label="default", subjects=args.subjects, cal_epochs=args.cal_epochs,
             n_channels=12, snr_db=6.0, separability=1.0, drift=0.0)

    print("\nSNR SWEEP (12ch):")
    for snr in (0.0, 3.0, 6.0, 12.0):
        run_cell(device, label=f"SNR {snr:.0f} dB", subjects=args.subjects,
                 cal_epochs=args.cal_epochs, n_channels=12, snr_db=snr,
                 separability=1.0, drift=0.0)

    print("\nCHANNEL-COUNT SWEEP (SNR 6 dB) -- DB2 has 12, ReAble hardware 16:")
    for ch in (8, 12, 16):
        run_cell(device, label=f"{ch} channels", subjects=args.subjects,
                 cal_epochs=args.cal_epochs, n_channels=ch, snr_db=6.0,
                 separability=1.0, drift=0.0)

    print("\nELECTRODE-DRIFT SWEEP (12ch, SNR 6 dB; drift between cal and test):")
    for d in (0.0, 0.2, 0.4):
        run_cell(device, label=f"drift {d}", subjects=args.subjects,
                 cal_epochs=args.cal_epochs, n_channels=12, snr_db=6.0,
                 separability=1.0, drift=d)
    print("\nReads: accuracy should rise with SNR and channel count, and fall with drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

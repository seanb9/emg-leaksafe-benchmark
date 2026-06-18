#!/usr/bin/env python
"""Lightweight causal TCN baseline under the IDENTICAL leak-safe within-user
protocol (reviewer request: a learned temporal baseline beyond classical TD+LDA).

Population base on the pretraining subjects, per-user full fine-tune calibration on
5 reps, test on the held-out later repetition. We report per-window balanced
accuracy (raw evidence, no HMM grammar) and per-execution majority-voted balanced
accuracy, directly comparable to the 'CNN, raw evidence' row of the ablation table.

RUN: python -u scripts/tcn_baseline.py configs/db2_reable_hand.yaml
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn
from emgv2.config import load_config, seed_everything
from emgv2.data.splits import within_user_split, per_subject_zscore_fit_apply, cap_rest_indices
from emgv2.train import engine
from emgv2.eval.metrics import balanced_accuracy, majority_vote_by_group
from emgv2.utils.device import pick_device, machine_label
from run_within_user import load_subject, in_channels


class CausalTCN(nn.Module):
    """Standard dilated causal TCN (Bai et al. 2018): residual blocks with
    exponentially increasing dilation, left-padding for causality, GAP + linear head.
    Input [B, C, T] (the engine permutes [N,W,C]->[N,C,W])."""
    def __init__(self, c_in, n_classes, ch=48, levels=4, k=3, p_drop=0.1):
        super().__init__()
        layers = []
        prev = c_in
        for i in range(levels):
            d = 2 ** i
            pad = (k - 1) * d
            layers.append(_TCNBlock(prev, ch, k, d, pad, p_drop))
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.BatchNorm1d(ch), nn.Dropout(0.5), nn.Linear(ch, n_classes))

    def embed(self, x):
        h = self.tcn(x)
        return h.mean(dim=-1)                      # global average pool over time

    def forward(self, x, *, return_embed=False):
        z = self.embed(x)
        logits = self.head(z)
        return (logits, z) if return_embed else logits

    def deploy_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, pad, p_drop):
        super().__init__()
        self.pad = pad
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=d)
        self.bn1 = nn.BatchNorm1d(c_out); self.do1 = nn.Dropout(p_drop)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=d)
        self.bn2 = nn.BatchNorm1d(c_out); self.do2 = nn.Dropout(p_drop)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.act = nn.ReLU(inplace=True)

    def _causal(self, x, conv):
        return conv(nn.functional.pad(x, (self.pad, 0)))   # left-pad only -> causal

    def forward(self, x):
        h = self.do1(self.act(self.bn1(self._causal(x, self.conv1))))
        h = self.do2(self.act(self.bn2(self._causal(h, self.conv2))))
        res = x if self.down is None else self.down(x)
        return self.act(h + res)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--base-epochs", type=int, default=40); ap.add_argument("--cal-epochs", type=int, default=25)
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default="results/tcn_baseline.csv")
    args = ap.parse_args()
    cfg = load_config(args.config); seed_everything(int(cfg.seed))
    device = pick_device(args.device); mlabel = machine_label(device)
    nC = len(cfg.dataset.classes); rest_idx = int(cfg.get_path("control_loop.rest_class_index", 0))
    c_in = in_channels(cfg); seed = int(cfg.seed)
    rc = cfg.get_path("balance.rest_cap_mult")
    pre = cfg.get_path("within_user.pretrain_subjects"); tst = cfg.get_path("within_user.test_subjects")
    if pre is None or tst is None:
        from emgv2.data import ninapro_db2 as db2
        avail = db2.available_subjects(cfg); half = len(avail) // 2
        tst = avail[:half]; pre = avail[half:]      # test = lower-indexed half, base = upper (matches headline)
    print(f"=== TCN BASELINE (leak-safe within-user) === {mlabel} | {cfg.name} | {c_in}ch {nC} classes | base {len(pre)} test {len(tst)}")

    def zfit(X, idx): m = X[idx].mean((0, 1)); s = X[idx].std((0, 1)) + 1e-6; return ((X - m) / s).astype(np.float32)

    # population base on pretraining subjects (per-subject z-scored on all their windows)
    Xtr, ytr = [], []
    for s in pre:
        ws = load_subject(cfg, s)
        if ws is None or len(ws) == 0: continue
        Xn = zfit(ws.X, np.arange(len(ws.y))); Xtr.append(Xn); ytr.append(ws.y)
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    keep = cap_rest_indices(ytr, rest_class_idx=rest_idx, mult=float(rc or 1.5), seed=seed); Xtr, ytr = Xtr[keep], ytr[keep]
    base = CausalTCN(c_in, nC).to(device); nval = max(1, len(Xtr) // 10)
    print(f"  base TCN: {base.deploy_params():,} params | {len(Xtr)} win | {args.base_epochs} ep")
    engine.train_supervised(base, Xtr[:-nval], ytr[:-nval], Xtr[-nval:], ytr[-nval:], device,
                            n_classes=nC, epochs=args.base_epochs, aug_kw=dict(channel_shift=2), seed=seed, log=lambda *_: None)
    bstate = {k: v.clone() for k, v in base.state_dict().items()}
    del Xtr, ytr

    rows = []
    for s in tst:
        ws = load_subject(cfg, s)
        if ws is None or len(ws) == 0: continue
        cal_idx, test_idx = within_user_split(ws.y, ws.group, ws.rep, cal_reps=[1, 2, 3, 4, 5],
                                              test_reps=cfg.get_path("within_user.test_reps", [6]), rest_class_idx=rest_idx)
        if rc is not None:
            cal_idx = cal_idx[cap_rest_indices(ws.y[cal_idx], rest_class_idx=rest_idx, mult=float(rc), seed=seed)]
        Xn = zfit(ws.X, cal_idx)
        m = CausalTCN(c_in, nC).to(device); m.load_state_dict(bstate, strict=True)
        m = engine.fewshot_adapt(m, Xn[cal_idx], ws.y[cal_idx], device, n_classes=nC,
                                 epochs=args.cal_epochs, lr=float(cfg.get_path("within_user.cal_lr", 3e-4)),
                                 head_only=False, seed=seed)
        order = np.argsort(ws.group[test_idx], kind="stable"); sidx = test_idx[order]
        yte, gte = ws.y[sidx], ws.group[sidx]
        preds, _ = engine.predict(m, Xn[sidx], device)
        pw = balanced_accuracy(yte, preds, nC) * 100
        st, sp = majority_vote_by_group(yte, preds, gte, nC); pe = balanced_accuracy(st, sp, nC) * 100
        rows.append(dict(subject=s, per_window=pw, per_exec=pe))
        print(f"  S{s}: per-window {pw:.1f} | per-exec {pe:.1f}")

    import csv
    out = Path(args.out); out = out if out.is_absolute() else Path(__file__).resolve().parents[1] / out
    with open(out, "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    a = lambda k: np.array([r[k] for r in rows])
    print(f"\n=== TCN BASELINE ({len(rows)} subjects), balanced accuracy ===")
    print(f"  per-window : {a('per_window').mean():.1f} +/- {a('per_window').std():.1f}")
    print(f"  per-exec   : {a('per_exec').mean():.1f} +/- {a('per_exec').std():.1f}")
    print(f"  (compare: CNN raw 72.6/94.0 ; classical TD+LDA 66.9/94.7 ; CNN+HMM 75.2/92.7)")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

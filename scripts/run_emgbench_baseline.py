#!/usr/bin/env python
"""In-harness EMGBench baseline (ResNet18 on jet-colormap images) run under OUR
LOSO splits — the "our re-implementation" side of the gold-standard cross-check.

Emits per (leftout_subject, proportion) Macro_Acc in the SAME CSV schema as
scripts/emgbench_repo_crosscheck.py, so scripts/compare_baseline.py can decide
whether our re-impl is faithful to EMGBench's actual repo (within noise) or not.

Uses the EMGBench-matched protocol (db2_emgbench_matched.yaml: 2 kHz, 250 ms
non-overlap, their 10 classes) and per-window image normalization (fold-
independent, matching --turn_off_scaler_normalization=True), so per-subject image
caches are computed once and reused across folds.

RIG:
    python scripts/run_emgbench_baseline.py configs/db2_emgbench_matched.yaml \
        --subjects 1 2 3 4 5 6 7 8 --proportion 0.0 --proportion 0.2 \
        --epochs 100 --finetune-epochs 50 --out results/emgbench_inhouse.csv
DEV (structural, Mac):
    python scripts/run_emgbench_baseline.py configs/db2_emgbench_matched.yaml \
        --subjects 1 2 --proportion 0.0 --epochs 1 --cap 200 --no-pretrained --out /tmp/x.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emgv2.config import load_config, seed_everything
from emgv2.data import ninapro_db2 as db2
from emgv2.data.splits import make_loso_folds, leak_safe_calibration_split
from emgv2.models.emgbench_baseline import EMGBenchImage, build_emgbench_baseline
from emgv2.eval.metrics import balanced_accuracy
from emgv2.utils.device import pick_device, machine_label


def cache_images(cfg, subject, tf, cache_dir, cap=None, rng=None):
    """Compute (or load) a subject's EMGBench-style images. Returns (images, y, group, rep)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"S{subject:02d}_img.npy"
    mpath = cache_dir / f"S{subject:02d}_meta.npz"
    ws = db2.get_subject(cfg, subject)
    if ws is None:
        return None
    idx = np.arange(len(ws))
    if cap and len(ws) > cap:
        idx = np.sort(rng.choice(len(ws), cap, replace=False))
    if cpath.exists() and mpath.exists() and cap is None:
        return np.load(cpath, mmap_mode="r"), *(np.load(mpath)[k] for k in ("y", "group", "rep"))
    imgs = tf.batch(ws.X[idx]).astype(np.float16)
    if cap is None:
        np.save(cpath, imgs)
        np.savez(mpath, y=ws.y[idx], group=ws.group[idx], rep=ws.rep[idx])
    return imgs, ws.y[idx], ws.group[idx], ws.rep[idx]


def train_resnet(model, X, y, device, epochs, lr, bs=32, rng=None):
    import torch, torch.nn as nn
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    y_t = torch.from_numpy(np.asarray(y)).long()
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for s in range(0, len(perm), bs):
            ch = perm[s:s + bs]
            if len(ch) < 2:
                continue
            xb = torch.from_numpy(np.asarray(X[ch])).float().to(device)
            loss = ce(model(xb), y_t[ch].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def predict(model, X, device, bs=64):
    import torch
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            xb = torch.from_numpy(np.asarray(X[s:s + bs])).float().to(device)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    import copy
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--proportion", type=float, action="append", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--finetune-epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--cap", type=int, default=None, help="cap windows/subject (dev only)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/emgbench_inhouse.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    device = pick_device(args.device)
    rng = np.random.default_rng(cfg.seed)
    props = args.proportion or [0.0, 0.2]
    n_classes = len(cfg.dataset.classes)
    tf = EMGBenchImage(mode="per_window").fit(np.zeros((1, 1, 1)))
    cache_dir = Path(__file__).resolve().parents[1] / "data_cache" / f"emgbench_img_{cfg.name}"

    print(f"=== IN-HARNESS EMGBench baseline === {machine_label(device)} | {cfg.name} | "
          f"subjects={args.subjects} | props={props}")

    # cache images for all needed subjects
    data = {}
    for s in args.subjects:
        d = cache_images(cfg, s, tf, cache_dir, cap=args.cap, rng=rng)
        if d is not None:
            data[s] = d
            print(f"  S{s}: {len(d[1])} images cached")

    folds = make_loso_folds(sorted(data.keys()),
                            val_n=min(int(cfg.split.val_n_subjects), max(1, len(data) - 2)))
    rows = []
    for f in folds:
        test = f["test"]
        tr = [s for s in f["train"] + f["val"] if s in data]
        if not tr:
            continue
        Xtr = np.concatenate([np.asarray(data[s][0]) for s in tr])
        ytr = np.concatenate([data[s][1] for s in tr])
        base = build_emgbench_baseline(n_classes, pretrained=not args.no_pretrained).to(device)
        t0 = time.time()
        train_resnet(base, Xtr, ytr, device, args.epochs, args.lr, rng=rng)
        Ite, yte, gte, rte = data[test]
        Ite = np.asarray(Ite)
        for prop in props:
            model = base
            if prop > 0:
                cal_idx, eval_idx = leak_safe_calibration_split(
                    yte, gte, rte, ratio=prop,
                    eval_reps=cfg.split.calibration.eval_reps,
                    cal_reps=cfg.split.calibration.cal_reps, seed=cfg.seed)
                model = train_resnet(copy.deepcopy(base), Ite[cal_idx], yte[cal_idx],
                                     device, args.finetune_epochs, args.lr, rng=rng)
            else:
                _, eval_idx = leak_safe_calibration_split(
                    yte, gte, rte, ratio=0.0,
                    eval_reps=cfg.split.calibration.eval_reps,
                    cal_reps=cfg.split.calibration.cal_reps, seed=cfg.seed)
            preds = predict(model, Ite[eval_idx], device)
            macc = balanced_accuracy(yte[eval_idx], preds, n_classes)
            rows.append(dict(machine=machine_label(device), source="inhouse_reimpl",
                             model="resnet18", representation="raw_perwindow",
                             leftout_subject=test, proportion=prop, macro_acc=macc))
            print(f"  S{test} prop={prop}: Macro_Acc={macc*100:.1f}% ({time.time()-t0:.0f}s)")

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out}")
    print("Compare to EMGBench repo with scripts/compare_baseline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

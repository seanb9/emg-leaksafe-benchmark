#!/usr/bin/env python
"""Full 40-fold LOSO for V2Net — crash-proof + parallelisable (rig/CUDA).

Each fold is checkpointed to its own file the moment it finishes, so:
  * a power-off only loses the in-progress fold — just re-run to resume, and
  * you can split the 40 folds across several PowerShell windows that share the
    GPU, each given a different --test-subjects range, for a big speedup.

Single window (resumable):
    python scripts/run_loso_full.py configs/db2_reable_10class.yaml --size S --supcon-epochs 20 --ft-epochs 60

Parallel — 4 windows on one GPU (do this with 32 GB RAM), then read the combined
result from any window once all 40 folds exist:
    win1:  python scripts/run_loso_full.py configs/db2_reable_10class.yaml --size S --test-subjects 1-10
    win2:  python scripts/run_loso_full.py configs/db2_reable_10class.yaml --size S --test-subjects 11-20
    win3:  python scripts/run_loso_full.py configs/db2_reable_10class.yaml --size S --test-subjects 21-30
    win4:  python scripts/run_loso_full.py configs/db2_reable_10class.yaml --size S --test-subjects 31-40
All four write to the same results/ckpt_<config>_S/ dir (one file per fold, no
contention). The last to finish prints the complete 40-fold SUMMARY.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emgv2.config import load_config, seed_everything
from emgv2.models.presets import preset
from emgv2.train.loso import run_loso_checkpointed
from emgv2.utils.device import pick_device, machine_label, is_headline_capable
from emgv2.utils.results import append_results_csv


def parse_subjects(spec):
    """'1-10' or '1,2,3' or '1-10,15,20-25' -> sorted list of ints."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--size", choices=["S", "M", "L"], default="S")
    ap.add_argument("--test-subjects", default=None,
                    help="folds this worker runs, e.g. '1-10' (default: all)")
    ap.add_argument("--target-fs", type=int, default=None)
    ap.add_argument("--supcon-epochs", type=int, default=20)
    ap.add_argument("--ft-epochs", type=int, default=60)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--csv", default="results/loso_full.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.target_fs:
        cfg["signal"]["target_fs"] = args.target_fs
    seed_everything(int(cfg.seed))
    device = pick_device(args.device)
    subjects = parse_subjects(args.test_subjects) if args.test_subjects else None
    run_tag = args.size + (f"_fs{args.target_fs}" if args.target_fs else "")

    if not is_headline_capable(device):
        print(f"WARNING: device={machine_label(device)} is not CUDA — not headline-eligible.")

    t0 = time.time()
    rows, recs, total = run_loso_checkpointed(
        cfg, model_override=preset(args.size), test_subjects=subjects,
        epochs_supcon=args.supcon_epochs, epochs_ft=args.ft_epochs,
        device=device, run_tag=run_tag,
    )
    print(f"\n=== LOSO {cfg.name} size {args.size} | {machine_label(device)} | "
          f"{len(recs)}/{total} folds done | {time.time()-t0:.0f}s ===")
    print(f"{'cal%':>5} {'bal_acc':>9} {'±std':>7} {'+HMM':>8} {'+HMM_vote':>10} {'headline':>9}")
    for r in rows:
        ex = r.extra or {}
        hmm = ex.get("hmm_bal_acc_mean"); hmmv = ex.get("hmm_vote_bal_acc_mean")
        hs = f"{hmm*100:>7.1f}%" if hmm is not None else f"{'-':>8}"
        hvs = f"{hmmv*100:>9.1f}%" if hmmv is not None else f"{'-':>10}"
        print(f"{int(r.cal_ratio*100):>5} {r.value_mean*100:>8.1f}% {r.value_std*100:>6.1f}% "
              f"{hs} {hvs} {str(r.headline_eligible):>9}")
    print("bal_acc=baseline per-window;  +HMM=per-window with sequence logic;  "
          "+HMM_vote=per-repetition voted (EMGBench protocol). All zero-extra-calibration.")
    if len(recs) >= total:
        csv_path = str(Path(__file__).resolve().parents[1] / args.csv)
        append_results_csv(rows, csv_path)
        print(f"\nALL {total} folds done. Wrote headline rows -> {args.csv}")
    else:
        print(f"\n{len(recs)}/{total} folds so far (this is the combined view across "
              f"workers). Re-run any time to resume; rows saved only when all folds done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

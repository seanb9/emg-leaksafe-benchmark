#!/usr/bin/env python
"""Gold-standard EMGBench baseline cross-check: run THEIR actual repo on a chosen
set of LOSO folds (leftout subjects) and capture per-fold Macro_Acc, so we can
compare it to our in-harness EMGBench-style re-implementation on the SAME folds.

This is the load-bearing comparison. We do NOT tune our re-impl to hit a target
number (circular). Instead:
  * If their repo and our in-harness re-impl agree within noise on these folds,
    the in-harness version is faithful and becomes the baseline for the full sweep.
  * If they disagree, THEIR repo is the baseline, the gap is reported, and we fix
    the re-impl.

Runs ONLY where their repo runs (CUDA rig with their env + DB2 data). On the rig:
  # one-time setup
  ln -s "$(pwd)/../../reable_emg_classifier/data/ninapro_db2" external/emgbench/NinaproDB2
  cd external/emgbench && pip install -r requirements.txt   # their deps
  # cross-check (zero-shot and 20%) on 8 folds
  WANDB_MODE=offline python ../../scripts/emgbench_repo_crosscheck.py \
      --emgbench-dir external/emgbench --subjects 1 2 3 4 5 6 7 8 \
      --proportion 0.0 --proportion 0.2 --out results/emgbench_repo_crosscheck.csv

Their exact DB2 LOSO command (from EMGBench README) is reproduced below; only
--leftout_subject and --proportion_transfer_learning vary per run.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

MACRO_ACC_RE = re.compile(r"Macro_Acc:\s*([0-9.]+)")


def build_command(subject: int, proportion: float, *, model="resnet18", epochs=100,
                  finetuning_epochs=750, lr="5e-4", representation_flag=None) -> list[str]:
    """Reproduce EMGBench's DB2 partial-dataset (10-class) LOSO command."""
    cmd = [
        sys.executable, "CNN_EMG.py",
        "--dataset=ninapro-db2", "--seed=0", f"--model={model}", f"--epochs={epochs}",
        "--turn_off_scaler_normalization=True", f"--leftout_subject={subject}",
        "--leave_one_subject_out=True", "--train_test_split_for_time_series=True",
        "--save_images=True", f"--learning_rate={lr}",
        "--proportion_data_from_training_subjects=1.0",
        "--partial_dataset_ninapro=True",
    ]
    if proportion and proportion > 0:
        cmd += [
            "--transfer_learning=True", f"--proportion_transfer_learning={proportion}",
            f"--finetuning_epochs={finetuning_epochs}", "--pretrain_and_finetune=True",
        ]
    if representation_flag:  # e.g. --turn_on_rms=True ; omit for 'raw'
        cmd.append(representation_flag)
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emgbench-dir", required=True)
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--proportion", type=float, action="append", default=None,
                    help="calibration proportion; repeat for several (0.0=zero-shot)")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--finetuning-epochs", type=int, default=750)
    ap.add_argument("--representation", default="raw",
                    choices=["raw", "rms", "spectrogram", "cwt"])
    ap.add_argument("--out", default="results/emgbench_repo_crosscheck.csv")
    ap.add_argument("--dry-run", action="store_true", help="print commands, do not run")
    args = ap.parse_args()

    props = args.proportion or [0.0, 0.2]
    rep_flag = {"raw": None, "rms": "--turn_on_rms=True",
                "spectrogram": "--turn_on_spectrogram=True",
                "cwt": "--turn_on_cwt=True"}[args.representation]
    eb_dir = Path(args.emgbench_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for subj in args.subjects:
        for prop in props:
            cmd = build_command(subj, prop, model=args.model, epochs=args.epochs,
                                finetuning_epochs=args.finetuning_epochs,
                                representation_flag=rep_flag)
            print(f"[fold S{subj} | prop {prop}] {' '.join(cmd)}")
            if args.dry_run:
                continue
            env = {**os.environ, "WANDB_MODE": "offline"}
            proc = subprocess.run(cmd, cwd=eb_dir, env=env, capture_output=True, text=True)
            macros = MACRO_ACC_RE.findall(proc.stdout)
            macro_acc = float(macros[-1]) if macros else float("nan")
            if not macros:
                print(f"  WARNING: no Macro_Acc parsed (exit {proc.returncode}). "
                      f"stderr tail:\n{proc.stderr[-500:]}")
            print(f"  -> Macro_Acc = {macro_acc:.4f}")
            rows.append(dict(machine="rig-cuda", source="emgbench_repo",
                             model=args.model, representation=args.representation,
                             leftout_subject=subj, proportion=prop, macro_acc=macro_acc))

    if rows:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {out_path}")
        print("Next: compare to our in-harness baseline on the same folds (scripts/compare_baseline.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Result rows with mandatory provenance stamps.

Every reported number carries: which machine produced it, the protocol/config,
the class set and count, the sampling rate, the calibration ratio, and whether it
is a headline-eligible run (full 40-fold LOSO on the rig) or a dev smoke. This is
enforced by making those fields required on the dataclass.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class ResultRow:
    # provenance (all required)
    run_kind: str          # "dev_smoke" | "loso_full" | "loso_partial"
    headline_eligible: bool  # True only for full 40-fold LOSO on CUDA rig
    machine: str           # from utils.machine_label()
    config_name: str
    model: str             # "v2net" | "emgbench_resnet18" | ...
    protocol: str          # "loso"
    class_set: str         # "emgbench10" | "reable12" | ...
    n_classes: int
    sampling_hz: int
    window_ms: int
    # result
    cal_ratio: float
    metric_name: str       # "balanced_accuracy" (== EMGBench Macro_Acc) | "macro_f1" | ...
    value_mean: float
    value_std: float
    n_folds: int
    decision_unit: str = "window"   # "window" | "segment_majority_vote"
    notes: str = ""
    extra: dict = field(default_factory=dict)


CSV_FIELDS = [
    "run_kind", "headline_eligible", "machine", "config_name", "model", "protocol",
    "class_set", "n_classes", "sampling_hz", "window_ms", "cal_ratio", "metric_name",
    "value_mean", "value_std", "n_folds", "decision_unit", "notes",
]


def results_to_dataframe(rows: list[ResultRow]):
    import pandas as pd

    return pd.DataFrame([{k: getattr(r, k) for k in CSV_FIELDS} for r in rows])


def append_results_csv(rows: list[ResultRow], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in CSV_FIELDS})

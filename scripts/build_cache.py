#!/usr/bin/env python
"""Window + cache all available subjects for a config, and print dataset stats.

Usage:
    python scripts/build_cache.py configs/db2_reable_10class.yaml
    python scripts/build_cache.py configs/db2_reable_10class.yaml --no-cache --subjects 1 2 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emgv2.config import load_config, seed_everything
from emgv2.data import ninapro_db2 as db2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--subjects", type=int, nargs="*", default=None,
                    help="subset of subject ids (default: all available)")
    ap.add_argument("--no-cache", action="store_true", help="window without writing cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    subjects = args.subjects or db2.available_subjects(cfg)
    print(f"config={cfg.name} | fs={cfg.signal.target_fs} | classes={len(cfg.dataset.classes)} "
          f"| win={cfg.window.size_ms}ms/{cfg.window.stride_ms}ms | subjects={len(subjects)}")

    totals = np.zeros(len(cfg.dataset.classes), dtype=np.int64)
    grand = 0
    t0 = time.time()
    for s in subjects:
        ts = time.time()
        ws = db2.get_subject(cfg, s, use_cache=not args.no_cache)
        if ws is None:
            print(f"  S{s:02d}: NOT FOUND")
            continue
        cnt = np.bincount(ws.y, minlength=len(cfg.dataset.classes))
        totals += cnt
        grand += len(ws)
        print(f"  S{s:02d}: {len(ws):6d} windows | {len(np.unique(ws.group)):3d} groups "
              f"| {time.time()-ts:4.1f}s")
    print(f"\nTotal windows: {grand} over {len(subjects)} subjects in {time.time()-t0:.0f}s")
    print("Class totals:")
    for name, c in zip(cfg.dataset.class_names, totals):
        print(f"  {name:20s} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

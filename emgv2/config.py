"""Config loading and run-determinism helpers.

A single YAML file fully specifies a run (datasets, signal, windowing, splits,
normalization, seed). This module loads it into a lightweight attribute-access
object and provides one place to seed all RNGs.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class Cfg(dict):
    """dict with attribute access and nested-dict wrapping.

    Cfg({'a': {'b': 1}}).a.b == 1
    """

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(val, dict) and not isinstance(val, Cfg):
            val = Cfg(val)
            self[name] = val
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted key, e.g. cfg.get_path('signal.target_fs')."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | os.PathLike) -> Cfg:
    """Load a YAML run-config. The config dir is the base for relative dataset paths."""
    path = Path(path).expanduser().resolve()
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    cfg = Cfg(raw)
    # Data dir resolution order:
    #   1. REABLE_DB2_DIR env var (set this on the rig once; never edit configs ->
    #      git pull stays clean). e.g. setx REABLE_DB2_DIR C:\data\ninapro_db2
    #   2. otherwise the config's dataset.raw_dir, resolved relative to the config
    #      file's directory so a run is portable regardless of cwd.
    env_dir = os.environ.get("REABLE_DB2_DIR")
    if env_dir:
        cfg["dataset"]["raw_dir"] = env_dir
    else:
        raw_dir = cfg.get_path("dataset.raw_dir")
        if raw_dir is not None and not os.path.isabs(raw_dir):
            cfg["dataset"]["raw_dir"] = str((path.parent / raw_dir).resolve())
    cfg["_config_path"] = str(path)
    return cfg


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy (and Torch if present) for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional in the data/step-1 stage
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch not installed yet
        pass

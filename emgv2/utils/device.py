"""Device selection and a human-readable machine label.

Every result is stamped with a machine label so dev-subset MPS numbers can never
be confused with full rig LOSO numbers (a hard requirement: MPS = "does it work",
only full 40-fold rig LOSO is a headline number).
"""
from __future__ import annotations

import platform


def pick_device(prefer: str = "auto"):
    import torch

    if prefer not in ("auto", "cuda", "mps", "cpu"):
        prefer = "auto"
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def machine_label(device=None) -> str:
    """e.g. 'mac-mps (Darwin arm64)' or 'rig-cuda:NVIDIA GeForce RTX 4080 SUPER'."""
    import torch

    if device is None:
        device = pick_device()
    dtype = device.type if hasattr(device, "type") else str(device)
    if dtype == "cuda":
        try:
            return f"rig-cuda:{torch.cuda.get_device_name(0)}"
        except Exception:
            return "rig-cuda"
    if dtype == "mps":
        return f"mac-mps ({platform.system()} {platform.machine()})"
    return f"cpu ({platform.system()} {platform.machine()})"


def is_headline_capable(device) -> bool:
    """Only CUDA (the rig) may produce a headline number; MPS/CPU are dev only."""
    return getattr(device, "type", str(device)) == "cuda"

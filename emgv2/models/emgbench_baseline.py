"""EMGBench-style baseline reproduction (ResNet18 on image-encoded EMG).

EMGBench's CNN pipeline turns each EMG window into an image: globally min/max
normalized window -> 'jet' colormap -> resize to 224x224 -> ImageNet
normalization -> pretrained ResNet (fc replaced). It is trained with Adam + CE
and reported with Macro accuracy (== mean per-class recall == our balanced
accuracy) on a class-balanced evaluation set.

This module reproduces that path INSIDE our harness so the matched comparison is
"their method on our splits", not just "their class list".

FIDELITY CAVEATS (to verify against their actual repo on the rig before any
"matched" claim):
  * EMGBench's exact `getImages` 'raw' layout (channel/time reshape) and default
    LR/epochs come from their Parse_Arguments; here we use Adam(1e-4), early
    stopped, which must be tuned on the rig until we land near their published
    DB2 number (~19.9% zero-shot LOSO, ~52.4% @20%). If we cannot reproduce it,
    the gap is reported, not hidden.
  * They cache images as zarr; we compute on the fly (cache added for the rig).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EMGBenchImage:
    """Window [T, C] -> EMGBench-style jet-colormap image tensor [3, 224, 224].

    fit() captures the global min/max scaler from the training windows (EMGBench
    normalizes with a global scaler fit on training data only). Apply with __call__.
    """

    def __init__(self, size: int = 224, mode: str = "global_scaler"):
        # mode 'global_scaler' : fit one min/max on training data (fold-dependent).
        # mode 'per_window'     : normalize each window by its own min/max. This is
        #   fold-INDEPENDENT, matching EMGBench's --turn_off_scaler_normalization=True
        #   path, so per-subject image caches can be precomputed once and reused.
        self.size = size
        self.mode = mode
        self.gmin = None
        self.gmax = None
        self._cmap = None

    def fit(self, X_train: np.ndarray) -> "EMGBenchImage":
        if self.mode == "per_window":
            self.gmin, self.gmax = 0.0, 1.0  # unused; set so apply() is callable
            return self
        # global scalar min/max over all train windows/channels.
        self.gmin = float(np.min(X_train))
        self.gmax = float(np.max(X_train))
        return self

    def _jet(self):
        if self._cmap is None:
            import matplotlib as mpl
            self._cmap = mpl.colormaps["jet"]
        return self._cmap

    def __call__(self, window: np.ndarray) -> np.ndarray:
        """window [T, C] -> [3, size, size] float32 (ImageNet-normalized)."""
        import torch.nn.functional as F

        assert self.gmin is not None, "call fit() before applying the transform"
        img = window.T.astype(np.float32)              # [C, T]
        if self.mode == "per_window":
            lo, hi = float(img.min()), float(img.max())
            denom = (hi - lo) or 1.0
            img = np.clip((img - lo) / denom, 0.0, 1.0)
        else:
            denom = (self.gmax - self.gmin) or 1.0
            img = np.clip((img - self.gmin) / denom, 0.0, 1.0)
        rgba = self._jet()(img)                        # [C, T, 4] in [0,1]
        rgb = rgba[..., :3].transpose(2, 0, 1)         # [3, C, T]
        t = torch.from_numpy(np.ascontiguousarray(rgb)).float().unsqueeze(0)
        t = F.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
        t = t.squeeze(0).numpy()
        for c in range(3):
            t[c] = (t[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
        return t.astype(np.float32)

    def batch(self, windows: np.ndarray) -> np.ndarray:
        return np.stack([self(w) for w in windows]).astype(np.float32)


class EMGBenchResNet(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.net = resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)

    def forward(self, x):
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_emgbench_baseline(n_classes: int, pretrained: bool = True) -> EMGBenchResNet:
    return EMGBenchResNet(n_classes, pretrained=pretrained)

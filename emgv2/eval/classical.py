"""Classical EMG baseline: Hudgins time-domain features + shrinkage LDA.

A reviewer-standard sanity baseline. The Hudgins TD set (MAV, WL, ZC, SSC) with a
linear classifier is the long-standing reference for myoelectric pattern
recognition and is competitive with lightweight CNNs at a fraction of the cost. We
report it so the learned network's footprint/accuracy is benchmarked against a
no-representation-learning baseline, not just against other deep models.
"""
from __future__ import annotations

import numpy as np


def td_features(X_raw: np.ndarray, *, zc_thr=0.01, ssc_thr=0.01) -> np.ndarray:
    """[N, T, C] raw windows -> [N, 4C] (MAV, WL, ZC, SSC per channel)."""
    X = np.asarray(X_raw, dtype=np.float64)
    N, T, C = X.shape
    mav = np.mean(np.abs(X), axis=1)                                  # [N, C]
    dx = np.diff(X, axis=1)                                           # [N, T-1, C]
    wl = np.sum(np.abs(dx), axis=1)                                   # [N, C]
    sd = np.std(X, axis=1, keepdims=True)
    zt = zc_thr * sd
    zc = np.sum((np.sign(X[:, :-1]) != np.sign(X[:, 1:])) & (np.abs(dx) > zt), axis=1)
    ddx = np.diff(dx, axis=1)
    st = ssc_thr * np.std(dx, axis=1, keepdims=True)
    ssc = np.sum((np.sign(dx[:, :-1]) != np.sign(dx[:, 1:])) & (np.abs(ddx) > st), axis=1)
    return np.concatenate([mav, wl, zc.astype(float), ssc.astype(float)], axis=1)


class ClassicalTD:
    """Hudgins TD features + standardisation + shrinkage LDA."""

    def __init__(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.mu = self.sd = None

    def fit(self, X_raw, y):
        F = td_features(X_raw)
        self.mu = F.mean(axis=0); self.sd = F.std(axis=0) + 1e-8
        self.clf.fit((F - self.mu) / self.sd, np.asarray(y))
        return self

    def predict(self, X_raw):
        F = (td_features(X_raw) - self.mu) / self.sd
        return self.clf.predict(F)

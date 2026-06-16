"""Riemannian tangent-space covariance classifier for EMG grasp decoding.

WHY THIS EXISTS
A grasp is a *spatial muscle co-activation pattern*: which forearm muscles fire
together, and how strongly. The mathematical object that captures exactly that is
the per-window spatial covariance matrix of the electrode signals (a symmetric
positive-definite, SPD, matrix). Classifying these on the curved Riemannian
manifold of SPD matrices — rather than feeding raw samples to a CNN — is the
state of the art for few-shot EMG: published results reach 92-93% on NinaPro DB2
(17 gestures, 12 electrodes) with almost no per-subject training data, because the
representation is hand-built to encode the grasp signature instead of being learned
from a handful of calibration reps (Barachant et al.; "Topology of sEMG", 2023).

THE LOG-EUCLIDEAN FORMULATION (what we implement)
Working in the log-Euclidean metric makes the whole pipeline plain linear algebra:
  1. per window:  Σ = cov(channels over time) + shrinkage  (SPD)
  2. S = logm(Σ)  (matrix log via symmetric eigendecomposition)
  3. tangent vector = upper-triangle(S) with √2 on the off-diagonals (isometry)
  4. PER-USER RECENTERING = subtract the user's mean tangent vector. In the
     log-Euclidean metric this *is* Riemannian Procrustes recentering — it aligns
     the user's covariance distribution to a common reference with ZERO labels
     (the established cross-subject/-session transfer trick from BCI), so a new
     user calibrates instantly.
  5. shrinkage-LDA on the recentered tangent vectors (Ledoit-Wolf shrinkage =
     robust with only a few calibration reps and an 78-D feature).

Dependency-light: numpy + scipy(eigh via numpy) + scikit-learn's LDA. Tiny and
deployable (the "model" is a mean vector + an LDA matrix).
"""
from __future__ import annotations

import numpy as np


def _delay_embed(X: np.ndarray, delays) -> np.ndarray:
    """[N, T, C] -> [N, T, C*len(delays)] by stacking time-delayed copies.

    The "augmented covariance matrix" trick (Carrara & Congedo, 2022): stacking the
    signal with delayed versions of itself folds temporal/spectral dynamics into the
    covariance, so the SPD representation captures not just *which* muscles co-fire
    but *how the activation evolves* — a substantial accuracy boost for Riemannian
    decoding. delays=(0,) recovers the plain spatial covariance."""
    delays = list(delays)
    if delays == [0]:
        return X
    parts = []
    for d in delays:
        if d == 0:
            parts.append(X)
        else:
            parts.append(np.concatenate([np.repeat(X[:, :1], d, axis=1), X[:, :-d]], axis=1))
    return np.concatenate(parts, axis=2)


def covariances(X_raw: np.ndarray, shrink: float = 0.1, delays=(0,)) -> np.ndarray:
    """[N, T, C] windows -> [N, C', C'] shrunk SPD covariances (vectorised),
    where C' = C*len(delays) for the augmented (delay-embedded) covariance.

    Shrinkage toward a scaled identity (Ledoit-Wolf style) guarantees the matrices
    are well-conditioned and strictly positive-definite even for short windows.
    Use the RAW filtered EMG channels (amplitude carries the activation level, so
    Rest -> low-power covariance is separable too)."""
    X = _delay_embed(np.asarray(X_raw, dtype=np.float64), delays)
    N, T, C = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = np.einsum("ntc,ntd->ncd", Xc, Xc) / max(1, T - 1)        # [N, C', C']
    tr = np.einsum("ncc->n", cov) / C                              # mean eigenvalue
    eye = np.eye(C)[None]
    cov = (1.0 - shrink) * cov + shrink * tr[:, None, None] * eye
    return cov


def _logm_sym_batch(cov: np.ndarray) -> np.ndarray:
    """Matrix log of a batch of symmetric PD matrices via eigendecomposition."""
    w, V = np.linalg.eigh(cov)                                     # w:[N,C], V:[N,C,C]
    w = np.clip(w, 1e-10, None)
    return (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))


def _upper_vec(S: np.ndarray) -> np.ndarray:
    """[N, C, C] symmetric -> [N, C(C+1)/2] tangent vectors (√2 off-diagonal)."""
    N, C, _ = S.shape
    iu = np.triu_indices(C, k=1)
    diag = S[:, np.arange(C), np.arange(C)]                        # [N, C]
    off = S[:, iu[0], iu[1]] * np.sqrt(2.0)                        # [N, C(C-1)/2]
    return np.concatenate([diag, off], axis=1)


def tangent_features(X_raw: np.ndarray, shrink: float = 0.1, ref_log: np.ndarray | None = None,
                     delays=(0,)):
    """Windows -> recentered tangent-space feature vectors.

    Returns (feats, ref_log). If ref_log is None it is computed as the mean log-cov
    over the given windows (use the CALIBRATION set's ref for both cal and test so
    the recentering is consistent and leak-free)."""
    cov = covariances(X_raw, shrink=shrink, delays=delays)
    S = _logm_sym_batch(cov)                                       # [N, C, C]
    if ref_log is None:
        ref_log = S.mean(axis=0)
    Sr = S - ref_log[None]                                         # log-Euclidean recentering
    return _upper_vec(Sr), ref_log


class RiemannTSM:
    """Riemannian tangent-space classifier with per-user log-Euclidean recentering.

    fit() on a user's calibration windows establishes the recentering reference and
    a shrinkage-LDA; predict()/predict_proba() apply the SAME reference to held-out
    windows. Drop-in grasp classifier — far more sample-efficient than fine-tuning a
    CNN head on a few reps."""

    def __init__(self, shrink: float = 0.1, recenter: bool = True, delays=(0,)):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.shrink = shrink
        self.recenter = recenter
        self.delays = tuple(delays)
        self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.ref_log = None
        self.classes_ = None

    def fit(self, X_raw, y):
        feats, ref = tangent_features(X_raw, self.shrink, delays=self.delays)
        self.ref_log = ref if self.recenter else np.zeros_like(ref)
        if not self.recenter:
            feats, _ = tangent_features(X_raw, self.shrink, ref_log=self.ref_log, delays=self.delays)
        self.clf.fit(feats, y)
        self.classes_ = self.clf.classes_
        return self

    def _feats(self, X_raw):
        feats, _ = tangent_features(X_raw, self.shrink, ref_log=self.ref_log, delays=self.delays)
        return feats

    def predict(self, X_raw):
        return self.clf.predict(self._feats(X_raw))

    def predict_proba(self, X_raw):
        return self.clf.predict_proba(self._feats(X_raw))

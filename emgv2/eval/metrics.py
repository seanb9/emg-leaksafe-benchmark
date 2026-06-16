"""Metric computation — implemented directly so the unit tests can pin behaviour
against scikit-learn and against hand-worked examples. These are exactly the
places (per-class F1, balanced accuracy, majority voting) where a silent bug can
inflate a reported number, so they are tested independently of the training code.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """[n_classes, n_classes] integer confusion matrix, rows=true, cols=pred."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.diag(cm) / support
    rec[support == 0] = np.nan
    return rec


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall over classes that have support (matches sklearn)."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    rec = per_class_recall(cm)
    rec = rec[~np.isnan(rec)]
    if rec.size == 0:
        return float("nan")
    return float(np.mean(rec))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """[n_classes] F1 per class; classes with no support or no prediction -> 0."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return f1


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean F1 over all classes (matches sklearn average='macro')."""
    return float(np.mean(per_class_f1(y_true, y_pred, n_classes)))


def majority_vote_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse window predictions to one prediction per group by majority vote.

    Returns (group_true, group_pred), one entry per unique group, where group_true
    is that group's (single) true label and group_pred is the modal predicted
    label (ties broken toward the lowest class index). This is the per-repetition
    voting scheme used by EMGBench-style high LOSO numbers; reporting both window
    and voted accuracy makes the comparison honest.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    group = np.asarray(group).reshape(-1)
    groups = np.unique(group)
    gt = np.empty(groups.shape[0], dtype=np.int64)
    gp = np.empty(groups.shape[0], dtype=np.int64)
    for i, g in enumerate(groups):
        mask = group == g
        gt[i] = y_true[mask][0]
        counts = np.bincount(y_pred[mask], minlength=n_classes)
        gp[i] = int(np.argmax(counts))  # ties -> lowest index
    return gt, gp

"""Control-loop layer for the clinical track: temporal voting + confidence
rejection + false-activation rate.

A real prosthetic controller does not act on a single 256 ms window — it votes
over several consecutive windows and refuses to act when unsure (holds the current
state). This raises effective accuracy and, crucially, suppresses false
activations (firing a movement during rest), which is the metric clinicians care
about most. These functions turn per-window predictions into the numbers a real
device would deliver.
"""
from __future__ import annotations

import numpy as np


def sliding_vote(preds: np.ndarray, group: np.ndarray, window: int) -> np.ndarray:
    """Majority vote over the trailing `window` predictions, within each group.

    Voting is confined to a single contiguous segment (`group`) so it never mixes
    predictions across a gesture boundary — mirrors a real-time controller that
    votes over the last N decisions. Ties break to the lowest class index.
    """
    preds = np.asarray(preds)
    group = np.asarray(group)
    if window <= 1:
        return preds.copy()
    out = preds.copy()
    n_classes = int(preds.max()) + 1 if preds.size else 1
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        gp = preds[idx]
        for i in range(len(idx)):
            lo = max(0, i - window + 1)
            out[idx[i]] = int(np.bincount(gp[lo:i + 1], minlength=n_classes).argmax())
    return out


def rest_gate(probs: np.ndarray, rest_idx: int, gate_thr: float) -> np.ndarray:
    """Two-stage stage 1: act only when confidently NOT rest, i.e. P(rest) < gate_thr.

    A dedicated rest/active gate is more reliable than thresholding the 7-way max
    probability, because it doesn't reject a clear movement just because the model
    is split between two grasps. This is the lever that controls false-activation.
    """
    return np.asarray(probs)[:, rest_idx] < gate_thr


def grasp_prediction(probs: np.ndarray, rest_idx: int) -> np.ndarray:
    """Two-stage stage 2: which grasp — argmax over the non-rest classes."""
    p = np.asarray(probs).copy()
    p[:, rest_idx] = -np.inf
    return p.argmax(axis=1)


def hysteresis_decode(
    active_score: np.ndarray,
    grasp_pred: np.ndarray,
    group: np.ndarray,
    rest_idx: int = 0,
    *,
    enter_gate: float = 0.5,
    exit_gate: float = 0.35,
    on_frames: int = 3,
    off_frames: int = 4,
    switch_frames: int = 3,
) -> np.ndarray:
    """Asymmetric finite-state decoder for the clinical control loop.

    The dominant error in within-user EMG grasp control is NOT grasp-vs-grasp
    confusion — it is the low-activation onset / offset / mid-movement-dip windows
    of a real grasp being labelled Rest (and, the other way, isolated spurious
    movement spikes during true rest). A symmetric majority vote cannot recover
    these: at movement onset the *majority* of the trailing windows genuinely look
    like Rest, so the vote stays Rest.

    A real myoelectric controller is asymmetric and hysteretic: it is *hard to
    start* a grasp (debounce: require ``on_frames`` consecutive confident-active
    windows agreeing on the same grasp — this crushes false activations) and *easy
    to keep* one (hysteresis: only fall back to Rest after ``off_frames``
    consecutive confident-rest windows, and only switch grasp after
    ``switch_frames`` consecutive windows voting a different grasp). This holds the
    grasp through brief activation dips and recovers the boundary windows that the
    flat classifier loses to Rest.

    Decoding is confined to a single contiguous segment (``group``) and each
    segment starts in the Rest state (you do not know the grasp before the user
    moves), mirroring a streaming controller. ``active_score`` is P(active) from the
    stage-1 rest/active detector (or 1 - P(rest) of a flat model); ``grasp_pred`` is
    the stage-2 grasp class in ORIGINAL class indices.

    Returns the decoded per-window label (original class indices), same length as
    the inputs — a drop-in replacement for ``sliding_vote``.
    """
    active_score = np.asarray(active_score)
    grasp_pred = np.asarray(grasp_pred)
    group = np.asarray(group)
    out = np.full(len(active_score), rest_idx, dtype=np.int64)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)              # already temporal order
        state = rest_idx
        on_streak = 0            # consecutive active windows agreeing on cand grasp
        cand = rest_idx          # the grasp those active windows agree on
        off_streak = 0           # consecutive rest windows while holding a grasp
        sw_streak = 0            # consecutive windows voting a *different* grasp
        sw_cand = rest_idx
        for i in idx:
            av = active_score[i] >= enter_gate
            gp = int(grasp_pred[i])
            if state == rest_idx:
                if av:
                    if gp == cand:
                        on_streak += 1
                    else:
                        cand, on_streak = gp, 1
                    if on_streak >= on_frames:
                        state = cand          # commit to the grasp
                        off_streak = sw_streak = 0
                else:
                    on_streak, cand = 0, rest_idx
            else:                                     # holding a grasp
                if active_score[i] < exit_gate:
                    off_streak += 1
                    sw_streak = 0
                    if off_streak >= off_frames:
                        state = rest_idx              # release back to Rest
                        on_streak, cand = 0, rest_idx
                else:
                    off_streak = 0
                    if av and gp != state:
                        if gp == sw_cand:
                            sw_streak += 1
                        else:
                            sw_cand, sw_streak = gp, 1
                        if sw_streak >= switch_frames:
                            state = gp                # switch grasp
                            sw_streak = 0
                    else:
                        sw_streak, sw_cand = 0, rest_idx
            out[i] = state
    return out


def confidence_mask(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Bool mask: True where the device acts (max class prob >= threshold)."""
    probs = np.asarray(probs)
    if probs.ndim != 2:
        raise ValueError("probs must be [N, n_classes]")
    return probs.max(axis=1) >= threshold


def coverage(acted_mask: np.ndarray) -> float:
    """Fraction of windows where the device acts (rest of the time it holds)."""
    acted_mask = np.asarray(acted_mask)
    return float(np.mean(acted_mask)) if acted_mask.size else float("nan")


def accuracy_on_acted(y_true, preds, acted_mask) -> float:
    """Accuracy over only the windows where the device acted."""
    y_true = np.asarray(y_true); preds = np.asarray(preds); acted_mask = np.asarray(acted_mask)
    if acted_mask.sum() == 0:
        return float("nan")
    return float(np.mean(preds[acted_mask] == y_true[acted_mask]))


def mean_onset_latency(decoded, y_true, group, rest_idx: int = 0):
    """Median windows between true movement start and the decoder committing.

    For every movement group (a contiguous grasp segment), the latency is the
    number of leading windows the decoder still calls Rest before it commits to a
    grasp (the debounce cost). Returns the median over movement groups in windows
    (multiply by stride_ms for milliseconds). NaN if no movement group ever fires.
    """
    decoded = np.asarray(decoded); y_true = np.asarray(y_true); group = np.asarray(group)
    lat = []
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        if y_true[idx][0] == rest_idx:
            continue
        fired = np.flatnonzero(decoded[idx] != rest_idx)
        if fired.size:
            lat.append(int(fired[0]))
    return float(np.median(lat)) if lat else float("nan")


def decode_clinical_metrics(decoded, y_true, group, rest_idx: int = 0, n_classes=None):
    """Clinical scorecard for a decoded label stream (used per-user and in sweeps).

    Returns a dict with:
      bal_acc    : balanced accuracy over all classes (the headline 'voted' number)
      grasp_acc  : accuracy on TRUE-movement windows (was it the right grip?)
      grasp_cov  : fraction of true-movement windows the decoder marked as a grasp
      false_act  : fraction of true-REST windows that fired a movement (clinical bar)
      latency    : median onset latency in windows (responsiveness cost)
    """
    from .metrics import balanced_accuracy
    decoded = np.asarray(decoded); y_true = np.asarray(y_true)
    if n_classes is None:
        n_classes = int(max(y_true.max(), decoded.max())) + 1
    move = y_true != rest_idx
    rest = ~move
    grasp_acc = float(np.mean(decoded[move] == y_true[move])) if move.any() else float("nan")
    grasp_cov = float(np.mean(decoded[move] != rest_idx)) if move.any() else float("nan")
    false_act = float(np.mean(decoded[rest] != rest_idx)) if rest.any() else float("nan")
    return dict(
        bal_acc=balanced_accuracy(y_true, decoded, n_classes),
        grasp_acc=grasp_acc, grasp_cov=grasp_cov, false_act=false_act,
        latency=mean_onset_latency(decoded, y_true, group, rest_idx),
    )


def false_activation_rate(y_true, preds, rest_class: int = 0, acted_mask=None) -> float:
    """Fraction of true-REST windows that fire a movement (clinical false-activation).

    Denominator = all rest windows (the whole time the hand should stay still);
    numerator = rest windows where the controller acted AND predicted a movement.
    With rejection on, holding during low-confidence rest drives this down — the
    clinical bar is roughly <= 2%.
    """
    y_true = np.asarray(y_true); preds = np.asarray(preds)
    rest = y_true == rest_class
    n_rest = int(rest.sum())
    if n_rest == 0:
        return float("nan")
    if acted_mask is None:
        acted_mask = np.ones(len(y_true), dtype=bool)
    else:
        acted_mask = np.asarray(acted_mask)
    false = int((rest & acted_mask & (preds != rest_class)).sum())
    return false / n_rest

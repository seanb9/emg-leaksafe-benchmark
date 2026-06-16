"""Sequence reasoning: a Hidden-Markov 'logic' layer over the CNN's evidence.

The CNN tells us, per window, how much each class looks present (the EMISSIONS).
But it has no notion of the RULES of how grasps unfold in time. A person doesn't
flicker Power->Pinch->Power frame by frame; they rest, form a grasp, HOLD it, then
relax. This module adds that logic as a Markov TRANSITION model learned from data:

    A[i, j] = P(state j now | state i a moment ago)

Learned by counting real label transitions in the population, so it encodes exactly
"if I was here, why would I go there": near-zero for grasp->different-grasp without
passing through rest, high for staying put (holding), rest as the hub.

Decoding combines evidence with logic. Two modes:
  * causal_forward_decode — online belief update (alpha_t uses only windows 0..t).
    Real-time, deployable, leak-free. This is the honest per-window 'thinking'.
  * viterbi_decode — most likely WHOLE-sequence path (offline upper bound).

A window whose emission is weak (an onset ramp) is held at the current grasp because
jumping out and back is implausible under A — so the boundary windows are recovered
by LOGIC, not smoothing.
"""
from __future__ import annotations

import numpy as np


def _logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def class_priors(label_seqs, n_classes, *, smoothing=1.0):
    """Empirical class frequencies P(j) over the training population (Laplace-
    smoothed), for the scaled-likelihood emission correction."""
    counts = np.full(n_classes, smoothing, dtype=np.float64)
    for seq in label_seqs:
        seq = np.asarray(seq)
        if seq.size:
            np.add.at(counts, seq, 1.0)
    return counts / counts.sum()


def learn_transition_structure(label_seqs, n_classes, *, smoothing=0.5):
    """Learn WHICH transitions are plausible (the off-diagonal 'logic'), separately
    from how sticky the chain is.

    Counting raw transitions on 64 ms-stride data yields self-transitions ~0.999
    (segments are long), which makes a decoder ignore the evidence and lock up. So
    we return only the OFF-diagonal structure O (each row a distribution over where
    you go GIVEN that you leave the current state) plus the start distribution pi.
    The actual hold-probability is applied later via build_transition(), as a knob.

    O encodes the real logic: rest->grasp onsets, grasp->rest releases, and near-zero
    grasp->different-grasp (you return through rest). Rows of O sum to 1 (or are 0 if
    a state was never left)."""
    C = np.full((n_classes, n_classes), smoothing, dtype=np.float64)
    pi = np.full(n_classes, smoothing, dtype=np.float64)
    for seq in label_seqs:
        seq = np.asarray(seq)
        if seq.size == 0:
            continue
        pi[seq[0]] += 1.0
        if seq.size > 1:
            np.add.at(C, (seq[:-1], seq[1:]), 1.0)
    O = C.copy()
    np.fill_diagonal(O, 0.0)
    rs = O.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    O /= rs
    pi /= pi.sum()
    return O, pi


def build_transition(O, hold):
    """Combine the learned off-diagonal logic O with a chosen hold-probability.

    A[i,i] = hold;  A[i,j!=i] = (1-hold) * O[i,j].  hold near 1 = very sticky
    (strong holding, recovers dips/onsets but slow to switch); lower hold lets the
    evidence move it more readily. This is the one knob worth sweeping."""
    O = np.asarray(O, dtype=np.float64)
    n = O.shape[0]
    A = (1.0 - hold) * O
    np.fill_diagonal(A, 0.0)
    A[np.diag_indices(n)] = hold
    # rows where the state was never left (O row all zero) -> pure hold
    bad = A.sum(axis=1) <= 0
    A[bad] = 0.0
    A[bad, np.flatnonzero(bad)] = 1.0
    A /= A.sum(axis=1, keepdims=True)
    return A


def _log_emissions(emis, priors, eps):
    """Convert classifier POSTERIORS P(j|x) to scaled emission LIKELIHOODS for the
    HMM. A softmax head outputs posteriors; the HMM forward pass requires
    likelihoods P(x|j). By Bayes, P(x|j) ∝ P(j|x)/P(j) (the hybrid NN-HMM scaled-
    likelihood trick, Bourlard & Morgan), so we divide by the empirical class priors
    P(j). Without this the emissions are mismatched with the transition matrix and
    predictions are biased toward frequent classes. priors=None falls back to the
    posterior (an ablation)."""
    logE = np.log(np.asarray(emis, dtype=np.float64) + eps)
    if priors is not None:
        logE = logE - np.log(np.asarray(priors, dtype=np.float64) + eps)[None, :]
    return logE


def causal_forward_decode(emis, groups, A, pi, eps=1e-8, priors=None):
    """Online HMM filtering: per window, argmax of the belief using ONLY the past.

    emis: [N, K] per-window class posteriors (the CNN's evidence). priors: [K]
    empirical class frequencies P(j) from the training population; when given,
    posteriors are converted to scaled likelihoods (see _log_emissions). Returns
    per-window decoded states [N]. Confined to each contiguous segment (group)."""
    emis = np.asarray(emis, dtype=np.float64)
    groups = np.asarray(groups)
    K = emis.shape[1]
    logA = np.log(A + eps); logpi = np.log(pi + eps); logE = _log_emissions(emis, priors, eps)
    out = np.zeros(len(emis), dtype=np.int64)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)               # temporal order
        alpha = logpi + logE[idx[0]]                    # belief at t0
        out[idx[0]] = int(alpha.argmax())
        for t in idx[1:]:
            # alpha_t(j) = logE[t,j] + logsumexp_i( alpha_{t-1}(i) + logA[i,j] )
            alpha = logE[t] + _logsumexp(alpha[:, None] + logA, axis=0)
            alpha -= alpha.max()                        # keep numerically bounded
            out[t] = int(alpha.argmax())
    return out


def viterbi_decode(emis, groups, A, pi, eps=1e-8, priors=None):
    """Most-likely whole-sequence path per segment (offline upper bound)."""
    emis = np.asarray(emis, dtype=np.float64)
    groups = np.asarray(groups)
    K = emis.shape[1]
    logA = np.log(A + eps); logpi = np.log(pi + eps); logE = _log_emissions(emis, priors, eps)
    out = np.zeros(len(emis), dtype=np.int64)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        L = len(idx)
        delta = logpi + logE[idx[0]]
        back = np.zeros((L, K), dtype=np.int64)
        for ti in range(1, L):
            scores = delta[:, None] + logA            # [i, j]
            back[ti] = scores.argmax(axis=0)
            delta = logE[idx[ti]] + scores.max(axis=0)
        s = int(delta.argmax())
        out[idx[L - 1]] = s
        for ti in range(L - 1, 0, -1):
            s = int(back[ti][s])
            out[idx[ti - 1]] = s
    return out

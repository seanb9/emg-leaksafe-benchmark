"""Causal temporal-memory head — give the memoryless window classifier a 'brain'.

The per-window CNN decides each 256 ms window in isolation, so the low-activation
ONSET/OFFSET windows of a grasp (the muscle hasn't fully contracted yet) look like
Rest and are lost — the single biggest cap on the honest per-window accuracy.

A real controller — like a brain — uses CONTEXT: having just seen rest with
activation now rising in a particular muscle pattern, it recognises "this is the
onset of grasp X" before the contraction is complete. This module adds exactly that:
a small UNIDIRECTIONAL (causal) GRU on top of the frozen 28K backbone's per-window
embeddings. At timestep t its output depends only on inputs 0..t, so it is
real-time, deployable on the same MCU (a GRU over a 128-d embedding is a few K
params), and leak-free — it never sees the future or any label.

It is trained on the population (the 'experience' that builds the brain) and
fine-tuned on the user's calibration sequences. Operates on contiguous segments
(groups) in temporal order, which is how a streaming device sees the signal.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMemory(nn.Module):
    def __init__(self, d_in, n_classes, hidden=64, layers=1, p_drop=0.3):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, layers, batch_first=True)   # unidirectional = causal
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):                       # x: [B, L, D]
        out, _ = self.gru(x)                    # out[t] depends only on x[0..t]
        return self.head(self.drop(out))        # [B, L, n_classes]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def group_sequences(emb, y, groups):
    """Split per-window embeddings into per-segment temporal sequences.

    Windows within a group are already in temporal order (sorted indices), so each
    group becomes one ordered sequence. Returns (list[seq_emb], list[seq_y],
    list[idx]) where idx maps each sequence position back to the row in `emb`."""
    seqs, labs, idxs = [], [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        seqs.append(emb[idx]); labs.append(y[idx] if y is not None else None); idxs.append(idx)
    return seqs, labs, idxs


def _pad_batch(seqs, labs, device):
    L = max(len(s) for s in seqs)
    B, D = len(seqs), seqs[0].shape[1]
    X = torch.zeros(B, L, D, dtype=torch.float32)
    Y = torch.full((B, L), -100, dtype=torch.long)         # -100 = ignore_index
    M = torch.zeros(B, L, dtype=torch.bool)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        n = len(s)
        X[i, :n] = torch.from_numpy(np.asarray(s, dtype=np.float32))
        if l is not None:
            Y[i, :n] = torch.from_numpy(np.asarray(l, dtype=np.int64))
        M[i, :n] = True
    return X.to(device), Y.to(device), M.to(device)


def train_temporal(brain, emb, y, groups, device, *, epochs=30, lr=1e-3, bs=32,
                   weight=None, seed=0, warm=None, log=lambda *_: None):
    """Train (or fine-tune from `warm`) the temporal brain on grouped sequences."""
    rng = np.random.default_rng(seed)
    if warm is not None:
        brain.load_state_dict(warm)
    seqs, labs, _ = group_sequences(emb, y, groups)
    order = np.arange(len(seqs))
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    w = None if weight is None else torch.tensor(weight, dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    brain.train()
    for _ in range(epochs):
        rng.shuffle(order)
        for s in range(0, len(order), bs):
            batch = [seqs[i] for i in order[s:s + bs]]
            blab = [labs[i] for i in order[s:s + bs]]
            X, Y, _ = _pad_batch(batch, blab, device)
            logits = brain(X)                                    # [B, L, K]
            loss = ce(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    brain.eval()
    return brain


@torch.no_grad()
def predict_temporal(brain, emb, groups, n_classes, device):
    """Causal per-window prediction: returns (preds[N], logits[N, K]) in row order."""
    seqs, _, idxs = group_sequences(emb, None, groups)
    N = emb.shape[0]
    logits_full = np.zeros((N, n_classes), dtype=np.float32)
    brain.eval()
    for s, idx in zip(seqs, idxs):
        x = torch.from_numpy(np.asarray(s, dtype=np.float32))[None].to(device)   # [1, L, D]
        lg = brain(x)[0].cpu().numpy()                                            # [L, K]
        logits_full[idx] = lg
    return logits_full.argmax(1), logits_full

"""Phase-Aware Predictive Brain — a generative (grasp x phase) state-space decoder.

THE NOVEL IDEA
Every decoder in myoelectric control treats a grasp as a single static pattern, so
the low-activation ONSET of a grasp (the muscle still ramping) gets read as Rest —
the wall that caps instantaneous accuracy. But a grasp is not a state, it is a
TRAJECTORY WITH A PHASE: onset -> hold -> release. A human brain does not ask "is
this a full Power grasp?"; it asks "where am I inside a movement, and which movement
is this becoming?" An onset window is then perfectly meaningful — not "weak Power"
but "the BEGINNING of Power", which differs from the beginning of Pinch.

So we decode intent as joint Bayesian inference over (grasp g, phase p) under a
learned generative model of how a grasp unfolds:

  * STATES = Rest + {each grasp} x {onset, hold, release}.
  * PHASE SUPERVISION FOR FREE: within each training segment, position gives the
    phase (first frac = onset, last frac = release, middle = hold). No new labels.
    The emission head therefore learns "Power-onset" as its own concept.
  * MOVEMENT GRAMMAR (the logic + the trajectory): you may only ENTER a grasp through
    its onset, progress onset->hold->release, and leave release->rest. Direct
    grasp->other-grasp is forbidden. This is the structured transition matrix.
  * CAUSAL BAYESIAN FILTERING over the states -> infer (g, p) per window, online,
    using only the past -> real-time, deployable, leak-free.
  * ANTICIPATION: because (g, onset) is its own recognised state and the grammar
    enters grasps through onset, a ramp window is decoded as (g, onset) and the
    grasp is committed EARLY -> lower latency, the novel clinical win.

This subsumes the flat HMM (n_phase=1) and is the 'brain that interprets where it is
in a movement'. Honest positioning: transient/onset EMG classification exists; the
novelty is the unified generative phase-state formulation with anticipatory causal
decoding and auto-derived phase supervision.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from ..eval.sequence import causal_forward_decode, viterbi_decode


class PhaseStateSpace:
    """Maps between (grasp, phase) and a flat state index. State 0 is Rest."""

    def __init__(self, n_classes, rest_idx, n_phase=3):
        self.n_classes = n_classes
        self.rest_idx = rest_idx
        self.n_phase = n_phase
        self.grasps = [c for c in range(n_classes) if c != rest_idx]
        self.states = [("rest", -1)]
        for g in self.grasps:
            for p in range(n_phase):
                self.states.append((g, p))
        self.n_states = len(self.states)
        self._idx = {s: i for i, s in enumerate(self.states)}
        self.rest_state = self._idx[("rest", -1)]
        self.grasp_of = np.array([rest_idx if s[0] == "rest" else s[0] for s in self.states])
        self.phase_of = np.array([s[1] for s in self.states])

    def state(self, grasp, phase):
        if grasp == self.rest_idx:
            return self.rest_state
        return self._idx[(grasp, min(phase, self.n_phase - 1))]


def derive_phase_states(y, group, ss: PhaseStateSpace, *, onset_frac=0.25, release_frac=0.25):
    """Auto-derive (grasp, phase) state labels from position within each segment."""
    y = np.asarray(y); group = np.asarray(group)
    out = np.zeros(len(y), dtype=np.int64)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)               # temporal order
        c = int(y[idx][0])
        if c == ss.rest_idx:
            out[idx] = ss.rest_state
            continue
        L = len(idx)
        pos = np.arange(L) / max(1, L - 1)
        for k, t in enumerate(idx):
            if ss.n_phase == 1:
                ph = 0
            elif pos[k] < onset_frac:
                ph = 0
            elif pos[k] >= 1.0 - release_frac:
                ph = ss.n_phase - 1
            else:
                ph = 1
            out[t] = ss.state(c, ph)
    return out


def _allowed_mask(ss: PhaseStateSpace):
    n = ss.n_states
    allowed = np.zeros((n, n), dtype=bool)
    r = ss.rest_state
    allowed[r, r] = True
    for g in ss.grasps:
        allowed[r, ss.state(g, 0)] = True            # enter through onset
        for p in range(ss.n_phase):
            sp = ss.state(g, p)
            allowed[sp, sp] = True                    # hold the phase
            if p + 1 < ss.n_phase:
                allowed[sp, ss.state(g, p + 1)] = True   # advance phase
            else:
                allowed[sp, r] = True                 # release -> rest
    return allowed


def build_phase_transition(ss: PhaseStateSpace, hold=0.9, learned=None, smoothing=0.5,
                           rest_hold=None):
    """Structured transition matrix: only grammar-allowed moves, off-diagonal mass
    shaped by learned counts, diagonal set to the tunable hold-probability.

    rest_hold (default = hold) is the SEPARATE probability of staying in Rest. Making
    it high is the clinical safety knob: it makes the brain reluctant to leave rest,
    which crushes false activation (entering a grasp must be a deliberate, sustained
    event), traded against onset latency. This is the anticipation-vs-safety dial."""
    n = ss.n_states
    if rest_hold is None:
        rest_hold = hold
    allowed = _allowed_mask(ss)
    C = np.where(allowed, smoothing, 0.0)
    if learned is not None:
        C = C + np.where(allowed, learned, 0.0)
    A = np.zeros((n, n))
    for i in range(n):
        h = rest_hold if i == ss.rest_state else hold
        off = C[i].copy(); off[i] = 0.0
        ssum = off.sum()
        if ssum <= 0:
            A[i, i] = 1.0
        else:
            A[i] = (1.0 - h) * off / ssum
            A[i, i] = h
    A /= A.sum(axis=1, keepdims=True)
    pi = np.zeros(n); pi[ss.rest_state] = 1.0          # movements start from rest
    return A, pi


def count_phase_transitions(states, group, n_states):
    C = np.zeros((n_states, n_states))
    for g in np.unique(group):
        seq = states[group == g]
        if seq.size > 1:
            np.add.at(C, (seq[:-1], seq[1:]), 1.0)
    return C


class PhaseHead(nn.Module):
    """Emission model: backbone embedding -> P(state). The 'interpreter' that names
    what it sees, including phase ('this is Power-onset')."""

    def __init__(self, d, n_states, hidden=128, p_drop=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(d), nn.Dropout(p_drop),
            nn.Linear(d, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_states),
        )

    def forward(self, x):
        return self.net(x)


def _train_head(head, emb, states, device, *, epochs, lr, bs=256, weight=None,
                warm=None, seed=0):
    if warm is not None:
        head.load_state_dict(warm)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    w = None if weight is None else torch.tensor(weight, dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=w)
    E = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
    Y = torch.from_numpy(np.asarray(states, dtype=np.int64)).to(device)
    n = len(emb)
    head.train()
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            c = idx[s:s + bs]
            if len(c) < 2:
                continue
            ci = torch.as_tensor(c, device=device)
            loss = ce(head(E.index_select(0, ci)), Y.index_select(0, ci))
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def _head_proba(head, emb, device, bs=1024):
    head.eval()
    out = []
    E = torch.from_numpy(np.asarray(emb, dtype=np.float32))
    for s in range(0, len(emb), bs):
        out.append(torch.softmax(head(E[s:s + bs].to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


class PhaseBrain:
    """The full phase-aware predictive decoder. fit_population builds the emission
    head + grammar from population data; predict_user fine-tunes the head on a user's
    calibration and decodes their test stream causally."""

    def __init__(self, d, n_classes, rest_idx, *, n_phase=3, hidden=128, hold=0.9,
                 rest_hold=None, onset_frac=0.25, release_frac=0.25, device="cpu", seed=0):
        self.ss = PhaseStateSpace(n_classes, rest_idx, n_phase)
        self.head = PhaseHead(d, self.ss.n_states, hidden).to(device)
        self.device = device
        self.hold = hold
        self.rest_hold = rest_hold if rest_hold is not None else hold
        self.onset_frac = onset_frac
        self.release_frac = release_frac
        self.seed = seed
        self.A = self.pi = self._pop = self.learned = None

    def _states(self, y, group):
        return derive_phase_states(y, group, self.ss,
                                   onset_frac=self.onset_frac, release_frac=self.release_frac)

    def fit_population(self, emb, y, group, *, epochs, lr=1e-3, weight=None):
        states = self._states(y, group)
        _train_head(self.head, emb, states, self.device, epochs=epochs, lr=lr,
                    weight=weight, seed=self.seed)
        self._pop = copy.deepcopy(self.head.state_dict())
        self.learned = count_phase_transitions(states, np.asarray(group), self.ss.n_states)
        self.A, self.pi = build_phase_transition(
            self.ss, hold=self.hold, rest_hold=self.rest_hold, learned=self.learned)
        return self

    def rebuild(self, hold=None, rest_hold=None):
        """Rebuild the transition matrix with new stickiness knobs (for sweeping)."""
        if hold is not None:
            self.hold = hold
        if rest_hold is not None:
            self.rest_hold = rest_hold
        self.A, self.pi = build_phase_transition(
            self.ss, hold=self.hold, rest_hold=self.rest_hold, learned=self.learned)
        return self.A, self.pi

    def predict_user(self, emb_cal, y_cal, g_cal, emb_test, g_test, *, epochs, lr,
                     weight=None, decode="causal"):
        head = copy.deepcopy(self.head)
        states = self._states(y_cal, g_cal)
        _train_head(head, emb_cal, states, self.device, epochs=epochs, lr=lr,
                    weight=weight, warm=self._pop, seed=self.seed)
        emis = _head_proba(head, emb_test, self.device)        # [N, n_states]
        dec = viterbi_decode if decode == "viterbi" else causal_forward_decode
        st = dec(emis, np.asarray(g_test), self.A, self.pi)
        grasp = self.ss.grasp_of[st]                            # collapse to grasp index
        return grasp, st, emis

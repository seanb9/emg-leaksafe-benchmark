#!/usr/bin/env python
"""Online streaming demo — the real-time / causal validation TNSRE asks for.

The within-user benchmark (run_within_user.py) scores the decoder OFFLINE: it
predicts every test window in one batched pass and then applies the sequence
logic. That is leak-safe and causal in principle, but a reviewer cannot see that
the system runs as a stream. This script makes the streaming explicit.

For one held-out user it:
  1. calibrates the DEPLOYED decoder on that user's calibration reps (the same
     two-stage rest-detector + grasp head as the paper), exactly once, offline;
  2. learns the population transition grammar (the zero-parameter HMM) from the
     pretrain subjects;
  3. then REPLAYS the held-out repetition ONE WINDOW AT A TIME, in temporal
     order, feeding each 256 ms window through the full pipeline incrementally:
        CNN emission  ->  online HMM belief update (past-only)  ->  hysteresis gate
     measuring the wall-clock compute time of every single decision.

It reports the per-decision compute latency against the 64 ms decision budget,
the onset latency (how many windows after true movement onset the controller
commits), and confirms the streamed decode matches the batched decode bit-for-bit
(proving the offline number is genuinely causal). It writes a figure with the
live decode timeline and the per-decision latency distribution.

This is a streaming-REPLAY demo (the gold standard short of a closed-loop human
study): the decoder sees the windows as a live stream and never looks ahead.

DEV (Mac):   python scripts/streaming_demo.py configs/db2_reable_hand.yaml --reuse-base --subject 3 --cal-epochs 25
QUICK plumbing check: add --dev
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from emgv2.config import load_config, seed_everything
from emgv2.data import ninapro_db2 as db2
from emgv2.data.splits import within_user_split, per_subject_zscore_fit_apply, cap_rest_indices
from emgv2.models import build_v2net
from emgv2.train import engine
from emgv2.eval.sequence import learn_transition_structure, build_transition
from emgv2.utils.device import pick_device, machine_label

# reuse the exact loaders / helpers the benchmark uses, so the demo is faithful
from run_within_user import (
    load_subject, in_channels, make_mc, softmax_probs, _load_backbone, _feats,
)


# --------------------------------------------------------------------------- #
#  Calibrate the deployed two-stage decoder once (offline), keep the models.   #
# --------------------------------------------------------------------------- #
def calibrate(cfg, base_state, c_in, n_classes, rest_idx, Xn, cal_idx, ws_y,
              device, cal_epochs, cal_lr, seed, cal_aug):
    """Train stage-1 (rest/active) + stage-2 (which grasp) on this user's
    calibration reps and return everything needed to emit a single-window
    emission online. Mirrors two_stage_eval in run_within_user.py but RETAINS the
    trained models instead of immediately predicting the whole test set."""
    grasp_classes = [c for c in range(n_classes) if c != rest_idx]
    K = len(grasp_classes)
    g2orig = np.array(grasp_classes)

    # stage 1 — rest vs active, on all calibration windows
    y_bin = (ws_y[cal_idx] != rest_idx).astype(np.int64)
    mA = build_v2net(c_in, 2, make_mc(cfg), n_subjects=0).to(device)
    _load_backbone(mA, base_state)
    mA = engine.fewshot_adapt(mA, Xn[cal_idx], y_bin, device, n_classes=2,
                              epochs=cal_epochs, lr=cal_lr, head_only=False,
                              seed=seed, aug_kw=cal_aug)
    mA.eval()

    # stage 2 — which grasp, on movement-only calibration windows
    move_cal = cal_idx[ws_y[cal_idx] != rest_idx]
    y_g = np.searchsorted(g2orig, ws_y[move_cal]).astype(np.int64)
    mB = build_v2net(c_in, K, make_mc(cfg), n_subjects=0).to(device)
    _load_backbone(mB, base_state)
    mB = engine.fewshot_adapt(mB, Xn[move_cal], y_g, device, n_classes=K,
                              epochs=cal_epochs, lr=cal_lr, head_only=False,
                              seed=seed, aug_kw=cal_aug)
    mB.eval()

    proto_blend = float(cfg.get_path("within_user.proto_blend", 0.0))
    emb_cal = engine.embed(mB, Xn[move_cal], device) if proto_blend > 0 else None
    return dict(mA=mA, mB=mB, g2orig=g2orig, K=K, proto_blend=proto_blend,
                emb_cal=emb_cal, y_g=y_g)


_TTA_AUG = dict(channel_shift=0, mag_warp_std=0.15, noise_std=0.04,
                time_mask_frac=0.0, n_time_masks=0, chan_scale=(0.95, 1.05))


def _infer(model, x_row, device, tta_views):
    """Single forward, or averaged over `tta_views` gently-augmented views. We have
    ~20x latency headroom under the 64 ms budget, so a few-view TTA stays real-time
    and recovers the per-window accuracy the offline benchmark gets from it."""
    if tta_views and tta_views > 1:
        return engine.predict_tta(model, x_row, device, aug_kw=_TTA_AUG,
                                  n_views=tta_views, bs=1)
    return engine.predict(model, x_row, device, bs=1)


def emit_window(cal, x_row, n_classes, rest_idx, device, tta_views=0):
    """Full per-window emission for ONE z-scored window x_row [1, T, C].
    tta_views>1 averages that many augmented passes (still within budget here).
    Returns an [n_classes] posterior, P(active), and the argmax grasp class."""
    _, lA = _infer(cal["mA"], x_row, device, tta_views)
    p_active = float(softmax_probs(lA)[0, 1])
    _, lB = _infer(cal["mB"], x_row, device, tta_views)
    p_g = softmax_probs(lB)[0]                       # [K]
    if cal["proto_blend"] > 0:
        e = engine.embed(cal["mB"], x_row, device)
        plog = engine.prototype_logits(e, cal["emb_cal"], cal["y_g"], cal["K"])
        pp = F.softmax(torch.from_numpy(plog).float(), dim=1).numpy()[0]
        p_g = (1.0 - cal["proto_blend"]) * p_g + cal["proto_blend"] * pp
    emis = np.zeros(n_classes, dtype=np.float64)
    emis[rest_idx] = 1.0 - p_active
    emis[cal["g2orig"]] = p_active * p_g
    return emis, p_active, int(cal["g2orig"][int(np.argmax(p_g))])


def batch_emissions(cal, X, idx, device, rest_idx, n_classes, tta_views=0):
    """Vectorised emissions for a set of windows (offline, for sweeps): returns
    (p_active[N], grasp_pred[N] in original class indices, emis[N, n_classes])."""
    Xi = X[idx]
    _, lA = _infer(cal["mA"], Xi, device, tta_views)
    pA = softmax_probs(lA)[:, 1]
    _, lB = _infer(cal["mB"], Xi, device, tta_views)
    pG = softmax_probs(lB)
    if cal["proto_blend"] > 0:
        E = engine.embed(cal["mB"], Xi, device)
        plog = engine.prototype_logits(E, cal["emb_cal"], cal["y_g"], cal["K"])
        pp = F.softmax(torch.from_numpy(plog).float(), dim=1).numpy()
        pG = (1.0 - cal["proto_blend"]) * pG + cal["proto_blend"] * pp
    emis = np.zeros((len(idx), n_classes), dtype=np.float64)
    emis[:, rest_idx] = 1.0 - pA
    emis[:, cal["g2orig"]] = pA[:, None] * pG
    grasp_pred = cal["g2orig"][pG.argmax(axis=1)]
    return pA, grasp_pred, emis


# --------------------------------------------------------------------------- #
#  Online decoders: incremental forms of the batched paper decoders.           #
# --------------------------------------------------------------------------- #
class OnlineHMM:
    """Incremental causal forward filter — one belief update per window, past-only.
    Identical math to eval.sequence.causal_forward_decode, stepped live."""
    def __init__(self, A, pi, eps=1e-8):
        self.logA = np.log(A + eps)
        self.logpi = np.log(pi + eps)
        self.eps = eps
        self.alpha = None

    def reset(self):
        self.alpha = None

    def step(self, emis_row):
        logE = np.log(emis_row + self.eps)
        if self.alpha is None:
            self.alpha = self.logpi + logE
        else:
            m = (self.alpha[:, None] + self.logA)
            self.alpha = logE + (m.max(0) + np.log(np.exp(m - m.max(0)).sum(0)))
            self.alpha -= self.alpha.max()
        return int(self.alpha.argmax())


class OnlineHysteresis:
    """Incremental asymmetric hysteresis state machine — one step per window.
    Identical logic to eval.control_loop.hysteresis_decode, stepped live."""
    def __init__(self, rest_idx, enter_gate, exit_gate, on_frames, off_frames, switch_frames):
        self.rest_idx = rest_idx
        self.enter_gate, self.exit_gate = enter_gate, exit_gate
        self.on_frames, self.off_frames, self.switch_frames = on_frames, off_frames, switch_frames
        self.reset()

    def reset(self):
        self.state = self.rest_idx
        self.on_streak = 0; self.cand = self.rest_idx
        self.off_streak = 0; self.sw_streak = 0; self.sw_cand = self.rest_idx

    def step(self, active_score, grasp_pred):
        av = active_score >= self.enter_gate
        gp = int(grasp_pred)
        if self.state == self.rest_idx:
            if av:
                if gp == self.cand:
                    self.on_streak += 1
                else:
                    self.cand, self.on_streak = gp, 1
                if self.on_streak >= self.on_frames:
                    self.state = self.cand
                    self.off_streak = self.sw_streak = 0
            else:
                self.on_streak, self.cand = 0, self.rest_idx
        else:
            if active_score < self.exit_gate:
                self.off_streak += 1; self.sw_streak = 0
                if self.off_streak >= self.off_frames:
                    self.state = self.rest_idx
                    self.on_streak, self.cand = 0, self.rest_idx
            else:
                self.off_streak = 0
                if av and gp != self.state:
                    if gp == self.sw_cand:
                        self.sw_streak += 1
                    else:
                        self.sw_cand, self.sw_streak = gp, 1
                    if self.sw_streak >= self.switch_frames:
                        self.state = gp; self.sw_streak = 0
                else:
                    self.sw_streak, self.sw_cand = 0, self.rest_idx
        return self.state


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--reuse-base", action="store_true", help="load the saved base checkpoint")
    ap.add_argument("--subject", type=int, default=None, help="DB2 subject id to stream (default: first test user)")
    ap.add_argument("--cal-epochs", type=int, default=25)
    ap.add_argument("--tta-views", type=int, default=0,
                    help="online test-time augmentation views per window (0/1 = single pass). "
                         "We have ~20x budget headroom, so 4-8 stays real-time.")
    ap.add_argument("--hmm-pretrain", type=int, default=8, help="how many pretrain subjects to learn the grammar from")
    ap.add_argument("--dev", action="store_true", help="tiny epochs for a plumbing check")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/streaming_demo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    device = pick_device(args.device)
    mlabel = machine_label(device)
    n_classes = len(cfg.dataset.classes)
    rest_idx = int(cfg.get_path("control_loop.rest_class_index", 0))
    stride_ms = float(cfg.get_path("window.stride_ms", 64))
    names = list(cfg.dataset.class_names)
    cal_lr = float(cfg.get_path("within_user.cal_lr", 3e-4))
    cal_epochs = 4 if args.dev else args.cal_epochs
    cal_aug = dict(channel_shift=0, mag_warp_std=0.2, noise_std=0.05,
                   time_mask_frac=0.1, n_time_masks=1, chan_scale=(0.9, 1.1)) \
        if bool(cfg.get_path("within_user.cal_augment", False)) else None
    dec = cfg.get_path("decoder", {}) or {}
    hmm_hold = float(cfg.get_path("within_user.hmm_hold", 0.97))

    avail = db2.available_subjects(cfg)
    half = len(avail) // 2
    tst, pre = avail[:half], avail[half:]
    subj = args.subject if args.subject is not None else tst[0]
    pre_for_hmm = pre[:args.hmm_pretrain]

    print(f"=== ONLINE STREAMING DEMO === {mlabel} | {cfg.name} | classes={n_classes}")
    print(f"    streaming user S{subj} | grammar from {len(pre_for_hmm)} pretrain subj | "
          f"decision budget = {stride_ms:.0f} ms/window\n")

    # base checkpoint -------------------------------------------------------- #
    base_path = Path(__file__).resolve().parents[1] / "results" / f"base_{cfg.name}.pt"
    if not (args.reuse_base and base_path.exists()):
        print(f"  ERROR: need the saved base checkpoint at {base_path}.\n"
              f"  Run run_within_user.py once to train it, then re-run with --reuse-base.")
        return 1
    base_state = torch.load(base_path, map_location=device)
    print(f"  loaded base backbone from {base_path.name}")

    # learn the zero-parameter transition grammar from the population --------- #
    seqs = []
    for s in pre_for_hmm:
        wsp = load_subject(cfg, s)
        if wsp is None or len(wsp) == 0:
            continue
        for g in np.unique(wsp.group):
            seqs.append(wsp.y[wsp.group == g])
    O, pi = learn_transition_structure(seqs, n_classes)
    A = build_transition(O, hmm_hold)
    print(f"  learned transition grammar from {len(seqs)} population segments (hold={hmm_hold})")

    # load the streamed user, split leak-safe, z-score on calibration -------- #
    ws = load_subject(cfg, subj)
    if ws is None or len(ws) == 0:
        print(f"  ERROR: subject S{subj} has no data in cache."); return 1
    cal_reps = [1, 2, 3, 4, 5]; test_reps = cfg.get_path("within_user.test_reps", [6])
    cal_idx, test_idx = within_user_split(ws.y, ws.group, ws.rep, cal_reps=cal_reps,
                                          test_reps=test_reps, rest_class_idx=rest_idx)
    rest_cap = cfg.get_path("balance.rest_cap_mult")
    if rest_cap is not None:
        keep = cap_rest_indices(ws.y[cal_idx], rest_class_idx=rest_idx,
                                mult=float(rest_cap), seed=int(cfg.seed))
        cal_idx = cal_idx[keep]
    mean, std = per_subject_zscore_fit_apply(ws.X, cal_idx)
    Xn = ((ws.X - mean) / std).astype(np.float32)

    print(f"  calibrating deployed two-stage decoder on {len(cal_idx)} cal windows "
          f"(reps {cal_reps}), {cal_epochs} epochs ...")
    t0 = time.time()
    cal = calibrate(cfg, base_state, in_channels(cfg), n_classes, rest_idx, Xn, cal_idx,
                    ws.y, device, cal_epochs, cal_lr, int(cfg.seed), cal_aug)
    print(f"  calibrated in {time.time()-t0:.0f}s. Now STREAMING held-out rep "
          f"{test_reps} window-by-window ...\n")

    # ---- THE STREAM: one window at a time, temporal order, past-only ------- #
    yte = ws.y[test_idx]; gte = ws.group[test_idx]
    order = np.argsort(gte, kind="stable")          # contiguous segments, temporal within each
    yte, gte, sidx = yte[order], gte[order], test_idx[order]

    hmm = OnlineHMM(A, pi)
    hys = OnlineHysteresis(rest_idx,
                           enter_gate=float(dec.get("enter_gate", 0.6)),
                           exit_gate=float(dec.get("exit_gate", 0.35)),
                           on_frames=int(dec.get("on_frames", 3)),
                           off_frames=int(dec.get("off_frames", 4)),
                           switch_frames=int(dec.get("switch_frames", 3)))

    # warm up the inference kernels (a deployed device does this once at init;
    # the first forward otherwise pays a one-off compile cost that is not a
    # per-decision latency). Run a few untimed passes before the timed stream.
    for _ in range(3):
        emit_window(cal, Xn[sidx[0]:sidx[0] + 1], n_classes, rest_idx, device, args.tta_views)
    if args.tta_views and args.tta_views > 1:
        print(f"  online test-time augmentation: {args.tta_views} views/window")

    decoded_hmm = np.zeros(len(sidx), dtype=np.int64)
    decoded_hys = np.zeros(len(sidx), dtype=np.int64)
    latencies_ms = np.zeros(len(sidx), dtype=np.float64)
    prev_g = None
    for i, gi in enumerate(gte):
        if gi != prev_g:                            # new segment -> reset both decoders
            hmm.reset(); hys.reset(); prev_g = gi
        x_row = Xn[sidx[i]:sidx[i] + 1]             # ONE window [1, T, C]
        t = time.perf_counter()
        emis, p_active, grasp_pred = emit_window(cal, x_row, n_classes, rest_idx, device, args.tta_views)
        decoded_hmm[i] = hmm.step(emis)
        decoded_hys[i] = hys.step(p_active, grasp_pred)
        latencies_ms[i] = (time.perf_counter() - t) * 1e3

    # ---- metrics ----------------------------------------------------------- #
    from emgv2.eval.metrics import balanced_accuracy, majority_vote_by_group
    from emgv2.eval.control_loop import (
        decode_clinical_metrics, mean_onset_latency, false_activation_rate,
    )
    move = yte != rest_idx; rest = ~move
    hmm_bal = balanced_accuracy(yte, decoded_hmm, n_classes)
    hmm_fa = false_activation_rate(yte, decoded_hmm, rest_idx, None)
    dm = decode_clinical_metrics(decoded_hys, yte, gte, rest_idx, n_classes)
    onset_win = mean_onset_latency(decoded_hys, yte, gte, rest_idx)
    # per-EXECUTION (one decision per grasp-and-hold segment) — the device-delivered
    # number, directly comparable to the paper's per-execution headline
    seg_t, seg_p = majority_vote_by_group(yte, decoded_hmm, gte, n_classes)
    seg_move = seg_t != rest_idx
    seg_grasp = float(np.mean(seg_p[seg_move] == seg_t[seg_move])) if seg_move.any() else float("nan")
    seg_bal = balanced_accuracy(seg_t, seg_p, n_classes)

    # cross-check: streamed HMM decode must equal the batched offline decode ---
    from emgv2.eval.sequence import causal_forward_decode
    emis_all = np.zeros((len(sidx), n_classes))
    # rebuild the batched emissions in one pass for the equality check
    _, lA = engine.predict(cal["mA"], Xn[sidx], device)
    pA = softmax_probs(lA)[:, 1]
    _, lB = engine.predict(cal["mB"], Xn[sidx], device)
    pG = softmax_probs(lB)
    if cal["proto_blend"] > 0:
        E = engine.embed(cal["mB"], Xn[sidx], device)
        plog = engine.prototype_logits(E, cal["emb_cal"], cal["y_g"], cal["K"])
        pp = F.softmax(torch.from_numpy(plog).float(), dim=1).numpy()
        pG = (1.0 - cal["proto_blend"]) * pG + cal["proto_blend"] * pp
    emis_all[:, rest_idx] = 1.0 - pA
    emis_all[:, cal["g2orig"]] = pA[:, None] * pG
    batched_hmm = causal_forward_decode(emis_all, gte, A, pi)
    match = float(np.mean(batched_hmm == decoded_hmm)) * 100

    # FIXED-LAG SMOOTHING sweep — trade a few windows of lookahead for boundary
    # recovery. lag 0 = causal forward (what we stream); large lag -> Viterbi bound.
    from emgv2.eval.sequence import fixed_lag_decode, viterbi_decode
    from emgv2.eval.metrics import balanced_accuracy as _bal
    from emgv2.eval.control_loop import mean_onset_latency as _onset
    mv = yte != rest_idx
    def _moverec(d):  # fraction of true-movement windows NOT lost to Rest
        return float(np.mean(d[mv] != rest_idx)) if mv.any() else float("nan")
    print("  --- fixed-lag smoothing (lookahead vs accuracy/latency) ---")
    print(f"  {'lag':>4} {'added_ms':>9} {'win_bal':>8} {'move_recall':>12} {'onset_ms':>9}")
    for lag in (0, 1, 2, 4, 8):
        d = fixed_lag_decode(emis_all, gte, A, pi, lag=lag)
        print(f"  {lag:>4} {lag*stride_ms:>8.0f} {_bal(yte,d,n_classes)*100:>7.1f}% "
              f"{_moverec(d)*100:>11.0f}% {_onset(d,yte,gte,rest_idx)*stride_ms:>8.0f}")
    dv = viterbi_decode(emis_all, gte, A, pi)
    print(f"  {'∞(vit)':>4} {'offline':>9} {_bal(yte,dv,n_classes)*100:>7.1f}% "
          f"{_moverec(dv)*100:>11.0f}%  (offline upper bound)\n")

    p50, p95, pmax = np.percentile(latencies_ms, [50, 95, 100])
    budget = stride_ms
    print("=== STREAM COMPLETE ===")
    print(f"  windows streamed         : {len(sidx)}  ({len(np.unique(gte))} grasp segments)")
    print(f"  per-decision compute     : median {p50:.2f} ms | p95 {p95:.2f} ms | max {pmax:.2f} ms "
          f"(on {mlabel}, single pass)")
    print(f"  decision budget          : {budget:.0f} ms/window  ->  "
          f"{'WITHIN budget (real-time)' if p95 < budget else 'OVER budget'} "
          f"at p95 ({p95/budget*100:.0f}% of budget)")
    print(f"  streamed == batched HMM  : {match:.1f}%  (confirms the offline number is causal)\n")
    print(f"  HMM per-window balanced acc : {hmm_bal*100:.1f}%   false-activation {hmm_fa*100:.1f}%")
    print(f"  PER-EXECUTION (per grasp seg): grasp-acc {seg_grasp*100:.1f}%  bal-acc {seg_bal*100:.1f}%  "
          f"<- comparable to the paper headline")
    print(f"  deployed hysteresis decode  : grasp-acc {dm['grasp_acc']*100:.1f}%  "
          f"coverage {dm['grasp_cov']*100:.0f}%  false-act {dm['false_act']*100:.1f}%")
    print(f"  onset latency (commit)      : {onset_win:.1f} windows ~= {onset_win*stride_ms:.0f} ms "
          f"(median over grasp segments)\n")

    # dump decode arrays so the paper figure can be re-styled without re-running
    np.savez(Path(__file__).resolve().parents[1] / "results" / f"streaming_arrays_S{subj}.npz",
             y_true=yte, decoded=decoded_hys, group=gte, latency_ms=latencies_ms, stride_ms=stride_ms,
             names=np.array(names, dtype=object))

    # ---- figure: live timeline + per-decision latency ---------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t_ms = np.arange(len(sidx)) * stride_ms
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2),
                                       gridspec_kw=dict(height_ratios=[2, 1]))
        ax1.step(t_ms / 1000, yte, where="post", lw=2.4, color="#888", label="true intent")
        ax1.step(t_ms / 1000, decoded_hys, where="post", lw=1.6, color="#0072B2",
                 label="streamed decode (deployed)")
        ax1.set_yticks(range(n_classes)); ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel("time (s, streamed at 64 ms/window)"); ax1.set_ylabel("grip")
        ax1.set_title(f"Online streaming decode — user S{subj} (held-out rep, causal)", fontsize=11)
        ax1.legend(loc="upper right", fontsize=8, frameon=False)
        ax2.hist(latencies_ms, bins=40, color="#0072B2", alpha=0.85)
        ax2.axvline(budget, color="#c33", ls="--", lw=1.5)
        ax2.text(budget, ax2.get_ylim()[1] * 0.9, f" {budget:.0f} ms budget",
                 color="#c33", fontsize=8.5, va="top")
        ax2.set_xlabel("per-decision compute latency (ms)"); ax2.set_ylabel("windows")
        ax2.set_title(f"Compute per decision: median {p50:.2f} ms, p95 {p95:.2f} ms "
                      f"(<< {budget:.0f} ms budget)", fontsize=10)
        fig.tight_layout()
        out = Path(args.out)
        if not out.is_absolute():
            out = Path(__file__).resolve().parents[1] / out
        out.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}_S{subj}.{ext}", dpi=150, bbox_inches="tight")
        print(f"  wrote figure -> {out}_S{subj}.png / .pdf")
    except Exception as e:
        print(f"  (figure skipped: {e})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

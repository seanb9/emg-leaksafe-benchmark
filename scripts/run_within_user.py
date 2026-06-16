#!/usr/bin/env python
"""Within-user clinical-90 benchmark (the product track).

Each DB2 subject = a stand-in "user". A base model is trained ONCE on a held-out
population of subjects, then for each test user we fine-tune on THEIR OWN
calibration reps and evaluate on THEIR HELD-OUT reps (leak-safe, no rep overlap),
with temporal voting + confidence rejection. Reports the within-user calibration
curve (where it crosses 90%), voted accuracy, and false-activation rate.

Labelled WITHIN-USER / CLINICAL — NOT comparable to the cross-subject LOSO number.

RIG (PowerShell):
    $env:PYTHONUTF8=1
    python -u scripts/run_within_user.py configs/db2_clinical_grips.yaml --supcon-epochs 15 --base-ft-epochs 40 --cal-epochs 30 2>&1 | Tee-Object -FilePath results/within_user_clinical.log
DEV (Mac, structural):
    python scripts/run_within_user.py configs/db2_clinical_grips.yaml --dev
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
from emgv2.models.presets import preset
from emgv2.train import engine
from emgv2.eval.metrics import balanced_accuracy, accuracy, majority_vote_by_group
from emgv2.eval.control_loop import (
    sliding_vote, confidence_mask, coverage, accuracy_on_acted, false_activation_rate,
    rest_gate, grasp_prediction, hysteresis_decode, decode_clinical_metrics,
    mean_onset_latency,
)
from emgv2.utils.device import pick_device, machine_label


def _zscore(X, fit_idx):
    mean, std = per_subject_zscore_fit_apply(X, fit_idx)
    return ((X - mean) / std).astype(np.float32)


def _feats(cfg):
    fc = cfg.get_path("signal.feature_channels")
    return list(fc) if fc else None


def in_channels(cfg):
    base = int(cfg.signal.channels)
    feats = _feats(cfg)
    return base * (1 + len(feats)) if feats else base


def load_subject(cfg, s):
    """get_subject + optional per-channel activation features (raw + RMS/WL envelopes)."""
    ws = db2.get_subject(cfg, s)
    if ws is None or len(ws) == 0:
        return ws
    feats = _feats(cfg)
    if feats:
        from emgv2.data.features import add_envelope_features
        ws.X = add_envelope_features(ws.X, win=int(cfg.get_path("signal.feature_win", 16)),
                                     feats=feats)
    return ws


def softmax_probs(logits):
    return F.softmax(torch.from_numpy(logits), dim=1).numpy()


def _load_backbone(model, base_state):
    """Load the base model's backbone (+ shared heads) but NOT its classifier head,
    so a fresh 2-way / N-grasp-way head can be trained on top."""
    bb = {k: v for k, v in base_state.items() if not k.startswith("classifier")}
    model.load_state_dict(bb, strict=False)


def _predict(model, X, device, tta):
    """predict, with test-time augmentation when tta=(aug_kw, n_views) is enabled."""
    if tta and tta[1] > 1:
        return engine.predict_tta(model, X, device, aug_kw=tta[0], n_views=tta[1])
    return engine.predict(model, X, device)


def _aligned_proba(model_proba, classes_, K):
    """Map a classifier's predict_proba (columns in classes_ order) to a full
    [N, K] array in 0..K-1 grasp-index order (missing classes get 0)."""
    full = np.zeros((model_proba.shape[0], K), dtype=np.float64)
    full[:, np.asarray(classes_, dtype=int)] = model_proba
    return full


def two_stage_eval(base_state, mc, c_in, n_classes, rest_idx, Xn, cal_idx, test_idx,
                   ws_y, device, cal_epochs, cal_lr, seed, proto_blend=0.0, aug_kw=None,
                   tta=None, X_raw=None, grasp_clf="cnn", riemann_shrink=0.1,
                   riemann_delays=(0, 4)):
    """Hierarchical: a dedicated rest/active detector (stage 1) + a grasp classifier
    trained on movement-only data (stage 2). Returns (p_active[test], grasp_pred[test])
    where grasp_pred is in ORIGINAL class indices. Decoupling the two stages stops
    Rest from competing with grasps and gives a much cleaner false-activation gate.

    grasp_clf: 'cnn' (fine-tuned head, optionally proto-blended), 'riemann'
    (Riemannian tangent-space covariance + shrinkage-LDA, per-user recentered), or
    'fuse' (average the two). Riemann uses the RAW EMG channels in X_raw.
    """
    grasp_classes = [c for c in range(n_classes) if c != rest_idx]
    K = len(grasp_classes)
    g2orig = np.array(grasp_classes)
    # Stage 1 — rest vs active (binary), on ALL calibration windows
    y_bin = (ws_y[cal_idx] != rest_idx).astype(np.int64)
    mA = build_v2net(c_in, 2, mc, n_subjects=0).to(device)
    _load_backbone(mA, base_state)
    mA = engine.fewshot_adapt(mA, Xn[cal_idx], y_bin, device, n_classes=2,
                              epochs=cal_epochs, lr=cal_lr, head_only=False, seed=seed,
                              aug_kw=aug_kw)
    _, logitsA = _predict(mA, Xn[test_idx], device, tta)
    p_active = softmax_probs(logitsA)[:, 1]
    # Stage 2 — which grasp, on MOVEMENT-only calibration windows
    move_cal = cal_idx[ws_y[cal_idx] != rest_idx]
    y_g = np.searchsorted(g2orig, ws_y[move_cal]).astype(np.int64)

    proba = None                        # [N_test, K] grasp posterior in 0..K-1 order
    extra = {"cnn": None, "riemann": None, "riemann_aug": None}   # component grasp preds for diagnosis
    if grasp_clf in ("cnn", "fuse"):
        mB = build_v2net(c_in, K, mc, n_subjects=0).to(device)
        _load_backbone(mB, base_state)
        mB = engine.fewshot_adapt(mB, Xn[move_cal], y_g, device, n_classes=K,
                                  epochs=cal_epochs, lr=cal_lr, head_only=False, seed=seed,
                                  aug_kw=aug_kw)
        _, logitsB = _predict(mB, Xn[test_idx], device, tta)
        p_cnn = softmax_probs(logitsB)
        if proto_blend > 0.0:
            emb_cal = engine.embed(mB, Xn[move_cal], device)
            emb_test = engine.embed(mB, Xn[test_idx], device)
            plog = engine.prototype_logits(emb_test, emb_cal, y_g, K)
            pp = F.softmax(torch.from_numpy(plog).float(), dim=1).numpy()
            p_cnn = (1.0 - proto_blend) * p_cnn + proto_blend * pp
        extra["cnn"] = g2orig[p_cnn.argmax(axis=1)]
        proba = p_cnn
    if grasp_clf in ("riemann", "fuse"):
        from emgv2.models.riemann import RiemannTSM
        # plain spatial covariance
        rtp = RiemannTSM(shrink=riemann_shrink, delays=(0,)).fit(X_raw[move_cal], y_g)
        pr_p = _aligned_proba(rtp.predict_proba(X_raw[test_idx]), rtp.classes_, K)
        extra["riemann"] = g2orig[pr_p.argmax(axis=1)]
        # augmented (delay-embedded) covariance — captures temporal dynamics
        rta = RiemannTSM(shrink=riemann_shrink, delays=riemann_delays).fit(X_raw[move_cal], y_g)
        pr_a = _aligned_proba(rta.predict_proba(X_raw[test_idx]), rta.classes_, K)
        extra["riemann_aug"] = g2orig[pr_a.argmax(axis=1)]   # diagnostic only (can overfit few-shot)
        proba = pr_p if proba is None else 0.5 * proba + 0.5 * pr_p   # SHIP: fuse CNN + plain riemann
    # full per-window class posterior (emission) for the HMM sequence-logic layer
    emis = np.zeros((len(p_active), n_classes), dtype=np.float64)
    emis[:, rest_idx] = 1.0 - p_active
    emis[:, g2orig] = p_active[:, None] * proba
    extra["emis"] = emis
    return p_active, g2orig[proba.argmax(axis=1)], extra


def make_mc(cfg):
    return preset(cfg.get_path("model.size", "S"),
                  supcon=bool(cfg.get_path("model.supcon", True)),
                  adversarial=bool(cfg.get_path("model.adversarial", False)))


def train_base(cfg, pretrain_subj, device, epochs_supcon, epochs_ft, log):
    """Train the population base model on subjects the test users are NOT in."""
    n_classes = len(cfg.dataset.classes)
    mc = make_mc(cfg)
    Xs, ys = [], []
    for s in pretrain_subj:
        ws = load_subject(cfg, s)
        if ws is None or len(ws) == 0:
            continue
        ws.X = _zscore(ws.X, np.arange(len(ws)))   # per-subject z-score on own windows
        Xs.append(ws.X); ys.append(ws.y)
    # hold out the last 2 pretrain subjects' data as validation
    Xtr = np.concatenate(Xs[:-1]); ytr = np.concatenate(ys[:-1])
    Xval, yval = Xs[-1], ys[-1]
    # cap Rest in the base training pool (fights the Rest-majority bias that makes
    # movements get classified as Rest). Eval/val untouched.
    rest_idx = int(cfg.get_path("control_loop.rest_class_index", 0))
    rest_cap = cfg.get_path("balance.rest_cap_mult")
    if rest_cap is not None:
        keep = cap_rest_indices(ytr, rest_class_idx=rest_idx, mult=float(rest_cap), seed=int(cfg.seed))
        log(f"  rest-cap: base train {len(ytr)} -> {len(keep)} windows (mult={rest_cap})")
        Xtr, ytr = Xtr[keep], ytr[keep]
    model = build_v2net(in_channels(cfg), n_classes, mc, n_subjects=0).to(device)
    log(f"  base: train {len(Xtr)}w on {len(pretrain_subj)-1} subj | val {len(Xval)}w | "
        f"params {model.num_params()} (deploy {model.deploy_params()})")
    aug = dict(channel_shift=2)
    if bool(cfg.get_path("model.ssl_pretrain", False)):
        ssl_ep = int(cfg.get_path("model.ssl_epochs", 20))
        log(f"  SSL masked pretrain: {ssl_ep} epochs (brain learns move structure unlabelled)")
        engine.ssl_pretrain(model, Xtr, device, epochs=ssl_ep,
                            mask_frac=float(cfg.get_path("model.ssl_mask_frac", 0.4)),
                            seed=cfg.seed, log=log)
    if mc.get("supcon"):
        engine.supcon_pretrain(model, Xtr, ytr, device, epochs=epochs_supcon, aug_kw=aug,
                               seed=cfg.seed, log=log)
    engine.train_supervised(model, Xtr, ytr, Xval, yval, device, n_classes=n_classes,
                            epochs=epochs_ft, aug_kw=aug, seed=cfg.seed, log=log)
    return model, mc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--supcon-epochs", type=int, default=15)
    ap.add_argument("--base-ft-epochs", type=int, default=40)
    ap.add_argument("--cal-epochs", type=int, default=30)
    ap.add_argument("--reuse-base", action="store_true", help="load saved base if present")
    ap.add_argument("--quick", action="store_true",
                    help="only the max calibration level (skip the cal-curve sweep) — ~5x faster")
    ap.add_argument("--dev", action="store_true", help="tiny structural run (Mac)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/within_user_clinical.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    device = pick_device(args.device)
    n_classes = len(cfg.dataset.classes)
    rest_idx = int(cfg.get_path("control_loop.rest_class_index", 0))
    vote_w = int(cfg.get_path("control_loop.vote_window", 5))
    thr = float(cfg.get_path("control_loop.reject_threshold", 0.6))
    two_stage = bool(cfg.get_path("within_user.two_stage", False))
    gate0 = float(cfg.get_path("control_loop.gate_threshold", 0.5))
    cal_lr = float(cfg.get_path("within_user.cal_lr", 1e-3))
    cal_sweep = cfg.get_path("within_user.cal_reps_sweep", [[1], [1, 2], [1, 2, 3]])
    test_reps = cfg.get_path("within_user.test_reps", [4, 5, 6])
    proto_blend = float(cfg.get_path("within_user.proto_blend", 0.0))
    grasp_clf = str(cfg.get_path("within_user.grasp_classifier", "cnn"))   # cnn|riemann|fuse
    riemann_shrink = float(cfg.get_path("within_user.riemann_shrink", 0.1))
    riemann_delays = tuple(cfg.get_path("within_user.riemann_delays", [0, 4]))
    classical_on = bool(cfg.get_path("within_user.classical_baseline", True))
    raw_ch = int(cfg.signal.channels)
    # gentle calibration-time augmentation (no channel shift within a session)
    cal_aug = None
    if bool(cfg.get_path("within_user.cal_augment", False)):
        cal_aug = dict(channel_shift=0, mag_warp_std=0.2, noise_std=0.05,
                       time_mask_frac=0.1, n_time_masks=1, chan_scale=(0.9, 1.1))
    # test-time augmentation: average logits over N gently-augmented views of each
    # test window (denoises the per-window prediction; labels never used).
    tta_views = int(cfg.get_path("within_user.tta_views", 0))
    tta = (dict(channel_shift=0, mag_warp_std=0.15, noise_std=0.04,
                time_mask_frac=0.0, n_time_masks=0, chan_scale=(0.95, 1.05)),
           tta_views) if tta_views > 1 else None
    # clinical sequential decoder (asymmetric hysteresis state machine)
    dec = cfg.get_path("decoder", {}) or {}
    dec_on = str(dec.get("type", "hysteresis")) == "hysteresis"
    dkw = dict(enter_gate=float(dec.get("enter_gate", gate0)),
               exit_gate=float(dec.get("exit_gate", 0.35)),
               on_frames=int(dec.get("on_frames", 3)),
               off_frames=int(dec.get("off_frames", 4)),
               switch_frames=int(dec.get("switch_frames", 3)))

    avail = db2.available_subjects(cfg)
    pre = cfg.get_path("within_user.pretrain_subjects")
    tst = cfg.get_path("within_user.test_subjects")
    if pre is None or tst is None:
        half = len(avail) // 2
        tst = avail[:half]; pre = avail[half:]   # disjoint: base never sees test users
    if args.dev:
        pre = pre[:4]; tst = tst[:3]
        args.supcon_epochs = min(args.supcon_epochs, 1)
        args.base_ft_epochs = min(args.base_ft_epochs, 3)
        args.cal_epochs = min(args.cal_epochs, 5)
    if args.quick:
        cal_sweep = [max(cal_sweep, key=len)]   # only the deepest cal level -> ~5x faster

    mlabel = machine_label(device)
    print(f"=== WITHIN-USER CLINICAL === {mlabel} | {cfg.name} | classes={n_classes} "
          f"| grips={cfg.dataset.class_names}")
    print(f"    base on {len(pre)} subj (never tested) | test on {len(tst)} users | "
          f"vote={vote_w} reject={thr} | grasp_clf={grasp_clf} | WITHIN-USER, not the LOSO number\n")

    base_path = Path(__file__).resolve().parents[1] / "results" / f"base_{cfg.name}.pt"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    want_cin = in_channels(cfg)
    reuse_ok = False
    if args.reuse_base and base_path.exists():
        cand = torch.load(base_path, map_location=device)
        saved_cin = cand.get("backbone.stem.0.weight")
        saved_cin = int(saved_cin.shape[1]) if saved_cin is not None else None
        if saved_cin == want_cin:
            mc = make_mc(cfg); base_state = cand; reuse_ok = True
            print(f"  reusing base from {base_path} (skipped base training)\n")
        else:
            print(f"  saved base has {saved_cin} in-channels but config needs {want_cin} "
                  f"(feature set changed) -> retraining base\n")
    if not reuse_ok:
        base_model, mc = train_base(cfg, pre, device, args.supcon_epochs, args.base_ft_epochs, print)
        base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
        torch.save(base_state, base_path)
        print(f"  base trained in {time.time()-t0:.0f}s (saved to {base_path.name})\n")

    # ---- TEMPORAL MEMORY ('brain') ---------------------------------------------
    # A causal GRU over the frozen backbone's per-window embeddings, trained on the
    # population then fine-tuned per user. Uses temporal CONTEXT to recover the
    # onset/offset windows a memoryless classifier loses to Rest. Full n_classes.
    temporal_on = bool(cfg.get_path("within_user.temporal", False))
    phase_on = bool(cfg.get_path("within_user.phase_brain", False))
    brain_pop = brain_model = phase_brain = None
    Ep = Yp = Gp = None
    if temporal_on or phase_on:
        # shared embedding model (frozen backbone) + population embeddings/labels
        brain_model = build_v2net(in_channels(cfg), n_classes, mc, n_subjects=0).to(device)
        brain_model.load_state_dict(base_state, strict=False); brain_model.eval()
        Ep, Yp, Gp, gb = [], [], [], 0
        for s in pre:
            wsp = load_subject(cfg, s)
            if wsp is None or len(wsp) == 0:
                continue
            Xzp = _zscore(wsp.X, np.arange(len(wsp)))
            Ep.append(engine.embed(brain_model, Xzp, device))
            Yp.append(wsp.y); Gp.append(wsp.group + gb); gb += int(wsp.group.max()) + 1
        Ep = np.concatenate(Ep); Yp = np.concatenate(Yp); Gp = np.concatenate(Gp)
        d_emb = Ep.shape[1]
    if temporal_on:
        from emgv2.models.temporal import TemporalMemory, train_temporal
        hid = int(cfg.get_path("within_user.temporal_hidden", 64))
        brain_pop = TemporalMemory(d_emb, n_classes, hidden=hid).to(device)
        w = engine.class_weights(Yp, n_classes, device).cpu().numpy()
        tb_ep = int(cfg.get_path("within_user.temporal_pop_epochs", 20))
        print(f"  temporal brain: GRU(d={d_emb},h={hid}) {brain_pop.num_params()} params | "
              f"pretrain on {len(np.unique(Gp))} population segments, {tb_ep} epochs")
        train_temporal(brain_pop, Ep, Yp, Gp, device, epochs=tb_ep, lr=1e-3,
                       weight=w, seed=int(cfg.seed)); print()
    if phase_on:
        from emgv2.models.phase_brain import PhaseBrain
        n_phase = int(cfg.get_path("within_user.phase_n", 3))
        phase_brain = PhaseBrain(
            d_emb, n_classes, rest_idx, n_phase=n_phase,
            hidden=int(cfg.get_path("within_user.phase_hidden", 128)),
            hold=float(cfg.get_path("within_user.phase_hold", 0.9)),
            rest_hold=float(cfg.get_path("within_user.phase_rest_hold", 0.99)),
            onset_frac=float(cfg.get_path("within_user.phase_onset_frac", 0.25)),
            release_frac=float(cfg.get_path("within_user.phase_release_frac", 0.25)),
            device=device, seed=int(cfg.seed))
        # cap rest in the population pool so the head doesn't drown in Rest
        keep = cap_rest_indices(Yp, rest_class_idx=rest_idx, mult=2.0, seed=int(cfg.seed))
        # keep groups intact for transition counting: recompute group ids on kept set
        wgt = engine.class_weights(
            phase_brain._states(Yp[keep], Gp[keep]), phase_brain.ss.n_states, device).cpu().numpy()
        print(f"  PHASE-AWARE PREDICTIVE BRAIN: {phase_brain.ss.n_states} states "
              f"(rest + {len(phase_brain.ss.grasps)} grasps x {n_phase} phases) | "
              f"pretrain on {len(np.unique(Gp))} population segments")
        phase_brain.fit_population(Ep[keep], Yp[keep], Gp[keep],
                                   epochs=int(cfg.get_path("within_user.phase_pop_epochs", 25)),
                                   weight=wgt)
        print()

    # ---- HMM SEQUENCE LOGIC ('if I was here, why would I go there') -------------
    # Learn the transition matrix from population label sequences, then decode each
    # test segment causally (online forward filtering) over the CNN's per-window
    # posteriors. Recovers held/onset windows by LOGIC, not smoothing.
    hmm_on = bool(cfg.get_path("within_user.hmm", True))
    hmm_O = hmm_pi = hmm_A = hmm_priors = None
    hmm_hold = float(cfg.get_path("within_user.hmm_hold", 0.9))
    if hmm_on:
        from emgv2.eval.sequence import (learn_transition_structure, build_transition,
                                          causal_forward_decode, class_priors)
        seqs = []
        for s in pre:
            wsp = load_subject(cfg, s)
            if wsp is None or len(wsp) == 0:
                continue
            for g in np.unique(wsp.group):
                seqs.append(wsp.y[wsp.group == g])
        hmm_O, hmm_pi = learn_transition_structure(seqs, n_classes)
        hmm_A = build_transition(hmm_O, hmm_hold)
        # scaled-likelihood emission correction: divide posteriors by class priors
        # NOTE: the scaled-likelihood prior correction is OFF by default for the
        # clinical decoder. It improves balanced accuracy a little but removes the
        # protective rest prior, inflating false-activation ~10x (see paper ablation).
        # The per-frame posterior keeps rest as the safe default state.
        hmm_priors = class_priors(seqs, n_classes) \
            if bool(cfg.get_path("within_user.hmm_prior_correct", False)) else None
        print(f"  HMM sequence logic: learned from {len(seqs)} population segments | "
              f"hold={hmm_hold} | top off-diagonal moves encode rest<->grasp\n")

    rows = []
    cm_true, cm_pred = [], []          # accumulate at the max cal level for a confusion matrix
    sweep_pred, sweep_prob, sweep_full = [], [], []   # for threshold + soft two-stage sweeps
    ts_pact, ts_gpred = [], []                         # full two-stage: p_active + grasp pred
    # unified stream for the clinical decoder sweep (works in both modes)
    dec_score, dec_grasp, dec_true, dec_group = [], [], [], []
    seg_true_pool, seg_pred_pool = [], []     # pooled per-segment decisions (headline)
    emis_pool, emis_true, emis_grp = [], [], []   # pooled emissions for the HMM hold-sweep
    phase_true_pool, phase_pred_pool = [], []     # pooled phase-brain per-window stream
    # ablation pools (per-window preds per method) at max cal — reviewer baselines
    abl = dict(true=[], group=[], raw=[], vote3=[], vote5=[], hmm=[], classical=[])
    phemis_pool, phemis_true, phemis_grp = [], [], []   # pooled phase emissions for the safety sweep
    comp = dict(true=[], cnn=[], riemann=[], riemann_aug=[], fuse=[])   # grasp-classifier face-off
    grp_base = 0
    max_cal = max(len(c) for c in cal_sweep)
    for s in tst:
        ws = load_subject(cfg, s)
        if ws is None or len(ws) == 0:
            continue
        for cal_reps in cal_sweep:
            cal_idx, test_idx = within_user_split(ws.y, ws.group, ws.rep,
                                                  cal_reps=cal_reps, test_reps=test_reps,
                                                  rest_class_idx=rest_idx)
            if len(cal_idx) == 0 or len(test_idx) == 0:
                continue
            # cap Rest in the user's CALIBRATION set (not the test set) so the head
            # fine-tune doesn't default to Rest
            rest_cap = cfg.get_path("balance.rest_cap_mult")
            if rest_cap is not None:
                keep = cap_rest_indices(ws.y[cal_idx], rest_class_idx=rest_idx,
                                        mult=float(rest_cap), seed=int(cfg.seed))
                cal_idx = cal_idx[keep]
            # z-score fit on this user's calibration windows only
            mean, std = per_subject_zscore_fit_apply(ws.X, cal_idx)
            Xn = ((ws.X - mean) / std).astype(np.float32)
            yte = ws.y[test_idx]; gte = ws.group[test_idx]

            if temporal_on:
                from emgv2.models.temporal import TemporalMemory, train_temporal, predict_temporal
                emb_cal = engine.embed(brain_model, Xn[cal_idx], device)
                emb_test = engine.embed(brain_model, Xn[test_idx], device)
                brain_u = TemporalMemory(emb_cal.shape[1], n_classes,
                                         hidden=int(cfg.get_path("within_user.temporal_hidden", 64))).to(device)
                wgt = engine.class_weights(ws.y[cal_idx], n_classes, device).cpu().numpy()
                train_temporal(brain_u, emb_cal, ws.y[cal_idx], ws.group[cal_idx], device,
                               epochs=args.cal_epochs, lr=cal_lr, weight=wgt,
                               warm=brain_pop.state_dict(), seed=int(cfg.seed))
                preds, logits = predict_temporal(brain_u, emb_test, gte, n_classes, device)
                probs = softmax_probs(logits)
                acted = confidence_mask(probs, thr)
                active_score = 1.0 - probs[:, rest_idx]
                grasp_for_dec = grasp_prediction(probs, rest_idx)
                gextra = None
            elif two_stage:
                p_active, grasp_pred, gextra = two_stage_eval(
                    base_state, mc, in_channels(cfg), n_classes, rest_idx,
                    Xn, cal_idx, test_idx, ws.y, device, args.cal_epochs, cal_lr,
                    int(cfg.seed), proto_blend=proto_blend, aug_kw=cal_aug, tta=tta,
                    X_raw=ws.X[:, :, :raw_ch], grasp_clf=grasp_clf,
                    riemann_shrink=riemann_shrink, riemann_delays=riemann_delays)
                acted = p_active >= gate0
                preds = np.where(acted, grasp_pred, rest_idx)
                probs = None
                active_score = p_active
                grasp_for_dec = grasp_pred
            else:
                model = build_v2net(in_channels(cfg), n_classes, mc, n_subjects=0).to(device)
                model.load_state_dict(base_state, strict=False)   # ssl_decoder may be absent
                model = engine.fewshot_adapt(
                    model, Xn[cal_idx], ws.y[cal_idx], device, n_classes=n_classes,
                    epochs=args.cal_epochs, lr=cal_lr,
                    head_only=bool(cfg.get_path("within_user.cal_head_only", True)),
                    seed=cfg.seed, aug_kw=cal_aug)
                preds, logits = _predict(model, Xn[test_idx], device, tta)
                probs = softmax_probs(logits)
                acted = confidence_mask(probs, thr)
                active_score = 1.0 - probs[:, rest_idx]
                grasp_for_dec = grasp_prediction(probs, rest_idx)

            voted = sliding_vote(preds, gte, vote_w)
            # HMM sequence-logic decode (causal): combine CNN evidence with learned
            # transition logic. This is the 'thinking' per-window output.
            emis = gextra["emis"] if (two_stage and not temporal_on and gextra) else probs
            preds_hmm = causal_forward_decode(emis, gte, hmm_A, hmm_pi, priors=hmm_priors) \
                if hmm_on else preds
            # classical Hudgins-TD + LDA baseline (reviewer sanity baseline)
            preds_classical = None
            if classical_on and len(cal_reps) == max_cal:
                from emgv2.eval.classical import ClassicalTD
                Xr = ws.X[:, :, :raw_ch]
                ct = ClassicalTD().fit(Xr[cal_idx], ws.y[cal_idx])
                preds_classical = ct.predict(Xr[test_idx])
            # PHASE-AWARE PREDICTIVE BRAIN: joint (grasp, phase) Bayesian decode
            phase_preds = None
            if phase_on:
                emb_cal = engine.embed(brain_model, Xn[cal_idx], device)
                emb_test = engine.embed(brain_model, Xn[test_idx], device)
                pw = engine.class_weights(
                    phase_brain._states(ws.y[cal_idx], ws.group[cal_idx]),
                    phase_brain.ss.n_states, device).cpu().numpy()
                phase_preds, _, phase_emis = phase_brain.predict_user(
                    emb_cal, ws.y[cal_idx], ws.group[cal_idx], emb_test, gte,
                    epochs=args.cal_epochs, lr=cal_lr, weight=pw)
            # clinical sequential decoder: asymmetric hysteresis over the stream
            decoded = hysteresis_decode(active_score, grasp_for_dec, gte, rest_idx, **dkw) \
                if dec_on else voted
            dm = decode_clinical_metrics(decoded, yte, gte, rest_idx, n_classes)
            # SEGMENT-LEVEL (per-repetition) decision — one intent per grasp hold,
            # the clinical / EMGBench-style headline. Majority over each contiguous
            # segment turns 64ms windows (incl. the rest-like onset ramp) into the
            # single decision a grasp-and-hold device actually makes.
            seg_t, seg_p = majority_vote_by_group(yte, preds, gte, n_classes)
            seg_move = seg_t != rest_idx; seg_rest = ~seg_move
            rows.append(dict(
                subject=s, cal_reps=len(cal_reps),
                bal_acc=balanced_accuracy(yte, preds, n_classes),
                voted_bal_acc=balanced_accuracy(yte, voted, n_classes),
                voted_acc=accuracy(yte, voted),
                seg_bal_acc=balanced_accuracy(seg_t, seg_p, n_classes),
                seg_grasp_acc=float(np.mean(seg_p[seg_move] == seg_t[seg_move])) if seg_move.any() else float("nan"),
                seg_false_act=float(np.mean(seg_p[seg_rest] != rest_idx)) if seg_rest.any() else float("nan"),
                hmm_bal_acc=balanced_accuracy(yte, preds_hmm, n_classes),
                hmm_false_act=false_activation_rate(yte, preds_hmm, rest_idx, None),
                hmm_latency=mean_onset_latency(preds_hmm, yte, gte, rest_idx),
                phase_bal_acc=balanced_accuracy(yte, phase_preds, n_classes) if phase_on else float("nan"),
                phase_false_act=false_activation_rate(yte, phase_preds, rest_idx, None) if phase_on else float("nan"),
                phase_latency=mean_onset_latency(phase_preds, yte, gte, rest_idx) if phase_on else float("nan"),
                dec_bal_acc=dm["bal_acc"], dec_grasp_acc=dm["grasp_acc"],
                dec_false_act=dm["false_act"], dec_latency=dm["latency"],
                coverage=coverage(acted),
                acc_on_acted=accuracy_on_acted(yte, preds, acted),
                false_act=false_activation_rate(yte, preds, rest_idx, acted),
                false_act_noreject=false_activation_rate(yte, preds, rest_idx, None),
            ))
            if len(cal_reps) == max_cal:
                cm_true.append(yte)
                cm_pred.append(decoded if dec_on else voted)   # CM shows the shipped decoder
                seg_true_pool.append(seg_t); seg_pred_pool.append(seg_p)
                if two_stage and gextra:    # per-window grasp face-off on movement windows
                    mv = yte != rest_idx
                    comp["true"].append(yte[mv]); comp["fuse"].append(grasp_pred[mv])
                    for key in ("cnn", "riemann", "riemann_aug"):
                        if gextra.get(key) is not None: comp[key].append(gextra[key][mv])
                # unified decoder-sweep stream (re-offset group ids so users don't collide)
                dec_score.append(active_score); dec_grasp.append(grasp_for_dec)
                dec_true.append(yte); dec_group.append(gte + grp_base)
                if hmm_on:
                    emis_pool.append(emis); emis_true.append(yte); emis_grp.append(gte + grp_base)
                if phase_on and phase_preds is not None:
                    phase_true_pool.append(yte); phase_pred_pool.append(phase_preds)
                    phemis_pool.append(phase_emis); phemis_true.append(yte)
                    phemis_grp.append(gte + grp_base)
                # ablation: pool per-window preds for raw / vote / HMM / classical
                abl["true"].append(yte); abl["group"].append(gte + grp_base)
                abl["raw"].append(preds); abl["hmm"].append(preds_hmm)
                # window-matched majority vote (reviewer: match the HMM's effective lag)
                abl["vote3"].append(sliding_vote(preds, gte, 3))
                abl["vote5"].append(sliding_vote(preds, gte, 5))
                if preds_classical is not None:
                    abl["classical"].append(preds_classical)
                grp_base += int(gte.max()) + 1
                if two_stage and not temporal_on:
                    ts_pact.append(p_active); ts_gpred.append(grasp_pred)
                else:
                    sweep_pred.append(preds); sweep_prob.append(probs.max(axis=1))
                    sweep_full.append(probs)
        done = [r for r in rows if r["subject"] == s]
        if done:
            best = max(done, key=lambda r: r["cal_reps"])
            print(f"  user S{s}: window_bal={best['voted_bal_acc']*100:.0f}% | "
                  f"SEGMENT bal={best['seg_bal_acc']*100:.0f}% grasp={best['seg_grasp_acc']*100:.0f}% "
                  f"false-act={best['seg_false_act']*100:.1f}% (at {best['cal_reps']} cal reps)")

    # aggregate by calibration level
    print(f"\n=== WITHIN-USER CLINICAL SUMMARY ({cfg.name}, {mlabel}) — NOT the LOSO number ===")
    print(f"  raw_win = per-window argmax (no logic);  HMM_win = causal sequence-logic decode (per-window);  "
          f"SEG = per-grasp-execution")
    print(f"  values are mean +/- SD across test subjects (per-subject balanced accuracy)")
    print(f"{'cal_reps':>8} {'raw_win':>13} {'HMM_win':>13} {'SEG_bal':>14} {'SEG_false':>10}")
    levels = sorted({r["cal_reps"] for r in rows})
    stride_ms = float(cfg.get_path("window.stride_ms", 64))
    for lv in levels:
        g = [r for r in rows if r["cal_reps"] == lv]
        def ms(k):
            v = np.array([r[k] for r in g], float)
            return f"{np.nanmean(v)*100:4.1f}+/-{np.nanstd(v)*100:3.1f}"
        print(f"{lv:>8} {ms('bal_acc'):>13} {ms('hmm_bal_acc'):>13} {ms('seg_bal_acc'):>14} "
              f"{np.nanmean([r['seg_false_act'] for r in g])*100:>9.1f}%")
    print("HMM_win = honest per-window WITH sequence logic (evidence + transition reasoning, causal).")

    # POOLED SEGMENT-LEVEL headline — one decision per grasp execution, pooled over
    # all users (EMGBench-style per-repetition number). This is the clinically
    # meaningful figure and the one to quote as "X% within-user".
    if seg_true_pool:
        st = np.concatenate(seg_true_pool); sp = np.concatenate(seg_pred_pool)
        from emgv2.eval.metrics import confusion_matrix as _cm
        scm = _cm(st, sp, n_classes).astype(float)
        srec = np.divide(np.diag(scm), scm.sum(axis=1),
                         out=np.full(n_classes, np.nan), where=scm.sum(axis=1) > 0)
        names = [str(c)[:9] for c in cfg.dataset.class_names]
        seg_move = st != rest_idx; seg_rest = ~seg_move
        seg_bal = float(np.nanmean(srec))
        seg_grasp = float(np.mean(sp[seg_move] == st[seg_move])) if seg_move.any() else float("nan")
        seg_fa = float(np.mean(sp[seg_rest] != rest_idx)) if seg_rest.any() else float("nan")
        print(f"\n=== POOLED SEGMENT-LEVEL (per grasp execution, {max_cal} cal reps) — THE CLINICAL NUMBER ===")
        print(f"  balanced accuracy = {seg_bal*100:.1f}%   grasp accuracy = {seg_grasp*100:.1f}%   "
              f"rest false-activation = {seg_fa*100:.1f}%")
        print("  per-grip recall: " + "  ".join(
            f"{names[i]}={srec[i]*100:.0f}%" for i in range(n_classes) if not np.isnan(srec[i])))

    # PHASE-AWARE PREDICTIVE BRAIN — the novel method. Per-window (with grammar),
    # per-execution, and the ANTICIPATION result: onset latency vs the flat HMM.
    if phase_true_pool:
        pt = np.concatenate(phase_true_pool); pp = np.concatenate(phase_pred_pool)
        from emgv2.eval.metrics import confusion_matrix as _cm2
        pcm = _cm2(pt, pp, n_classes).astype(float)
        prec = np.divide(np.diag(pcm), pcm.sum(axis=1),
                         out=np.full(n_classes, np.nan), where=pcm.sum(axis=1) > 0)
        names = [str(c)[:9] for c in cfg.dataset.class_names]
        def _m(k): g = [r for r in rows if r["cal_reps"] == max_cal]; return float(np.nanmean([r[k] for r in g]))
        print(f"\n=== PHASE-AWARE PREDICTIVE BRAIN ({max_cal} cal reps) — the novel decoder ===")
        print(f"  per-window balanced acc: HMM(flat)={_m('hmm_bal_acc')*100:.1f}%  ->  "
              f"PHASE={_m('phase_bal_acc')*100:.1f}%   (false-act {_m('phase_false_act')*100:.1f}%)")
        print(f"  ANTICIPATION — median onset latency: HMM={_m('hmm_latency')*stride_ms:.0f}ms  ->  "
              f"PHASE={_m('phase_latency')*stride_ms:.0f}ms  (lower = commits to the grasp sooner)")
        print("  per-grip recall (phase brain, per-window): " + "  ".join(
            f"{names[i]}={prec[i]*100:.0f}%" for i in range(n_classes) if not np.isnan(prec[i])))
        # ANTICIPATION-vs-SAFETY sweep over rest-stickiness: high rest_hold = reluctant
        # to leave rest = low false activation but more onset latency. The clinical dial.
        if phemis_pool:
            from emgv2.eval.sequence import causal_forward_decode as _fwd
            PE = np.concatenate(phemis_pool); PT = np.concatenate(phemis_true)
            PG = np.concatenate(phemis_grp); go = phase_brain.ss.grasp_of
            print(f"  {'rest_hold':>9} {'win_bal':>8} {'false_act':>10} {'latency_ms':>11}")
            for rh in (0.90, 0.95, 0.98, 0.995, 0.999):
                A, pi = phase_brain.rebuild(rest_hold=rh)
                st = _fwd(PE, PG, A, pi); gp = go[st]
                b = balanced_accuracy(PT, gp, n_classes)
                fa = false_activation_rate(PT, gp, rest_idx, None)
                lat = mean_onset_latency(gp, PT, PG, rest_idx) * stride_ms
                print(f"  {rh:>9.3f} {b*100:>7.1f}% {fa*100:>9.1f}% {lat:>10.0f}")
            print("  pick the rest_hold giving false_act<2% at the lowest latency; "
                  "set within_user.phase_rest_hold.")

    # BASELINES & ABLATION — PER-SUBJECT mean +/- SD (consistent with Table I), with a
    # Wilcoxon signed-rank test of the HMM grammar vs the matched-window vote across
    # subjects. classical TD+LDA / raw CNN / sliding vote (w3,w5) / HMM grammar.
    if abl["true"]:
        def per_user(key, voted=False):
            """per-subject balanced accuracy for a method (per-window, or per-exec)."""
            out = []
            for t, p, g in zip(abl["true"], abl[key], abl["group"]):
                if voted:
                    gt2, gp2 = majority_vote_by_group(t, p, g, n_classes)
                    out.append(balanced_accuracy(gt2, gp2, n_classes))
                else:
                    out.append(balanced_accuracy(t, p, n_classes))
            return np.array(out, float)
        print(f"\n=== BASELINES & ABLATION (per-subject mean +/- SD, {max_cal} cal reps) ===")
        print(f"  {'method':<26} {'per-window':>14} {'per-exec':>14}")
        rowspec = [("Classical TD+LDA", "classical"), ("CNN raw evidence", "raw"),
                   ("CNN + majority vote (w=3)", "vote3"), ("CNN + majority vote (w=5)", "vote5"),
                   ("CNN + HMM grammar (ours)", "hmm")]
        for label, key in rowspec:
            if not abl[key]:
                continue
            pw, pe = per_user(key), per_user(key, voted=True)
            print(f"  {label:<26} {pw.mean()*100:5.1f}+/-{pw.std()*100:3.1f}   "
                  f"{pe.mean()*100:5.1f}+/-{pe.std()*100:3.1f}")
        def _rank_biserial(a, b):
            """matched-pairs rank-biserial effect size r = (W+ - W-)/(W+ + W-)."""
            from scipy.stats import rankdata
            d = np.asarray(a, float) - np.asarray(b, float)
            d = d[d != 0]
            if len(d) == 0:
                return 0.0
            rk = rankdata(np.abs(d))
            wp, wn = rk[d > 0].sum(), rk[d < 0].sum()
            return float((wp - wn) / (wp + wn))
        if abl["hmm"] and abl["vote5"]:
            try:
                from scipy.stats import wilcoxon
                a, b = per_user("hmm"), per_user("vote5")
                w = wilcoxon(a, b); rb = _rank_biserial(a, b)
                print(f"  Wilcoxon HMM vs vote(w=5), per-window: W={w.statistic:.0f} "
                      f"p={w.pvalue:.3f} rank-biserial r={rb:.2f} "
                      f"({'significant' if w.pvalue < 0.05 else 'NOT significant'} at 0.05)")
            except Exception as e:
                print(f"  (wilcoxon unavailable: {e})")
        # ours (CNN+HMM) vs classical TD+LDA — is the learned model significantly
        # better? Reviewer-requested; Holm-corrected across the two metrics.
        if abl["hmm"] and abl["classical"]:
            try:
                from scipy.stats import wilcoxon
                res = []
                for metric, voted in [("per-window", False), ("per-exec", True)]:
                    a, b = per_user("hmm", voted), per_user("classical", voted)
                    ww = wilcoxon(a, b)
                    res.append((metric, a.mean()*100, b.mean()*100, ww.pvalue,
                                ww.statistic, _rank_biserial(a, b)))
                # Holm-Bonferroni over the 2 comparisons
                order = sorted(range(len(res)), key=lambda i: res[i][3])
                m = len(res); holm = [None]*m
                for rank, i in enumerate(order):
                    holm[i] = min(1.0, res[i][3]*(m-rank))
                print("  --- ours (CNN+HMM) vs Classical TD+LDA (Wilcoxon, Holm-corrected) ---")
                for (metric, ma, mb, p, W, rb), pc in zip(res, holm):
                    sig = "significant" if pc < 0.05 else "NOT significant"
                    print(f"    {metric:<11}: ours {ma:.1f} vs classical {mb:.1f} | "
                          f"W={W:.0f} p={p:.4f} (Holm p={pc:.4f}) r={rb:.2f} {sig}")
            except Exception as e:
                print(f"  (ours-vs-classical wilcoxon unavailable: {e})")
        print("  (HMM vs matched-window vote isolates learned grammar from naive smoothing.)")

    # GRASP-CLASSIFIER FACE-OFF — per-window grasp accuracy on true-movement windows,
    # CNN vs Riemannian tangent-space vs fused. This is the honest per-window number
    # that proves the model (no voting). Riemann should win clearly.
    if comp["true"]:
        ct = np.concatenate(comp["true"])
        print(f"\n=== GRASP-CLASSIFIER FACE-OFF (per-window, movement windows, {max_cal} cal reps) ===")
        for key in ("cnn", "riemann", "riemann_aug", "fuse"):
            if comp[key]:
                cp = np.concatenate(comp[key])
                print(f"  {key:>12}: per-window grasp accuracy = {np.mean(cp == ct)*100:.1f}%")
        print("  riemann=spatial covariance; riemann_aug=delay-embedded covariance (temporal dynamics);")
        print("  both with per-user log-Euclidean recentering + shrinkage-LDA. fuse=CNN+riemann_aug.")

    # confusion matrix at the max calibration level (which grips still confuse)
    if cm_true:
        from emgv2.eval.metrics import confusion_matrix
        yt = np.concatenate(cm_true); yp = np.concatenate(cm_pred)
        cm = confusion_matrix(yt, yp, n_classes).astype(float)
        recall = np.divide(np.diag(cm), cm.sum(axis=1),
                           out=np.full(n_classes, np.nan), where=cm.sum(axis=1) > 0)
        names = [str(c)[:9] for c in cfg.dataset.class_names]
        tag = "hysteresis-decoded" if dec_on else "voted"
        print(f"\n=== CONFUSION MATRIX ({tag}, {max_cal} cal reps, row-normalized %) — pick/drop grips ===")
        print(" " * 11 + "".join(f"{n:>10}" for n in names))
        for i, n in enumerate(names):
            rowsum = cm[i].sum()
            cells = "".join(f"{(cm[i, j]/rowsum*100 if rowsum else 0):>10.0f}" for j in range(n_classes))
            print(f"{n:>10} {cells}   recall={recall[i]*100:.0f}%")
        worst = int(np.nanargmin(recall))
        print(f"\nWeakest grip: '{names[worst]}' (recall {recall[worst]*100:.0f}%) — candidate to drop. "
              f"Off-diagonal = what it's confused with.")

    # rejection-threshold operating-point sweep (free — re-thresholds the same probs)
    if sweep_prob:
        yt = np.concatenate(cm_true); pr = np.concatenate(sweep_pred)
        mp = np.concatenate(sweep_prob)
        print(f"\n=== REJECTION THRESHOLD SWEEP ({max_cal} cal reps) — pick the operating point ===")
        print(f"{'thresh':>7} {'coverage':>9} {'acc@act':>9} {'false_act':>10}")
        for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
            acted = mp >= thr
            cov = float(np.mean(acted))
            accact = float(np.mean(pr[acted] == yt[acted])) if acted.sum() else float("nan")
            fa = false_activation_rate(yt, pr, rest_idx, acted)
            print(f"{thr:>7.1f} {cov*100:>8.0f}% {accact*100:>8.1f}% {fa*100:>9.1f}%")
        print("Goal: false_act <= ~2% at the highest coverage/accuracy you can keep.")

    # TWO-STAGE gate sweep: act when active-score >= gate, then pick the grasp.
    # FULL two-stage uses a dedicated rest-detector's p_active; otherwise the soft
    # version reinterprets the flat model's 1 - P(rest).
    active_score = gpred_ts = yt_ts = None
    if ts_pact:
        yt_ts = np.concatenate(cm_true); active_score = np.concatenate(ts_pact)
        gpred_ts = np.concatenate(ts_gpred); label = "full (dedicated rest-detector)"
    elif sweep_full:
        yt_ts = np.concatenate(cm_true); P = np.concatenate(sweep_full)
        active_score = 1.0 - P[:, rest_idx]; gpred_ts = grasp_prediction(P, rest_idx)
        label = "soft (reinterpret flat model)"
    if active_score is not None:
        move = yt_ts != rest_idx; n_move = int(move.sum()); n_rest = int((~move).sum())
        print(f"\n=== TWO-STAGE GATE SWEEP [{label}] ({max_cal} cal reps) — clinical operating points ===")
        print(f"{'gate':>6} {'move_cov':>9} {'grasp_acc':>10} {'false_act':>10}")
        for gate in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            acted = active_score >= gate
            am = acted & move
            gacc = float(np.mean(gpred_ts[am] == yt_ts[am])) * 100 if am.sum() else float("nan")
            mcov = am.sum() / n_move * 100 if n_move else float("nan")
            fa = (acted & ~move).sum() / n_rest * 100 if n_rest else float("nan")
            print(f"{gate:>6.1f} {mcov:>8.0f}% {gacc:>9.1f}% {fa:>9.1f}%")
        print("gate=act when active-score>=gate; move_cov=movement windows acted; "
              "grasp_acc=of those, correct grip; false_act=rest fired. Want false_act<2%.")

    # HMM SEQUENCE-LOGIC sweep over the hold-probability (stickiness). For each hold
    # it reports the per-window balanced accuracy WITH logic vs the raw argmax — so
    # we see how much "reasoning about the sequence" recovers, and the best knob.
    if hmm_on and emis_pool:
        from emgv2.eval.sequence import build_transition, causal_forward_decode
        Em = np.concatenate(emis_pool); Et = np.concatenate(emis_true); Eg = np.concatenate(emis_grp)
        raw_bal = balanced_accuracy(Et, Em.argmax(1), n_classes)
        print(f"\n=== HMM SEQUENCE-LOGIC SWEEP ({max_cal} cal reps) — reasoning vs raw evidence ===")
        print(f"  raw per-window (no logic) balanced acc = {raw_bal*100:.1f}%")
        print(f"  {'hold':>6} {'HMM_bal':>9} {'false_act':>10}")
        best = (hmm_hold, -1.0)
        for hold in (0.80, 0.90, 0.95, 0.97, 0.99):
            A = build_transition(hmm_O, hold)
            d = causal_forward_decode(Em, Eg, A, hmm_pi, priors=hmm_priors)
            b = balanced_accuracy(Et, d, n_classes)
            fa = false_activation_rate(Et, d, rest_idx, None)
            print(f"  {hold:>6.2f} {b*100:>8.1f}% {fa*100:>9.1f}%")
            if b > best[1]:
                best = (hold, b)
        print(f"  best hold={best[0]} -> {best[1]*100:.1f}% (set within_user.hmm_hold). "
              f"Logic lifts per-window by {(best[1]-raw_bal)*100:+.1f} pts.")

    # HYSTERESIS DECODER sweep — the headline clinical table. Sweeps the asymmetric
    # state machine over the pooled stream (max cal). For each operating point it
    # reports the all-class balanced accuracy (the "voted ~90%" number), the grasp
    # accuracy on true-movement windows, the false-activation rate, and the onset
    # latency. This is where ~90% at <2% false-activation should appear.
    if dec_score:
        ds = np.concatenate(dec_score); dg = np.concatenate(dec_grasp)
        dt = np.concatenate(dec_true); dgrp = np.concatenate(dec_group)
        print(f"\n=== HYSTERESIS DECODER SWEEP ({max_cal} cal reps) — the clinical 90% table ===")
        print(f"{'enter':>6} {'exit':>5} {'on':>3} {'off':>4} | "
              f"{'bal_acc':>8} {'grasp':>7} {'false':>7} {'lat_ms':>7}")
        grid = []
        for enter in (0.5, 0.6, 0.7):
            for ex in (0.25, 0.35, 0.45):
                for on in (2, 3, 4):
                    for off in (3, 4, 6):
                        d = hysteresis_decode(ds, dg, dgrp, rest_idx, enter_gate=enter,
                                              exit_gate=ex, on_frames=on, off_frames=off,
                                              switch_frames=dkw["switch_frames"])
                        mm = decode_clinical_metrics(d, dt, dgrp, rest_idx, n_classes)
                        grid.append((enter, ex, on, off, mm))
        # show the best-balanced-accuracy points that also keep false-act < 2%
        clean = [r for r in grid if r[4]["false_act"] < 0.02]
        show = sorted(clean or grid, key=lambda r: -r[4]["bal_acc"])[:8]
        for enter, ex, on, off, mm in show:
            print(f"{enter:>6.2f} {ex:>5.2f} {on:>3d} {off:>4d} | "
                  f"{mm['bal_acc']*100:>7.1f}% {mm['grasp_acc']*100:>6.1f}% "
                  f"{mm['false_act']*100:>6.1f}% {mm['latency']*stride_ms:>7.0f}")
        best = max(grid, key=lambda r: r[4]["bal_acc"])
        print(f"BEST bal_acc={best[4]['bal_acc']*100:.1f}% at enter={best[0]} exit={best[1]} "
              f"on={best[2]} off={best[3]} (false_act={best[4]['false_act']*100:.1f}%). "
              f"Set these under `decoder:` in the config to lock the operating point.")

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out} | {time.time()-t0:.0f}s on {mlabel}")
    print("Quote the POOLED SEGMENT-LEVEL balanced accuracy as the within-user clinical number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

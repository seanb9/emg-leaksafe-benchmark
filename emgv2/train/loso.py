"""LOSO orchestration for V2Net.

Per fold: per-subject z-score (test subject fit on its calibration-pool only,
never the eval reps), pool training subjects, optional SupCon pre-train, fine-tune
with validation early stopping, then build the leak-safe calibration curve on the
held-out test subject (0% = BN+TENT only; >0% = few-shot head adaptation on the
disjoint calibration pool). Every aggregate is returned as a provenance-stamped
ResultRow so dev-subset MPS runs can never be mistaken for rig headline numbers.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data import ninapro_db2 as db2
from ..data.splits import (
    make_loso_folds,
    leak_safe_calibration_split,
    calibration_pool_indices,
    per_subject_zscore_fit_apply,
)
from ..eval.metrics import balanced_accuracy, accuracy, macro_f1, majority_vote_by_group
from ..eval.sequence import (learn_transition_structure, build_transition,
                             causal_forward_decode, class_priors)
from ..models import build_v2net
from ..utils.device import pick_device, machine_label, is_headline_capable
from ..utils.results import ResultRow
from . import engine


def _apply_norm(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


@dataclass
class FoldData:
    test: int
    val: list
    train: list
    subjects: dict = field(default_factory=dict)  # subject -> WindowedSubject


def _normalize_fold(fd: FoldData, cfg):
    """Per-subject z-score. Train/val: fit on own windows. Test: fit on cal pool."""
    method = cfg.get_path("normalize.method", "per_subject_zscore")
    if method == "none":
        return
    for s, ws in fd.subjects.items():
        if s == fd.test:
            pool = calibration_pool_indices(
                ws.group, ws.rep,
                eval_reps=cfg.split.calibration.eval_reps,
                cal_reps=cfg.split.calibration.cal_reps,
            )
            fit_idx = pool if len(pool) else np.arange(len(ws))
        else:
            fit_idx = np.arange(len(ws))
        mean, std = per_subject_zscore_fit_apply(ws.X, fit_idx)
        ws.X = _apply_norm(ws.X, mean, std)


def run_fold_v2(cfg, fold, device, *, epochs_supcon, epochs_ft, log=lambda *_: None):
    """Run one LOSO fold; return dict: ratio -> metrics dict (per-window + vote)."""
    fd = FoldData(test=fold["test"], val=fold["val"], train=fold["train"])
    needed = [fold["test"], *fold["val"], *fold["train"]]
    for s in needed:
        ws = db2.get_subject(cfg, s)
        if ws is not None and len(ws) > 0:
            fd.subjects[s] = ws
    _normalize_fold(fd, cfg)

    n_classes = len(cfg.dataset.classes)
    c_in = int(cfg.signal.channels)
    mc = cfg.get_path("model", {}) or {}

    # Pool training + validation
    tr_subj = [s for s in fd.train if s in fd.subjects]
    Xtr = np.concatenate([fd.subjects[s].X for s in tr_subj])
    ytr = np.concatenate([fd.subjects[s].y for s in tr_subj])
    subj_tr = np.concatenate([np.full(len(fd.subjects[s]), i) for i, s in enumerate(tr_subj)])
    val_subj = [s for s in fd.val if s in fd.subjects]
    Xval = np.concatenate([fd.subjects[s].X for s in val_subj])
    yval = np.concatenate([fd.subjects[s].y for s in val_subj])

    model = build_v2net(c_in, n_classes, mc, n_subjects=len(tr_subj)).to(device)
    log(f"  fold S{fold['test']}: train {len(Xtr)}w/{len(tr_subj)}subj | "
        f"val {len(Xval)}w | params {model.num_params()}")

    aug_kw = dict(channel_shift=int(cfg.get_path("signal.channels", 12) > 8 and 2 or 1))
    if bool(cfg.get_path("model.ssl_pretrain", False)):
        engine.ssl_pretrain(model, Xtr, device,
                            epochs=int(cfg.get_path("model.ssl_epochs", 20)),
                            mask_frac=float(cfg.get_path("model.ssl_mask_frac", 0.4)),
                            seed=cfg.seed, log=log)
    if mc.get("supcon", False) and epochs_supcon > 0:
        engine.supcon_pretrain(model, Xtr, ytr, device, epochs=epochs_supcon,
                               bs=int(mc.get("supcon_bs", 256)), aug_kw=aug_kw,
                               seed=cfg.seed, log=log)
    engine.train_supervised(
        model, Xtr, ytr, Xval, yval, device, n_classes=n_classes, epochs=epochs_ft,
        bs=int(mc.get("batch_size", 64)), lr=float(mc.get("lr", 5e-4)),
        label_smoothing=float(mc.get("label_smoothing", 0.1)),
        patience=int(mc.get("patience", 15)), aug_kw=aug_kw, subj_tr=subj_tr,
        adv_lambda_max=float(mc.get("adv_lambda_max", 0.0)), seed=cfg.seed, log=log,
    )

    # HMM sequence-logic: learn the transition grammar from the TRAINING subjects'
    # label sequences (zero test info), to reason over the held-out subject's stream.
    hmm_on = bool(cfg.get_path("loso.hmm", True))
    hmm_A = hmm_pi = None
    if hmm_on:
        seqs = []
        for s in tr_subj:
            ws = fd.subjects[s]
            for g in np.unique(ws.group):
                seqs.append(ws.y[ws.group == g])
        O, hmm_pi = learn_transition_structure(seqs, n_classes)
        hmm_A = build_transition(O, hold=float(cfg.get_path("loso.hmm_hold", 0.97)))
        hmm_priors = class_priors(seqs, n_classes) \
            if bool(cfg.get_path("loso.hmm_prior_correct", False)) else None

    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

    # Test subject calibration curve
    tws = fd.subjects[fold["test"]]
    out = {}
    for ratio in cfg.split.calibration.ratios:
        cal_idx, eval_idx = leak_safe_calibration_split(
            tws.y, tws.group, tws.rep, ratio=ratio,
            eval_reps=cfg.split.calibration.eval_reps,
            cal_reps=cfg.split.calibration.cal_reps, seed=cfg.seed,
        )
        adapted = model
        if cfg.get_path("model.tta", True):
            adapted = engine.adapt_bn_tent(model, tws.X[eval_idx], device,
                                           bn_passes=1, tent_steps=10, seed=cfg.seed)
        if ratio > 0 and len(cal_idx) > 0:
            adapted = engine.fewshot_adapt(adapted, tws.X[cal_idx], tws.y[cal_idx],
                                           device, n_classes=n_classes,
                                           epochs=int(mc.get("fewshot_epochs", 20)),
                                           seed=cfg.seed)
        preds, logits = engine.predict(adapted, tws.X[eval_idx], device)
        yte = tws.y[eval_idx]; geval = tws.group[eval_idx]
        gt, gp = majority_vote_by_group(yte, preds, geval, n_classes)
        rec = dict(
            bal_acc=balanced_accuracy(yte, preds, n_classes),
            acc=accuracy(yte, preds),
            macro_f1=macro_f1(yte, preds, n_classes),
            vote_bal_acc=balanced_accuracy(gt, gp, n_classes),
            n_eval=len(eval_idx), n_cal=len(cal_idx),
        )
        if hmm_on:
            # decode the per-window posteriors with the learned transition logic
            preds_h = causal_forward_decode(_softmax(logits), geval, hmm_A, hmm_pi,
                                            priors=hmm_priors)
            gt_h, gp_h = majority_vote_by_group(yte, preds_h, geval, n_classes)
            rec["hmm_bal_acc"] = balanced_accuracy(yte, preds_h, n_classes)
            rec["hmm_vote_bal_acc"] = balanced_accuracy(gt_h, gp_h, n_classes)
        out[ratio] = rec
    return {"res": out, "params": model.num_params(), "deploy_params": model.deploy_params()}


def run_loso(cfg, *, subjects=None, dev=True, eval_folds=None, epochs_supcon=2,
             epochs_ft=3, device=None, model_override=None, log=print):
    """Run LOSO and return (rows, raw).

    dev=True      -> tiny smoke: subset BOTH training universe and #folds; never headline.
    eval_folds=K  -> train on the FULL universe each fold, but evaluate only the
                     first K test subjects (capacity sweep / reduced-fold). Not a
                     headline unless K covers all folds.
    Headline-eligible only on CUDA, full fold coverage, dev=False.
    """
    device = device or pick_device()
    mlabel = machine_label(device)

    avail = sorted(subjects or db2.available_subjects(cfg))
    if dev:
        avail = avail[: int(cfg.get_path("dev.n_subjects", 4))]
    val_n = min(int(cfg.split.val_n_subjects), max(1, len(avail) - 2))
    folds = make_loso_folds(avail, val_n=val_n)
    folds = [f for f in folds if f["test"] in avail
             and len([s for s in f["train"] if s in avail]) >= 1]
    full_coverage = True
    if dev:
        folds = folds[: int(cfg.get_path("dev.n_folds", 2))]
        full_coverage = False
    elif eval_folds is not None:
        full_coverage = eval_folds >= len(folds)
        folds = folds[:eval_folds]

    if model_override:
        cfg = type(cfg)({**cfg})
        cfg["model"] = {**(cfg.get_path("model", {}) or {}), **model_override}

    headline = is_headline_capable(device) and not dev and full_coverage
    kind = "dev_smoke" if dev else ("loso_full" if full_coverage else "loso_partial")
    log(f"[{kind.upper()}] {cfg.name} | device={mlabel} | {len(folds)} folds | "
        f"classes={len(cfg.dataset.classes)} @ {cfg.signal.target_fs}Hz | "
        f"model={cfg.get_path('model.widths', [32,64,128])}")

    per_ratio = {r: {k: [] for k in ("bal_acc", "acc", "macro_f1", "vote_bal_acc")}
                 for r in cfg.split.calibration.ratios}
    raw = []
    for f in folds:
        fr = run_fold_v2(cfg, f, device, epochs_supcon=epochs_supcon,
                         epochs_ft=epochs_ft, log=log)
        res = fr["res"]
        raw.append({"test": f["test"], "res": res,
                    "params": fr["params"], "deploy_params": fr["deploy_params"]})
        for r, m in res.items():
            for k in ("bal_acc", "acc", "macro_f1", "vote_bal_acc"):
                per_ratio[r][k].append(m[k])
        log(f"  -> S{f['test']}: " + " ".join(
            f"r{int(r*100)}={res[r]['bal_acc']*100:.0f}%" for r in cfg.split.calibration.ratios))

    rows = []
    class_set = "emgbench10" if list(cfg.dataset.classes) == [0,5,6,8,9,10,13,14,15,16] else \
                ("reable12" if len(cfg.dataset.classes) == 12 else f"custom{len(cfg.dataset.classes)}")
    widths = cfg.get_path("model.widths", [32, 64, 128])
    deploy_p = raw[0].get("deploy_params") if raw else None
    note = "DEV SMOKE — not a headline number" if dev else \
           ("" if full_coverage else f"REDUCED {len(folds)}-fold — not headline")
    for r in cfg.split.calibration.ratios:
        vals = np.array(per_ratio[r]["bal_acc"], dtype=float)
        vote = np.array(per_ratio[r]["vote_bal_acc"], dtype=float)
        rows.append(ResultRow(
            run_kind=kind, headline_eligible=headline,
            machine=mlabel, config_name=cfg.name, model=f"v2net{widths}", protocol="loso",
            class_set=class_set, n_classes=len(cfg.dataset.classes),
            sampling_hz=int(cfg.signal.target_fs), window_ms=int(cfg.window.size_ms),
            cal_ratio=float(r), metric_name="balanced_accuracy",
            value_mean=float(np.mean(vals)) if len(vals) else float("nan"),
            value_std=float(np.std(vals)) if len(vals) else float("nan"),
            n_folds=len(folds), decision_unit="window", notes=note,
            extra={"deploy_params": deploy_p,
                   "vote_bal_acc_mean": float(np.mean(vote)) if len(vote) else None},
        ))
    return rows, raw


def _class_set_tag(cfg):
    if list(cfg.dataset.classes) == [0, 5, 6, 8, 9, 10, 13, 14, 15, 16]:
        return "emgbench10"
    if len(cfg.dataset.classes) == 12:
        return "reable12"
    return f"custom{len(cfg.dataset.classes)}"


def run_loso_checkpointed(cfg, *, model_override=None, test_subjects=None,
                          epochs_supcon=20, epochs_ft=60, device=None,
                          run_tag="run", log=print):
    """Crash-proof, parallelisable full 40-fold LOSO.

    Each fold's result is written to its own JSON file under
    results/ckpt_<config>_<run_tag>/ as soon as it completes, so:
      * a power-off only loses the fold in progress (re-run resumes),
      * several workers can run at once on different `test_subjects` (folds),
        all writing to the same dir with no contention (one file per fold).
    The final aggregate is computed by reading ALL fold files present, so whichever
    worker finishes last produces the complete 40-fold summary. Headline-eligible
    only on CUDA when all folds are done.
    """
    device = device or pick_device()
    mlabel = machine_label(device)
    if model_override:
        cfg = type(cfg)({**cfg})
        cfg["model"] = {**(cfg.get_path("model", {}) or {}), **model_override}

    avail = sorted(db2.available_subjects(cfg))
    val_n = min(int(cfg.split.val_n_subjects), max(1, len(avail) - 2))
    folds = make_loso_folds(avail, val_n=val_n)
    folds = [f for f in folds if f["test"] in avail
             and len([s for s in f["train"] if s in avail]) >= 1]
    total = len(folds)

    ckpt_dir = Path(__file__).resolve().parents[2] / "results" / f"ckpt_{cfg.name}_{run_tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if test_subjects is not None:
        test_subjects = set(int(s) for s in test_subjects)

    log(f"[LOSO-CKPT] {cfg.name} | {mlabel} | {total} folds total | this worker tests: "
        f"{sorted(test_subjects) if test_subjects is not None else 'ALL'} | dir={ckpt_dir.name}")

    for f in folds:
        subj = int(f["test"])
        ckpt = ckpt_dir / f"fold_{subj:02d}.json"
        if ckpt.exists():
            continue                                   # already done (this run or another worker)
        if test_subjects is not None and subj not in test_subjects:
            continue                                   # another worker owns this fold
        fr = run_fold_v2(cfg, f, device, epochs_supcon=epochs_supcon,
                         epochs_ft=epochs_ft, log=log)
        rec = {"test": subj, "res": fr["res"],
               "params": fr["params"], "deploy_params": fr["deploy_params"]}
        tmp = ckpt.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, ckpt)                          # atomic — no half-written files
        log(f"  [saved S{subj}] " + " ".join(
            f"r{int(r*100)}={fr['res'][r]['bal_acc']*100:.0f}"
            f"->{fr['res'][r].get('hmm_bal_acc', fr['res'][r]['bal_acc'])*100:.0f}%"
            for r in cfg.split.calibration.ratios) + "  (baseline->HMM per-window)")

    # aggregate from ALL fold files on disk (covers every worker)
    recs = []
    for p in sorted(ckpt_dir.glob("fold_*.json")):
        try:
            recs.append(json.load(open(p)))
        except Exception:
            pass
    rows = _aggregate_ckpt_rows(cfg, recs, total, mlabel, device, run_tag)
    return rows, recs, total


def _aggregate_ckpt_rows(cfg, recs, total, mlabel, device, run_tag):
    ratios = list(cfg.split.calibration.ratios)
    n_done = len(recs)
    headline = is_headline_capable(device) and n_done >= total
    deploy_p = recs[0].get("deploy_params") if recs else None
    note = "" if n_done >= total else f"PARTIAL {n_done}/{total} folds — not headline"
    rows = []
    def _collect(key):
        out = []
        for rec in recs:
            res = {float(k): v for k, v in rec["res"].items()}
            if r in res and key in res[r]:
                out.append(res[r][key])
        return np.array(out, dtype=float)

    for r in ratios:
        vals = _collect("bal_acc")
        hmm = _collect("hmm_bal_acc"); hmm_v = _collect("hmm_vote_bal_acc")
        vote = _collect("vote_bal_acc")
        rows.append(ResultRow(
            run_kind="loso_full" if n_done >= total else "loso_partial",
            headline_eligible=headline, machine=mlabel, config_name=cfg.name,
            model=f"v2net_{run_tag}", protocol="loso", class_set=_class_set_tag(cfg),
            n_classes=len(cfg.dataset.classes), sampling_hz=int(cfg.signal.target_fs),
            window_ms=int(cfg.window.size_ms), cal_ratio=float(r),
            metric_name="balanced_accuracy",
            value_mean=float(np.mean(vals)) if len(vals) else float("nan"),
            value_std=float(np.std(vals)) if len(vals) else float("nan"),
            n_folds=n_done, decision_unit="window", notes=note,
            extra={"deploy_params": deploy_p,
                   "vote_bal_acc_mean": float(np.mean(vote)) if len(vote) else None,
                   "hmm_bal_acc_mean": float(np.mean(hmm)) if len(hmm) else None,
                   "hmm_vote_bal_acc_mean": float(np.mean(hmm_v)) if len(hmm_v) else None},
        ))
    return rows

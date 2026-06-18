#!/usr/bin/env python
"""DB6 cross-session (re-donning) study -- the binding deployment constraint.

NinaPro DB6 records 10 subjects over 5 days x 2 sessions with the electrodes
removed and re-applied between sessions. This measures what the within-session
benchmark cannot: the accuracy drop when the device is re-donned on a later day.

Protocol (leak-safe, per test subject): a population base is trained on a disjoint
set of subjects; the test subject is calibrated on DAY 1 and evaluated on (a)
held-out DAY 1 repetitions [within-session] and (b) DAY 5 [cross-session]. The
within-minus-cross gap is the re-donning cost. DB6 has 16 channels and its own
8-class set (rest + 7 grasps, ids 0,1,3,4,6,9,10,11).

Only Day 1 and Day 5 are loaded (all the protocol needs); windows are cached per
subject. RUN: python -u scripts/db6_crosssession.py --root <ninapro_db6 dir>
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scipy.io as sio
import torch
from emgv2.data.signal import resample_signal, apply_filter
from emgv2.data.windowing import window_subject
from emgv2.data.features import add_envelope_features
from emgv2.data.splits import per_subject_zscore_fit_apply, cap_rest_indices
from emgv2.models import build_v2net
from emgv2.models.presets import preset
from emgv2.train import engine
from emgv2.eval.metrics import balanced_accuracy, majority_vote_by_group
from emgv2.utils.device import pick_device, machine_label

DB6_CLASSES = [0, 1, 3, 4, 6, 9, 10, 11]      # rest + 7 grasps (DB6 numbering)
RAW_FS, TGT_FS = 2000, 250
WIN_MS, STRIDE_MS = 256, 64
FEATS = ["rms", "wl"]; FEAT_WIN = 16
DAYS = {1: ("a", [1, 2]), 5: ("b", [1, 2])}   # day -> (zip half, times)


def _sess_path(root, subj, day, t):
    half = DAYS[day][0]
    return Path(root) / f"DB6_s{subj}_{half}" / f"S{subj}_D{day}_T{t}.mat"


def load_db6_subject(root, subj, cache_dir):
    """Window DB6 Day-1 and Day-5 for one subject; cache. Returns dict with
    X[N,W,C(+feats)], y, group, rep, day (1 or 5)."""
    cpath = Path(cache_dir) / f"db6_s{subj}_d1d5.npz"
    if cpath.exists():
        d = np.load(cpath)
        return {k: d[k] for k in d.files}
    Xs, ys, gs, rs, ds = [], [], [], [], []
    gbase = 0
    for day in (1, 5):
        for t in DAYS[day][1]:
            p = _sess_path(root, subj, day, t)
            if not p.exists():
                continue
            m = sio.loadmat(str(p), variable_names=["emg", "restimulus", "rerepetition"])
            emg = np.asarray(m["emg"], dtype=np.float64)
            emg = resample_signal(emg, RAW_FS, TGT_FS)
            emg = apply_filter(emg, TGT_FS, "reable_bandpass").astype(np.float32)
            # resample the label/rep channels by nearest index
            rs_full = np.asarray(m["restimulus"]).reshape(-1)
            rr_full = np.asarray(m["rerepetition"]).reshape(-1)
            idx = np.linspace(0, len(rs_full) - 1, emg.shape[0]).round().astype(int)
            restim, rerep = rs_full[idx], rr_full[idx]
            ws = window_subject(subj, emg, restim, rerep, class_ids=DB6_CLASSES,
                                fs=TGT_FS, win_samples=int(WIN_MS * TGT_FS / 1000),
                                stride_samples=int(STRIDE_MS * TGT_FS / 1000))
            if ws is None or len(ws) == 0:
                continue
            Xs.append(ws.X); ys.append(ws.y); gs.append(ws.group + gbase)
            rs.append(ws.rep); ds.append(np.full(len(ws), day, np.int64))
            gbase += int(ws.group.max()) + 1
    X = add_envelope_features(np.concatenate(Xs), win=FEAT_WIN, feats=FEATS)
    out = dict(X=X.astype(np.float32), y=np.concatenate(ys), group=np.concatenate(gs),
               rep=np.concatenate(rs), day=np.concatenate(ds))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cpath, **out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/sean/Desktop/ReAble Matlab/reable_emg_classifier/data/ninapro_db6")
    ap.add_argument("--test", default="1,2,3,4,5"); ap.add_argument("--pretrain", default="6,7,8,9,10")
    ap.add_argument("--base-epochs", type=int, default=20); ap.add_argument("--cal-epochs", type=int, default=25)
    ap.add_argument("--reuse-base", action="store_true")
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default="results/db6_crosssession.csv")
    args = ap.parse_args()
    device = pick_device(args.device); mlabel = machine_label(device)
    cache = Path(__file__).resolve().parents[1] / "data_cache" / "db6"
    nC = len(DB6_CLASSES); rest_idx = 0
    test = [int(x) for x in args.test.split(",")]; pre = [int(x) for x in args.pretrain.split(",")]
    print(f"=== DB6 CROSS-SESSION + RECOVERY === {mlabel} | classes={nC} | test {test} | base {pre}")

    def _znorm(X, fit_idx):
        m = X[fit_idx].mean((0, 1)); sd = X[fit_idx].std((0, 1)) + 1e-6
        return ((X - m) / sd).astype(np.float32)

    raw = {s: load_db6_subject(args.root, s, cache) for s in set(test) | set(pre)}
    c_in = raw[test[0]]["X"].shape[2]
    mc = preset("S", supcon=False, adversarial=False)
    base_path = Path(__file__).resolve().parents[1] / "results" / "base_db6.pt"

    # population base: supervised CNN on pretrain subjects (per-subject z-scored)
    t0 = time.time()
    if args.reuse_base and base_path.exists():
        base_state = torch.load(base_path, map_location=device); print(f"  reusing base {base_path.name}")
    else:
        Xtr, ytr = [], []
        for s in pre:
            d = raw[s]; Xn = _znorm(d["X"], np.arange(len(d["y"])))
            Xtr.append(Xn); ytr.append(d["y"]); print(f"  loaded pretrain S{s}: {len(d['y'])} win")
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
        keep = cap_rest_indices(ytr, rest_class_idx=rest_idx, mult=1.5, seed=1337); Xtr, ytr = Xtr[keep], ytr[keep]
        base = build_v2net(c_in, nC, mc, n_subjects=0).to(device); nval = max(1, len(Xtr)//10)
        print(f"  training base: {len(Xtr)} win, {c_in} ch, {nC} classes, {args.base_epochs} ep")
        engine.train_supervised(base, Xtr[:-nval], ytr[:-nval], Xtr[-nval:], ytr[-nval:], device,
                                n_classes=nC, epochs=args.base_epochs, aug_kw=dict(channel_shift=2),
                                seed=1337, log=lambda *_: None)
        base_state = {k: v.clone() for k, v in base.state_dict().items()}
        torch.save(base_state, base_path)
        print(f"  base trained in {time.time()-t0:.0f}s (saved {base_path.name})\n")

    def fit_model(Xn, ci):
        m = build_v2net(c_in, nC, mc, n_subjects=0).to(device); m.load_state_dict(base_state, strict=False)
        return engine.fewshot_adapt(m, Xn[ci], raw_y[ci], device, n_classes=nC, epochs=args.cal_epochs,
                                    lr=3e-4, head_only=False, seed=1337)
    def evl(model, Xn, mask):
        idx = np.flatnonzero(mask); order = np.argsort(grp[idx], kind="stable"); idx = idx[order]
        preds, _ = engine.predict(model, Xn[idx], device)
        st, sp = majority_vote_by_group(raw_y[idx], preds, grp[idx], nC)
        return balanced_accuracy(raw_y[idx], preds, nC)*100, balanced_accuracy(st, sp, nC)*100

    rows = []
    for s in test:
        d = raw[s]; X, raw_y, grp, rep, day = d["X"], d["y"], d["group"], d["rep"], d["day"]
        d1 = day == 1; d5 = day == 5
        d1cal = d1 & np.isin(rep, [1,2,3,4,5,6,7,8]); d1te = d1 & np.isin(rep, [9,10,11,12])
        d5cal = d5 & np.isin(rep, [1,2,3,4,5,6,7,8]); d5te = d5 & np.isin(rep, [9,10,11,12])
        if min(d1cal.sum(), d1te.sum(), d5cal.sum(), d5te.sum()) == 0:
            print(f"  S{s}: missing day/rep data, skip"); continue
        d1ci = np.flatnonzero(d1cal); d1ci = d1ci[cap_rest_indices(raw_y[d1ci], rest_class_idx=rest_idx, mult=1.5, seed=1337)]
        d5ci = np.flatnonzero(d5cal); d5ci = d5ci[cap_rest_indices(raw_y[d5ci], rest_class_idx=rest_idx, mult=1.5, seed=1337)]
        # normalisations: day-1 stats (deployed), day-5 stats (label-free renorm)
        Xn_d1 = _znorm(X, d1ci); Xn_d5 = _znorm(X, np.flatnonzero(d5))
        m_d1 = fit_model(Xn_d1, d1ci)               # model calibrated on day 1
        within_pw, within_pe = evl(m_d1, Xn_d1, d1te)             # within-session day 1
        noad_pw, noad_pe     = evl(m_d1, Xn_d1, d5te)             # cross, no adaptation
        unsup_pw, unsup_pe   = evl(m_d1, Xn_d5, d5te)             # cross + unsupervised renorm
        m_d5 = fit_model(Xn_d5, d5ci)               # full per-don recalibration
        recal_pw, recal_pe   = evl(m_d5, Xn_d5, d5te)
        rows.append(dict(subject=s, within_pe=within_pe, noad_pe=noad_pe, unsup_pe=unsup_pe, recal_pe=recal_pe,
                         within_pw=within_pw, noad_pw=noad_pw, unsup_pw=unsup_pw, recal_pw=recal_pw))
        print(f"  S{s}: within {within_pe:.0f}% | d5 no-adapt {noad_pe:.0f}% | +unsup-renorm {unsup_pe:.0f}% | "
              f"+recal {recal_pe:.0f}%  (per-exec)")

    if rows:
        import csv
        a = lambda k: np.array([r[k] for r in rows])
        def line(lab, k):
            return f"  {lab:<34} per-exec {a(k+'_pe').mean():5.1f}+/-{a(k+'_pe').std():4.1f}   per-window {a(k+'_pw').mean():5.1f}+/-{a(k+'_pw').std():4.1f}"
        print(f"\n=== DB6 CROSS-SESSION RECOVERY ({len(rows)} test subjects) ===")
        print(line("within-session (day 1)", "within"))
        print(line("cross (day 5, no adaptation)", "noad"))
        print(line("cross + unsupervised renorm", "unsup"))
        print(line("cross + full recalibration", "recal"))
        print(f"\n  re-donning gap (within -> no-adapt): {a('within_pe').mean()-a('noad_pe').mean():.1f} pts per-exec")
        print(f"  recovered by unsup renorm:           {a('unsup_pe').mean()-a('noad_pe').mean():+.1f} pts")
        print(f"  recovered by full recalibration:     {a('recal_pe').mean()-a('noad_pe').mean():+.1f} pts "
              f"(-> {a('recal_pe').mean():.1f}%, vs {a('within_pe').mean():.1f}% within-session)")
        out = Path(args.out); out = out if out.is_absolute() else Path(__file__).resolve().parents[1]/out
        with open(out,"w",newline="") as fh:
            w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"  wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

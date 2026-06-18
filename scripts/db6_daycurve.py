#!/usr/bin/env python
"""DB6 day-separation curve (full cohort, n=10): how the re-donning gap grows with
days elapsed, and how per-don recalibration recovers it at every separation.

Calibrate each subject on DAY 1; test on day K (K=1..5, increasing re-donning
separation) both WITHOUT adaptation (day-1 model) and WITH full day-K
recalibration. Two-fold population base (subjects 1-5 <-> 6-10) so every subject is
tested leak-free. Produces the degradation trajectory + recovery line and a figure.

RUN: python -u scripts/db6_daycurve.py
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
from emgv2.data.splits import cap_rest_indices
from emgv2.models import build_v2net
from emgv2.models.presets import preset
from emgv2.train import engine
from emgv2.eval.metrics import balanced_accuracy, majority_vote_by_group
from emgv2.utils.device import pick_device, machine_label

DB6_CLASSES = [0,1,3,4,6,9,10,11]; RAW_FS, TGT_FS = 2000, 250
WIN_MS, STRIDE_MS = 256, 64; FEATS = ["rms","wl"]; FEAT_WIN = 16
DAY_HALF = {1:"a",2:"a",3:"a",4:"b",5:"b"}


def load_all_days(root, subj, cache_dir):
    cpath = Path(cache_dir) / f"db6_s{subj}_all.npz"
    if cpath.exists():
        d = np.load(cpath); return {k: d[k] for k in d.files}
    Xs, ys, gs, rs, ds = [], [], [], [], []; gbase = 0
    for day in (1,2,3,4,5):
        for t in (1,2):
            p = Path(root)/f"DB6_s{subj}_{DAY_HALF[day]}"/f"S{subj}_D{day}_T{t}.mat"
            if not p.exists(): continue
            m = sio.loadmat(str(p), variable_names=["emg","restimulus","rerepetition"])
            emg = resample_signal(np.asarray(m["emg"],dtype=np.float64), RAW_FS, TGT_FS)
            emg = apply_filter(emg, TGT_FS, "reable_bandpass").astype(np.float32)
            rsf = np.asarray(m["restimulus"]).reshape(-1); rrf = np.asarray(m["rerepetition"]).reshape(-1)
            idx = np.linspace(0, len(rsf)-1, emg.shape[0]).round().astype(int)
            ws = window_subject(subj, emg, rsf[idx], rrf[idx], class_ids=DB6_CLASSES, fs=TGT_FS,
                                win_samples=int(WIN_MS*TGT_FS/1000), stride_samples=int(STRIDE_MS*TGT_FS/1000))
            if ws is None or len(ws)==0: continue
            Xs.append(ws.X); ys.append(ws.y); gs.append(ws.group+gbase); rs.append(ws.rep)
            ds.append(np.full(len(ws), day, np.int64)); gbase += int(ws.group.max())+1
    X = add_envelope_features(np.concatenate(Xs), win=FEAT_WIN, feats=FEATS).astype(np.float32)
    out = dict(X=X, y=np.concatenate(ys), group=np.concatenate(gs), rep=np.concatenate(rs), day=np.concatenate(ds))
    Path(cache_dir).mkdir(parents=True, exist_ok=True); np.savez_compressed(cpath, **out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/sean/Desktop/ReAble Matlab/reable_emg_classifier/data/ninapro_db6")
    ap.add_argument("--base-epochs", type=int, default=20); ap.add_argument("--cal-epochs", type=int, default=25)
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default="results/db6_daycurve.csv")
    args = ap.parse_args()
    device = pick_device(args.device); mlabel = machine_label(device)
    cache = Path(__file__).resolve().parents[1]/"data_cache"/"db6"; nC=len(DB6_CLASSES); rest_idx=0
    mc = preset("S", supcon=False, adversarial=False)
    folds = [([1,2,3,4,5],[6,7,8,9,10]), ([6,7,8,9,10],[1,2,3,4,5])]   # (test, base)
    print(f"=== DB6 DAY-SEPARATION CURVE (n=10, 2-fold) === {mlabel}")

    def znorm(X, fit): m=X[fit].mean((0,1)); sd=X[fit].std((0,1))+1e-6; return ((X-m)/sd).astype(np.float32)
    rows = []
    for test_subs, base_subs in folds:
        # base on pretrain subjects' DAY-1 windows only (population prior, memory-light)
        Xtr, ytr = [], []
        for s in base_subs:
            d = load_all_days(args.root, s, cache); d1 = d["day"]==1
            Xn = znorm(d["X"], np.flatnonzero(d1))[d1]; Xtr.append(Xn); ytr.append(d["y"][d1])
        Xtr=np.concatenate(Xtr); ytr=np.concatenate(ytr); c_in=Xtr.shape[2]
        keep=cap_rest_indices(ytr, rest_class_idx=rest_idx, mult=1.5, seed=1337); Xtr,ytr=Xtr[keep],ytr[keep]
        base=build_v2net(c_in,nC,mc,n_subjects=0).to(device); nval=max(1,len(Xtr)//10)
        print(f"  fold base on {base_subs}: {len(Xtr)} win, train {args.base_epochs} ep")
        engine.train_supervised(base, Xtr[:-nval],ytr[:-nval],Xtr[-nval:],ytr[-nval:],device,n_classes=nC,
                                epochs=args.base_epochs, aug_kw=dict(channel_shift=2), seed=1337, log=lambda *_:None)
        bstate={k:v.clone() for k,v in base.state_dict().items()}
        del Xtr,ytr
        for s in test_subs:
            d = load_all_days(args.root, s, cache); X,y,g,rep,day = d["X"],d["y"],d["group"],d["rep"],d["day"]
            d1cal = (day==1)&np.isin(rep,[1,2,3,4,5,6,7,8])
            if d1cal.sum()==0: continue
            Xn1 = znorm(X, np.flatnonzero(d1cal))
            ci1 = np.flatnonzero(d1cal); ci1=ci1[cap_rest_indices(y[ci1],rest_class_idx=rest_idx,mult=1.5,seed=1337)]
            m1 = build_v2net(c_in,nC,mc,n_subjects=0).to(device); m1.load_state_dict(bstate,strict=False)
            m1 = engine.fewshot_adapt(m1, Xn1[ci1], y[ci1], device, n_classes=nC, epochs=args.cal_epochs, lr=3e-4, head_only=False, seed=1337)
            def pe(model, Xn, mask):
                idx=np.flatnonzero(mask); idx=idx[np.argsort(g[idx],kind="stable")]
                preds,_=engine.predict(model,Xn[idx],device); st,sp=majority_vote_by_group(y[idx],preds,g[idx],nC)
                return balanced_accuracy(st,sp,nC)*100
            rec = dict(subject=s)
            for K in (1,2,3,4,5):
                te = (day==K)&np.isin(rep,[9,10,11,12])
                if te.sum()==0: rec[f"noadapt_d{K}"]=np.nan; rec[f"recal_d{K}"]=np.nan; continue
                rec[f"noadapt_d{K}"] = pe(m1, Xn1, te)               # day-1 model + day-1 norm on day K
                # recal: calibrate on day-K reps 1-8 with day-K norm
                dkc=(day==K)&np.isin(rep,[1,2,3,4,5,6,7,8]); Xnk=znorm(X,np.flatnonzero(day==K))
                cik=np.flatnonzero(dkc); cik=cik[cap_rest_indices(y[cik],rest_class_idx=rest_idx,mult=1.5,seed=1337)]
                mk=build_v2net(c_in,nC,mc,n_subjects=0).to(device); mk.load_state_dict(bstate,strict=False)
                mk=engine.fewshot_adapt(mk,Xnk[cik],y[cik],device,n_classes=nC,epochs=args.cal_epochs,lr=3e-4,head_only=False,seed=1337)
                rec[f"recal_d{K}"]=pe(mk,Xnk,te)
            rows.append(rec)
            print(f"  S{s}: no-adapt d1-d5 = " + " ".join(f"{rec[f'noadapt_d{K}']:.0f}" for K in (1,2,3,4,5))
                  + " | recal = " + " ".join(f"{rec[f'recal_d{K}']:.0f}" for K in (1,2,3,4,5)))

    import csv
    out=Path(args.out); out=out if out.is_absolute() else Path(__file__).resolve().parents[1]/out
    with open(out,"w",newline="") as fh: w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    a=lambda k:np.array([r[k] for r in rows if not np.isnan(r.get(k,np.nan))])
    print(f"\n=== DAY-SEPARATION CURVE (n={len(rows)}) per-execution bal acc ===")
    print(f"  {'test day':>9} {'no-adapt':>12} {'recal':>12}")
    for K in (1,2,3,4,5):
        na,rc=a(f'noadapt_d{K}'),a(f'recal_d{K}')
        print(f"  {('D'+str(K)+(' (within)' if K==1 else '')):>9} {na.mean():5.1f}+/-{na.std():4.1f} {rc.mean():5.1f}+/-{rc.std():4.1f}")
    na1,na5=a('noadapt_d1'),a('noadapt_d5'); rc5=a('recal_d5')
    print(f"\n  day1->day5 no-adapt drop: {na1.mean()-na5.mean():.1f} pts | recovered by recal to {rc5.mean():.1f}% (vs {na1.mean():.1f}% within)")
    print(f"  wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

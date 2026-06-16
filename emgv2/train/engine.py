"""Training / adaptation engine for V2Net.

Operates on in-memory numpy arrays ([N, T, C]) to keep things fast and
dependency-light. Provides: SupCon pre-training, supervised fine-tuning (with
optional adversarial subject alignment), prediction, test-time BN+TENT
adaptation, and leak-safe few-shot calibration.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import augment_batch, two_views
from .objectives import supcon_loss


def to_ct(x_np: np.ndarray, device) -> torch.Tensor:
    """[N, T, C] numpy -> [N, C, T] float tensor on device."""
    t = torch.from_numpy(np.ascontiguousarray(x_np)).float()
    return t.permute(0, 2, 1).contiguous().to(device)


def _batches(n, bs, shuffle, rng, drop_last=False):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, n, bs):
        chunk = idx[s:s + bs]
        if drop_last and len(chunk) < 2:
            continue
        yield chunk


def predict(model, X_np, device, bs=256):
    model.eval()
    preds, logits_all = [], []
    with torch.no_grad():
        for s in range(0, len(X_np), bs):
            xb = to_ct(X_np[s:s + bs], device)
            out = model(xb)
            logits_all.append(out.cpu())
            preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds), torch.cat(logits_all).numpy()


def predict_tta(model, X_np, device, *, aug_kw, n_views=8, bs=256, seed=0):
    """Test-time augmentation: average the logits over n_views gently-augmented
    forward passes. Each view perturbs the window the way intra-session variability
    would (magnitude warp, small noise, no channel shift), and averaging denoises
    the per-window prediction — a real, leak-free per-window accuracy gain (the
    test labels are never used; only the test *inputs* are augmented). Falls back to
    a single clean pass when n_views<=1. Drop-in for predict()."""
    if not aug_kw or n_views <= 1:
        return predict(model, X_np, device, bs=bs)
    model.eval()
    torch.manual_seed(seed)
    acc = None
    with torch.no_grad():
        for v in range(n_views):
            logits_all = []
            for s in range(0, len(X_np), bs):
                xb = to_ct(X_np[s:s + bs], device)
                xb = augment_batch(xb, **aug_kw)
                logits_all.append(model(xb).cpu())
            lv = torch.cat(logits_all)
            acc = lv if acc is None else acc + lv
    mean_logits = (acc / n_views).numpy()
    return mean_logits.argmax(1), mean_logits


def embed(model, X_np, device, bs=256):
    """Backbone embeddings [N, D] for a window array (no grad)."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X_np), bs):
            out.append(model.embed(to_ct(X_np[s:s + bs], device)).cpu())
    return torch.cat(out).numpy()


def prototype_logits(emb_test, emb_cal, y_cal, n_classes, *, tau=0.1):
    """Cosine-similarity-to-class-prototype logits (a nearest-class-mean head).

    With only a handful of calibration reps, a fine-tuned linear head overfits;
    classifying by similarity to per-class mean embeddings (prototypical networks,
    Snell 2017) is far more sample-efficient and uses the SupCon embedding the way
    it was trained to be used. Returns [N_test, n_classes] logits; classes absent
    from calibration get -inf so they are never predicted. Blend these with the
    linear head's logits for the best of both.
    """
    emb_cal = np.asarray(emb_cal); emb_test = np.asarray(emb_test)
    ec = emb_cal / (np.linalg.norm(emb_cal, axis=1, keepdims=True) + 1e-8)
    et = emb_test / (np.linalg.norm(emb_test, axis=1, keepdims=True) + 1e-8)
    protos = np.full((n_classes, ec.shape[1]), np.nan, dtype=np.float64)
    for c in range(n_classes):
        m = y_cal == c
        if m.any():
            p = ec[m].mean(axis=0)
            protos[c] = p / (np.linalg.norm(p) + 1e-8)
    logits = np.full((len(et), n_classes), -np.inf, dtype=np.float64)
    present = ~np.isnan(protos).any(axis=1)
    logits[:, present] = (et @ protos[present].T) / tau
    return logits


def class_weights(y, n_classes, device):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    med = np.median(counts[counts > 0]) if np.any(counts > 0) else 1.0
    w = med / np.maximum(counts, 1.0)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def ssl_pretrain(model, X_np, device, *, epochs=20, bs=256, lr=1e-3, mask_frac=0.4,
                 n_spans=3, seed=0, log=lambda *_: None):
    """Self-supervised masked pretraining — let the backbone learn the structure of
    EMG 'moves' before it ever sees a label.

    Random time-spans of each window are hidden (zeroed) and the model must
    reconstruct the missing muscle activity from what remains (a 1-D masked
    autoencoder, MAE-style). To fill the gaps it is forced to learn how muscles
    co-fire and how contractions build — the actual concept of a move — which the
    label-supervised stages then just put names on. Trains backbone + ssl_decoder."""
    rng = np.random.default_rng(seed)
    params = [p for n, p in model.named_parameters()
              if n.startswith("backbone") or n.startswith("ssl_decoder")]
    opt = torch.optim.Adam(params, lr=lr)
    X_g = to_ct(X_np, device)                                  # [N, C, T]
    T = X_g.shape[-1]
    span_w = max(1, int(mask_frac * T / max(1, n_spans)))
    ar = torch.arange(T, device=device)[None, :]
    model.train()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for chunk in _batches(len(X_np), bs, True, rng, drop_last=True):
            ci = torch.as_tensor(chunk, device=device)
            xb = X_g.index_select(0, ci)                       # [B, C, T]
            B = xb.shape[0]
            m = torch.zeros(B, T, dtype=torch.bool, device=device)
            for _ in range(n_spans):
                start = torch.randint(0, max(1, T - span_w + 1), (B,), device=device)
                m |= (ar >= start[:, None]) & (ar < (start + span_w)[:, None])
            xin = xb * (~m)[:, None, :]                        # hide the masked spans
            recon = model.reconstruct(xin)
            diff = (recon - xb) ** 2 * m[:, None, :].float()   # score only hidden parts
            loss = diff.sum() / (m.float().sum() * xb.shape[1] + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        log(f"    ssl ep{ep+1}/{epochs} recon_loss={tot/max(1,nb):.4f}")
    return model


def supcon_pretrain(model, X_np, y, device, *, epochs=30, bs=256, tau=0.1,
                    lr=1e-3, aug_kw=None, seed=0, log=lambda *_: None):
    aug_kw = aug_kw or {}
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam([p for n, p in model.named_parameters()
                            if "classifier" not in n and "adv" not in n], lr=lr)
    X_g = to_ct(X_np, device)                    # preload once on GPU (no per-batch copy)
    y_t = torch.from_numpy(y).long().to(device)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for chunk in _batches(len(X_np), bs // 2, True, rng, drop_last=True):
            ci = torch.as_tensor(chunk, device=device)
            xb = X_g.index_select(0, ci)
            v1, v2 = two_views(xb, **aug_kw)
            x2 = torch.cat([v1, v2], 0)
            z = model.project(x2)
            lbl = torch.cat([y_t.index_select(0, ci), y_t.index_select(0, ci)])
            loss = supcon_loss(z, lbl, tau=tau)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        log(f"    supcon ep{ep+1}/{epochs} loss={tot/max(1,len(X_np)//(bs//2)):.4f}")
    return model


def train_supervised(model, Xtr, ytr, Xval, yval, device, *, n_classes, epochs=60,
                     bs=64, lr=5e-4, weight_decay=5e-4, label_smoothing=0.1,
                     patience=15, aug_kw=None, use_class_weights=True,
                     subj_tr=None, adv_lambda_max=0.0, seed=0, log=lambda *_: None):
    aug_kw = aug_kw or {}
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    cw = class_weights(ytr, n_classes, device) if use_class_weights else None
    ce = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    adv_ce = nn.CrossEntropyLoss()
    ytr_t = torch.from_numpy(ytr).long()
    subj_t = torch.from_numpy(subj_tr).long() if subj_tr is not None else None
    use_adv = (adv_lambda_max > 0) and getattr(model, "adversarial", False) and subj_t is not None

    n_subj = int(subj_t.max().item()) + 1 if subj_t is not None else 0
    Xtr_g = to_ct(Xtr, device)                   # preload once on GPU (no per-batch copy)
    ytr_t = ytr_t.to(device)
    if subj_t is not None:
        subj_t = subj_t.to(device)
    best_val, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(epochs):
        model.train()
        p = ep / max(1, epochs - 1)
        lambd = adv_lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) if use_adv else 0.0
        adv_loss_sum, adv_correct, adv_total = 0.0, 0, 0
        for chunk in _batches(len(Xtr), bs, True, rng, drop_last=True):
            ci = torch.as_tensor(chunk, device=device)
            xb = augment_batch(Xtr_g.index_select(0, ci), **aug_kw)
            yb = ytr_t.index_select(0, ci)
            logits = model(xb)
            loss = ce(logits, yb)
            if use_adv and lambd > 0:
                sb = subj_t.index_select(0, ci)
                advl = model.adv_logits(xb, lambd)
                aloss = adv_ce(advl, sb)
                loss = loss + aloss
                adv_loss_sum += float(aloss.detach())
                adv_correct += int((advl.argmax(1) == sb).sum())
                adv_total += len(sb)
            opt.zero_grad(); loss.backward(); opt.step()

        # validation (balanced accuracy proxy = plain acc here; cheap)
        vp, _ = predict(model, Xval, device, bs=256)
        val = float(np.mean(vp == yval))
        if val > best_val:
            best_val, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        # log every epoch (sanity / progress visibility)
        if True:
            msg = f"    ft ep{ep+1}/{epochs} val_acc={val*100:.1f}% (best {best_val*100:.1f}%)"
            if use_adv and adv_total > 0:
                # discriminator acc vs chance (1/n_subj): low => encoder is confusing it
                msg += (f" | adv lambda={lambd:.2f} disc_acc={adv_correct/adv_total*100:.0f}% "
                        f"(chance {100.0/max(1,n_subj):.0f}%)")
            log(msg)
        if bad >= patience:
            log(f"    early stop @ ep{ep+1}")
            break
    model.load_state_dict(best_state)
    return model


def adapt_bn_tent(model, X_eval, device, *, bn_passes=1, tent_steps=10,
                  tent_lr=1e-4, bs=64, seed=0):
    """Unsupervised, label-free test-time adaptation (transductive).

    Phase 1: forward passes in train mode recalibrate BN running stats.
    Phase 2 (TENT): minimise prediction entropy w.r.t. BN affine params only.
    Returns an adapted *copy* of the model (original untouched).
    """
    m = copy.deepcopy(model)
    rng = np.random.default_rng(seed)
    # Phase 1: BN stat recalibration
    m.train()
    with torch.no_grad():
        for _ in range(bn_passes):
            # drop_last: BN in train mode rejects a final chunk of size 1
            # (Expected more than 1 value per channel). Phase 2 already guards.
            for chunk in _batches(len(X_eval), bs, True, rng, drop_last=True):
                m(to_ct(X_eval[chunk], device))
    # Phase 2: TENT on BN affine params
    bn_params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d,)):
            mod.requires_grad_(True)
            if mod.weight is not None:
                bn_params += [mod.weight, mod.bias]
    if bn_params and tent_steps > 0:
        opt = torch.optim.Adam(bn_params, lr=tent_lr)
        m.train()
        for _ in range(tent_steps):
            for chunk in _batches(len(X_eval), bs, True, rng, drop_last=True):
                logits = m(to_ct(X_eval[chunk], device))
                pr = F.softmax(logits, dim=1)
                ent = -(pr * torch.log(pr + 1e-8)).sum(1).mean()
                opt.zero_grad(); ent.backward(); opt.step()
    m.eval()
    return m


def fewshot_adapt(model, X_cal, y_cal, device, *, n_classes, epochs=20, lr=1e-3,
                  head_only=True, seed=0, class_weights_on=True, aug_kw=None,
                  label_smoothing=0.0, weight_decay=0.0):
    """Fine-tune on a small leak-safe calibration set; return adapted copy.

    class_weights_on weights the loss by inverse class frequency so the head does
    not default to the majority (Rest) class on Rest-heavy calibration data — the
    main cause of movements being misclassified as Rest.

    aug_kw (when given) augments each calibration batch. With only a few cal reps,
    augmenting models the intra-session variability between the calibration reps and
    the held-out test rep (contraction strength, slight timing, sensor noise) — a
    real generalization gain. Use a GENTLE recipe (no channel shift: electrodes do
    not move within one session). label_smoothing + weight_decay further regularize
    the tiny-data fit.
    """
    m = copy.deepcopy(model)
    if head_only:
        for n, p in m.named_parameters():
            p.requires_grad_("classifier" in n)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    cw = class_weights(y_cal, n_classes, device) if class_weights_on else None
    ce = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    X_g = to_ct(X_cal, device)                   # preload once on GPU
    y_t = torch.from_numpy(np.asarray(y_cal)).long().to(device)
    rng = np.random.default_rng(seed)
    m.train()
    bs = min(256, max(2, len(X_cal)))
    for _ in range(epochs):
        for chunk in _batches(len(X_cal), bs, True, rng, drop_last=len(X_cal) > 4):
            ci = torch.as_tensor(chunk, device=device)
            xb = X_g.index_select(0, ci)
            if aug_kw:
                xb = augment_batch(xb, **aug_kw)
            logits = m(xb)
            loss = ce(logits, y_t.index_select(0, ci))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m

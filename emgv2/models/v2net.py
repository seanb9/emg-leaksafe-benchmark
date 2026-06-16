"""V2Net — the ReAble v2 model.

A compact depthwise-separable 1D temporal CNN backbone (32->64->128) producing a
128-d embedding, with toggleable heads:
  * classifier head            (always)
  * SupCon projection head     (config.model.supcon)
  * adversarial subject head   (config.model.adversarial) via gradient reversal

The v1 anatomical muscle-demix dual-path is intentionally NOT included by default
(decision: drop it; v1's lightweight variant without it beat the full model at
every calibration level). It can be reintroduced as an ablation toggle later.

Input tensor: [B, C, T]  (C = electrode channels, T = window samples).
Small enough for STM32-class / ESP32 deployment; param count is reported.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverse.apply(x, lambd)


class SepConv1d(nn.Module):
    """Depthwise temporal conv + pointwise conv (MobileNet-style), BN + ReLU."""

    def __init__(self, c_in, c_out, k=5, pool=2, p_drop=0.1):
        super().__init__()
        self.dw = nn.Conv1d(c_in, c_in, k, padding=k // 2, groups=c_in, bias=False)
        self.pw = nn.Conv1d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        x = self.pw(self.dw(x))
        x = self.act(self.bn(x))
        return self.drop(self.pool(x))


class Backbone(nn.Module):
    """Variable-depth depthwise-separable temporal CNN.

    `widths` = (stem_out, block1_out, block2_out, ...). The stem is a plain conv;
    each subsequent width adds a depthwise-separable block with 2x temporal pool.
    Embedding dim = widths[-1]. This lets the capacity sweep span ~56K..~1M params
    by changing widths/depth, since hardware is no longer ESP32-locked.
    """

    def __init__(self, c_in, widths=(32, 64, 128), kernel=7, p_drop=0.1):
        super().__init__()
        assert len(widths) >= 2, "need a stem width + at least one block"
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, widths[0], kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(p_drop),
        )
        blocks = []
        for i in range(1, len(widths)):
            drop = p_drop if i < len(widths) - 1 else min(2 * p_drop, 0.3)
            blocks.append(SepConv1d(widths[i - 1], widths[i], k=5, pool=2, p_drop=drop))
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embed_dim = widths[-1]

    def forward(self, x, return_map=False):   # x: [B, C, T]
        x = self.stem(x)
        x = self.blocks(x)                     # [B, embed_dim, T']
        if return_map:
            return x
        return self.gap(x).squeeze(-1)         # [B, embed_dim]


class V2Net(nn.Module):
    def __init__(self, c_in, n_classes, *, widths=(32, 64, 128), kernel=7,
                 p_drop=0.1, fc=64, clf_drop=0.5, supcon=False, proj_dim=128,
                 adversarial=False, n_subjects=0):
        super().__init__()
        self.c_in = c_in
        self.backbone = Backbone(c_in, widths, kernel, p_drop)
        d = self.backbone.embed_dim
        # self-supervised reconstruction decoder (training-only; dropped at deploy).
        # Upsamples the pre-GAP feature map back to the input window so the backbone
        # can be pretrained by masked reconstruction (learn the structure of moves).
        n_pools = len(widths)                  # stem + (len-1) blocks, each /2
        dec, ch = [], d
        for i in range(n_pools):
            out_ch = c_in if i == n_pools - 1 else max(c_in, ch // 2)
            dec.append(nn.ConvTranspose1d(ch, out_ch, 4, stride=2, padding=1))
            if i < n_pools - 1:
                dec += [nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)]
            ch = out_ch
        self.ssl_decoder = nn.Sequential(*dec)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(d), nn.Dropout(clf_drop),
            nn.Linear(d, fc), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(fc, n_classes),
        )
        self.supcon = supcon
        if supcon:
            self.proj = nn.Sequential(
                nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True),
                nn.Linear(d, proj_dim),
            )
        self.adversarial = adversarial and n_subjects > 0
        if self.adversarial:
            self.adv = nn.Sequential(
                nn.Linear(d, 64), nn.ReLU(inplace=True), nn.Linear(64, n_subjects)
            )

    def embed(self, x):
        return self.backbone(x)

    def reconstruct(self, x):
        """Self-supervised reconstruction of the input window from its (masked)
        feature map. Returns [B, C, T] matched to the input length."""
        T = x.shape[-1]
        fmap = self.backbone(x, return_map=True)
        r = self.ssl_decoder(fmap)
        if r.shape[-1] != T:                    # crop/pad to exact input length
            if r.shape[-1] > T:
                r = r[..., :T]
            else:
                r = nn.functional.pad(r, (0, T - r.shape[-1]))
        return r

    def forward(self, x, *, return_embed=False):
        z = self.backbone(x)
        logits = self.classifier(z)
        if return_embed:
            return logits, z
        return logits

    def project(self, x):
        assert self.supcon, "model built without SupCon projection head"
        z = self.backbone(x)
        p = self.proj(z)
        return nn.functional.normalize(p, dim=1)

    def adv_logits(self, x, lambd=1.0):
        assert self.adversarial, "model built without adversarial head"
        z = grad_reverse(self.backbone(x), lambd)
        return self.adv(z)

    def num_params(self) -> int:
        """All trainable params (includes training-only SupCon/adversarial heads)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def deploy_params(self) -> int:
        """Params that actually ship: backbone + classifier only.

        The SupCon projection head and adversarial subject discriminator are used
        only during training and are dropped at inference, so the deployed model
        is much smaller than num_params() for wide backbones.
        """
        n = sum(p.numel() for p in self.backbone.parameters())
        n += sum(p.numel() for p in self.classifier.parameters())
        return n


def build_v2net(c_in, n_classes, model_cfg=None, n_subjects=0) -> V2Net:
    mc = model_cfg or {}
    return V2Net(
        c_in=c_in, n_classes=n_classes,
        widths=tuple(mc.get("widths", (32, 64, 128))),
        kernel=int(mc.get("kernel", 7)),
        p_drop=float(mc.get("backbone_dropout", 0.1)),
        fc=int(mc.get("fc_units", 64)),
        clf_drop=float(mc.get("classifier_dropout", 0.5)),
        supcon=bool(mc.get("supcon", False)),
        proj_dim=int(mc.get("proj_dim", 128)),
        adversarial=bool(mc.get("adversarial", False)),
        n_subjects=n_subjects,
    )

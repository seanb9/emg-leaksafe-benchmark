"""EMG augmentations operating on [B, C, T] tensors.

Used both for fine-tuning and (two independent views) for SupCon pre-training.
All augmentations are per-sample. Cross-subject invariance augmentations
(channel shift, magnitude warp, channel scale, gaussian noise, time mask) mirror
the v1 recipe but vectorised in torch.
"""
from __future__ import annotations

import torch


def augment_batch(x: torch.Tensor, *, channel_shift=2, time_mask_frac=0.125,
                  n_time_masks=2, mag_warp_std=0.3, noise_std=0.05,
                  chan_scale=(0.8, 1.2)) -> torch.Tensor:
    """Return an augmented copy of x [B, C, T]. Fully vectorised (no per-sample
    Python loops) so it runs on the GPU at batch speed."""
    B, C, T = x.shape
    dev = x.device
    out = x

    # per-sample circular channel shift, vectorised via gather:
    # roll(x_i, s_i) == x_i[(c - s_i) % C]
    if channel_shift > 0:
        shifts = torch.randint(-channel_shift, channel_shift + 1, (B,), device=dev)
        idx = (torch.arange(C, device=dev)[None, :] - shifts[:, None]) % C   # [B, C]
        out = torch.gather(out, 1, idx[:, :, None].expand(B, C, T))
    else:
        out = out.clone()

    # per-sample, per-channel magnitude warp + scale
    warp = 1.0 + mag_warp_std * torch.randn(B, C, 1, device=dev)
    lo, hi = chan_scale
    scale = lo + (hi - lo) * torch.rand(B, C, 1, device=dev)
    out = out * warp * scale

    # gaussian noise
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)

    # per-sample time masking, vectorised over the batch (loop only over #masks)
    if n_time_masks > 0 and time_mask_frac > 0:
        max_w = max(1, int(T * time_mask_frac))
        ar = torch.arange(T, device=dev)[None, :]                            # [1, T]
        for _ in range(n_time_masks):
            w = torch.randint(1, max_w + 1, (B,), device=dev)
            start = (torch.rand(B, device=dev) * (T - w).clamp(min=1).float()).long()
            m = (ar >= start[:, None]) & (ar < (start + w)[:, None])         # [B, T]
            out = out.masked_fill(m[:, None, :], 0.0)
    return out


def two_views(x: torch.Tensor, **kw) -> tuple[torch.Tensor, torch.Tensor]:
    return augment_batch(x, **kw), augment_batch(x, **kw)

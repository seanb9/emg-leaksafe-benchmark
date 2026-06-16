"""Training objectives: supervised contrastive (SupCon) loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive loss (Khosla et al. 2020).

    z: [B, D] L2-normalized embeddings (e.g. concatenated two-view batch).
    labels: [B] class indices. All same-class samples (except self) are positives.
    Pulls same-gesture embeddings together across subjects.
    """
    device = z.device
    B = z.shape[0]
    sim = z @ z.t() / tau                                  # [B, B]
    # numerical stability
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & ~self_mask         # same class, not self

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.zeros((), device=device, requires_grad=True)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()

"""Original CG3 structural contrastive loss (local ↔ global views).

Faithful extraction of `GNNModel._contrastive_loss` from the snapshot port:
unsupervised pairwise InfoNCE-style term + supervised contrastive on labeled
nodes using intra/inter class indicator matrices.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.losses.base import BaseViewLoss


class StructuralContrastiveLoss(BaseViewLoss):
    name = "structural"

    def __init__(self, temperature: float = 0.5, hp1: float = 0.9):
        super().__init__()
        self.temperature = float(temperature)
        self.hp1 = float(hp1)

    def forward(
        self,
        local: torch.Tensor,
        global_: torch.Tensor,
        ctx: Dict[str, Any],
        semantic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del semantic  # structural loss only aligns local ↔ global
        device = local.device
        loss = torch.zeros((), device=device)
        temp = self.temperature
        hp1 = self.hp1

        # Eq. 4 — pairwise unsupervised between local & global (and reverse).
        cos_dist = torch.exp(torch.matmul(local, global_.t()) / temp)
        neg = torch.mean(cos_dist, dim=1)
        diag_cos = torch.diagonal(cos_dist, 0)
        pos_neg1 = diag_cos / (neg + 1e-8)

        cos_dist = torch.exp(torch.matmul(global_, local.t()) / temp)
        neg = torch.mean(cos_dist, dim=1)
        diag_cos = torch.diagonal(cos_dist, 0)
        pos_neg2 = diag_cos / (neg + 1e-8)

        pos_neg3 = torch.cat([pos_neg1, pos_neg2], dim=0)
        loss = loss + (-hp1 * torch.mean(torch.log(pos_neg3.clamp(min=1e-8))))

        train_idx = ctx["train_idx"]
        mat01_intra = ctx["mat01_intra"]
        mat01_inter = ctx["mat01_inter"]
        mat01_intra_rowsum = ctx["mat01_intra_rowsum"]
        train_idx_size = int(ctx["train_idx_size"])

        # Supervised contrastive (round 1).
        h1 = local.index_select(0, train_idx)
        h2 = global_.index_select(0, train_idx)
        h_cos = torch.exp(torch.matmul(h1, h2.t()) / temp)
        sup_pos = torch.sum(h_cos * mat01_intra, dim=1)
        sup_neg = (torch.sum(h_cos * mat01_inter, dim=1) + sup_pos) / max(train_idx_size - 1, 1)
        sup_pos = sup_pos / (mat01_intra_rowsum + 1e-8)
        pos_neg_sup_1 = sup_pos / (sup_neg + 1e-8)

        # Supervised contrastive (round 2, swapped).
        h2_b = local.index_select(0, train_idx)
        h1_b = global_.index_select(0, train_idx)
        h_cos = torch.exp(torch.matmul(h1_b, h2_b.t()) / temp)
        sup_pos = torch.sum(h_cos * mat01_intra, dim=1)
        sup_neg = (torch.sum(h_cos * mat01_inter, dim=1) + sup_pos) / max(train_idx_size - 1, 1)
        sup_pos = sup_pos / (mat01_intra_rowsum + 1e-8)
        pos_neg_sup_2 = sup_pos / (sup_neg + 1e-8)

        pos_neg_sup_3 = torch.cat([pos_neg_sup_1, pos_neg_sup_2], dim=0)
        loss = loss + (-hp1 * torch.mean(torch.log(pos_neg_sup_3.clamp(min=1e-8))))
        return loss

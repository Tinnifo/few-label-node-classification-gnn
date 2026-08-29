"""Pluggable view-alignment losses for multi-view GNNs."""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class BaseViewLoss(nn.Module, abc.ABC):
    """Interface every pluggable view loss must implement."""

    name: str = "base"

    @abc.abstractmethod
    def forward(
        self,
        local: torch.Tensor,
        global_: torch.Tensor,
        ctx: Dict[str, Any],
        semantic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a scalar loss term."""

    def regularizer_value(self) -> Optional[torch.Tensor]:
        """Optional detached scalar for logging a separate `loss_reg` term."""
        return None


class StructuralContrastiveLoss(BaseViewLoss):
    """Original CG3 structural contrastive loss (local ↔ global views)."""

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
        del semantic
        device = local.device
        loss = torch.zeros((), device=device)
        temp = self.temperature
        hp1 = self.hp1

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

        h1 = local.index_select(0, train_idx)
        h2 = global_.index_select(0, train_idx)
        h_cos = torch.exp(torch.matmul(h1, h2.t()) / temp)
        sup_pos = torch.sum(h_cos * mat01_intra, dim=1)
        sup_neg = (torch.sum(h_cos * mat01_inter, dim=1) + sup_pos) / max(
            train_idx_size - 1, 1
        )
        sup_pos = sup_pos / (mat01_intra_rowsum + 1e-8)
        pos_neg_sup_1 = sup_pos / (sup_neg + 1e-8)

        h2_b = local.index_select(0, train_idx)
        h1_b = global_.index_select(0, train_idx)
        h_cos = torch.exp(torch.matmul(h1_b, h2_b.t()) / temp)
        sup_pos = torch.sum(h_cos * mat01_intra, dim=1)
        sup_neg = (torch.sum(h_cos * mat01_inter, dim=1) + sup_pos) / max(
            train_idx_size - 1, 1
        )
        sup_pos = sup_pos / (mat01_intra_rowsum + 1e-8)
        pos_neg_sup_2 = sup_pos / (sup_neg + 1e-8)

        pos_neg_sup_3 = torch.cat([pos_neg_sup_1, pos_neg_sup_2], dim=0)
        loss = loss + (-hp1 * torch.mean(torch.log(pos_neg_sup_3.clamp(min=1e-8))))
        return loss


class DisparityLoss(BaseViewLoss):
    """Placeholder: push semantic embeddings away from structural views."""

    name = "disparity"

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)
        self._last_reg: Optional[torch.Tensor] = None

    def forward(
        self,
        local: torch.Tensor,
        global_: torch.Tensor,
        ctx: Dict[str, Any],
        semantic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del global_, ctx, semantic
        zero = torch.zeros((), device=local.device)
        self._last_reg = zero.detach()
        return self.weight * zero

    def regularizer_value(self) -> Optional[torch.Tensor]:
        return self._last_reg


class HSICLoss(BaseViewLoss):
    """Placeholder: HSIC between semantic and structural views."""

    name = "hsic"

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)
        self._last_reg: Optional[torch.Tensor] = None

    def forward(
        self,
        local: torch.Tensor,
        global_: torch.Tensor,
        ctx: Dict[str, Any],
        semantic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del global_, ctx, semantic
        zero = torch.zeros((), device=local.device)
        self._last_reg = zero.detach()
        return self.weight * zero

    def regularizer_value(self) -> Optional[torch.Tensor]:
        return self._last_reg


class CompositeViewLoss(BaseViewLoss):
    """`structural + lambda_reg * regularizer` (disparity / HSIC / …)."""

    name = "composite"

    def __init__(
        self,
        structural: BaseViewLoss,
        regularizer: BaseViewLoss,
        lambda_reg: float = 1.0,
    ):
        super().__init__()
        self.structural = structural
        self.regularizer = regularizer
        self.lambda_reg = float(lambda_reg)
        self._last_reg: Optional[torch.Tensor] = None

    def forward(
        self,
        local: torch.Tensor,
        global_: torch.Tensor,
        ctx: Dict[str, Any],
        semantic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        structural_term = self.structural(local, global_, ctx, semantic=semantic)
        reg_term = self.regularizer(local, global_, ctx, semantic=semantic)
        self._last_reg = reg_term.detach()
        return structural_term + self.lambda_reg * reg_term

    def regularizer_value(self) -> Optional[torch.Tensor]:
        return self._last_reg


def build_loss(
    name_or_cfg: Union[str, Any] = "structural",
    **kwargs,
) -> BaseViewLoss:
    """Instantiate a view loss from a name string or an object with `.name`."""
    if isinstance(name_or_cfg, str):
        name = name_or_cfg
        get = kwargs.get
    else:
        cfg = name_or_cfg
        loss_cfg = getattr(cfg, "loss", cfg)
        name = str(getattr(loss_cfg, "name", "structural"))
        get = lambda key, default=None: getattr(loss_cfg, key, default)

    if name == "structural":
        return StructuralContrastiveLoss(
            temperature=float(get("temperature", 0.5)),
            hp1=float(get("hp1", 0.9)),
        )
    if name == "disparity":
        return DisparityLoss(weight=float(get("weight", 1.0)))
    if name == "hsic":
        return HSICLoss(weight=float(get("weight", 1.0)))
    if name == "structural_plus_disparity":
        return CompositeViewLoss(
            structural=StructuralContrastiveLoss(
                temperature=float(get("temperature", 0.5)),
                hp1=float(get("hp1", 0.9)),
            ),
            regularizer=DisparityLoss(weight=1.0),
            lambda_reg=float(get("lambda_reg", 1.0)),
        )
    if name == "structural_plus_hsic":
        return CompositeViewLoss(
            structural=StructuralContrastiveLoss(
                temperature=float(get("temperature", 0.5)),
                hp1=float(get("hp1", 0.9)),
            ),
            regularizer=HSICLoss(weight=1.0),
            lambda_reg=float(get("lambda_reg", 1.0)),
        )
    raise ValueError(f"Unknown loss '{name}'.")

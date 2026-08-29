"""Disparity regularization placeholder (semantic vs structural views).

Intended to push the LLM semantic embedding away from structural views so
complementary information is preserved. Replace `forward` with e.g. negative
cosine similarity or L2 distance between semantic and (local/global) means.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.losses.base import BaseViewLoss


class DisparityLoss(BaseViewLoss):
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
        # TODO: implement disparity between semantic and structural embeddings.
        zero = torch.zeros((), device=local.device)
        self._last_reg = zero.detach()
        return self.weight * zero

    def regularizer_value(self) -> Optional[torch.Tensor]:
        return self._last_reg


"""Composite view loss: structural contrastive + λ * regularizer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.losses.base import BaseViewLoss


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


"""HSIC regularization placeholder (semantic vs structural views).

Hilbert-Schmidt Independence Criterion can discourage dependence between the
LLM semantic view and structural views while keeping structural contrastive
alignment intact. Replace `forward` with a real HSIC estimator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.losses.base import BaseViewLoss


class HSICLoss(BaseViewLoss):
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
        # TODO: implement HSIC(semantic, structural) regularization.
        zero = torch.zeros((), device=local.device)
        self._last_reg = zero.detach()
        return self.weight * zero

    def regularizer_value(self) -> Optional[torch.Tensor]:
        return self._last_reg

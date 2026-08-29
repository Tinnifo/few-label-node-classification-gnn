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

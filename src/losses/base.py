"""Pluggable view-alignment / regularization losses for multi-view GNNs.

A loss module takes local/global (and optionally semantic) embeddings plus a
context dict of buffers (train indices, intra/inter mats, etc.) and returns a
scalar term that CG3 (or a future method) adds into its total loss.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

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
        """Return a scalar loss term.

        `ctx` holds method-specific buffers, e.g. for CG3 structural contrastive:
          train_idx, mat01_intra, mat01_inter, mat01_intra_rowsum, train_idx_size
        """

    def regularizer_value(self) -> Optional[torch.Tensor]:
        """Optional detached scalar for logging a separate `loss_reg` term."""
        return None

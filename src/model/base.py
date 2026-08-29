"""Protocol for GNN backbones in `src.model`. CG3 lives in `utils.graph.GNNModel`."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseGNN(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return class logits with shape `[num_nodes, num_classes]`."""

    def prepare(self, data):
        """One-shot pre-training hook. Default: identity."""
        return data

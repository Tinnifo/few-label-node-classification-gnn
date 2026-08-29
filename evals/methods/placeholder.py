"""Placeholder methods that raise until implemented.

Shared base so each stub only needs a short docstring describing intent.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.methods.base import BaseMethod


class PlaceholderMethod(BaseMethod):
    """Raises NotImplementedError from build_model / train_step."""

    method_name: str = "placeholder"
    todo: str = "Implement this method."

    def build_model(self, in_channels: int, num_classes: int, *, data=None) -> torch.nn.Module:
        raise NotImplementedError(
            f"Method '{self.method_name}' is a placeholder. {self.todo}"
        )

    def train_step(self, model: torch.nn.Module, data,
                   optimizer: torch.optim.Optimizer, epoch: int) -> Dict[str, float]:
        raise NotImplementedError(
            f"Method '{self.method_name}' is a placeholder. {self.todo}"
        )

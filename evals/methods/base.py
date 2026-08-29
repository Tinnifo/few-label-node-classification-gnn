"""Protocol every training method follows.

`main.py` treats every method behind this surface. `cg3` ignores `cfg.model`
and builds its own architecture in `build_model`.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import torch

from evals.metrics import compute_metrics


class BaseMethod(abc.ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abc.abstractmethod
    def build_model(self, in_channels: int, num_classes: int, *, data=None) -> torch.nn.Module:
        """Build and return the trainable model."""

    def prepare(self, model: torch.nn.Module, data) -> Any:
        if hasattr(model, "prepare"):
            return model.prepare(data)
        return data

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(),
            lr=float(self.cfg.method.lr),
            weight_decay=float(self.cfg.method.weight_decay),
        )

    @abc.abstractmethod
    def train_step(self, model: torch.nn.Module, data, optimizer: torch.optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """One training step. Must call backward+step. Returns scalars to log."""

    def predict_logits(self, model: torch.nn.Module, data) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model(data.x, data.edge_index)

    def evaluate(self, model: torch.nn.Module, data,
                 mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        if mask is None:
            mask = data.test_mask
        logits = self.predict_logits(model, data)
        pred = logits.argmax(dim=1)
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        return compute_metrics(y_true, y_pred)

    def validate(self, model: torch.nn.Module, data) -> Dict[str, float]:
        import torch.nn.functional as F
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=1)
            out = {
                "train_acc": float((pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()),
            }
            if hasattr(data, "val_mask") and data.val_mask.sum() > 0:
                val_loss = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask]).item()
                val_acc = float((pred[data.val_mask] == data.y[data.val_mask]).float().mean().item())
                out["val_loss"] = val_loss
                out["val_acc"] = val_acc
                out["early_stop_metric"] = val_acc
            return out

"""Loss registry — build a pluggable view loss from a Hydra `cfg.loss` node."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.losses.base import BaseViewLoss
from src.losses.composite import CompositeViewLoss
from src.losses.disparity import DisparityLoss
from src.losses.hsic import HSICLoss
from src.losses.structural import StructuralContrastiveLoss

__all__ = [
    "BaseViewLoss",
    "StructuralContrastiveLoss",
    "DisparityLoss",
    "HSICLoss",
    "CompositeViewLoss",
    "build_loss",
]


def build_loss(cfg: DictConfig) -> BaseViewLoss:
    """Instantiate the loss named by `cfg.loss.name` (or a bare loss cfg)."""
    loss_cfg = cfg.loss if "loss" in cfg and OmegaConf.is_dict(cfg.loss) and "name" in cfg.loss else cfg
    name = str(loss_cfg.name)

    if name == "structural":
        return StructuralContrastiveLoss(
            temperature=float(loss_cfg.get("temperature", 0.5)),
            hp1=float(loss_cfg.get("hp1", 0.9)),
        )
    if name == "disparity":
        return DisparityLoss(weight=float(loss_cfg.get("weight", 1.0)))
    if name == "hsic":
        return HSICLoss(weight=float(loss_cfg.get("weight", 1.0)))
    if name == "structural_plus_disparity":
        return CompositeViewLoss(
            structural=StructuralContrastiveLoss(
                temperature=float(loss_cfg.get("temperature", 0.5)),
                hp1=float(loss_cfg.get("hp1", 0.9)),
            ),
            regularizer=DisparityLoss(weight=1.0),
            lambda_reg=float(loss_cfg.get("lambda_reg", 1.0)),
        )
    if name == "structural_plus_hsic":
        return CompositeViewLoss(
            structural=StructuralContrastiveLoss(
                temperature=float(loss_cfg.get("temperature", 0.5)),
                hp1=float(loss_cfg.get("hp1", 0.9)),
            ),
            regularizer=HSICLoss(weight=1.0),
            lambda_reg=float(loss_cfg.get("lambda_reg", 1.0)),
        )
    raise ValueError(f"Unknown loss '{name}'. Add it to build_loss in src/losses/__init__.py.")

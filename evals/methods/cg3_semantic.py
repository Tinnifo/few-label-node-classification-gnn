"""CG3 plus an LLM semantic view, routed through `SemanticGNNModel`.

Tags are optional: Planetoid runs pass `tags=None` until TAG data is wired.
"""

from __future__ import annotations

from typing import Dict

import torch

from evals.methods.cg3 import CG3Method
from src.method import SemanticGNNModel
from src.model.hgcn import HGAT, HGCN
from utils.graph import build_cg3_artifacts
from utils.losses import build_loss


class CG3SemanticMethod(CG3Method):
    method_name = "cg3_semantic"

    def build_model(self, in_channels: int, num_classes: int, *, data=None) -> torch.nn.Module:
        if data is None:
            raise RuntimeError(
                "CG3SemanticMethod.build_model requires `data=` (it preprocesses "
                "the graph hierarchy before constructing the model)."
            )

        view_loss = build_loss(self.cfg)
        artifacts = build_cg3_artifacts(
            data,
            coarsen_level=self.coarsen_level,
            max_node_wgt=self.max_node_wgt,
            channel_num=self.channel_num,
        )
        self._artifacts = artifacts

        if self.global_model_name == "hgcn":
            GlobalCls = HGCN
        elif self.global_model_name == "hgat":
            GlobalCls = HGAT
        else:
            raise ValueError(f"Unknown global_model: {self.global_model_name}")

        global_model = GlobalCls(
            input_dim=artifacts.input_dim,
            output_dim=artifacts.num_classes,
            hidden=self.hidden_global,
            transfer_list=artifacts.transfer_list,
            adj_list=artifacts.adj_list,
            node_wgt_list=artifacts.node_wgt_list,
            coarsen_level=self.coarsen_level,
            max_node_wgt=self.max_node_wgt,
            node_wgt_embed_dim=self.node_wgt_embed_dim,
            weight_decay=self.weight_decay,
            channel_num=self.channel_num,
            dropout=self.dropout,
        )

        semantic_channel = getattr(self.cfg.method, "semantic_channel", None)

        return SemanticGNNModel(
            num_classes=artifacts.num_classes,
            hidden=self.hidden_local,
            input_dim=artifacts.input_dim,
            global_model=global_model,
            semantic_channel=semantic_channel,
            train_idx=artifacts.train_idx_np,
            edge_pos=artifacts.edge_pos,
            mat01_tr_te=artifacts.mats_intra_inter,
            weight_decay=self.weight_decay,
            local_model=self.local_model,
            dropout=self.dropout,
            num_features_nonzero=artifacts.num_features_nonzero,
            view_loss=view_loss,
        )

    def _tags(self, data):
        return getattr(data, "tags", None)

    def train_step(
        self,
        model: torch.nn.Module,
        data,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        model.train()
        optimizer.zero_grad()
        outputs, loss, accuracy = model(
            data._cg3_feature_sp,
            data._cg3_support_sp,
            data._cg3_y_train_oh,
            data._cg3_train_mask_int,
            tags=self._tags(data),
        )
        loss.backward()
        optimizer.step()

        train_loss = loss.detach()
        loss_ce = getattr(model, "loss_ce", None)
        loss_gen = getattr(model, "loss_gen", None)
        loss_contrastive = getattr(model, "loss_contrastive", None)
        loss_reg = getattr(model, "loss_reg", None)
        loss_hsic = getattr(model, "loss_hsic", None)
        loss_total = getattr(model, "loss_total", loss)

        out = {
            "train_loss": float(train_loss.item()),
            "train_acc": float(accuracy.detach().item()),
            "loss_ce": float(loss_ce.detach().item()) if loss_ce is not None else None,
            "loss_gen": float(loss_gen.detach().item()) if loss_gen is not None else None,
            "loss_contrastive": float(loss_contrastive.detach().item())
            if loss_contrastive is not None
            else None,
            "loss_reg": float(loss_reg.detach().item()) if loss_reg is not None else None,
            "loss_total": float(loss_total.detach().item()),
        }
        if loss_hsic is not None:
            out["loss_hsic"] = float(loss_hsic.detach().item())
        return out

    def predict_logits(self, model: torch.nn.Module, data) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            outputs, _, _ = model(
                data._cg3_feature_sp,
                data._cg3_support_sp,
                data._cg3_y_train_oh,
                data._cg3_train_mask_int,
                tags=self._tags(data),
            )
        return outputs

    def validate(self, model: torch.nn.Module, data) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            outputs, _, _ = model(
                data._cg3_feature_sp,
                data._cg3_support_sp,
                data._cg3_y_val_oh,
                data._cg3_val_mask_int,
                tags=self._tags(data),
            )
            pred = outputs.argmax(dim=1)
        out: Dict[str, float] = {
            "train_acc": float(
                (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            ),
        }
        if hasattr(data, "val_mask") and data.val_mask.sum() > 0:
            import torch.nn.functional as F

            val_loss = F.cross_entropy(outputs[data.val_mask], data.y[data.val_mask]).item()
            val_acc = float(
                (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            )
            out["val_loss"] = val_loss
            out["val_acc"] = val_acc
            out["early_stop_metric"] = val_acc
        return out

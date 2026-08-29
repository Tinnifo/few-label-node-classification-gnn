"""CG3 local/global fusion plus a semantic channel, gated by HSIC.

Low HSIC concatenates structural and semantic representations and classifies
the joint vector. High HSIC keeps the views separate and mixes their logits
with entropy-based attention.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.graph import GraphAttention, GraphConvolution, MLP
from utils.losses import BaseViewLoss, StructuralContrastiveLoss


def rbf_kernel(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if x.size(0) <= 1:
        return torch.ones((x.size(0), x.size(0)), device=x.device, dtype=x.dtype)
    dist_sq = torch.cdist(x, x, p=2).pow(2)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


def hsic_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sigma: float = 1.0,
    max_samples: int = 1024,
) -> torch.Tensor:
    if z1.size(0) <= 1:
        return z1.new_zeros(())

    if z1.size(0) > max_samples:
        idx = torch.randperm(z1.size(0), device=z1.device)[:max_samples]
        z1 = z1.index_select(0, idx)
        z2 = z2.index_select(0, idx)

    n = z1.size(0)
    K = rbf_kernel(z1, sigma)
    L = rbf_kernel(z2, sigma)
    K_centered = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()
    L_centered = L - L.mean(dim=0, keepdim=True) - L.mean(dim=1, keepdim=True) + L.mean()
    return (K_centered * L_centered).sum() / ((n - 1) ** 2)


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -torch.sum(probs * log_probs, dim=-1)


def entropy_attention(logits_structural: torch.Tensor, logits_semantic: torch.Tensor):
    entropy_structural = compute_entropy(logits_structural)
    entropy_semantic = compute_entropy(logits_semantic)
    attention = F.softmax(
        torch.stack([-entropy_structural, -entropy_semantic], dim=-1),
        dim=-1,
    )
    alpha_structural = attention[:, 0:1]
    alpha_semantic = attention[:, 1:2]
    fused_logits = alpha_structural * logits_structural + alpha_semantic * logits_semantic
    return (
        fused_logits,
        alpha_structural,
        alpha_semantic,
        entropy_structural,
        entropy_semantic,
    )


def masked_softmax_cross_entropy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(preds, dim=1)
    loss = -(labels * log_probs).sum(dim=1)
    mask = mask.float()
    mean = mask.mean()
    if mean.item() == 0:
        return torch.zeros((), device=preds.device, dtype=preds.dtype)
    return (loss * (mask / mean)).mean()


def masked_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    correct = torch.eq(torch.argmax(preds, dim=1), torch.argmax(labels, dim=1)).float()
    mask = mask.float()
    mean = mask.mean()
    if mean.item() == 0:
        return torch.zeros((), device=preds.device, dtype=preds.dtype)
    return (correct * (mask / mean)).mean()


class SemanticGNNModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        hidden: int,
        input_dim: int,
        global_model: nn.Module,
        semantic_channel: nn.Module | None,
        train_idx,
        edge_pos,
        mat01_tr_te,
        weight_decay: float,
        local_model: str,
        dropout: float,
        num_features_nonzero: int,
        semantic_dim: int = 128,
        hsic_threshold: float = 0.1,
        hsic_sigma: float = 1.0,
        hsic_weight: float = 0.1,
        hsic_max_samples: int = 1024,
        view_loss: BaseViewLoss | None = None,
    ):
        super().__init__()
        self.weight_decay = float(weight_decay)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden1 = hidden
        self.global_model = global_model
        self.semantic_channel = semantic_channel
        self.dropout = float(dropout)
        self.semantic_dim = semantic_dim
        self.hsic_threshold = float(hsic_threshold)
        self.hsic_sigma = float(hsic_sigma)
        self.hsic_weight = float(hsic_weight)
        self.hsic_max_samples = int(hsic_max_samples)
        self.view_loss = view_loss if view_loss is not None else StructuralContrastiveLoss()

        if local_model == "gat":
            LocalLayer = GraphAttention
            hidden_act = F.elu
            hidden_dropout = self.dropout
            output_dropout = self.dropout
        elif local_model == "gcn":
            LocalLayer = GraphConvolution
            hidden_act = F.relu
            hidden_dropout = self.dropout
            output_dropout = 0.0
        else:
            raise ValueError(f"Unknown local_model: {local_model}")

        self.register_buffer(
            "edge_pos_i",
            torch.from_numpy(np.asarray(edge_pos[:, 0]).astype("int64")),
        )
        self.register_buffer(
            "edge_pos_j",
            torch.from_numpy(np.asarray(edge_pos[:, 1]).astype("int64")),
        )
        self.register_buffer(
            "train_idx_buf",
            torch.from_numpy(np.asarray(train_idx).astype("int64")),
        )
        self.register_buffer(
            "mat01_intra",
            torch.from_numpy(mat01_tr_te[0].astype("float32")),
        )
        self.register_buffer(
            "mat01_inter",
            torch.from_numpy(mat01_tr_te[1].astype("float32")),
        )
        self.register_buffer(
            "mat01_intra_rowsum",
            torch.from_numpy(np.sum(mat01_tr_te[0], axis=1).astype("float32")),
        )
        self.train_idx_size = int(np.shape(train_idx)[0])

        self.classlayers = nn.ModuleList()
        self.classlayers.append(
            LocalLayer(
                act=hidden_act,
                input_dim=self.input_dim,
                output_dim=self.hidden1,
                support=None,
                sparse_inputs=True,
                isSparse=True,
                dropout=hidden_dropout,
                num_features_nonzero=num_features_nonzero,
                bias=True,
            )
        )
        self.classlayers.append(
            LocalLayer(
                act=(lambda x: x),
                input_dim=self.hidden1,
                output_dim=self.num_classes,
                support=None,
                sparse_inputs=False,
                isSparse=True,
                dropout=output_dropout,
                num_features_nonzero=num_features_nonzero,
                bias=True,
            )
        )

        self.classifier_struct = nn.Linear(self.num_classes, self.num_classes)
        self.classifier_semantic = nn.Linear(self.semantic_dim, self.num_classes)
        self.classifier_fused = nn.Linear(
            self.num_classes + self.semantic_dim,
            self.num_classes,
        )
        self.p_e_yy_w_contra = MLP(
            act=(lambda x: x),
            input_dim=2 * self.num_classes,
            output_dim=1,
            sparse_inputs=False,
            isSparse=True,
            bias=True,
        )

        self.outputs = None
        self.concat_vec_local = None
        self.concat_vec_global = None
        self.z_structural = None
        self.z_semantic = None
        self.semantic_descriptors = None
        self.semantic_embeddings = None
        self.semantic_logits = None
        self.logits_structural = None
        self.logits_semantic = None
        self.alpha_structural = None
        self.alpha_semantic = None
        self.entropy_structural = None
        self.entropy_semantic = None
        self.hsic_value = None
        self.semantic_enabled = None
        self.loss = torch.tensor(0.0)
        self.accuracy = torch.tensor(0.0)
        self.p_e_xy = torch.tensor(0.0)

    @property
    def train_idx(self) -> torch.Tensor:
        return self.train_idx_buf

    def forward(
        self,
        features: torch.Tensor,
        support: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        tags: list[str] | None = None,
    ):
        self.classlayers[0].support = support
        self.classlayers[0].sparse_inputs = True
        h0 = self.classlayers[0](features)

        self.classlayers[1].support = support
        self.classlayers[1].sparse_inputs = False
        h1 = self.classlayers[1](h0)
        self.concat_vec_local = F.normalize(h1, p=2, dim=1)

        global_out = self.global_model(features)
        self.concat_vec_global = F.normalize(global_out, p=2, dim=1)
        self.z_structural = F.normalize(
            0.6 * self.concat_vec_local + 0.4 * self.concat_vec_global,
            p=2,
            dim=1,
        )

        semantic_available = (
            self.semantic_channel is not None
            and tags is not None
            and len(tags) == features.size(0)
        )
        if semantic_available:
            generated_descriptors, x_semantic, h_semantic, semantic_logits = (
                self.semantic_channel(tags)
            )
            structural_device = self.z_structural.device
            x_semantic = x_semantic.to(structural_device)
            h_semantic = h_semantic.to(structural_device)
            semantic_logits = semantic_logits.to(structural_device)
            self.semantic_descriptors = generated_descriptors
            self.semantic_embeddings = x_semantic
            self.semantic_logits = semantic_logits
            self.z_semantic = F.normalize(h_semantic, p=2, dim=1)
        else:
            self.semantic_descriptors = None
            self.semantic_embeddings = None
            self.semantic_logits = None
            self.z_semantic = torch.zeros(
                (self.z_structural.size(0), self.semantic_dim),
                device=self.z_structural.device,
                dtype=self.z_structural.dtype,
            )

        if semantic_available:
            loss_hsic = hsic_loss(
                self.z_structural,
                self.z_semantic,
                sigma=self.hsic_sigma,
                max_samples=self.hsic_max_samples,
            )
        else:
            loss_hsic = self.z_structural.new_zeros(())
        self.hsic_value = loss_hsic.detach()

        low_hsic = semantic_available and loss_hsic.detach().item() < self.hsic_threshold
        self.semantic_enabled = low_hsic

        logits_structural = self.classifier_struct(self.z_structural)
        self.logits_structural = logits_structural

        if low_hsic:
            z_combined = torch.cat([self.z_structural, self.z_semantic], dim=-1)
            self.outputs = self.classifier_fused(z_combined)
            self.logits_semantic = self.semantic_logits
            self.alpha_structural = None
            self.alpha_semantic = None
            self.entropy_structural = None
            self.entropy_semantic = None
        else:
            if semantic_available:
                logits_semantic = self.semantic_logits
            else:
                logits_semantic = self.classifier_semantic(self.z_semantic)
            self.logits_semantic = logits_semantic
            (
                fused_logits,
                alpha_structural,
                alpha_semantic,
                entropy_structural,
                entropy_semantic,
            ) = entropy_attention(logits_structural, logits_semantic)
            self.outputs = fused_logits
            self.alpha_structural = alpha_structural
            self.alpha_semantic = alpha_semantic
            self.entropy_structural = entropy_structural
            self.entropy_semantic = entropy_semantic

        loss_q_yobs_x_g = masked_softmax_cross_entropy(self.outputs, labels, mask)

        y_ei_local = self.concat_vec_local.index_select(0, self.edge_pos_i)
        y_ej_global = self.concat_vec_global.index_select(0, self.edge_pos_j)
        y_ei_global = self.concat_vec_global.index_select(0, self.edge_pos_i)
        y_ej_local = self.concat_vec_local.index_select(0, self.edge_pos_j)

        p_e_xy_1 = -torch.mean(
            torch.log(
                torch.sigmoid(
                    self.p_e_yy_w_contra(torch.cat([y_ei_local, y_ej_global], dim=1))
                ).clamp(min=1e-8)
            )
        )
        p_e_xy_2 = -torch.mean(
            torch.log(
                torch.sigmoid(
                    self.p_e_yy_w_contra(torch.cat([y_ei_global, y_ej_local], dim=1))
                ).clamp(min=1e-8)
            )
        )
        self.p_e_xy = p_e_xy_1 + p_e_xy_2

        loss_ctx = {
            "train_idx": self.train_idx_buf,
            "mat01_intra": self.mat01_intra,
            "mat01_inter": self.mat01_inter,
            "mat01_intra_rowsum": self.mat01_intra_rowsum,
            "train_idx_size": self.train_idx_size,
        }
        loss_contrastive = self.view_loss(
            self.concat_vec_local,
            self.concat_vec_global,
            loss_ctx,
        )

        loss_ce = loss_q_yobs_x_g
        loss_gen = self.p_e_xy
        total = loss_ce + 0.4 * loss_gen + loss_contrastive + self.hsic_weight * loss_hsic

        for i in range(2):
            for var in self.classlayers[i].vars.values():
                total = total + self.weight_decay * 0.5 * torch.sum(var ** 2)
        for var in self.p_e_yy_w_contra.vars.values():
            total = total + self.weight_decay * 0.5 * torch.sum(var ** 2)
        total = total + self.global_model.loss

        self.loss = total
        self.accuracy = masked_accuracy(self.outputs, labels, mask)
        self.loss_ce = loss_ce.detach()
        self.loss_gen = loss_gen.detach()
        self.loss_contrastive = loss_contrastive.detach()
        self.loss_hsic = loss_hsic.detach()
        self.loss_total = total.detach()
        reg = self.view_loss.regularizer_value()
        self.loss_reg = reg.detach() if reg is not None else None

        return self.outputs, self.loss, self.accuracy

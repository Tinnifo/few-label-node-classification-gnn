"""
CG3 GNNModel — local-view GCN/GAT fused with a global HGCN/HGAT view,
augmented with:

    1. Semantic channel
       TAG → Granite → descriptor → MiniLM → semantic MLP → classifier

    2. HSIC complementary-information checking

    3. HSIC-dependent decision:
         LOW HSIC
             → structural + semantic representations are combined
             → combined classifier

         HIGH HSIC
             → structural and semantic views remain separate
             → each produces its own class logits
             → entropy is calculated for both
             → adaptive attention weights are calculated
             → final prediction is attention-weighted logits

Pipeline
--------
                    LOCAL GNN
                       |
                       v
                local representation
                       |
                       |
                    GLOBAL GNN
                       |
                       v
               global representation
                       |
                       v
             CG3 structural fusion
                       |
                       v
                z_structural
                       |
                       +----------------------+
                       |                      |
                       |                      |
                     HSIC              Semantic Channel
                       |                      |
                       |                   TAGS
                       |                      |
                       |                  Granite LLM
                       |                      |
                       |                  descriptor
                       |                      |
                       |                    MiniLM
                       |                      |
                       |                 x_semantic
                       |                      |
                       |                  semantic MLP
                       |                      |
                       |                 h_semantic
                       |                      |
                       |                semantic logits
                       |                      |
                       +----------+-----------+
                                  |
                                  v
                           HSIC decision
                           /          \
                       LOW HSIC      HIGH HSIC
                          |              |
                          v              v
                    concatenate      separate views
                    structural +     structural logits
                    semantic         semantic logits
                          |              |
                          v              v
                  fused classifier    entropy(T)
                          |            entropy(S)
                          |              |
                          |              v
                          |        adaptive attention
                          |              |
                          |              v
                          |       weighted prediction
                          |              |
                          +------+-------+
                                 |
                                 v
                         final prediction
"""


from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.base import BaseViewLoss
from src.losses.structural import StructuralContrastiveLoss

from .cg3_layers import (
    MLP,
    GraphAttention,
    GraphConvolution,
)


# =====================================================================
# HSIC
# =====================================================================

def rbf_kernel(
    x: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute an RBF kernel matrix.

    Args:
        x: [N, D]

    Returns:
        K: [N, N]
    """

    if x.size(0) <= 1:
        return torch.ones(
            (x.size(0), x.size(0)),
            device=x.device,
            dtype=x.dtype,
        )

    dist_sq = torch.cdist(
        x,
        x,
        p=2,
    ).pow(2)

    return torch.exp(
        -dist_sq / (2.0 * sigma ** 2)
    )


def hsic_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sigma: float = 1.0,
    max_samples: int = 1024,
) -> torch.Tensor:
    """
    Compute HSIC between two representations.

    HSIC is used here as the compatibility/complementarity
    criterion between the structural and semantic views.

    For large graphs, only a subset of nodes is used because
    the kernel matrices are O(N^2) in memory.

    Args:
        z1:
            Structural representation [N, D1]

        z2:
            Semantic representation [N, D2]

        sigma:
            RBF kernel bandwidth.

        max_samples:
            Maximum number of nodes used for HSIC.

    Returns:
        Scalar HSIC value.
    """

    if z1.size(0) <= 1:
        return z1.new_zeros(())

    # --------------------------------------------------------------
    # Limit computational cost for large graphs.
    # --------------------------------------------------------------

    if z1.size(0) > max_samples:

        idx = torch.randperm(
            z1.size(0),
            device=z1.device,
        )[:max_samples]

        z1 = z1.index_select(
            0,
            idx,
        )

        z2 = z2.index_select(
            0,
            idx,
        )

    n = z1.size(0)

    # --------------------------------------------------------------
    # Kernel matrices.
    # --------------------------------------------------------------

    K = rbf_kernel(
        z1,
        sigma,
    )

    L = rbf_kernel(
        z2,
        sigma,
    )

    # --------------------------------------------------------------
    # Center kernels.
    #
    # HKH =
    # K - row_mean - column_mean + global_mean
    #
    # This avoids explicitly creating the NxN centering matrix H.
    # --------------------------------------------------------------

    K_centered = (
        K
        - K.mean(
            dim=0,
            keepdim=True,
        )
        - K.mean(
            dim=1,
            keepdim=True,
        )
        + K.mean()
    )

    L_centered = (
        L
        - L.mean(
            dim=0,
            keepdim=True,
        )
        - L.mean(
            dim=1,
            keepdim=True,
        )
        + L.mean()
    )

    # --------------------------------------------------------------
    # HSIC.
    # --------------------------------------------------------------

    hsic = (
        K_centered * L_centered
    ).sum() / ((n - 1) ** 2)

    return hsic


# =====================================================================
# ENTROPY
# =====================================================================

def compute_entropy(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute prediction entropy for every node.

        H(P) = -sum_c P(c) log P(c)

    Lower entropy:
        more confident prediction.

    Higher entropy:
        more uncertain prediction.

    Args:
        logits:
            [N, C]

    Returns:
        entropy:
            [N]
    """

    log_probs = F.log_softmax(
        logits,
        dim=-1,
    )

    probs = log_probs.exp()

    entropy = -torch.sum(
        probs * log_probs,
        dim=-1,
    )

    return entropy


def entropy_attention(
    logits_structural: torch.Tensor,
    logits_semantic: torch.Tensor,
):
    """
    Calculate entropy-based adaptive attention between the
    structural and semantic classification views.

    This function is used ONLY when HSIC is HIGH.

    Lower entropy → higher confidence → higher attention.

    Returns:
        fused_logits
        alpha_structural
        alpha_semantic
        entropy_structural
        entropy_semantic
    """

    # --------------------------------------------------------------
    # 1. Calculate entropy for each classification view.
    # --------------------------------------------------------------

    entropy_structural = compute_entropy(
        logits_structural
    )

    entropy_semantic = compute_entropy(
        logits_semantic
    )

    # --------------------------------------------------------------
    # 2. Convert negative entropy into confidence scores.
    #
    # Lower entropy = larger negative-entropy score
    #               = larger attention.
    # --------------------------------------------------------------

    confidence_scores = torch.stack(
        [
            -entropy_structural,
            -entropy_semantic,
        ],
        dim=-1,
    )

    # --------------------------------------------------------------
    # 3. Normalize into adaptive attention weights.
    #
    # For every node:
    #
    # alpha_structural + alpha_semantic = 1
    # --------------------------------------------------------------

    attention = F.softmax(
        confidence_scores,
        dim=-1,
    )

    alpha_structural = attention[
        :, 0:1
    ]

    alpha_semantic = attention[
        :, 1:2
    ]

    # --------------------------------------------------------------
    # 4. Entropy-adaptive final prediction.
    # --------------------------------------------------------------

    fused_logits = (
        alpha_structural * logits_structural
        + alpha_semantic * logits_semantic
    )

    return (
        fused_logits,
        alpha_structural,
        alpha_semantic,
        entropy_structural,
        entropy_semantic,
    )


# =====================================================================
# MASKED CLASSIFICATION LOSS
# =====================================================================

def masked_softmax_cross_entropy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked softmax cross entropy.

    Labels are expected to be one-hot encoded.
    """

    log_probs = F.log_softmax(
        preds,
        dim=1,
    )

    loss = -(
        labels * log_probs
    ).sum(dim=1)

    mask = mask.float()

    mean = mask.mean()

    if mean.item() == 0:
        return torch.zeros(
            (),
            device=preds.device,
            dtype=preds.dtype,
        )

    mask = mask / mean

    loss = loss * mask

    return loss.mean()


# =====================================================================
# MASKED ACCURACY
# =====================================================================

def masked_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked classification accuracy.
    """

    correct = torch.eq(
        torch.argmax(
            preds,
            dim=1,
        ),
        torch.argmax(
            labels,
            dim=1,
        ),
    ).float()

    mask = mask.float()

    mean = mask.mean()

    if mean.item() == 0:
        return torch.zeros(
            (),
            device=preds.device,
            dtype=preds.dtype,
        )

    mask = mask / mean

    return (
        correct * mask
    ).mean()


# =====================================================================
# MAIN GNN MODEL
# =====================================================================

class GNNModel(nn.Module):

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

        # --------------------------------------------------------------
        # Configuration
        # --------------------------------------------------------------

        self.weight_decay = float(
            weight_decay
        )

        self.input_dim = input_dim

        self.num_classes = num_classes

        self.hidden1 = hidden

        self.global_model = global_model

        self.semantic_channel = semantic_channel

        self.dropout = float(
            dropout
        )

        self.semantic_dim = semantic_dim

        self.hsic_threshold = float(
            hsic_threshold
        )

        self.hsic_sigma = float(
            hsic_sigma
        )

        self.hsic_weight = float(
            hsic_weight
        )

        self.hsic_max_samples = int(
            hsic_max_samples
        )

        # --------------------------------------------------------------
        # Original CG3 structural loss.
        # --------------------------------------------------------------

        self.view_loss = (
            view_loss
            if view_loss is not None
            else StructuralContrastiveLoss()
        )

        # --------------------------------------------------------------
        # Select local GNN.
        # --------------------------------------------------------------

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

            raise ValueError(
                f"Unknown local_model: {local_model}"
            )

        # --------------------------------------------------------------
        # Preprocessing buffers.
        # --------------------------------------------------------------

        self.register_buffer(
            "edge_pos_i",
            torch.from_numpy(
                np.asarray(
                    edge_pos[:, 0]
                ).astype("int64")
            ),
        )

        self.register_buffer(
            "edge_pos_j",
            torch.from_numpy(
                np.asarray(
                    edge_pos[:, 1]
                ).astype("int64")
            ),
        )

        self.register_buffer(
            "train_idx_buf",
            torch.from_numpy(
                np.asarray(
                    train_idx
                ).astype("int64")
            ),
        )

        self.register_buffer(
            "mat01_intra",
            torch.from_numpy(
                mat01_tr_te[0].astype(
                    "float32"
                )
            ),
        )

        self.register_buffer(
            "mat01_inter",
            torch.from_numpy(
                mat01_tr_te[1].astype(
                    "float32"
                )
            ),
        )

        self.register_buffer(
            "mat01_intra_rowsum",
            torch.from_numpy(
                np.sum(
                    mat01_tr_te[0],
                    axis=1,
                ).astype("float32")
            ),
        )

        self.train_idx_size = int(
            np.shape(train_idx)[0]
        )

        # --------------------------------------------------------------
        # Local CG3 GNN layers.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Structural classifier.
        #
        # Input:
        #     z_structural [N, num_classes]
        #
        # Output:
        #     structural logits [N, num_classes]
        # --------------------------------------------------------------

        self.classifier_struct = nn.Linear(
            self.num_classes,
            self.num_classes,
        )

        # --------------------------------------------------------------
        # Semantic classifier.
        #
        # This is kept here for the adaptive high-HSIC branch.
        #
        # Input:
        #     z_semantic [N, semantic_dim]
        #
        # Output:
        #     semantic logits [N, num_classes]
        #
        # NOTE:
        # SemanticChannel ALSO contains a classifier. We do not need
        # to use this head if SemanticChannel returns its logits.
        # --------------------------------------------------------------

        self.classifier_semantic = nn.Linear(
            self.semantic_dim,
            self.num_classes,
        )

        # --------------------------------------------------------------
        # Low-HSIC combined classifier.
        #
        # Structural + semantic representations are concatenated
        # only when HSIC is LOW.
        # --------------------------------------------------------------

        self.classifier_fused = nn.Linear(
            self.num_classes + self.semantic_dim,
            self.num_classes,
        )

        # --------------------------------------------------------------
        # Original CG3 edge generation MLP.
        # --------------------------------------------------------------

        self.p_e_yy_w_contra = MLP(
            act=(lambda x: x),
            input_dim=2 * self.num_classes,
            output_dim=1,
            sparse_inputs=False,
            isSparse=True,
            bias=True,
        )

        # --------------------------------------------------------------
        # Runtime attributes.
        # --------------------------------------------------------------

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

        self.loss = torch.tensor(
            0.0
        )

        self.accuracy = torch.tensor(
            0.0
        )

        self.p_e_xy = torch.tensor(
            0.0
        )

    # =================================================================
    # PROPERTY
    # =================================================================

    @property
    def train_idx(self) -> torch.Tensor:

        return self.train_idx_buf

    # =================================================================
    # FORWARD
    # =================================================================

    def forward(
        self,
        features: torch.Tensor,
        support: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,

        # IMPORTANT:
        # SemanticChannel expects RAW TAGS.
        tags: list[str] | None = None,
    ):

        # ==============================================================
        # 1. LOCAL STRUCTURAL VIEW
        # ==============================================================

        self.classlayers[0].support = support

        self.classlayers[0].sparse_inputs = True

        h0 = self.classlayers[0](
            features
        )

        # ==============================================================
        # 2. LOCAL CLASSIFICATION REPRESENTATION
        # ==============================================================

        self.classlayers[1].support = support

        self.classlayers[1].sparse_inputs = False

        h1 = self.classlayers[1](
            h0
        )

        self.concat_vec_local = F.normalize(
            h1,
            p=2,
            dim=1,
        )

        # ==============================================================
        # 3. GLOBAL STRUCTURAL VIEW
        # ==============================================================

        global_out = self.global_model(
            features
        )

        self.concat_vec_global = F.normalize(
            global_out,
            p=2,
            dim=1,
        )

        # ==============================================================
        # 4. ORIGINAL CG3 STRUCTURAL FUSION
        #
        # Local = 0.6
        # Global = 0.4
        # ==============================================================

        self.z_structural = F.normalize(
            0.6 * self.concat_vec_local
            + 0.4 * self.concat_vec_global,
            p=2,
            dim=1,
        )

        # ==============================================================
        # 5. SEMANTIC CHANNEL
        #
        # IMPORTANT:
        #
        # We call the actual SemanticChannel here.
        #
        # It performs:
        #
        # TAG
        #   ↓
        # GraniteDescriptorGenerator
        #   ↓
        # descriptor
        #   ↓
        # HuggingFaceSentenceEncoder
        #   ↓
        # x_semantic
        #   ↓
        # Semantic MLP
        #   ↓
        # h_semantic
        #   ↓
        # Semantic classifier
        #   ↓
        # semantic_logits
        #
        # SemanticChannel returns:
        #
        #     descriptors
        #     x_semantic
        #     h_semantic
        #     z_semantic
        # ==============================================================

        semantic_available = (
            self.semantic_channel is not None
            and tags is not None
            and len(tags) == features.size(0)
        )

        if semantic_available:

            (
                generated_descriptors,
                x_semantic,
                h_semantic,
                semantic_logits,
            ) = self.semantic_channel(
                tags
            )

            # ----------------------------------------------------------
            # Move semantic tensors to the structural model device.
            # ----------------------------------------------------------

            structural_device = (
                self.z_structural.device
            )

            x_semantic = x_semantic.to(
                structural_device
            )

            h_semantic = h_semantic.to(
                structural_device
            )

            semantic_logits = semantic_logits.to(
                structural_device
            )

            # ----------------------------------------------------------
            # Store semantic outputs.
            # ----------------------------------------------------------

            self.semantic_descriptors = (
                generated_descriptors
            )

            self.semantic_embeddings = (
                x_semantic
            )

            self.semantic_logits = (
                semantic_logits
            )

            # ----------------------------------------------------------
            # Learned semantic representation.
            # ----------------------------------------------------------

            self.z_semantic = F.normalize(
                h_semantic,
                p=2,
                dim=1,
            )

        else:

            # ----------------------------------------------------------
            # Semantic channel unavailable.
            # ----------------------------------------------------------

            self.semantic_descriptors = None

            self.semantic_embeddings = None

            self.semantic_logits = None

            self.z_semantic = torch.zeros(
                (
                    self.z_structural.size(0),
                    self.semantic_dim,
                ),
                device=self.z_structural.device,
                dtype=self.z_structural.dtype,
            )

        # ==============================================================
        # 6. HSIC
        # ==============================================================

        if semantic_available:

            loss_hsic = hsic_loss(
                self.z_structural,
                self.z_semantic,
                sigma=self.hsic_sigma,
                max_samples=self.hsic_max_samples,
            )

        else:

            loss_hsic = self.z_structural.new_zeros(
                ()
            )

        self.hsic_value = (
            loss_hsic.detach()
        )

        # ==============================================================
        # 7. HSIC DECISION
        #
        # LOW HSIC:
        #
        #     Structural and semantic views are considered compatible.
        #
        #     → combine them
        #     → combined classification
        #
        # HIGH HSIC:
        #
        #     Structural and semantic views remain separate.
        #
        #     → structural classifier
        #     → semantic classifier
        #     → entropy calculation
        #     → adaptive attention
        #     → final prediction
        #
        # This is the central routing decision.
        # ==============================================================

        low_hsic = (
            semantic_available
            and loss_hsic.detach().item()
            < self.hsic_threshold
        )

        self.semantic_enabled = low_hsic

        # ==============================================================
        # 8. STRUCTURAL CLASSIFICATION
        #
        # This exists in BOTH branches.
        # ==============================================================

        logits_structural = self.classifier_struct(
            self.z_structural
        )

        self.logits_structural = (
            logits_structural
        )

        # ==============================================================
        # 9. LOW-HSIC BRANCH
        #
        # Structural + semantic are COMBINED.
        #
        # NO entropy attention is used here.
        # ==============================================================

        if low_hsic:

            # ----------------------------------------------------------
            # Concatenate the two learned representations.
            # ----------------------------------------------------------

            z_combined = torch.cat(
                [
                    self.z_structural,
                    self.z_semantic,
                ],
                dim=-1,
            )

            # ----------------------------------------------------------
            # Combined classifier.
            # ----------------------------------------------------------

            self.outputs = self.classifier_fused(
                z_combined
            )

            # ----------------------------------------------------------
            # Record that this branch does not use independent
            # entropy attention.
            # ----------------------------------------------------------

            self.logits_semantic = (
                self.semantic_logits
            )

            self.alpha_structural = None

            self.alpha_semantic = None

            self.entropy_structural = None

            self.entropy_semantic = None

        # ==============================================================
        # 10. HIGH-HSIC BRANCH
        #
        # The two views remain SEPARATE.
        #
        # Structural:
        #     z_structural → structural logits
        #
        # Semantic:
        #     z_semantic → semantic logits
        #
        # Then:
        #     entropy(structural)
        #     entropy(semantic)
        #
        # Then:
        #     adaptive attention
        #
        # Then:
        #     final weighted prediction
        # ==============================================================

        else:

            # ----------------------------------------------------------
            # Structural logits already computed above.
            # ----------------------------------------------------------

            logits_structural = (
                self.logits_structural
            )

            # ----------------------------------------------------------
            # Semantic classification.
            #
            # Prefer the logits produced by SemanticChannel because
            # that classifier is part of the semantic pathway.
            #
            # If SemanticChannel is unavailable, use the local semantic
            # classifier only as a fallback.
            # ----------------------------------------------------------

            if semantic_available:

                logits_semantic = (
                    self.semantic_logits
                )

            else:

                logits_semantic = (
                    self.classifier_semantic(
                        self.z_semantic
                    )
                )

            self.logits_semantic = (
                logits_semantic
            )

            # ----------------------------------------------------------
            # Entropy + adaptive attention.
            # ----------------------------------------------------------

            (
                fused_logits,
                alpha_structural,
                alpha_semantic,
                entropy_structural,
                entropy_semantic,
            ) = entropy_attention(
                logits_structural,
                logits_semantic,
            )

            # ----------------------------------------------------------
            # Final prediction.
            # ----------------------------------------------------------

            self.outputs = fused_logits

            self.alpha_structural = (
                alpha_structural
            )

            self.alpha_semantic = (
                alpha_semantic
            )

            self.entropy_structural = (
                entropy_structural
            )

            self.entropy_semantic = (
                entropy_semantic
            )

        # ==============================================================
        # 11. CLASSIFICATION LOSS
        # ==============================================================

        loss_q_yobs_x_g = (
            masked_softmax_cross_entropy(
                self.outputs,
                labels,
                mask,
            )
        )

        # ==============================================================
        # 12. ORIGINAL CG3 EDGE GENERATION
        # ==============================================================

        y_ei_local = (
            self.concat_vec_local.index_select(
                0,
                self.edge_pos_i,
            )
        )

        y_ej_global = (
            self.concat_vec_global.index_select(
                0,
                self.edge_pos_j,
            )
        )

        y_ei_global = (
            self.concat_vec_global.index_select(
                0,
                self.edge_pos_i,
            )
        )

        y_ej_local = (
            self.concat_vec_local.index_select(
                0,
                self.edge_pos_j,
            )
        )

        p_e_xy_1 = -torch.mean(
            torch.log(
                torch.sigmoid(
                    self.p_e_yy_w_contra(
                        torch.cat(
                            [
                                y_ei_local,
                                y_ej_global,
                            ],
                            dim=1,
                        )
                    )
                ).clamp(
                    min=1e-8
                )
            )
        )

        p_e_xy_2 = -torch.mean(
            torch.log(
                torch.sigmoid(
                    self.p_e_yy_w_contra(
                        torch.cat(
                            [
                                y_ei_global,
                                y_ej_local,
                            ],
                            dim=1,
                        )
                    )
                ).clamp(
                    min=1e-8
                )
            )
        )

        self.p_e_xy = (
            p_e_xy_1
            + p_e_xy_2
        )

        # ==============================================================
        # 13. ORIGINAL CG3 STRUCTURAL CONTRASTIVE LOSS
        # ==============================================================

        loss_ctx = {
            "train_idx": self.train_idx_buf,

            "mat01_intra": self.mat01_intra,

            "mat01_inter": self.mat01_inter,

            "mat01_intra_rowsum":
                self.mat01_intra_rowsum,

            "train_idx_size":
                self.train_idx_size,
        }

        loss_contrastive = (
            self.view_loss(
                self.concat_vec_local,
                self.concat_vec_global,
                loss_ctx,
            )
        )

        # ==============================================================
        # 14. TOTAL LOSS
        #
        # Classification:
        #     loss_ce
        #
        # Original CG3:
        #     + 0.4 * generation
        #     + contrastive
        #
        # Added:
        #     + hsic_weight * HSIC
        #
        # HSIC is therefore used both as:
        #     1. the routing criterion
        #     2. a regularization term
        # ==============================================================

        loss_ce = loss_q_yobs_x_g

        loss_gen = self.p_e_xy

        total = (
            loss_ce
            + 0.4 * loss_gen
            + loss_contrastive
            + self.hsic_weight * loss_hsic
        )

        # ==============================================================
        # 15. ORIGINAL CG3 L2 REGULARIZATION
        # ==============================================================

        for i in range(2):

            for var in (
                self.classlayers[i].vars.values()
            ):

                total = (
                    total
                    + self.weight_decay
                    * 0.5
                    * torch.sum(
                        var ** 2
                    )
                )

        for var in (
            self.p_e_yy_w_contra.vars.values()
        ):

            total = (
                total
                + self.weight_decay
                * 0.5
                * torch.sum(
                    var ** 2
                )
            )

        # --------------------------------------------------------------
        # Global model regularization.
        # --------------------------------------------------------------

        total = (
            total
            + self.global_model.loss
        )

        # ==============================================================
        # 16. FINAL METRICS
        # ==============================================================

        self.loss = total

        self.accuracy = masked_accuracy(
            self.outputs,
            labels,
            mask,
        )

        self.loss_ce = (
            loss_ce.detach()
        )

        self.loss_gen = (
            loss_gen.detach()
        )

        self.loss_contrastive = (
            loss_contrastive.detach()
        )

        self.loss_hsic = (
            loss_hsic.detach()
        )

        self.loss_total = (
            total.detach()
        )

        reg = (
            self.view_loss
            .regularizer_value()
        )

        self.loss_reg = (
            reg.detach()
            if reg is not None
            else None
        )

        return (
            self.outputs,
            self.loss,
            self.accuracy,
        )

"""H-GCN / H-GAT: the global view of CG3.

Hu et al., "Hierarchical Graph Convolutional Networks for Semi-supervised Node
Classification", IJCAI 2019. One layer per hierarchy level going down
(coarsen) and back up (refine), skip connections between mirrored levels, and
an output layer on the refined node embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import HGCNGraphAttention, HGCNGraphConvolution, identity
from src.preprocess import Hierarchy


class HierarchicalGNN(nn.Module):
    HIDDEN_ACT = staticmethod(F.relu)

    def __init__(self, input_dim: int, output_dim: int, hidden: int, hierarchy: Hierarchy, *,
                 coarsen_level: int, max_node_wgt: int, node_wgt_embed_dim: int,
                 channel_num: int, dropout: float, weight_decay: float):
        super().__init__()
        self.coarsen_level = int(coarsen_level)
        self.weight_decay = float(weight_decay)

        # Embedding of a node's weight (how many input nodes it stands for),
        # concatenated to the input of every coarsen / refine layer.
        bound = math.sqrt(6.0 / (3.0 * node_wgt_embed_dim + 3.0 * input_dim))
        self.W_node_wgt = nn.Parameter(
            torch.empty(max_node_wgt + 1, node_wgt_embed_dim).uniform_(-bound, bound)
        )
        for level, node_wgt in enumerate(hierarchy.node_wgt):
            self.register_buffer(f"node_wgt_{level}", node_wgt.clamp(max=max_node_wgt), persistent=False)

        L = self.coarsen_level
        self.levels = [0] + list(range(1, L)) + list(range(L, 0, -1)) + [0]  # hierarchy level of every layer
        common = dict(channels=channel_num, node_wgt_embed_dim=node_wgt_embed_dim, dropout=dropout)
        act = type(self).HIDDEN_ACT
        layers = [self._make_layer(input_dim, hidden, hierarchy.supports[0], hierarchy.pool[0], "input",
                                   sparse_inputs=True, act=act, **common)]
        for k in range(1, L):
            layers.append(self._make_layer(hidden, hidden, hierarchy.supports[k], hierarchy.pool[k], "coarsen",
                                           act=act, **common))
        for k in range(L, 0, -1):
            layers.append(self._make_layer(hidden, hidden, hierarchy.supports[k], hierarchy.unpool[k - 1], "refine",
                                           act=act, **common))
        layers.append(self._make_layer(hidden, output_dim, hierarchy.supports[0], None, "output",
                                       act=identity, **common))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, input_dim, output_dim, support, transfer, mod, **kwargs) -> nn.Module:
        raise NotImplementedError

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        L = self.coarsen_level
        activations = [features]
        outputs = []  # every layer's output before pooling / unpooling, for the skip connections
        for i, layer in enumerate(self.layers):
            if layer.mod in ("coarsen", "refine"):
                node_emb = self.W_node_wgt[getattr(self, f"node_wgt_{self.levels[i]}")]
                hidden, out = layer(activations[-1], node_emb=node_emb)
            else:
                hidden, out = layer(activations[-1])
            outputs.append(out)
            if L <= i < 2 * L:  # refine layer: add the mirrored coarsen layer's output
                hidden = hidden + outputs[2 * L - i - 1]
            activations.append(hidden)
        return activations[-1]

    def l2(self) -> torch.Tensor:
        """CG3's explicit weight decay on the layer weights."""
        return self.weight_decay * 0.5 * sum((p ** 2).sum() for layer in self.layers for p in layer.vars.values())


class HGCN(HierarchicalGNN):
    """Hierarchical GCN global view."""

    HIDDEN_ACT = staticmethod(F.relu)

    def _make_layer(self, input_dim, output_dim, support, transfer, mod, **kwargs):
        return HGCNGraphConvolution(input_dim, output_dim, support, transfer, mod, **kwargs)


class HGAT(HierarchicalGNN):
    """Hierarchical GAT global view."""

    HIDDEN_ACT = staticmethod(F.elu)

    def _make_layer(self, input_dim, output_dim, support, transfer, mod, **kwargs):
        return HGCNGraphAttention(input_dim, output_dim, support, transfer, mod, **kwargs)

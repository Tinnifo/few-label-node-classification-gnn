"""Layers for the HGCN / HGAT global-view models.

Faithful port of the snapshot/86b0818 `CG3Method/layers.py`. Differences vs.
the snapshot: (1) the TF-style `placeholders` dict is replaced by explicit
ctor kwargs (`dropout: float`, `num_features_nonzero: int`,
`node_wgt_embed_dim: int`); (2) `print(...)` debug statements removed; (3)
multi-channel `support` (a list of `(coords, values, shape)` tuples) is
turned into per-layer sparse buffers in `__init__`.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from .cg3_layers import glorot, sparse_dropout, zeros


def _scipy_to_torch_sparse(X) -> torch.Tensor:
    coo = X.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values = torch.from_numpy(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def _support_to_sparse(support_tuple) -> torch.Tensor:
    coords, values, shape = support_tuple
    rows = coords[:, 0]
    cols = coords[:, 1]
    sp_mat = sp.csr_matrix((values, (rows, cols)), shape=shape, dtype=np.float32)
    return _scipy_to_torch_sparse(sp_mat)


def _dot(x, y, sparse: bool):
    if sparse:
        return torch.sparse.mm(x, y)
    return torch.matmul(x, y)


class HGCNGraphConvolution(nn.Module):
    """Multi-channel, hierarchy-aware GCN layer for HGCN."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        support,
        transfer,
        mod: str,
        layer_index: int,
        dropout: float = 0.0,
        sparse_inputs: bool = False,
        act=F.relu,
        bias: bool = False,
        featureless: bool = False,
        num_features_nonzero: int | None = None,
        node_wgt_embed_dim: int = 5,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.act = act
        self.support = support
        self.transfer = transfer
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.use_bias = bias
        self.mod = mod
        self.layer_index = layer_index
        self.output_dim = output_dim
        self.num_features_nonzero = num_features_nonzero
        self.node_wgt_embed_dim = node_wgt_embed_dim

        self.vars: dict[str, nn.Parameter] = {}
        if self.mod in ("coarsen", "refine"):
            in_dim = input_dim + node_wgt_embed_dim
        else:
            in_dim = input_dim

        for i in range(len(self.support)):
            w = glorot([in_dim, output_dim])
            self.vars[f"weights_{i}"] = w
            self.register_parameter(f"weights_{i}", w)
        if self.use_bias:
            b = zeros([output_dim])
            self.vars["bias"] = b
            self.register_parameter("bias", b)

        for i in range(len(self.support)):
            self.register_buffer(
                f"support_tensor_{i}", _support_to_sparse(self.support[i])
            )

        if self.mod in ("coarsen", "input"):
            transfer_opo = normalize(self.transfer.T, norm="l2", axis=1).astype(
                np.float32
            )
            self.register_buffer(
                "transfer_tensor", _scipy_to_torch_sparse(transfer_opo)
            )
        elif self.mod == "refine":
            self.register_buffer(
                "transfer_tensor",
                _scipy_to_torch_sparse(self.transfer.astype(np.float32)),
            )
        else:
            self.transfer_tensor = None

        self.channel_combine = nn.Conv1d(
            in_channels=len(self.support),
            out_channels=1,
            kernel_size=1,
            bias=False,
        )

    def _dropout_rate(self) -> float:
        return self.dropout if self.training else 0.0

    def forward(self, inputs, node_emb=None):
        if self.mod in ("coarsen", "refine"):
            x = torch.cat([inputs, node_emb], dim=1)
        else:
            x = inputs

        rate = self._dropout_rate()
        if self.sparse_inputs:
            x = sparse_dropout(x, 1.0 - rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=rate, training=self.training)

        supports = []
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = _dot(x, self.vars[f"weights_{i}"], sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars[f"weights_{i}"]
            sp_tensor = getattr(self, f"support_tensor_{i}")
            supports.append(_dot(sp_tensor, pre_sup, sparse=True))

        # (N, output_dim, num_channels) -> (N, num_channels, output_dim) -> Conv1d
        supports = torch.stack(supports, dim=2).permute(0, 2, 1)
        output = self.channel_combine(supports).squeeze(1)

        if self.use_bias:
            output = output + self.vars["bias"]
        output = self.act(output)

        gcn_output = output

        if self.mod == "output":
            return output, gcn_output
        if self.mod in ("coarsen", "input", "refine"):
            output = _dot(self.transfer_tensor, gcn_output, sparse=True)
        return output, gcn_output


class HGCNGraphAttention(nn.Module):
    """Multi-channel, hierarchy-aware GAT layer for HGAT."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        support,
        transfer,
        mod: str,
        layer_index: int,
        dropout: float = 0.0,
        sparse_inputs: bool = False,
        act=F.elu,
        bias: bool = False,
        featureless: bool = False,
        num_features_nonzero: int | None = None,
        node_wgt_embed_dim: int = 5,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.act = act
        self.support = support
        self.transfer = transfer
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.use_bias = bias
        self.mod = mod
        self.layer_index = layer_index
        self.output_dim = output_dim
        self.num_features_nonzero = num_features_nonzero
        self.node_wgt_embed_dim = node_wgt_embed_dim
        self.alpha = alpha

        self.vars: dict[str, nn.Parameter] = {}
        if self.mod in ("coarsen", "refine"):
            in_dim = input_dim + node_wgt_embed_dim
        else:
            in_dim = input_dim

        # one (W, a) pair per channel
        for i in range(len(self.support)):
            w = glorot([in_dim, output_dim])
            self.vars[f"weights_{i}"] = w
            self.register_parameter(f"weights_{i}", w)
            a = glorot([2 * output_dim, 1])
            self.vars[f"attn_{i}"] = a
            self.register_parameter(f"attn_{i}", a)
        if self.use_bias:
            b = zeros([output_dim])
            self.vars["bias"] = b
            self.register_parameter("bias", b)

        for i in range(len(self.support)):
            self.register_buffer(
                f"support_tensor_{i}", _support_to_sparse(self.support[i])
            )

        if self.mod in ("coarsen", "input"):
            transfer_opo = normalize(self.transfer.T, norm="l2", axis=1).astype(
                np.float32
            )
            self.register_buffer(
                "transfer_tensor", _scipy_to_torch_sparse(transfer_opo)
            )
        elif self.mod == "refine":
            self.register_buffer(
                "transfer_tensor",
                _scipy_to_torch_sparse(self.transfer.astype(np.float32)),
            )
        else:
            self.transfer_tensor = None

        self.leakyrelu = nn.LeakyReLU(alpha)

    def _dropout_rate(self) -> float:
        return self.dropout if self.training else 0.0

    def _attention(
        self, Wh: torch.Tensor, attn: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        N = Wh.shape[0]
        indices = adj.coalesce().indices()
        row, col = indices[0], indices[1]

        Wh_i = Wh[row]
        Wh_j = Wh[col]
        edge_features = torch.cat([Wh_i, Wh_j], dim=1)

        e = self.leakyrelu(torch.matmul(edge_features, attn).squeeze(1))
        # stable softmax
        e = e - e.max()
        exp_e = torch.exp(e)

        denom = torch.zeros(N, device=Wh.device).scatter_add_(0, row, exp_e)
        alpha = exp_e / (denom[row] + 1e-16)

        out = torch.zeros_like(Wh)
        out = out.index_add(0, row, Wh_j * alpha.unsqueeze(1))
        return out

    def forward(self, inputs, node_emb=None):
        if self.mod in ("coarsen", "refine"):
            x = torch.cat([inputs, node_emb], dim=1)
        else:
            x = inputs

        rate = self._dropout_rate()
        if self.sparse_inputs:
            # The snapshot raises NotImplementedError here; HGAT is only
            # composed with dense intermediates (the input layer in HGAT is
            # 'input' mod which sets sparse_inputs from the constructor —
            # but in our pipeline we pass the sparse feature tensor only to
            # the first layer of the *local* GNNModel, never to HGAT).
            x = sparse_dropout(x, 1.0 - rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=rate, training=self.training)

        outs = []
        for i in range(len(self.support)):
            Wh = _dot(x, self.vars[f"weights_{i}"], sparse=self.sparse_inputs)
            adj = getattr(self, f"support_tensor_{i}")
            outs.append(self._attention(Wh, self.vars[f"attn_{i}"], adj))

        out = torch.stack(outs, dim=2).mean(dim=2)
        if self.use_bias:
            out = out + self.vars["bias"]
        out = self.act(out)
        gat_output = out

        if self.mod == "output":
            return out, gat_output
        if self.mod in ("coarsen", "input", "refine"):
            out = torch.sparse.mm(self.transfer_tensor, gat_output)
        return out, gat_output


import numpy as np


class Graph(object):
    """Note: adj_list shows each edge twice. So edge_num is really two times of edge number for undirected graph."""

    def __init__(self, node_num, edge_num):
        self.node_num = node_num  # n
        self.edge_num = edge_num  # m
        self.adj_list = (
            np.zeros(edge_num, dtype=np.int32) - 1
        )  # a big array for all the neighbors.
        self.adj_idx = np.zeros(
            node_num + 1, dtype=np.int32
        )  # idx of the beginning neighbors in the adj_list. Pad one additional element at the end with value equal to the edge_num, i.e., self.adj_idx[-1] = edge_num
        self.adj_wgt = np.zeros(
            edge_num, dtype=np.float32
        )  # same dimension as adj_list, wgt on the edge. CAN be float numbers.
        self.node_wgt = np.zeros(node_num, dtype=np.int32)
        self.cmap = np.zeros(node_num, dtype=np.int32) - 1  # mapped to coarser graph

        # weighted degree: the sum of the adjacency weight of each vertex, including self-loop.
        self.degree = np.zeros(node_num, dtype=np.float32)
        self.A = None
        self.C = None  # Matching Matrix

        self.coarser = None
        self.finer = None

    def resize_adj(self, edge_num):
        """Resize the adjacency list/wgts based on the number of edges."""
        self.adj_list = np.resize(self.adj_list, edge_num)
        self.adj_wgt = np.resize(self.adj_wgt, edge_num)

    def get_neighs(self, idx):
        """obtain the list of neigbors given a node."""
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_list[istart:iend]

    def get_neigh_edge_wgts(self, idx):
        """obtain the weights of neighbors given a node."""
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_wgt[istart:iend]


from collections import defaultdict
from .graph import Graph
import numpy as np
import scipy.sparse as sp


def cmap2C(cmap):
    node_num = len(cmap)
    i_arr = []
    j_arr = []
    data_arr = []

    for i in range(node_num):
        i_arr.append(i)
        j_arr.append(cmap[i])
        data_arr.append(1)

    return sp.csr_matrix((data_arr, (i_arr, j_arr)))


def normalized_adj_wgt(graph):
    adj_wgt = graph.adj_wgt
    adj_idx = graph.adj_idx
    norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
    degree = graph.degree
    for i in range(graph.node_num):
        for j in range(adj_idx[i], adj_idx[i + 1]):
            neigh = graph.adj_list[j]
            norm_wgt[j] = adj_wgt[neigh] / np.sqrt(degree[i] * degree[neigh])
    return norm_wgt


###############################################
# This section of code adapted from jiongqianliang/MILE #
# http://jiongqianliang.com/MILE/ #
###############################################


def generate_hybrid_matching(max_node_wgt, graph):
    """Generate matchings using the hybrid method. It changes the cmap in graph object,
    return groups array and coarse_graph_size."""
    node_num = graph.node_num
    adj_list = graph.adj_list  # big array for neighbors.
    adj_idx = graph.adj_idx  # beginning idx of neighbors.
    adj_wgt = graph.adj_wgt  # weight on edge
    node_wgt = graph.node_wgt  # weight on node
    cmap = graph.cmap
    norm_adj_wgt = normalized_adj_wgt(graph)
    groups = []  # a list of groups, each group corresponding to one coarse node.
    matched = [False] * node_num

    # SEM: structural equivalence matching.
    jaccard_idx_preprocess(graph, matched, groups)
    # print("# groups have perfect jaccard idx (1.0): %d" % len(groups))
    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]

    sorted_idx = np.argsort(degree)
    for idx in sorted_idx:
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if (
                neigh == idx
            ):  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
                continue
            curr_wgt = norm_adj_wgt[j]
            if (
                (not matched[neigh])
                and max_wgt < curr_wgt
                and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt
            ):
                max_idx = neigh
                max_wgt = curr_wgt
        # it might happen that max_idx is idx, which means cannot find a match for the node.
        matched[idx] = matched[max_idx] = True
        if idx == max_idx:
            groups.append([idx])
        else:
            groups.append([idx, max_idx])
    coarse_graph_size = 0
    for idx in range(len(groups)):
        for ele in groups[idx]:
            cmap[ele] = coarse_graph_size
        coarse_graph_size += 1
    return (groups, coarse_graph_size)


def jaccard_idx_preprocess(graph, matched, groups):
    """Use hashmap to find out nodes with exactly same neighbors."""
    neighs2node = defaultdict(list)
    for i in range(graph.node_num):
        neighs = str(sorted(graph.get_neighs(i)))
        neighs2node[neighs].append(i)
    for key in neighs2node.keys():
        g = neighs2node[key]
        if len(g) > 1:
            for node in g:
                matched[node] = True
            groups.append(g)
    return


def create_coarse_graph(graph, groups, coarse_graph_size):
    """create the coarser graph and return it based on the groups array and coarse_graph_size"""
    coarse_graph = Graph(coarse_graph_size, graph.edge_num)
    coarse_graph.finer = graph
    graph.coarser = coarse_graph
    cmap = graph.cmap
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt

    coarse_adj_list = coarse_graph.adj_list
    coarse_adj_idx = coarse_graph.adj_idx
    coarse_adj_wgt = coarse_graph.adj_wgt
    coarse_node_wgt = coarse_graph.node_wgt
    coarse_degree = coarse_graph.degree

    coarse_adj_idx[0] = 0
    nedges = 0  # number of edges in the coarse graph
    for idx in range(len(groups)):  # idx in the graph
        coarse_node_idx = idx
        neigh_dict = (
            dict()
        )  # coarser graph neighbor node --> its location idx in adj_list.
        group = groups[idx]
        for i in range(len(group)):
            merged_node = group[i]
            if i == 0:
                coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
            else:
                coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

            istart = adj_idx[merged_node]
            iend = adj_idx[merged_node + 1]
            for j in range(istart, iend):
                k = cmap[
                    adj_list[j]
                ]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
                if k not in neigh_dict:  # add new neigh
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:  # increase weight to the existing neigh
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                # add weights to the degree. For now, we retain the loop.
                coarse_degree[coarse_node_idx] += adj_wgt[j]

        coarse_node_idx += 1
        coarse_adj_idx[coarse_node_idx] = nedges

    coarse_graph.edge_num = nedges

    coarse_graph.resize_adj(nedges)
    C = cmap2C(cmap)  # construct the matching matrix.
    graph.C = C
    coarse_graph.A = C.transpose().dot(graph.A).dot(C)
    return coarse_graph


"""Original CG3 structural contrastive loss (local ↔ global views).

Faithful extraction of `GNNModel._contrastive_loss` from the snapshot port:
unsupervised pairwise InfoNCE-style term + supervised contrastive on labeled
nodes using intra/inter class indicator matrices.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.losses.base import BaseViewLoss


class StructuralContrastiveLoss(BaseViewLoss):
    name = "structural"

    def __init__(self, temperature: float = 0.5, hp1: float = 0.9):
        super().__init__()
        self.temperature = float(temperature)
        self.hp1 = float(hp1)

    def forward(
        self,
        local: torch.Tensor,
        global_: torch.Tensor,
        ctx: Dict[str, Any],
        semantic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del semantic  # structural loss only aligns local ↔ global
        device = local.device
        loss = torch.zeros((), device=device)
        temp = self.temperature
        hp1 = self.hp1

        # Eq. 4 — pairwise unsupervised between local & global (and reverse).
        cos_dist = torch.exp(torch.matmul(local, global_.t()) / temp)
        neg = torch.mean(cos_dist, dim=1)
        diag_cos = torch.diagonal(cos_dist, 0)
        pos_neg1 = diag_cos / (neg + 1e-8)

        cos_dist = torch.exp(torch.matmul(global_, local.t()) / temp)
        neg = torch.mean(cos_dist, dim=1)
        diag_cos = torch.diagonal(cos_dist, 0)
        pos_neg2 = diag_cos / (neg + 1e-8)

        pos_neg3 = torch.cat([pos_neg1, pos_neg2], dim=0)
        loss = loss + (-hp1 * torch.mean(torch.log(pos_neg3.clamp(min=1e-8))))

        train_idx = ctx["train_idx"]
        mat01_intra = ctx["mat01_intra"]
        mat01_inter = ctx["mat01_inter"]
        mat01_intra_rowsum = ctx["mat01_intra_rowsum"]
        train_idx_size = int(ctx["train_idx_size"])

        # Supervised contrastive (round 1).
        h1 = local.index_select(0, train_idx)
        h2 = global_.index_select(0, train_idx)
        h_cos = torch.exp(torch.matmul(h1, h2.t()) / temp)
        sup_pos = torch.sum(h_cos * mat01_intra, dim=1)
        sup_neg = (torch.sum(h_cos * mat01_inter, dim=1) + sup_pos) / max(
            train_idx_size - 1, 1
        )
        sup_pos = sup_pos / (mat01_intra_rowsum + 1e-8)
        pos_neg_sup_1 = sup_pos / (sup_neg + 1e-8)

        # Supervised contrastive (round 2, swapped).
        h2_b = local.index_select(0, train_idx)
        h1_b = global_.index_select(0, train_idx)
        h_cos = torch.exp(torch.matmul(h1_b, h2_b.t()) / temp)
        sup_pos = torch.sum(h_cos * mat01_intra, dim=1)
        sup_neg = (torch.sum(h_cos * mat01_inter, dim=1) + sup_pos) / max(
            train_idx_size - 1, 1
        )
        sup_pos = sup_pos / (mat01_intra_rowsum + 1e-8)
        pos_neg_sup_2 = sup_pos / (sup_neg + 1e-8)

        pos_neg_sup_3 = torch.cat([pos_neg_sup_1, pos_neg_sup_2], dim=0)
        loss = loss + (-hp1 * torch.mean(torch.log(pos_neg_sup_3.clamp(min=1e-8))))
        return loss


"""Preprocessing for CG3 — port of the snapshot/86b0818 hierarchy pipeline.

Replaces the previous main-branch hierarchy builder. The artifacts produced
here mirror what `CG3Method/main.py` + `CG3Method/train.py` set up:
  * `feature_sp`   — sparse, row-normalized feature tensor.
  * `support_sp`   — D⁻⁰·⁵ (A+I) D⁻⁰·⁵ as a sparse tensor.
  * `support_tuple` — `(coords, values, shape)` tuple kept for `edge_pos`.
  * `transfer_list / adj_list / node_wgt_list` — per-level coarsening artifacts
    consumed by `HGCN` / `HGAT` (each level's `adj` is replicated `channel_num`
    times to mimic the snapshot's TF code).
  * `train_mat01`, `mats_intra_inter`, `tr_idx`, `y_train_oh`, `train_mask_int`,
    `y_val_oh`, `val_mask_int`, `y_test_oh`, `test_mask_int` — the supervised
    contrastive book-keeping and one-hot labels per split.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix

from .coarsen import create_coarse_graph, generate_hybrid_matching
from .graph import Graph


# ---------------------------------------------------------------------------
# helpers ported from CG3Method/funcCNN.py
# ---------------------------------------------------------------------------


def _normalize_features_csr(features: sp.spmatrix) -> sp.spmatrix:
    rowsum = np.asarray(features.sum(axis=1)).flatten()
    r_inv = np.power(rowsum.astype(float), -1, where=rowsum > 0)
    r_inv[~np.isfinite(r_inv)] = 0.0
    return sp.diags(r_inv).dot(features)


def _normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum.astype(float), -0.5, where=rowsum > 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat).transpose().dot(d_mat).tocoo()


def _sparse_to_tuple(mx: sp.spmatrix):
    if not sp.issparse(mx):
        mx = sp.coo_matrix(mx)
    else:
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    return coords, values, mx.shape


def _preprocess_features(features: sp.spmatrix):
    if not sp.issparse(features):
        features = sp.lil_matrix(features)
    features = _normalize_features_csr(features)
    if not sp.issparse(features):
        features = sp.lil_matrix(features)
    return _sparse_to_tuple(features)


def _preprocess_adj(adj: sp.spmatrix):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj_normalized = _normalize_adj(adj + sp.eye(adj.shape[0]))
    return _sparse_to_tuple(adj_normalized)


def _tuple_to_torch_sparse(sparse_tuple) -> torch.Tensor:
    coords, values, shape = sparse_tuple
    indices = torch.from_numpy(np.asarray(coords).T.astype("int64"))
    values_t = torch.from_numpy(np.asarray(values, dtype=np.float32))
    return torch.sparse_coo_tensor(indices, values_t, torch.Size(shape)).coalesce()


def _cal_class01_mat(y_train_oh: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Pairwise same-class indicator matrix; rows for unlabeled nodes are 0."""
    y = np.argmax(y_train_oh, axis=1)
    not_train = np.argwhere(~train_mask.astype(bool))
    y[not_train] = -1
    num_classes = int(np.max(y)) + 1
    mat01 = np.zeros([y_train_oh.shape[0], y_train_oh.shape[0]])
    for i in range(num_classes):
        pos = np.argwhere(y == i)
        if pos.size == 0:
            continue
        for j in range(pos.shape[0]):
            mat01[pos[j, 0], pos[:, 0]] = 1
    np.fill_diagonal(mat01, 0)
    return mat01


def _cal_intra_class_mat01(y: np.ndarray) -> list[np.ndarray]:
    """Returns [intra, inter] over the labeled-node × labeled-node block."""
    n = int(np.shape(y)[0])
    num_classes = int(np.max(y)) + 1
    intra = np.zeros([n, n])
    inter = np.ones([n, n])
    for c in range(num_classes):
        pos = np.argwhere(y == c)
        for k in range(pos.shape[0]):
            intra[pos[k, 0], pos[:, 0]] = 1
    inter -= intra
    intra -= np.eye(n)
    return [intra, inter]


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------


@dataclass
class CG3Artifacts:
    # feature / adjacency
    input_dim: int
    num_classes: int
    num_nodes: int
    feature_sp: torch.Tensor
    support_sp: torch.Tensor
    support_tuple: tuple
    edge_pos: np.ndarray
    num_features_nonzero: int
    # hierarchy
    transfer_list: list = field(default_factory=list)
    adj_list: list = field(default_factory=list)
    node_wgt_list: list = field(default_factory=list)
    # labels / masks (per split)
    y_train_oh: torch.Tensor = None
    train_mask_int: torch.Tensor = None
    y_val_oh: torch.Tensor = None
    val_mask_int: torch.Tensor = None
    y_test_oh: torch.Tensor = None
    test_mask_int: torch.Tensor = None
    # supervised-contrastive book-keeping
    train_idx_np: np.ndarray = None
    train_mat01: np.ndarray = None
    mats_intra_inter: list = None


def build_cg3_artifacts(
    data, *, coarsen_level: int, max_node_wgt: int, channel_num: int
) -> CG3Artifacts:
    """Build everything CG3 / HGCN / HGAT need from a PyG Data object.

    Numpy-side preprocessing runs on CPU; the returned tensors are CPU tensors
    (the caller / `model.to(device)` move them to GPU)."""
    device_data = data
    edge_index_cpu = data.edge_index.detach().cpu()
    num_nodes = int(data.num_nodes)

    # ---- adjacency (scipy CSR) ----
    adj = to_scipy_sparse_matrix(edge_index_cpu, num_nodes=num_nodes).tocsr()

    # ---- features (scipy sparse) ----
    x_cpu = data.x.detach().cpu().numpy()
    features_sp = sp.csr_matrix(x_cpu.astype(np.float32))
    feature_tuple = _preprocess_features(features_sp)
    feature_sp = _tuple_to_torch_sparse(feature_tuple)
    input_dim = feature_tuple[2][1]
    num_features_nonzero = int(feature_tuple[1].shape[0])

    # ---- support (D^-0.5 (A+I) D^-0.5) ----
    support_tuple = _preprocess_adj(adj)
    support_sp = _tuple_to_torch_sparse(support_tuple)
    edge_pos = np.asarray(support_tuple[0])

    # ---- hierarchy via graph coarsening ----
    graph, _ = _read_graph_from_adj(adj)
    transfer_list = []
    adj_list = [copy.copy(graph.A)]
    node_wgt_list = [copy.copy(graph.node_wgt)]
    for _ in range(coarsen_level):
        match, coarse_size = generate_hybrid_matching(max_node_wgt, graph)
        coarse_graph = create_coarse_graph(graph, match, coarse_size)
        transfer_list.append(copy.copy(graph.C))
        graph = coarse_graph
        adj_list.append(copy.copy(graph.A))
        node_wgt_list.append(copy.copy(graph.node_wgt))
    adj_list = [[_preprocess_adj(a)] * channel_num for a in adj_list]

    # ---- labels & masks (per split, one-hot) ----
    y_int = data.y.detach().cpu().numpy().astype(np.int64)
    num_classes = int(y_int.max()) + 1
    one_hot_all = np.eye(num_classes, dtype=np.float32)[y_int]

    def _split_oh(mask_t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        m = mask_t.detach().cpu().numpy().astype(bool)
        oh = np.zeros_like(one_hot_all)
        oh[m] = one_hot_all[m]
        return oh, m.astype(np.int32)

    y_train_oh_np, train_mask_np = _split_oh(data.train_mask)
    if hasattr(data, "val_mask") and data.val_mask is not None:
        y_val_oh_np, val_mask_np = _split_oh(data.val_mask)
    else:
        y_val_oh_np = np.zeros_like(one_hot_all)
        val_mask_np = np.zeros(num_nodes, dtype=np.int32)
    if hasattr(data, "test_mask") and data.test_mask is not None:
        y_test_oh_np, test_mask_np = _split_oh(data.test_mask)
    else:
        y_test_oh_np = np.zeros_like(one_hot_all)
        test_mask_np = np.zeros(num_nodes, dtype=np.int32)

    # ---- supervised-contrastive support matrices ----
    tr_idx = np.argwhere(train_mask_np > 0)[:, 0]
    train_mat01 = _cal_class01_mat(y_train_oh_np, train_mask_np)
    if tr_idx.size > 0:
        y_dim1 = np.argmax(y_train_oh_np, axis=1)
        mats = _cal_intra_class_mat01(y_dim1[tr_idx])
        num_labeled = int(tr_idx.size)
        mats[0] = mats[0] + np.eye(num_labeled)
    else:
        mats = [np.zeros((0, 0)), np.zeros((0, 0))]

    return CG3Artifacts(
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        num_nodes=num_nodes,
        feature_sp=feature_sp,
        support_sp=support_sp,
        support_tuple=support_tuple,
        edge_pos=edge_pos,
        num_features_nonzero=num_features_nonzero,
        transfer_list=transfer_list,
        adj_list=adj_list,
        node_wgt_list=node_wgt_list,
        y_train_oh=torch.from_numpy(y_train_oh_np.astype(np.float32)),
        train_mask_int=torch.from_numpy(train_mask_np.astype(np.int32)),
        y_val_oh=torch.from_numpy(y_val_oh_np.astype(np.float32)),
        val_mask_int=torch.from_numpy(val_mask_np.astype(np.int32)),
        y_test_oh=torch.from_numpy(y_test_oh_np.astype(np.float32)),
        test_mask_int=torch.from_numpy(test_mask_np.astype(np.int32)),
        train_idx_np=tr_idx,
        train_mat01=train_mat01,
        mats_intra_inter=mats,
    )


def _read_graph_from_adj(adj: sp.spmatrix) -> tuple[Graph, dict]:
    """Build a `Graph` (the coarsening pipeline's input) from a scipy adj.

    Mirrors snapshot's `utils.read_graph_from_adj` for the case where we don't
    need a node-id mapping.
    """
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()
    node_num = adj.shape[0]
    edge_num = int(adj.nnz)
    graph = Graph(node_num, edge_num)
    edge_ptr = 0
    graph.adj_idx[0] = 0
    for i in range(node_num):
        neighbors = adj.indices[adj.indptr[i] : adj.indptr[i + 1]]
        weights = adj.data[adj.indptr[i] : adj.indptr[i + 1]]
        for j in range(len(neighbors)):
            neigh = int(neighbors[j])
            graph.adj_list[edge_ptr] = neigh
            graph.adj_wgt[edge_ptr] = float(weights[j])
            graph.degree[i] += float(weights[j])
            edge_ptr += 1
        graph.adj_idx[i + 1] = edge_ptr
        graph.node_wgt[i] = 1
    graph.A = adj
    return graph, {}


"""Preprocessing for CG3 — port of the snapshot/86b0818 hierarchy pipeline.

Replaces the previous main-branch hierarchy builder. The artifacts produced
here mirror what `CG3Method/main.py` + `CG3Method/train.py` set up:
  * `feature_sp`   — sparse, row-normalized feature tensor.
  * `support_sp`   — D⁻⁰·⁵ (A+I) D⁻⁰·⁵ as a sparse tensor.
  * `support_tuple` — `(coords, values, shape)` tuple kept for `edge_pos`.
  * `transfer_list / adj_list / node_wgt_list` — per-level coarsening artifacts
    consumed by `HGCN` / `HGAT` (each level's `adj` is replicated `channel_num`
    times to mimic the snapshot's TF code).
  * `train_mat01`, `mats_intra_inter`, `tr_idx`, `y_train_oh`, `train_mask_int`,
    `y_val_oh`, `val_mask_int`, `y_test_oh`, `test_mask_int` — the supervised
    contrastive book-keeping and one-hot labels per split.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix

from .coarsen import create_coarse_graph, generate_hybrid_matching
from .graph import Graph


# ---------------------------------------------------------------------------
# helpers ported from CG3Method/funcCNN.py
# ---------------------------------------------------------------------------


def _normalize_features_csr(features: sp.spmatrix) -> sp.spmatrix:
    rowsum = np.asarray(features.sum(axis=1)).flatten()
    r_inv = np.power(rowsum.astype(float), -1, where=rowsum > 0)
    r_inv[~np.isfinite(r_inv)] = 0.0
    return sp.diags(r_inv).dot(features)


def _normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum.astype(float), -0.5, where=rowsum > 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat).transpose().dot(d_mat).tocoo()


def _sparse_to_tuple(mx: sp.spmatrix):
    if not sp.issparse(mx):
        mx = sp.coo_matrix(mx)
    else:
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    return coords, values, mx.shape


def _preprocess_features(features: sp.spmatrix):
    if not sp.issparse(features):
        features = sp.lil_matrix(features)
    features = _normalize_features_csr(features)
    if not sp.issparse(features):
        features = sp.lil_matrix(features)
    return _sparse_to_tuple(features)


def _preprocess_adj(adj: sp.spmatrix):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj_normalized = _normalize_adj(adj + sp.eye(adj.shape[0]))
    return _sparse_to_tuple(adj_normalized)


def _tuple_to_torch_sparse(sparse_tuple) -> torch.Tensor:
    coords, values, shape = sparse_tuple
    indices = torch.from_numpy(np.asarray(coords).T.astype("int64"))
    values_t = torch.from_numpy(np.asarray(values, dtype=np.float32))
    return torch.sparse_coo_tensor(indices, values_t, torch.Size(shape)).coalesce()


def _cal_class01_mat(y_train_oh: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Pairwise same-class indicator matrix; rows for unlabeled nodes are 0."""
    y = np.argmax(y_train_oh, axis=1)
    not_train = np.argwhere(~train_mask.astype(bool))
    y[not_train] = -1
    num_classes = int(np.max(y)) + 1
    mat01 = np.zeros([y_train_oh.shape[0], y_train_oh.shape[0]])
    for i in range(num_classes):
        pos = np.argwhere(y == i)
        if pos.size == 0:
            continue
        for j in range(pos.shape[0]):
            mat01[pos[j, 0], pos[:, 0]] = 1
    np.fill_diagonal(mat01, 0)
    return mat01


def _cal_intra_class_mat01(y: np.ndarray) -> list[np.ndarray]:
    """Returns [intra, inter] over the labeled-node × labeled-node block."""
    n = int(np.shape(y)[0])
    num_classes = int(np.max(y)) + 1
    intra = np.zeros([n, n])
    inter = np.ones([n, n])
    for c in range(num_classes):
        pos = np.argwhere(y == c)
        for k in range(pos.shape[0]):
            intra[pos[k, 0], pos[:, 0]] = 1
    inter -= intra
    intra -= np.eye(n)
    return [intra, inter]


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------


@dataclass
class CG3Artifacts:
    # feature / adjacency
    input_dim: int
    num_classes: int
    num_nodes: int
    feature_sp: torch.Tensor
    support_sp: torch.Tensor
    support_tuple: tuple
    edge_pos: np.ndarray
    num_features_nonzero: int
    # hierarchy
    transfer_list: list = field(default_factory=list)
    adj_list: list = field(default_factory=list)
    node_wgt_list: list = field(default_factory=list)
    # labels / masks (per split)
    y_train_oh: torch.Tensor = None
    train_mask_int: torch.Tensor = None
    y_val_oh: torch.Tensor = None
    val_mask_int: torch.Tensor = None
    y_test_oh: torch.Tensor = None
    test_mask_int: torch.Tensor = None
    # supervised-contrastive book-keeping
    train_idx_np: np.ndarray = None
    train_mat01: np.ndarray = None
    mats_intra_inter: list = None


def build_cg3_artifacts(
    data, *, coarsen_level: int, max_node_wgt: int, channel_num: int
) -> CG3Artifacts:
    """Build everything CG3 / HGCN / HGAT need from a PyG Data object.

    Numpy-side preprocessing runs on CPU; the returned tensors are CPU tensors
    (the caller / `model.to(device)` move them to GPU)."""
    device_data = data
    edge_index_cpu = data.edge_index.detach().cpu()
    num_nodes = int(data.num_nodes)

    # ---- adjacency (scipy CSR) ----
    adj = to_scipy_sparse_matrix(edge_index_cpu, num_nodes=num_nodes).tocsr()

    # ---- features (scipy sparse) ----
    x_cpu = data.x.detach().cpu().numpy()
    features_sp = sp.csr_matrix(x_cpu.astype(np.float32))
    feature_tuple = _preprocess_features(features_sp)
    feature_sp = _tuple_to_torch_sparse(feature_tuple)
    input_dim = feature_tuple[2][1]
    num_features_nonzero = int(feature_tuple[1].shape[0])

    # ---- support (D^-0.5 (A+I) D^-0.5) ----
    support_tuple = _preprocess_adj(adj)
    support_sp = _tuple_to_torch_sparse(support_tuple)
    edge_pos = np.asarray(support_tuple[0])

    # ---- hierarchy via graph coarsening ----
    graph, _ = _read_graph_from_adj(adj)
    transfer_list = []
    adj_list = [copy.copy(graph.A)]
    node_wgt_list = [copy.copy(graph.node_wgt)]
    for _ in range(coarsen_level):
        match, coarse_size = generate_hybrid_matching(max_node_wgt, graph)
        coarse_graph = create_coarse_graph(graph, match, coarse_size)
        transfer_list.append(copy.copy(graph.C))
        graph = coarse_graph
        adj_list.append(copy.copy(graph.A))
        node_wgt_list.append(copy.copy(graph.node_wgt))
    adj_list = [[_preprocess_adj(a)] * channel_num for a in adj_list]

    # ---- labels & masks (per split, one-hot) ----
    y_int = data.y.detach().cpu().numpy().astype(np.int64)
    num_classes = int(y_int.max()) + 1
    one_hot_all = np.eye(num_classes, dtype=np.float32)[y_int]

    def _split_oh(mask_t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        m = mask_t.detach().cpu().numpy().astype(bool)
        oh = np.zeros_like(one_hot_all)
        oh[m] = one_hot_all[m]
        return oh, m.astype(np.int32)

    y_train_oh_np, train_mask_np = _split_oh(data.train_mask)
    if hasattr(data, "val_mask") and data.val_mask is not None:
        y_val_oh_np, val_mask_np = _split_oh(data.val_mask)
    else:
        y_val_oh_np = np.zeros_like(one_hot_all)
        val_mask_np = np.zeros(num_nodes, dtype=np.int32)
    if hasattr(data, "test_mask") and data.test_mask is not None:
        y_test_oh_np, test_mask_np = _split_oh(data.test_mask)
    else:
        y_test_oh_np = np.zeros_like(one_hot_all)
        test_mask_np = np.zeros(num_nodes, dtype=np.int32)

    # ---- supervised-contrastive support matrices ----
    tr_idx = np.argwhere(train_mask_np > 0)[:, 0]
    train_mat01 = _cal_class01_mat(y_train_oh_np, train_mask_np)
    if tr_idx.size > 0:
        y_dim1 = np.argmax(y_train_oh_np, axis=1)
        mats = _cal_intra_class_mat01(y_dim1[tr_idx])
        num_labeled = int(tr_idx.size)
        mats[0] = mats[0] + np.eye(num_labeled)
    else:
        mats = [np.zeros((0, 0)), np.zeros((0, 0))]

    return CG3Artifacts(
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        num_nodes=num_nodes,
        feature_sp=feature_sp,
        support_sp=support_sp,
        support_tuple=support_tuple,
        edge_pos=edge_pos,
        num_features_nonzero=num_features_nonzero,
        transfer_list=transfer_list,
        adj_list=adj_list,
        node_wgt_list=node_wgt_list,
        y_train_oh=torch.from_numpy(y_train_oh_np.astype(np.float32)),
        train_mask_int=torch.from_numpy(train_mask_np.astype(np.int32)),
        y_val_oh=torch.from_numpy(y_val_oh_np.astype(np.float32)),
        val_mask_int=torch.from_numpy(val_mask_np.astype(np.int32)),
        y_test_oh=torch.from_numpy(y_test_oh_np.astype(np.float32)),
        test_mask_int=torch.from_numpy(test_mask_np.astype(np.int32)),
        train_idx_np=tr_idx,
        train_mat01=train_mat01,
        mats_intra_inter=mats,
    )


def _read_graph_from_adj(adj: sp.spmatrix) -> tuple[Graph, dict]:
    """Build a `Graph` (the coarsening pipeline's input) from a scipy adj.

    Mirrors snapshot's `utils.read_graph_from_adj` for the case where we don't
    need a node-id mapping.
    """
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()
    node_num = adj.shape[0]
    edge_num = int(adj.nnz)
    graph = Graph(node_num, edge_num)
    edge_ptr = 0
    graph.adj_idx[0] = 0
    for i in range(node_num):
        neighbors = adj.indices[adj.indptr[i] : adj.indptr[i + 1]]
        weights = adj.data[adj.indptr[i] : adj.indptr[i + 1]]
        for j in range(len(neighbors)):
            neigh = int(neighbors[j])
            graph.adj_list[edge_ptr] = neigh
            graph.adj_wgt[edge_ptr] = float(weights[j])
            graph.degree[i] += float(weights[j])
            edge_ptr += 1
        graph.adj_idx[i + 1] = edge_ptr
        graph.node_wgt[i] = 1
    graph.A = adj
    return graph, {}


"""Local-view layers for the CG3 GNNModel.

Faithful port of the snapshot/86b0818 `CG3Method/CG3Layer.py`
(`GraphConvolution`, `GraphAttention`, `MLP`) and its init/dropout helpers.
The TF-style `placeholders` dict and `dp_fea0` getter pattern are dropped;
the dropout rate and `num_features_nonzero` are constructor kwargs and the
layer uses `self.training` to switch dropout on/off.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform(shape, scale=0.05):
    return nn.Parameter(
        torch.empty(*shape, dtype=torch.float32).uniform_(-scale, scale)
    )


def zeros(shape):
    return nn.Parameter(torch.zeros(*shape, dtype=torch.float32))


def glorot(shape):
    init_range = math.sqrt(6.0 / (shape[0] + shape[1]))
    return nn.Parameter(
        torch.empty(*shape, dtype=torch.float32).uniform_(-init_range, init_range)
    )


def sparse_dropout(x: torch.Tensor, keep_prob: float, noise_shape: int) -> torch.Tensor:
    if keep_prob >= 1.0:
        return x
    indices = x._indices()
    values = x._values()
    nnz = values.shape[0]
    random_tensor = keep_prob + torch.rand(nnz, device=values.device)
    dropout_mask = torch.floor(random_tensor).bool()
    new_indices = indices[:, dropout_mask]
    new_values = values[dropout_mask] * (1.0 / keep_prob)
    return torch.sparse_coo_tensor(new_indices, new_values, x.shape).coalesce()


def _dot(x, y, sparse: bool):
    if sparse:
        return torch.sparse.mm(x, y)
    return torch.matmul(x, y)


class GraphConvolution(nn.Module):
    """Single-channel GCN layer used by GNNModel's local view.

    Note: `support` (the normalized adjacency) is set once via the attribute
    by GNNModel before calling forward — this mirrors the snapshot.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        support=None,
        num_features_nonzero: int | None = None,
        act=F.softplus,
        bias: bool = False,
        sparse_inputs: bool = False,
        isnorm: bool = False,
        isSparse: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.act = act
        self.support = support
        self.use_bias = bias
        self.isnorm = isnorm
        self.isSparse = isSparse
        self.sparse_inputs = sparse_inputs
        self.dropout = float(dropout)
        self.num_features_nonzero = num_features_nonzero

        self.vars: dict[str, nn.Parameter] = {}
        w = glorot([input_dim, output_dim])
        self.vars["weights_0"] = w
        self.register_parameter("weights_0", w)
        if self.use_bias:
            b = zeros([output_dim])
            self.vars["bias"] = b
            self.register_parameter("bias", b)

    def _dropout_rate(self) -> float:
        return self.dropout if self.training else 0.0

    def forward(self, inputs):
        x = inputs
        rate = self._dropout_rate()

        if self.sparse_inputs:
            x = sparse_dropout(x, 1.0 - rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=rate, training=self.training)

        pre_sup = _dot(x, self.vars["weights_0"], sparse=self.sparse_inputs)
        output = _dot(self.support, pre_sup, sparse=self.isSparse)

        if self.use_bias:
            output = output + self.vars["bias"]
        if self.isnorm:
            output = F.normalize(output, p=2, dim=0)
        return self.act(output)


class GraphAttention(nn.Module):
    """Single-channel GAT-style layer used by GNNModel's local view.

    Uses scatter-add over `support`'s edge indices; supports sparse feature
    input on the very first layer (matching snapshot behavior).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        support=None,
        num_features_nonzero: int | None = None,
        act=F.softplus,
        bias: bool = False,
        sparse_inputs: bool = False,
        isnorm: bool = False,
        isSparse: bool = True,
        dropout: float = 0.0,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.act = act
        self.support = support
        self.use_bias = bias
        self.isnorm = isnorm
        self.isSparse = isSparse
        self.sparse_inputs = sparse_inputs
        self.dropout = float(dropout)
        self.num_features_nonzero = num_features_nonzero
        self.alpha = alpha

        self.vars: dict[str, nn.Parameter] = {}
        self.vars["weights"] = glorot([input_dim, output_dim])
        self.register_parameter("weights", self.vars["weights"])
        self.vars["attn_l"] = glorot([output_dim, 1])
        self.vars["attn_r"] = glorot([output_dim, 1])
        self.register_parameter("attn_l", self.vars["attn_l"])
        self.register_parameter("attn_r", self.vars["attn_r"])

        # Treat all params under self.vars uniformly for the L2-on-vars used
        # by GNNModel's weight-decay term — keep "weights_0" alias too.
        self.vars["weights_0"] = self.vars["weights"]

        if self.use_bias:
            b = zeros([output_dim])
            self.vars["bias"] = b
            self.register_parameter("bias", b)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def _dropout_rate(self) -> float:
        return self.dropout if self.training else 0.0

    def forward(self, inputs):
        x = inputs
        rate = self._dropout_rate()

        if self.sparse_inputs:
            x = sparse_dropout(x, 1.0 - rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=rate, training=self.training)

        Wh = _dot(x, self.vars["weights"], sparse=self.sparse_inputs)

        adj = self.support.coalesce() if self.support.is_sparse else self.support
        indices = adj.indices()
        row, col = indices[0], indices[1]

        Wh_i = Wh[row]
        Wh_j = Wh[col]

        e = torch.matmul(Wh_i, self.vars["attn_l"]).squeeze(-1) + torch.matmul(
            Wh_j, self.vars["attn_r"]
        ).squeeze(-1)
        e = self.leakyrelu(e)

        e = e - e.max()
        exp_e = torch.exp(e)

        denom = torch.zeros(Wh.size(0), device=Wh.device).scatter_add_(0, row, exp_e)
        alpha = exp_e / (denom[row] + 1e-9)

        alpha = F.dropout(alpha, p=rate, training=self.training)

        output = torch.zeros_like(Wh)
        output.index_add_(0, row, alpha.unsqueeze(1) * Wh_j)

        if self.use_bias:
            output = output + self.vars["bias"]
        if self.isnorm:
            output = F.normalize(output, p=2, dim=0)
        return self.act(output)


class MLP(nn.Module):
    """Single-matmul MLP layer (used as the edge-decoder W_edge in CG3)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        act=F.softplus,
        bias: bool = False,
        sparse_inputs: bool = False,
        isnorm: bool = False,
        isSparse: bool = False,
    ):
        super().__init__()
        self.act = act
        self.use_bias = bias
        self.isnorm = isnorm
        self.isSparse = isSparse
        self.sparse_inputs = sparse_inputs

        self.vars: dict[str, nn.Parameter] = {}
        w = glorot([input_dim, output_dim])
        self.vars["weights_0"] = w
        self.register_parameter("weights_0", w)
        if self.use_bias:
            b = zeros([output_dim])
            self.vars["bias"] = b
            self.register_parameter("bias", b)

    def forward(self, inputs):
        x = inputs
        output = _dot(x, self.vars["weights_0"], sparse=self.sparse_inputs)
        if self.use_bias:
            output = output + self.vars["bias"]
        if self.isnorm:
            output = F.normalize(output, p=2, dim=0)
        return self.act(output)


"""CG3 GNNModel — local-view GCN/GAT fused with a global HGCN/HGAT view.

Faithful port of the snapshot/86b0818 `CG3Method/CG3Model.py`. Key
differences vs. the original:
- The optimizer is owned by Hydra's `BaseMethod.build_optimizer`, not by the
  model.
- `dp_fea0` (dropout-getter list) is replaced by an explicit `dropout: float`
  ctor kwarg, with `self.training` toggling it on/off.
- Numpy buffers from preprocessing (`edge_pos`, `train_idx`, `train_mat01`,
  `mat01_intra/inter`) are converted to tensors and registered as buffers
  here so `.to(device)` moves them.
- `forward(features, support, labels, mask)` returns the same
  `(outputs, loss, accuracy)` 3-tuple as the snapshot. Inference paths can
  pass any labels/mask and ignore the loss/accuracy outputs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.base import BaseViewLoss
from src.losses.structural import StructuralContrastiveLoss

from .cg3_layers import MLP, GraphAttention, GraphConvolution

from .semantic_channel import SemanticChannel


def masked_softmax_cross_entropy(
    preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """One-hot labels + int mask, matching the snapshot's TF formulation."""
    log_probs = F.log_softmax(preds, dim=1)
    loss = -(labels * log_probs).sum(dim=1)
    mask = mask.float()
    mean = mask.mean()
    if mean.item() == 0:
        return torch.zeros((), device=preds.device)
    mask = mask / mean
    loss = loss * mask
    return loss.mean()


def masked_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    correct = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1)).float()
    mask = mask.float()
    mean = mask.mean()
    if mean.item() == 0:
        return torch.zeros((), device=preds.device)
    mask = mask / mean
    return (correct * mask).mean()


class GNNModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        hidden: int,
        input_dim: int,
        global_model: nn.Module,
        train_idx,
        edge_pos,
        mat01_tr_te,
        weight_decay: float,
        local_model: str,
        dropout: float,
        num_features_nonzero: int,
        view_loss: BaseViewLoss | None = None,
    ):
        super().__init__()

        self.weight_decay = float(weight_decay)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden1 = hidden
        self.global_model = global_model
        self.dropout = float(dropout)
        # Pluggable structural / composite view loss (default = original CG3).
        self.view_loss = (
            view_loss if view_loss is not None else StructuralContrastiveLoss()
        )

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

        # Numpy preprocessing buffers — converted to tensors, registered so
        # `.to(device)` moves them with the model.
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
        # train_mat01 (N×N) is registered as a buffer in the snapshot but
        # never read in forward — skipped here to avoid OOM on PubMed (~19K
        # nodes → ~1.5 GB float32). Drop without behavioral change.
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

        # Two GNN class layers — the snapshot's classifier head.
        self.classlayers = nn.ModuleList()
        self.classlayers.append(
            LocalLayer(
                act=hidden_act,
                input_dim=self.input_dim,
                output_dim=self.hidden1,
                support=None,  # set per forward
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

        # Edge generative MLP (p_e_xy decoder).
        self.p_e_yy_w_contra = MLP(
            act=(lambda x: x),
            input_dim=2 * self.num_classes,
            output_dim=1,
            sparse_inputs=False,
            isSparse=True,
            bias=True,
        )

        self.outputs: torch.Tensor | None = None
        self.concat_vec_local: torch.Tensor | None = None
        self.concat_vec_global: torch.Tensor | None = None
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
    ):
        # Class layer 1 → hidden
        self.classlayers[0].support = support
        self.classlayers[0].sparse_inputs = True
        h0 = self.classlayers[0](features)

        # Class layer 2 → num_classes
        self.classlayers[1].support = support
        self.classlayers[1].sparse_inputs = False
        h1 = self.classlayers[1](h0)

        self.concat_vec_local = F.normalize(h1, p=2, dim=1)

        # HGCN/HGAT global view — note: features must be the same sparse
        # input the local view consumes.
        global_out = self.global_model(features)
        self.concat_vec_global = F.normalize(global_out, p=2, dim=1)

        self.outputs = F.normalize(
            0.6 * self.concat_vec_local + 0.4 * self.concat_vec_global,
            p=2,
            dim=1,
        )

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

        # --- individual losses ---
        loss_ce = loss_q_yobs_x_g
        loss_gen = self.p_e_xy
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

        # hsic_loss comes here

        total = loss_ce + 0.4 * loss_gen + loss_contrastive

        # Manual L2 on classlayer + p_e_yy_w_contra weights — matches snapshot.
        for i in range(2):
            for var in self.classlayers[i].vars.values():
                total = total + self.weight_decay * 0.5 * torch.sum(var**2)
        for var in self.p_e_yy_w_contra.vars.values():
            total = total + self.weight_decay * 0.5 * torch.sum(var**2)

        # Add the global model's own weight-decay term.
        total = total + self.global_model.loss

        self.loss = total
        self.accuracy = masked_accuracy(self.outputs, labels, mask)

        self.loss_ce = loss_ce.detach()
        self.loss_gen = loss_gen.detach()
        self.loss_contrastive = loss_contrastive.detach()
        self.loss_total = total.detach()
        reg = self.view_loss.regularizer_value()
        self.loss_reg = reg.detach() if reg is not None else None

        return self.outputs, self.loss, self.accuracy


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

    return torch.exp(-dist_sq / (2.0 * sigma**2))


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

    hsic = (K_centered * L_centered).sum() / ((n - 1) ** 2)

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

    entropy_structural = compute_entropy(logits_structural)

    entropy_semantic = compute_entropy(logits_semantic)

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

    alpha_structural = attention[:, 0:1]

    alpha_semantic = attention[:, 1:2]

    # --------------------------------------------------------------
    # 4. Entropy-adaptive final prediction.
    # --------------------------------------------------------------

    fused_logits = (
        alpha_structural * logits_structural + alpha_semantic * logits_semantic
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

    loss = -(labels * log_probs).sum(dim=1)

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

    return (correct * mask).mean()


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

        # --------------------------------------------------------------
        # Original CG3 structural loss.
        # --------------------------------------------------------------

        self.view_loss = (
            view_loss if view_loss is not None else StructuralContrastiveLoss()
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
            raise ValueError(f"Unknown local_model: {local_model}")

        # --------------------------------------------------------------
        # Preprocessing buffers.
        # --------------------------------------------------------------

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
            torch.from_numpy(
                np.sum(
                    mat01_tr_te[0],
                    axis=1,
                ).astype("float32")
            ),
        )

        self.train_idx_size = int(np.shape(train_idx)[0])

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

        self.loss = torch.tensor(0.0)

        self.accuracy = torch.tensor(0.0)

        self.p_e_xy = torch.tensor(0.0)

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

        h0 = self.classlayers[0](features)

        # ==============================================================
        # 2. LOCAL CLASSIFICATION REPRESENTATION
        # ==============================================================

        self.classlayers[1].support = support

        self.classlayers[1].sparse_inputs = False

        h1 = self.classlayers[1](h0)

        self.concat_vec_local = F.normalize(
            h1,
            p=2,
            dim=1,
        )

        # ==============================================================
        # 3. GLOBAL STRUCTURAL VIEW
        # ==============================================================

        global_out = self.global_model(features)

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
            0.6 * self.concat_vec_local + 0.4 * self.concat_vec_global,
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
            ) = self.semantic_channel(tags)

            # ----------------------------------------------------------
            # Move semantic tensors to the structural model device.
            # ----------------------------------------------------------

            structural_device = self.z_structural.device

            x_semantic = x_semantic.to(structural_device)

            h_semantic = h_semantic.to(structural_device)

            semantic_logits = semantic_logits.to(structural_device)

            # ----------------------------------------------------------
            # Store semantic outputs.
            # ----------------------------------------------------------

            self.semantic_descriptors = generated_descriptors

            self.semantic_embeddings = x_semantic

            self.semantic_logits = semantic_logits

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
            loss_hsic = self.z_structural.new_zeros(())

        self.hsic_value = loss_hsic.detach()

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
            semantic_available and loss_hsic.detach().item() < self.hsic_threshold
        )

        self.semantic_enabled = low_hsic

        # ==============================================================
        # 8. STRUCTURAL CLASSIFICATION
        #
        # This exists in BOTH branches.
        # ==============================================================

        logits_structural = self.classifier_struct(self.z_structural)

        self.logits_structural = logits_structural

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

            self.outputs = self.classifier_fused(z_combined)

            # ----------------------------------------------------------
            # Record that this branch does not use independent
            # entropy attention.
            # ----------------------------------------------------------

            self.logits_semantic = self.semantic_logits

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

            logits_structural = self.logits_structural

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
                logits_semantic = self.semantic_logits

            else:
                logits_semantic = self.classifier_semantic(self.z_semantic)

            self.logits_semantic = logits_semantic

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

            self.alpha_structural = alpha_structural

            self.alpha_semantic = alpha_semantic

            self.entropy_structural = entropy_structural

            self.entropy_semantic = entropy_semantic

        # ==============================================================
        # 11. CLASSIFICATION LOSS
        # ==============================================================

        loss_q_yobs_x_g = masked_softmax_cross_entropy(
            self.outputs,
            labels,
            mask,
        )

        # ==============================================================
        # 12. ORIGINAL CG3 EDGE GENERATION
        # ==============================================================

        y_ei_local = self.concat_vec_local.index_select(
            0,
            self.edge_pos_i,
        )

        y_ej_global = self.concat_vec_global.index_select(
            0,
            self.edge_pos_j,
        )

        y_ei_global = self.concat_vec_global.index_select(
            0,
            self.edge_pos_i,
        )

        y_ej_local = self.concat_vec_local.index_select(
            0,
            self.edge_pos_j,
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
                ).clamp(min=1e-8)
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
                ).clamp(min=1e-8)
            )
        )

        self.p_e_xy = p_e_xy_1 + p_e_xy_2

        # ==============================================================
        # 13. ORIGINAL CG3 STRUCTURAL CONTRASTIVE LOSS
        # ==============================================================

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
            loss_ce + 0.4 * loss_gen + loss_contrastive + self.hsic_weight * loss_hsic
        )

        # ==============================================================
        # 15. ORIGINAL CG3 L2 REGULARIZATION
        # ==============================================================

        for i in range(2):
            for var in self.classlayers[i].vars.values():
                total = total + self.weight_decay * 0.5 * torch.sum(var**2)

        for var in self.p_e_yy_w_contra.vars.values():
            total = total + self.weight_decay * 0.5 * torch.sum(var**2)

        # --------------------------------------------------------------
        # Global model regularization.
        # --------------------------------------------------------------

        total = total + self.global_model.loss

        # ==============================================================
        # 16. FINAL METRICS
        # ==============================================================

        self.loss = total

        self.accuracy = masked_accuracy(
            self.outputs,
            labels,
            mask,
        )

        self.loss_ce = loss_ce.detach()

        self.loss_gen = loss_gen.detach()

        self.loss_contrastive = loss_contrastive.detach()

        self.loss_hsic = loss_hsic.detach()

        self.loss_total = total.detach()

        reg = self.view_loss.regularizer_value()

        self.loss_reg = reg.detach() if reg is not None else None

        return (
            self.outputs,
            self.loss,
            self.accuracy,
        )

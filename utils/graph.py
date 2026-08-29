"""Original CG3 model: hierarchy graph, coarsening, local/global layers, GNNModel."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch_geometric.utils import to_scipy_sparse_matrix

from utils.losses import BaseViewLoss, StructuralContrastiveLoss


# ---------------------------------------------------------------------------
# Graph (coarsening adjacency container)
# ---------------------------------------------------------------------------

class Graph(object):
    ''' Note: adj_list shows each edge twice. So edge_num is really two times of edge number for undirected graph.'''

    def __init__(self, node_num, edge_num):
        self.node_num = node_num  # n
        self.edge_num = edge_num  # m
        self.adj_list = np.zeros(edge_num, dtype=np.int32) - 1  # a big array for all the neighbors.
        self.adj_idx = np.zeros(node_num + 1,
                                dtype=np.int32)  # idx of the beginning neighbors in the adj_list. Pad one additional element at the end with value equal to the edge_num, i.e., self.adj_idx[-1] = edge_num
        self.adj_wgt = np.zeros(edge_num,
                                dtype=np.float32)  # same dimension as adj_list, wgt on the edge. CAN be float numbers.
        self.node_wgt = np.zeros(node_num, dtype=np.int32)
        self.cmap = np.zeros(node_num, dtype=np.int32) - 1  # mapped to coarser graph

        # weighted degree: the sum of the adjacency weight of each vertex, including self-loop.
        self.degree = np.zeros(node_num, dtype=np.float32)
        self.A = None
        self.C = None  # Matching Matrix

        self.coarser = None
        self.finer = None

    def resize_adj(self, edge_num):
        '''Resize the adjacency list/wgts based on the number of edges.'''
        self.adj_list = np.resize(self.adj_list, edge_num)
        self.adj_wgt = np.resize(self.adj_wgt, edge_num)

    def get_neighs(self, idx):
        '''obtain the list of neigbors given a node.'''
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_list[istart:iend]

    def get_neigh_edge_wgts(self, idx):
        '''obtain the weights of neighbors given a node.'''
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_wgt[istart:iend]


# ---------------------------------------------------------------------------
# Coarsening
# ---------------------------------------------------------------------------

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
    '''Generate matchings using the hybrid method. It changes the cmap in graph object, 
    return groups array and coarse_graph_size.'''
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
    #print("# groups have perfect jaccard idx (1.0): %d" % len(groups))
    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]

    sorted_idx = np.argsort(degree)
    for idx in sorted_idx:
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
                continue
            curr_wgt = norm_adj_wgt[j]
            if ((not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
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
    '''Use hashmap to find out nodes with exactly same neighbors.'''
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
    '''create the coarser graph and return it based on the groups array and coarse_graph_size'''
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
        neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
        group = groups[idx]
        for i in range(len(group)):
            merged_node = group[i]
            if (i == 0):
                coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
            else:
                coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

            istart = adj_idx[merged_node]
            iend = adj_idx[merged_node + 1]
            for j in range(istart, iend):
                k = cmap[adj_list[
                    j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
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


# ---------------------------------------------------------------------------
# Local-view layers (GCN / GAT / MLP)
# ---------------------------------------------------------------------------

def uniform(shape, scale=0.05):
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32).uniform_(-scale, scale))


def zeros(shape):
    return nn.Parameter(torch.zeros(*shape, dtype=torch.float32))


def glorot(shape):
    init_range = math.sqrt(6.0 / (shape[0] + shape[1]))
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32).uniform_(-init_range, init_range))


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

    def __init__(self, input_dim: int, output_dim: int, support=None,
                 num_features_nonzero: int | None = None,
                 act=F.softplus, bias: bool = False, sparse_inputs: bool = False,
                 isnorm: bool = False, isSparse: bool = False, dropout: float = 0.0):
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

    def __init__(self, input_dim: int, output_dim: int, support=None,
                 num_features_nonzero: int | None = None,
                 act=F.softplus, bias: bool = False, sparse_inputs: bool = False,
                 isnorm: bool = False, isSparse: bool = True,
                 dropout: float = 0.0, alpha: float = 0.2):
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

        e = (
            torch.matmul(Wh_i, self.vars["attn_l"]).squeeze(-1)
            + torch.matmul(Wh_j, self.vars["attn_r"]).squeeze(-1)
        )
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

    def __init__(self, input_dim: int, output_dim: int,
                 act=F.softplus, bias: bool = False,
                 sparse_inputs: bool = False, isnorm: bool = False, isSparse: bool = False):
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


# ---------------------------------------------------------------------------
# Global-view HGCN / HGAT layers
# ---------------------------------------------------------------------------

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

    def __init__(self, input_dim: int, output_dim: int, support, transfer,
                 mod: str, layer_index: int, dropout: float = 0.0,
                 sparse_inputs: bool = False, act=F.relu, bias: bool = False,
                 featureless: bool = False, num_features_nonzero: int | None = None,
                 node_wgt_embed_dim: int = 5):
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
            self.register_buffer(f"support_tensor_{i}", _support_to_sparse(self.support[i]))

        if self.mod in ("coarsen", "input"):
            transfer_opo = normalize(self.transfer.T, norm="l2", axis=1).astype(np.float32)
            self.register_buffer("transfer_tensor", _scipy_to_torch_sparse(transfer_opo))
        elif self.mod == "refine":
            self.register_buffer(
                "transfer_tensor",
                _scipy_to_torch_sparse(self.transfer.astype(np.float32)),
            )
        else:
            self.transfer_tensor = None

        self.channel_combine = nn.Conv1d(
            in_channels=len(self.support), out_channels=1, kernel_size=1, bias=False,
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

    def __init__(self, input_dim: int, output_dim: int, support, transfer,
                 mod: str, layer_index: int, dropout: float = 0.0,
                 sparse_inputs: bool = False, act=F.elu, bias: bool = False,
                 featureless: bool = False, num_features_nonzero: int | None = None,
                 node_wgt_embed_dim: int = 5, alpha: float = 0.2):
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
            self.register_buffer(f"support_tensor_{i}", _support_to_sparse(self.support[i]))

        if self.mod in ("coarsen", "input"):
            transfer_opo = normalize(self.transfer.T, norm="l2", axis=1).astype(np.float32)
            self.register_buffer("transfer_tensor", _scipy_to_torch_sparse(transfer_opo))
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

    def _attention(self, Wh: torch.Tensor, attn: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Hierarchy preprocessing
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


def build_cg3_artifacts(data, *, coarsen_level: int, max_node_wgt: int,
                         channel_num: int) -> CG3Artifacts:
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
        neighbors = adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
        weights = adj.data[adj.indptr[i]:adj.indptr[i + 1]]
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


# ---------------------------------------------------------------------------
# CG3 GNNModel (local + global fusion)
# ---------------------------------------------------------------------------

def masked_softmax_cross_entropy(preds: torch.Tensor, labels: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
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


def masked_accuracy(preds: torch.Tensor, labels: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    correct = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1)).float()
    mask = mask.float()
    mean = mask.mean()
    if mean.item() == 0:
        return torch.zeros((), device=preds.device)
    mask = mask / mean
    return (correct * mask).mean()


class GNNModel(nn.Module):
    def __init__(self, *, num_classes: int, hidden: int, input_dim: int,
                 global_model: nn.Module, train_idx, edge_pos,
                 mat01_tr_te, weight_decay: float,
                 local_model: str, dropout: float, num_features_nonzero: int,
                 view_loss: BaseViewLoss | None = None):
        super().__init__()

        self.weight_decay = float(weight_decay)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden1 = hidden
        self.global_model = global_model
        self.dropout = float(dropout)
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
            "edge_pos_i", torch.from_numpy(np.asarray(edge_pos[:, 0]).astype("int64")),
        )
        self.register_buffer(
            "edge_pos_j", torch.from_numpy(np.asarray(edge_pos[:, 1]).astype("int64")),
        )
        self.register_buffer(
            "train_idx_buf", torch.from_numpy(np.asarray(train_idx).astype("int64")),
        )
        # train_mat01 (N×N) is registered as a buffer in the snapshot but
        # never read in forward — skipped here to avoid OOM on PubMed (~19K
        # nodes → ~1.5 GB float32). Drop without behavioral change.
        self.register_buffer(
            "mat01_intra", torch.from_numpy(mat01_tr_te[0].astype("float32")),
        )
        self.register_buffer(
            "mat01_inter", torch.from_numpy(mat01_tr_te[1].astype("float32")),
        )
        self.register_buffer(
            "mat01_intra_rowsum",
            torch.from_numpy(np.sum(mat01_tr_te[0], axis=1).astype("float32")),
        )
        self.train_idx_size = int(np.shape(train_idx)[0])

        self.classlayers = nn.ModuleList()
        self.classlayers.append(LocalLayer(
            act=hidden_act,
            input_dim=self.input_dim,
            output_dim=self.hidden1,
            support=None,
            sparse_inputs=True,
            isSparse=True,
            dropout=hidden_dropout,
            num_features_nonzero=num_features_nonzero,
            bias=True,
        ))
        self.classlayers.append(LocalLayer(
            act=(lambda x: x),
            input_dim=self.hidden1,
            output_dim=self.num_classes,
            support=None,
            sparse_inputs=False,
            isSparse=True,
            dropout=output_dropout,
            num_features_nonzero=num_features_nonzero,
            bias=True,
        ))

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

    def forward(self, features: torch.Tensor, support: torch.Tensor,
                labels: torch.Tensor, mask: torch.Tensor):
        self.classlayers[0].support = support
        self.classlayers[0].sparse_inputs = True
        h0 = self.classlayers[0](features)

        self.classlayers[1].support = support
        self.classlayers[1].sparse_inputs = False
        h1 = self.classlayers[1](h0)

        self.concat_vec_local = F.normalize(h1, p=2, dim=1)
        global_out = self.global_model(features)
        self.concat_vec_global = F.normalize(global_out, p=2, dim=1)
        self.outputs = F.normalize(
            0.6 * self.concat_vec_local + 0.4 * self.concat_vec_global, p=2, dim=1,
        )

        loss_q_yobs_x_g = masked_softmax_cross_entropy(self.outputs, labels, mask)

        y_ei_local = self.concat_vec_local.index_select(0, self.edge_pos_i)
        y_ej_global = self.concat_vec_global.index_select(0, self.edge_pos_j)
        y_ei_global = self.concat_vec_global.index_select(0, self.edge_pos_i)
        y_ej_local = self.concat_vec_local.index_select(0, self.edge_pos_j)

        p_e_xy_1 = -torch.mean(torch.log(torch.sigmoid(
            self.p_e_yy_w_contra(torch.cat([y_ei_local, y_ej_global], dim=1))
        ).clamp(min=1e-8)))
        p_e_xy_2 = -torch.mean(torch.log(torch.sigmoid(
            self.p_e_yy_w_contra(torch.cat([y_ei_global, y_ej_local], dim=1))
        ).clamp(min=1e-8)))
        self.p_e_xy = p_e_xy_1 + p_e_xy_2

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
            self.concat_vec_local, self.concat_vec_global, loss_ctx,
        )
        total = loss_ce + 0.4 * loss_gen + loss_contrastive

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
        self.loss_total = total.detach()
        reg = self.view_loss.regularizer_value()
        self.loss_reg = reg.detach() if reg is not None else None

        return self.outputs, self.loss, self.accuracy

"""From a PyG `Data` object to what CG3 needs.

`build_hierarchy` coarsens the graph for the global view (once per run);
`build_inputs` turns one label split into the model's inputs (once per seed):
row-normalised sparse features, the GCN support D^-1/2 (A+I) D^-1/2 with its
edge list, one-hot training labels, and the same-class indicator matrices the
supervised contrastive term needs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch_geometric.utils import to_scipy_sparse_matrix

from src.coarsening import coarsen


# ---------------------------------------------------------------------------
# scipy <-> torch
# ---------------------------------------------------------------------------

def sparse_to_tuple(mx: sp.spmatrix):
    """(coords [nnz, 2], values [nnz], shape) of a sparse matrix."""
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    return coords, mx.data, mx.shape


def tuple_to_torch_sparse(sparse_tuple) -> torch.Tensor:
    coords, values, shape = sparse_tuple
    indices = torch.from_numpy(np.asarray(coords).T.astype(np.int64))
    values = torch.from_numpy(np.asarray(values, dtype=np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()


def scipy_to_torch_sparse(mx: sp.spmatrix) -> torch.Tensor:
    return tuple_to_torch_sparse(sparse_to_tuple(mx.astype(np.float32)))


# ---------------------------------------------------------------------------
# normalisation (Kipf & Welling)
# ---------------------------------------------------------------------------

def normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    """D^-1/2 A D^-1/2."""
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(axis=1)).flatten().astype(float)
    d_inv_sqrt = np.zeros_like(rowsum)
    nonzero = rowsum > 0
    d_inv_sqrt[nonzero] = rowsum[nonzero] ** -0.5
    d = sp.diags(d_inv_sqrt)
    return adj.dot(d).transpose().dot(d).tocoo()


def preprocess_adj(adj: sp.spmatrix):
    """Renormalised support D^-1/2 (A+I) D^-1/2, as a sparse tuple."""
    adj = sp.csr_matrix(adj)
    return sparse_to_tuple(normalize_adj(adj + sp.eye(adj.shape[0])))


def preprocess_features(features: sp.spmatrix):
    """Row-normalised features, as a sparse tuple."""
    rowsum = np.asarray(features.sum(axis=1)).flatten().astype(float)
    r_inv = np.zeros_like(rowsum)
    nonzero = rowsum > 0
    r_inv[nonzero] = 1.0 / rowsum[nonzero]
    return sparse_to_tuple(sp.diags(r_inv).dot(features))


# ---------------------------------------------------------------------------
# hierarchy (global view)
# ---------------------------------------------------------------------------

@dataclass
class Hierarchy:
    supports: list[torch.Tensor]  # renormalised adjacency per level, sparse [N_k, N_k]; level 0 is the input graph
    pool: list[torch.Tensor]      # level k -> k + 1: row-l2-normalised C_k^T, sparse [N_{k+1}, N_k]
    unpool: list[torch.Tensor]    # level k + 1 -> k: C_k, sparse [N_k, N_{k+1}]
    node_wgt: list[torch.Tensor]  # per level, how many input nodes each node stands for


def build_hierarchy(data, coarsen_level: int, max_node_wgt: int) -> Hierarchy:
    adj = to_scipy_sparse_matrix(data.edge_index.cpu(), num_nodes=data.num_nodes).tocsr()
    adj_list, transfer_list, node_wgt_list = coarsen(adj, coarsen_level, max_node_wgt)
    return Hierarchy(
        supports=[tuple_to_torch_sparse(preprocess_adj(a)) for a in adj_list],
        pool=[scipy_to_torch_sparse(normalize(C.T, norm="l2", axis=1)) for C in transfer_list],
        unpool=[scipy_to_torch_sparse(C) for C in transfer_list],
        node_wgt=[torch.from_numpy(nw.astype(np.int64)) for nw in node_wgt_list],
    )


# ---------------------------------------------------------------------------
# model inputs (one label split)
# ---------------------------------------------------------------------------

@dataclass
class CG3Inputs:
    input_dim: int
    num_classes: int
    features: torch.Tensor    # sparse [N, F], row-normalised
    support: torch.Tensor     # sparse [N, N], D^-1/2 (A+I) D^-1/2
    edge_pos: torch.Tensor    # [nnz, 2] (i, j) pairs of `support`, for the generative edge loss
    y: torch.Tensor           # [N] int64
    train_mask: torch.Tensor  # [N] bool — the few-label split
    val_mask: torch.Tensor    # [N] bool
    test_mask: torch.Tensor   # [N] bool
    y_train_oh: torch.Tensor  # [N, C] one-hot, zero outside train_mask
    train_idx: torch.Tensor   # [L] labelled node ids
    mat01_intra: torch.Tensor  # [L, L] 1 where the two labelled nodes share a class (diagonal 1)
    mat01_inter: torch.Tensor  # [L, L] 1 where they do not

    def to(self, device) -> "CG3Inputs":
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return CG3Inputs(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in values.items()})


def build_inputs(data) -> CG3Inputs:
    n = int(data.num_nodes)
    adj = to_scipy_sparse_matrix(data.edge_index.cpu(), num_nodes=n).tocsr()
    features = sp.csr_matrix(data.x.cpu().numpy().astype(np.float32))
    feature_tuple = preprocess_features(features)
    support_tuple = preprocess_adj(adj)

    y = data.y.cpu().long()
    num_classes = int(y.max()) + 1
    train_mask = data.train_mask.cpu().bool()
    val_mask = data.val_mask.cpu().bool() if getattr(data, "val_mask", None) is not None else torch.zeros(n, dtype=torch.bool)
    test_mask = data.test_mask.cpu().bool() if getattr(data, "test_mask", None) is not None else torch.zeros(n, dtype=torch.bool)

    y_train_oh = F.one_hot(y, num_classes).float() * train_mask.unsqueeze(1)
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    y_labelled = y[train_idx]
    mat01_intra = (y_labelled.unsqueeze(1) == y_labelled.unsqueeze(0)).float()

    return CG3Inputs(
        input_dim=int(feature_tuple[2][1]),
        num_classes=num_classes,
        features=tuple_to_torch_sparse(feature_tuple),
        support=tuple_to_torch_sparse(support_tuple),
        edge_pos=torch.from_numpy(np.asarray(support_tuple[0]).astype(np.int64)),
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        y_train_oh=y_train_oh,
        train_idx=train_idx,
        mat01_intra=mat01_intra,
        mat01_inter=1.0 - mat01_intra,
    )

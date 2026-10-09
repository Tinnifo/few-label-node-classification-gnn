"""Graph coarsening for the global view.

Builds the hierarchy that H-GCN — and with it CG3's global view — runs on:
repeated hybrid matching (structural-equivalence groups first, then normalised
heavy-edge matching) merges nodes into one coarser graph per level. The
matching code is adapted from MILE (jiongqianliang/MILE).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse as sp


class Graph:
    """Adjacency-list graph. Every undirected edge is stored twice, so
    `edge_num` is twice the number of undirected edges."""

    def __init__(self, node_num: int, edge_num: int):
        self.node_num = node_num
        self.edge_num = edge_num
        self.adj_list = np.zeros(edge_num, dtype=np.int32) - 1  # neighbours of every node, concatenated
        self.adj_idx = np.zeros(node_num + 1, dtype=np.int32)  # node i's neighbours: adj_list[adj_idx[i]:adj_idx[i + 1]]
        self.adj_wgt = np.zeros(edge_num, dtype=np.float32)  # edge weights, aligned with adj_list
        self.node_wgt = np.zeros(node_num, dtype=np.int32)  # how many input nodes each node stands for
        self.cmap = np.zeros(node_num, dtype=np.int32) - 1  # node -> node of the coarser graph
        self.degree = np.zeros(node_num, dtype=np.float32)  # weighted degree, self-loops included
        self.A: sp.spmatrix | None = None  # adjacency of this level
        self.C: sp.spmatrix | None = None  # matching matrix, this level x coarser level
        self.coarser: Graph | None = None
        self.finer: Graph | None = None

    def resize_adj(self, edge_num: int) -> None:
        self.adj_list = np.resize(self.adj_list, edge_num)
        self.adj_wgt = np.resize(self.adj_wgt, edge_num)

    def get_neighs(self, idx: int) -> np.ndarray:
        return self.adj_list[self.adj_idx[idx]:self.adj_idx[idx + 1]]


def read_graph_from_adj(adj: sp.spmatrix) -> Graph:
    """Build a `Graph` from a scipy adjacency matrix."""
    adj = adj.tocsr()
    node_num = adj.shape[0]
    graph = Graph(node_num, int(adj.nnz))
    edge_ptr = 0
    for i in range(node_num):
        neighbors = adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
        weights = adj.data[adj.indptr[i]:adj.indptr[i + 1]]
        for neigh, wgt in zip(neighbors, weights):
            graph.adj_list[edge_ptr] = int(neigh)
            graph.adj_wgt[edge_ptr] = float(wgt)
            graph.degree[i] += float(wgt)
            edge_ptr += 1
        graph.adj_idx[i + 1] = edge_ptr
        graph.node_wgt[i] = 1
    graph.A = adj
    return graph


def cmap2C(cmap: np.ndarray) -> sp.csr_matrix:
    """Matching matrix C with C[i, cmap[i]] = 1."""
    n = len(cmap)
    return sp.csr_matrix((np.ones(n, dtype=np.int64), (np.arange(n), cmap)))


def normalized_adj_wgt(graph: Graph) -> np.ndarray:
    """Edge weights normalised by the endpoint degrees (MILE)."""
    adj_wgt = graph.adj_wgt
    adj_idx = graph.adj_idx
    norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
    degree = graph.degree
    for i in range(graph.node_num):
        for j in range(adj_idx[i], adj_idx[i + 1]):
            neigh = graph.adj_list[j]
            norm_wgt[j] = adj_wgt[neigh] / np.sqrt(degree[i] * degree[neigh])
    return norm_wgt


def jaccard_idx_preprocess(graph: Graph, matched: list, groups: list) -> None:
    """Structural-equivalence matching: nodes with identical neighbour sets
    become one group."""
    neighs2node = defaultdict(list)
    for i in range(graph.node_num):
        neighs2node[str(sorted(graph.get_neighs(i)))].append(i)
    for group in neighs2node.values():
        if len(group) > 1:
            for node in group:
                matched[node] = True
            groups.append(group)


def generate_hybrid_matching(max_node_wgt: int, graph: Graph):
    """Hybrid matching (MILE): structural-equivalence groups first, then every
    unmatched node — lowest degree first — pairs with its unmatched neighbour
    of largest normalised edge weight, as long as the merged node weight stays
    within `max_node_wgt`. Writes `graph.cmap`; returns `(groups, coarse_size)`."""
    node_num = graph.node_num
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    node_wgt = graph.node_wgt
    cmap = graph.cmap
    norm_adj_wgt = normalized_adj_wgt(graph)
    groups = []  # one group per coarse node
    matched = [False] * node_num

    jaccard_idx_preprocess(graph, matched, groups)
    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(node_num)]

    for idx in np.argsort(degree):
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if neigh == idx:  # exclude the self-loop, or nodes mostly match themselves
                continue
            curr_wgt = norm_adj_wgt[j]
            if (not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt:
                max_idx = neigh
                max_wgt = curr_wgt
        matched[idx] = matched[max_idx] = True
        if idx == max_idx:  # no partner found: the node stays on its own
            groups.append([idx])
        else:
            groups.append([idx, max_idx])

    coarse_graph_size = 0
    for group in groups:
        for node in group:
            cmap[node] = coarse_graph_size
        coarse_graph_size += 1
    return groups, coarse_graph_size


def create_coarse_graph(graph: Graph, groups: list, coarse_graph_size: int) -> Graph:
    """Merge every group into one coarse node; parallel edge weights add up."""
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
    nedges = 0
    for coarse_node_idx, group in enumerate(groups):
        neigh_dict = {}  # coarse neighbour -> its position in coarse_adj_list
        for i, merged_node in enumerate(group):
            if i == 0:
                coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
            else:
                coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

            for j in range(adj_idx[merged_node], adj_idx[merged_node + 1]):
                k = cmap[adj_list[j]]  # the neighbour's node in the coarse graph
                if k not in neigh_dict:
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                coarse_degree[coarse_node_idx] += adj_wgt[j]  # self-loops kept in the degree

        coarse_adj_idx[coarse_node_idx + 1] = nedges

    coarse_graph.edge_num = nedges
    coarse_graph.resize_adj(nedges)
    C = cmap2C(cmap)
    graph.C = C
    coarse_graph.A = C.transpose().dot(graph.A).dot(C)
    return coarse_graph


def coarsen(adj: sp.spmatrix, coarsen_level: int, max_node_wgt: int):
    """Coarsen `adj` `coarsen_level` times.

    Returns `(adj_list, transfer_list, node_wgt_list)`: the adjacency of every
    level (level 0 is `adj` itself), the matching matrix between consecutive
    levels (`transfer_list[k]` maps level k to level k + 1) and the node
    weights of every level.
    """
    graph = read_graph_from_adj(adj)
    adj_list = [graph.A]
    transfer_list = []
    node_wgt_list = [graph.node_wgt.copy()]
    for _ in range(coarsen_level):
        groups, coarse_size = generate_hybrid_matching(max_node_wgt, graph)
        coarse_graph = create_coarse_graph(graph, groups, coarse_size)
        transfer_list.append(graph.C)
        graph = coarse_graph
        adj_list.append(graph.A)
        node_wgt_list.append(graph.node_wgt.copy())
    return adj_list, transfer_list, node_wgt_list

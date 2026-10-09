"""Layers of the two structural views.

Local view: `GraphConvolution` / `GraphAttention` on the input graph, and
`MLP`, CG3's edge decoder. Global view: `HGCNGraphConvolution` /
`HGCNGraphAttention`, the multi-channel, hierarchy-aware layers of H-GCN.

Every layer keeps its weight matrices in `self.vars`, an `nn.ParameterDict`,
so the model can put CG3's explicit L2 penalty on exactly those parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def identity(x):
    return x


def glorot(shape) -> nn.Parameter:
    bound = math.sqrt(6.0 / (shape[0] + shape[1]))
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32).uniform_(-bound, bound))


def zeros(shape) -> nn.Parameter:
    return nn.Parameter(torch.zeros(*shape, dtype=torch.float32))


def sparse_dropout(x: torch.Tensor, keep_prob: float) -> torch.Tensor:
    """Dropout on the non-zero entries of a sparse COO tensor."""
    if keep_prob >= 1.0:
        return x
    x = x.coalesce()
    values = x.values()
    keep = torch.floor(keep_prob + torch.rand(values.shape[0], device=values.device)).bool()
    return torch.sparse_coo_tensor(x.indices()[:, keep], values[keep] / keep_prob, x.shape).coalesce()


def dot(x: torch.Tensor, y: torch.Tensor, sparse: bool) -> torch.Tensor:
    return torch.sparse.mm(x, y) if sparse else torch.matmul(x, y)


def _dropout(x: torch.Tensor, rate: float, sparse_inputs: bool, training: bool) -> torch.Tensor:
    if sparse_inputs:
        return sparse_dropout(x, 1.0 - rate)
    return F.dropout(x, p=rate, training=training)


# ---------------------------------------------------------------------------
# Local view
# ---------------------------------------------------------------------------

class GraphConvolution(nn.Module):
    """GCN layer: act(support @ (x W) + b)."""

    def __init__(self, input_dim: int, output_dim: int, *, act=identity, bias: bool = False,
                 sparse_inputs: bool = False, dropout: float = 0.0):
        super().__init__()
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.dropout = float(dropout)
        self.vars = nn.ParameterDict({"weights_0": glorot([input_dim, output_dim])})
        if bias:
            self.vars["bias"] = zeros([output_dim])

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        rate = self.dropout if self.training else 0.0
        x = _dropout(x, rate, self.sparse_inputs, self.training)
        out = torch.sparse.mm(support, dot(x, self.vars["weights_0"], self.sparse_inputs))
        if "bias" in self.vars:
            out = out + self.vars["bias"]
        return self.act(out)


class GraphAttention(nn.Module):
    """GAT layer (one head) over the edges of `support`."""

    def __init__(self, input_dim: int, output_dim: int, *, act=identity, bias: bool = False,
                 sparse_inputs: bool = False, dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.dropout = float(dropout)
        self.vars = nn.ParameterDict({
            "weights": glorot([input_dim, output_dim]),
            "attn_l": glorot([output_dim, 1]),
            "attn_r": glorot([output_dim, 1]),
        })
        if bias:
            self.vars["bias"] = zeros([output_dim])
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        rate = self.dropout if self.training else 0.0
        x = _dropout(x, rate, self.sparse_inputs, self.training)
        Wh = dot(x, self.vars["weights"], self.sparse_inputs)

        row, col = support.coalesce().indices()
        e = (torch.matmul(Wh[row], self.vars["attn_l"]).squeeze(-1)
             + torch.matmul(Wh[col], self.vars["attn_r"]).squeeze(-1))
        e = self.leakyrelu(e)
        exp_e = torch.exp(e - e.max())
        denom = torch.zeros(Wh.size(0), device=Wh.device).scatter_add_(0, row, exp_e)
        attention = F.dropout(exp_e / (denom[row] + 1e-9), p=rate, training=self.training)

        out = torch.zeros_like(Wh).index_add_(0, row, attention.unsqueeze(1) * Wh[col])
        if "bias" in self.vars:
            out = out + self.vars["bias"]
        return self.act(out)


class MLP(nn.Module):
    """One linear map with CG3's parameter dict — the edge decoder."""

    def __init__(self, input_dim: int, output_dim: int, *, act=identity, bias: bool = False):
        super().__init__()
        self.act = act
        self.vars = nn.ParameterDict({"weights_0": glorot([input_dim, output_dim])})
        if bias:
            self.vars["bias"] = zeros([output_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.vars["weights_0"])
        if "bias" in self.vars:
            out = out + self.vars["bias"]
        return self.act(out)


# ---------------------------------------------------------------------------
# Global view (H-GCN / H-GAT)
# ---------------------------------------------------------------------------

class _HierarchicalLayer(nn.Module):
    """Shared plumbing of the H-GCN / H-GAT layers.

    `mod` says where the layer sits in the hierarchy: `input` and `coarsen`
    layers pool their output to the next coarser level through `transfer`,
    `refine` layers unpool it to the next finer level, the `output` layer
    returns it as is. Coarsen and refine layers concatenate a node-weight
    embedding to their input. `forward` returns `(transferred, output)`; the
    untransferred output feeds the skip connections.
    """

    def __init__(self, support: torch.Tensor, transfer: torch.Tensor | None, mod: str, *,
                 dropout: float, sparse_inputs: bool, act):
        super().__init__()
        self.mod = mod
        self.act = act
        self.dropout = float(dropout)
        self.sparse_inputs = sparse_inputs
        self.register_buffer("support", support, persistent=False)
        self.register_buffer("transfer", transfer, persistent=False)

    def _prepare(self, x: torch.Tensor, node_emb: torch.Tensor | None) -> torch.Tensor:
        if self.mod in ("coarsen", "refine"):
            x = torch.cat([x, node_emb], dim=1)
        rate = self.dropout if self.training else 0.0
        return _dropout(x, rate, self.sparse_inputs, self.training)

    def _finish(self, out: torch.Tensor):
        if "bias" in self.vars:
            out = out + self.vars["bias"]
        out = self.act(out)
        if self.transfer is None:
            return out, out
        return torch.sparse.mm(self.transfer, out), out


class HGCNGraphConvolution(_HierarchicalLayer):
    """Multi-channel GCN layer of H-GCN: `channels` weight matrices over one
    support, mixed by a 1x1 convolution."""

    def __init__(self, input_dim: int, output_dim: int, support, transfer, mod: str, *,
                 channels: int, node_wgt_embed_dim: int = 5, dropout: float = 0.0,
                 sparse_inputs: bool = False, act=F.relu, bias: bool = False):
        super().__init__(support, transfer, mod, dropout=dropout, sparse_inputs=sparse_inputs, act=act)
        in_dim = input_dim + node_wgt_embed_dim if mod in ("coarsen", "refine") else input_dim
        self.channels = channels
        self.vars = nn.ParameterDict({f"weights_{i}": glorot([in_dim, output_dim]) for i in range(channels)})
        if bias:
            self.vars["bias"] = zeros([output_dim])
        self.channel_combine = nn.Conv1d(channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, node_emb: torch.Tensor | None = None):
        x = self._prepare(x, node_emb)
        outs = [torch.sparse.mm(self.support, dot(x, self.vars[f"weights_{i}"], self.sparse_inputs))
                for i in range(self.channels)]
        out = self.channel_combine(torch.stack(outs, dim=1)).squeeze(1)  # [N, channels, d] -> [N, d]
        return self._finish(out)


class HGCNGraphAttention(_HierarchicalLayer):
    """Multi-channel GAT layer of H-GAT: one (W, a) pair per channel, channel
    outputs averaged."""

    def __init__(self, input_dim: int, output_dim: int, support, transfer, mod: str, *,
                 channels: int, node_wgt_embed_dim: int = 5, dropout: float = 0.0,
                 sparse_inputs: bool = False, act=F.elu, bias: bool = False, alpha: float = 0.2):
        super().__init__(support, transfer, mod, dropout=dropout, sparse_inputs=sparse_inputs, act=act)
        in_dim = input_dim + node_wgt_embed_dim if mod in ("coarsen", "refine") else input_dim
        self.channels = channels
        self.vars = nn.ParameterDict()
        for i in range(channels):
            self.vars[f"weights_{i}"] = glorot([in_dim, output_dim])
            self.vars[f"attn_{i}"] = glorot([2 * output_dim, 1])
        if bias:
            self.vars["bias"] = zeros([output_dim])
        self.leakyrelu = nn.LeakyReLU(alpha)

    def _attention(self, Wh: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        row, col = self.support.coalesce().indices()
        e = self.leakyrelu(torch.matmul(torch.cat([Wh[row], Wh[col]], dim=1), attn).squeeze(1))
        exp_e = torch.exp(e - e.max())
        denom = torch.zeros(Wh.size(0), device=Wh.device).scatter_add_(0, row, exp_e)
        attention = exp_e / (denom[row] + 1e-16)
        return torch.zeros_like(Wh).index_add(0, row, Wh[col] * attention.unsqueeze(1))

    def forward(self, x: torch.Tensor, node_emb: torch.Tensor | None = None):
        x = self._prepare(x, node_emb)
        outs = [self._attention(dot(x, self.vars[f"weights_{i}"], self.sparse_inputs), self.vars[f"attn_{i}"])
                for i in range(self.channels)]
        return self._finish(torch.stack(outs, dim=2).mean(dim=2))

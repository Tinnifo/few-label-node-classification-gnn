"""Loss terms of the model.

CG3's supervised structural contrastive loss between the local and global
views, the HSIC dependence between the structural and semantic embeddings, and
the masked cross-entropy / accuracy CG3 computes on one-hot labels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# masked supervision (one-hot labels, boolean mask)
# ---------------------------------------------------------------------------

def masked_softmax_cross_entropy(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(preds, dim=1)
    loss = -(labels * log_probs).sum(dim=1)
    mask = mask.float()
    mean = mask.mean()
    if mean.item() == 0:
        return torch.zeros((), device=preds.device, dtype=preds.dtype)
    return (loss * (mask / mean)).mean()


def masked_accuracy(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    correct = torch.eq(torch.argmax(preds, dim=1), torch.argmax(labels, dim=1)).float()
    mask = mask.float()
    mean = mask.mean()
    if mean.item() == 0:
        return torch.zeros((), device=preds.device, dtype=preds.dtype)
    return (correct * (mask / mean)).mean()


# ---------------------------------------------------------------------------
# CG3 structural contrastive loss (local view <-> global view)
# ---------------------------------------------------------------------------

def _contrast(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Per node i: exp(a_i·b_i/τ) / mean_j exp(a_i·b_j/τ)."""
    sim = torch.exp(torch.matmul(a, b.t()) / temperature)
    return torch.diagonal(sim, 0) / (sim.mean(dim=1) + 1e-8)


def _supervised_contrast(a: torch.Tensor, b: torch.Tensor, intra: torch.Tensor, inter: torch.Tensor,
                         temperature: float) -> torch.Tensor:
    """Over the labelled nodes: mean similarity to same-class nodes over mean
    similarity to all other nodes."""
    n = a.size(0)
    sim = torch.exp(torch.matmul(a, b.t()) / temperature)
    pos = (sim * intra).sum(dim=1)
    neg = ((sim * inter).sum(dim=1) + pos) / max(n - 1, 1)
    pos = pos / (intra.sum(dim=1) + 1e-8)
    return pos / (neg + 1e-8)


def structural_contrastive_loss(local: torch.Tensor, global_: torch.Tensor, train_idx: torch.Tensor,
                                mat01_intra: torch.Tensor, mat01_inter: torch.Tensor,
                                temperature: float = 0.5, hp1: float = 0.9) -> torch.Tensor:
    """CG3's contrastive term: every node's local view against the global
    views of all nodes (and the reverse), plus a supervised term on the
    labelled nodes that pulls same-class pairs together across the views."""
    ratios = torch.cat([_contrast(local, global_, temperature), _contrast(global_, local, temperature)])
    loss = -hp1 * torch.log(ratios.clamp(min=1e-8)).mean()

    h_local = local.index_select(0, train_idx)
    h_global = global_.index_select(0, train_idx)
    ratios = torch.cat([
        _supervised_contrast(h_local, h_global, mat01_intra, mat01_inter, temperature),
        _supervised_contrast(h_global, h_local, mat01_intra, mat01_inter, temperature),
    ])
    return loss - hp1 * torch.log(ratios.clamp(min=1e-8)).mean()


# ---------------------------------------------------------------------------
# HSIC (structural embedding <-> semantic embedding)
# ---------------------------------------------------------------------------

def rbf_kernel(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if x.size(0) <= 1:
        return torch.ones((x.size(0), x.size(0)), device=x.device, dtype=x.dtype)
    dist_sq = torch.cdist(x, x, p=2).pow(2)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


def hsic_loss(z1: torch.Tensor, z2: torch.Tensor, sigma: float = 1.0, max_samples: int = 1024) -> torch.Tensor:
    """Biased HSIC estimate with RBF kernels; subsamples rows above `max_samples`."""
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


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Linear CKA (Kornblith et al. 2019): normalized HSIC with a linear kernel.

    Scale-invariant, so it cannot be gamed by shrinking a representation's
    norm (the alpha^2 pathology of a raw HSIC penalty) — the estimator the
    shared+private disparity term uses. O(n * d^2): cheap at full batch for
    the d=64 projection heads.
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    num = (y.T @ x).pow(2).sum()
    den = torch.sqrt((x.T @ x).pow(2).sum() * (y.T @ y).pow(2).sum()) + eps
    return num / den

"""Few-label splits: which training nodes a run may see.

Both strategies sample from the dataset's standard training pool only, so the
fixed validation and test splits stay untouched and no node leaks across.

WebKB (Texas, Cornell, Wisconsin) and Actor ship 10 Geom-GCN splits as
[num_nodes, 10] boolean masks. `select_split` keeps one column so every mask
is 1D before the label strategy runs.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_split(data, split_idx: int):
    """Turn [N, S] train/val/test masks into [N] masks by keeping one column.

    Masks that are already 1D are left alone. `split_idx` wraps around, so
    seed 12 with 10 splits uses split 2.
    """
    for key in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, key, None)
        if mask is not None and mask.dim() == 2:
            setattr(data, key, mask[:, split_idx % mask.size(1)].clone())
    return data


def set_few_label_mask(data, num_labels_per_class: int, seed: int):
    """Keep `num_labels_per_class` training nodes of every class (or all of
    them, if a class has fewer)."""
    set_seed(seed)
    num_classes = int(data.y.max()) + 1
    pool = data.train_mask.clone()
    device = pool.device
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    for c in range(num_classes):
        idx = ((data.y == c) & pool).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.size(0)).to(device)]
        train_mask[idx[:num_labels_per_class]] = True
    data.train_mask = train_mask
    return data


def set_budget_percent(data, fraction: float, seed: int):
    """Keep `fraction` of all nodes as training nodes, drawn from the training
    pool (capped at the pool size)."""
    set_seed(seed)
    pool_idx = data.train_mask.nonzero(as_tuple=True)[0]
    device = pool_idx.device
    num_train = min(int(fraction * data.num_nodes), len(pool_idx))
    chosen = pool_idx[torch.randperm(len(pool_idx)).to(device)[:num_train]]
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[chosen] = True
    data.train_mask = train_mask
    return data


def apply_label_strategy(data, strategy: str, budget, seed: int, split_idx: int | None = None):
    """Build the training mask. `strategy` is `per_class` or `percentage`.

    `split_idx` picks which of the dataset's fixed splits to use; it defaults
    to `seed`, so each seed gets its own split.
    """
    data = select_split(data, seed if split_idx is None else split_idx)
    if strategy == "per_class":
        return set_few_label_mask(data, int(budget), seed)
    if strategy == "percentage":
        return set_budget_percent(data, float(budget), seed)
    raise ValueError(f"Unknown label strategy: {strategy}")


def format_budget(budget) -> str:
    """`20` for 20 labels per class, `5.00%` for a 0.05 fraction."""
    b = float(budget)
    if b >= 1:
        return f"{int(b)}"
    return f"{b * 100:.3f}%" if b < 0.001 else f"{b * 100:.2f}%"
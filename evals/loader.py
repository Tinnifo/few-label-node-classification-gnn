"""Dataset loading + label-strategy dispatch.

`kind` is `planetoid` (Cora / CiteSeer / PubMed) or `placeholder` (TAG stubs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from evals.labels import set_budget_percent, set_few_label_mask, set_seed


@dataclass
class LoadedDataset:
    data: object
    name: str
    in_channels: int
    num_classes: int


def load_dataset(
    name: str,
    root: str = "data",
    normalize_features: bool = False,
    *,
    kind: Optional[str] = None,
) -> LoadedDataset:
    """Load a graph dataset.

    `kind` is `planetoid` or `placeholder`. If omitted, Planetoid names are
    inferred for backward compatibility.
    """
    kind = (kind or _infer_kind(name)).lower()
    if kind == "planetoid":
        return _load_planetoid(name, root=root, normalize_features=normalize_features)
    if kind == "placeholder":
        raise NotImplementedError(
            f"Dataset '{name}' is a heterophilic placeholder (kind=placeholder). "
            "Wire a text-attributed heterophilic loader in evals/loader.py or data/."
        )
    raise ValueError(f"Unknown dataset kind '{kind}' for dataset '{name}'.")


def _infer_kind(name: str) -> str:
    if name.lower() in {"cora", "citeseer", "pubmed"}:
        return "planetoid"
    return "placeholder"


def _load_planetoid(name: str, root: str, normalize_features: bool) -> LoadedDataset:
    transform = T.NormalizeFeatures() if normalize_features else None
    dataset = Planetoid(root=root, name=name, transform=transform)
    data = dataset[0]
    return LoadedDataset(
        data=data,
        name=name,
        in_channels=int(dataset.num_features),
        num_classes=int(dataset.num_classes),
    )


def apply_label_strategy(data, strategy: str, budget, seed: int):
    """Build the train mask. `strategy` is `per_class` or `percentage`."""
    set_seed(seed)
    if strategy == "per_class":
        return set_few_label_mask(data, int(budget), seed)
    if strategy == "percentage":
        return set_budget_percent(data, float(budget), seed)
    raise ValueError(f"Unknown label strategy: {strategy}")


def format_budget(budget) -> str:
    b = float(budget)
    if b >= 1:
        return f"{int(b)}"
    return f"{b * 100:.3f}%" if b < 0.001 else f"{b * 100:.2f}%"

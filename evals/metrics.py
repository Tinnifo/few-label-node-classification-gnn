"""Evaluation metrics for few-label node classification.

For now: accuracy and macro F1. Extend `compute_metrics` / `conf/metrics/` when
adding more metrics without changing method code.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true: Sequence, y_pred: Sequence) -> Dict[str, float]:
    """Return a dict of scalar metrics. Empty inputs → zeros."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

#!/usr/bin/env python3
"""Graph-free logistic probes on the semantic views (FEW-31, E-probe).

For each dataset x view: sample `budget`/class train nodes from the non-test
pool (train|val masks), fit LogisticRegression(C=1.0, max_iter=2000) on the
embeddings ALONE (no graph), evaluate on the test mask. 10 seeds resample the
train nodes. Answers two FEW-30 anomalies:
  P1  is e5's -7.1 on cora an embedding problem or a fusion problem?
  P2  do the pubmed tape views solve the task without the graph (= leakage)?

Run after scripts/download_data.py (+ make_texts.py cora for the e5 encode).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

DATASETS = {"cora": "cora", "citeseer_tag": "citeseer", "pubmed": "pubmed"}
VIEWS = ["features", "sbert", "gpt3l", "tape_stripped", "tape_full"]
BUDGET, SEEDS = 20, range(10)


def load(ds_dir: str):
    base = Path("datasets/tag") / ds_dir
    z = np.load(base / f"{ds_dir}.npz", allow_pickle=True)
    split = np.load(base / f"{ds_dir}_planetoid_split.npz", allow_pickle=True)
    views = {"features": z["node_features"]}
    for v in VIEWS[1:]:
        views[v] = np.load(base / f"{ds_dir}_sem_{v}.npy")
    return z["node_labels"], split["test_mask"], views


def probe(X, y, test_mask, seed):
    rng = np.random.RandomState(seed)
    pool = np.where(~test_mask)[0]
    train = np.concatenate([rng.permutation(pool[y[pool] == c])[:BUDGET] for c in np.unique(y)])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[train], y[train])
    return clf.score(X[test_mask], y[test_mask])


def main() -> None:
    rows = []
    for name, ds_dir in DATASETS.items():
        y, test_mask, views = load(ds_dir)
        if name == "cora":  # P1: e5 embeddings, encoded here (GPU), 'query:' prefix as in the pipeline
            sys.path.insert(0, ".")
            from src.semantic import HuggingFaceSentenceEncoder
            texts = Path("datasets/tag/cora/cora_texts.txt").read_text().splitlines()
            enc = HuggingFaceSentenceEncoder("intfloat/e5-large-v2")
            if torch.cuda.is_available():
                enc = enc.to("cuda")
            views["e5"] = enc(texts).numpy()
        for view, X in views.items():
            accs = [probe(X, y, test_mask, s) for s in SEEDS]
            rows.append([name, view, round(float(np.mean(accs)), 4), round(float(np.std(accs, ddof=1)), 4)])
            print(f"{name:14} {view:14} {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}", flush=True)
    Path("out").mkdir(exist_ok=True)
    with open("out/probe_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "view", "mean_acc", "std"])
        w.writerows(rows)


if __name__ == "__main__":
    main()

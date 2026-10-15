"""Load the text-attributed Texas graph (Graph-COM/Text-Attributed-Graphs).

Texas.csv has one row per web page: node_id, raw_text, label, neighbor_ids.
This builds everything the pipeline needs from that one file, so graph,
labels and texts line up by construction:

  * x           TF-IDF of raw_text (the CSV has no bag-of-words features)
  * edge_index  from neighbor_ids (directed, like PyG's WebKB)
  * y           label
  * masks       [N, num_splits] random 60/20/20 splits, the same layout PyG's
                WebKB uses, so evaluation.labels.select_split works unchanged
  * texts       one single-line string per node, in node_id order

These are NOT the Geom-GCN splits (they are defined for the 183-node graph),
so results are not directly comparable with published Texas numbers.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops


def load_text_attributed(csv_path: str, num_features: int = 1703, num_splits: int = 10,
                         fractions: tuple = (0.6, 0.2, 0.2), split_seed: int = 0):
    df = pd.read_csv(csv_path).sort_values("node_id").reset_index(drop=True)
    n = len(df)
    if not (df["node_id"].to_numpy() == np.arange(n)).all():
        raise ValueError(f"{csv_path}: node_id is not 0..{n - 1}")

    # one line per node: collapse all whitespace, including newlines
    texts = [" ".join(str(t).split()) or "empty page" for t in df["raw_text"].fillna("")]

    # graph
    src, dst = [], []
    for i, neighbors in enumerate(df["neighbor_ids"]):
        for j in ast.literal_eval(neighbors):
            if 0 <= int(j) < n:
                src.append(i)
                dst.append(int(j))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=n)

    # features
    tfidf = TfidfVectorizer(max_features=num_features, sublinear_tf=True, stop_words="english")
    x = torch.tensor(tfidf.fit_transform(texts).toarray(), dtype=torch.float)

    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)

    # [N, num_splits] masks
    rng = np.random.RandomState(split_seed)
    n_train, n_val = int(fractions[0] * n), int(fractions[1] * n)
    train = torch.zeros(n, num_splits, dtype=torch.bool)
    val = torch.zeros(n, num_splits, dtype=torch.bool)
    test = torch.zeros(n, num_splits, dtype=torch.bool)
    for s in range(num_splits):
        perm = torch.tensor(rng.permutation(n))
        train[perm[:n_train], s] = True
        val[perm[n_train:n_train + n_val], s] = True
        test[perm[n_train + n_val:], s] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train, val_mask=val, test_mask=test)
    return data, texts
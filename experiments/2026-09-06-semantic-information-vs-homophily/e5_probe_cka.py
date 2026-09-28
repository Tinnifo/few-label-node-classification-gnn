#!/usr/bin/env python3
"""Exp 5 — graph-free linear probes + linear CKA across the homophily sweep.

Mirrors scripts/probe_views.py (LogisticRegression C=1.0, max_iter=2000, 20/class from the
non-test pool, 10 seeds). Adds Â²X (sym-normalised 2-hop propagation of X) as a parameter-free
structural stand-in, and linear CKA (Kornblith 2019) between each semantic view and X / Â²X.
Hetero bundles: class-stratified 25/25/50 split with RNG(0) — the load_tag_native rule.
"""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

ROOT = Path("datasets/tag"); OUT = Path("out/e5"); OUT.mkdir(parents=True, exist_ok=True)
TRIO = ["cora", "citeseer", "pubmed"]
HET = ["texas", "cornell", "washington", "wisconsin", "amazon_ratings", "actor"]
BUDGET, SEEDS = 20, range(10)


def split_masks(y, base, ds):
    f = base / f"{ds}_planetoid_split.npz"
    if f.exists():
        s = np.load(f, allow_pickle=True); return s["test_mask"].astype(bool)
    rng = np.random.RandomState(0); test = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); n_pool = max(1, int(.25 * len(idx))); n_val = max(1, int(.25 * len(idx)))
        test[idx[n_pool + n_val:]] = True
    return test


def prop2(X, edges, n):
    r, c = edges[:, 0], edges[:, 1]
    A = sp.coo_matrix((np.ones(len(r) * 2), (np.r_[r, c], np.r_[c, r])), shape=(n, n)).tocsr()
    A = A + sp.eye(n); d = np.asarray(A.sum(1)).ravel(); Dm = sp.diags(1 / np.sqrt(d))
    Ah = Dm @ A @ Dm
    return Ah @ (Ah @ X)


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    return float(num / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))


def probe(X, y, test_mask, seed):
    rng = np.random.RandomState(seed); pool = np.where(~test_mask)[0]; train = []
    for c in np.unique(y):
        cand = pool[y[pool] == c]; train += list(rng.choice(cand, min(BUDGET, len(cand)), replace=False))
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[train], y[train])
    return float((clf.predict(X[test_mask]) == y[test_mask]).mean())


def h_adj(y, edges):
    r, c = edges[:, 0], edges[:, 1]; E = len(r)
    h_edge = float((y[r] == y[c]).mean())
    deg = np.bincount(np.r_[r, c], minlength=len(y)); p = np.bincount(y, weights=deg) / (2 * E)
    return h_edge, float((h_edge - (p ** 2).sum()) / (1 - (p ** 2).sum()))


rows, cka_rows = [], []
for ds in TRIO + HET:
    base = ROOT / ds; t = time.time()
    z = np.load(base / f"{ds}.npz", allow_pickle=True)
    y, X, edges = z["node_labels"].astype(int), z["node_features"].astype(np.float32), z["edges"]
    n = len(y); test = split_masks(y, base, ds)
    views = {"features": X, "prop2": np.asarray(prop2(X, edges, n), dtype=np.float32)}
    for v in ("sbert", "gpt3l"):
        f = base / f"{ds}_sem_{v}.npy"
        if f.exists(): views[v] = np.load(f).astype(np.float32)
    he, ha = h_adj(y, edges)
    for v, M in views.items():
        accs = [probe(M, y, test, s) for s in SEEDS]
        for s, a in zip(SEEDS, accs): rows.append({"dataset": ds, "view": v, "seed": s, "acc": a, "h_edge": he, "h_adj": ha, "n": n})
    for v in ("sbert", "gpt3l"):
        if v in views:
            for sv in ("features", "prop2"):
                cka_rows.append({"dataset": ds, "semantic": v, "structural": sv, "cka": linear_cka(views[v], views[sv]), "h_adj": ha})
    print(ds, f"n={n} h_adj={ha:+.3f}", {v: round(float(np.mean([r['acc'] for r in rows if r['dataset']==ds and r['view']==v])), 3) for v in views}, f"{time.time()-t:.0f}s", flush=True)

with open(OUT / "probe.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(OUT / "cka.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cka_rows[0])); w.writeheader(); w.writerows(cka_rows)

summ = {}
for ds in TRIO + HET:
    R = {v: np.array([r["acc"] for r in rows if r["dataset"] == ds and r["view"] == v]) for v in set(r["view"] for r in rows if r["dataset"] == ds)}
    best_struct = max(("features", "prop2"), key=lambda v: R[v].mean())
    d = {}
    for v in ("sbert", "gpt3l"):
        if v in R:
            delta = R[v] - R[best_struct]; ci = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
            d[v] = {"mean_acc": float(R[v].mean()), "delta_vs_" + best_struct: float(delta.mean()), "ci95": float(ci), "sig": bool(abs(delta.mean()) > ci)}
    summ[ds] = {"h_adj": float([r["h_adj"] for r in rows if r["dataset"] == ds][0]), "best_struct": best_struct, "struct_acc": float(R[best_struct].mean()), **d}
sp_ = {}
for v in ("sbert", "gpt3l"):
    for sv in ("features", "prop2"):
        pts = [(r["cka"], r["h_adj"]) for r in cka_rows if r["semantic"] == v and r["structural"] == sv]
        if len(pts) >= 4:
            rho, p = spearmanr([q[0] for q in pts], [q[1] for q in pts]); sp_[f"{v}~{sv}"] = {"rho": float(rho), "p": float(p), "n": len(pts)}
json.dump({"per_dataset": summ, "spearman_cka_vs_hadj": sp_}, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(sp_, indent=1))

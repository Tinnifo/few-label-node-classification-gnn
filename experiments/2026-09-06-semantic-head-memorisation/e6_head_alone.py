#!/usr/bin/env python3
"""Exp 6 — the semantic classifier (MLP head) trained alone, per dataset × encoder, 10 seeds, 400 full-batch steps.

Head = PrecomputedSemanticChannel's MLP + classifier: Linear(d,256)-ReLU-Linear(256,128)-Linear(128,C).
Optimiser = Adam(lr=0.01, weight_decay=0.0) as in src/cg3_semantic.py:448; no dropout, no early stop.
Logs per step: train acc, val acc, test acc, mean test entropy (nats). Splits as e5.
"""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
ROOT = Path("datasets/tag"); OUT = Path("out/e6"); OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["cora", "citeseer", "pubmed", "texas", "cornell", "washington", "wisconsin", "amazon_ratings", "actor"]
BUDGET, SEEDS, STEPS, HID, DIM, LR = 20, range(10), 400, 256, 128, 0.01


def masks(y, base, ds):
    f = base / f"{ds}_planetoid_split.npz"
    if f.exists():
        s = np.load(f, allow_pickle=True); return ~(s["test_mask"] | s["val_mask"]), s["val_mask"].astype(bool), s["test_mask"].astype(bool)
    rng = np.random.RandomState(0); pool = np.zeros(len(y), bool); val = pool.copy(); test = pool.copy()
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); npool = max(1, int(.25 * len(idx))); nval = max(1, int(.25 * len(idx)))
        pool[idx[:npool]] = True; val[idx[npool:npool + nval]] = True; test[idx[npool + nval:]] = True
    return pool, val, test


def run(Xn, y, pool, val, test, seed, C):
    rng = np.random.RandomState(seed); torch.manual_seed(seed); train = []
    for c in np.unique(y):
        cand = np.where(pool & (y == c))[0]; train += list(rng.choice(cand, min(BUDGET, len(cand)), replace=False))
    tr = torch.tensor(train); X = torch.from_numpy(Xn); Y = torch.from_numpy(y)
    head = nn.Sequential(nn.Linear(X.size(1), HID), nn.ReLU(), nn.Linear(HID, DIM), nn.Linear(DIM, C))
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=0.0)
    vm, tm = torch.from_numpy(val), torch.from_numpy(test); log = []
    for step in range(1, STEPS + 1):
        head.train(); opt.zero_grad(); out = head(X); loss = F.cross_entropy(out[tr], Y[tr]); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            out = head(X); pred = out.argmax(1); p = F.softmax(out, 1)
            ent = -(p * torch.log(p + 1e-12)).sum(1)
            log.append({"step": step, "loss": loss.item(),
                        "train_acc": (pred[tr] == Y[tr]).float().mean().item(),
                        "val_acc": (pred[vm] == Y[vm]).float().mean().item(),
                        "test_acc": (pred[tm] == Y[tm]).float().mean().item(),
                        "test_entropy": ent[tm].mean().item(), "train_entropy": ent[tr].mean().item()})
    return log


rows, summary = [], {}
for ds in DATASETS:
    base = ROOT / ds; z = np.load(base / f"{ds}.npz", allow_pickle=True); y = z["node_labels"].astype(int); C = int(y.max() + 1)
    pool, val, test = masks(y, base, ds)
    for enc in ("sbert", "gpt3l"):
        f = base / f"{ds}_sem_{enc}.npy"
        if not f.exists(): continue
        Xn = np.load(f).astype(np.float32); t = time.time(); per_seed = []
        for s in SEEDS:
            log = run(Xn, y, pool, val, test, s, C)
            for r in log: rows.append({"dataset": ds, "encoder": enc, "seed": s, **r})
            ta = np.array([r["train_acc"] for r in log]); te = np.array([r["test_acc"] for r in log]); en = np.array([r["test_entropy"] for r in log]); va = np.array([r["val_acc"] for r in log])
            hit = np.where(ta >= 1.0)[0]; step1 = int(hit[0] + 1) if len(hit) else None
            plateau = int(np.where(te >= 0.95 * te.max())[0][0] + 1); es = int(va.argmax() + 1)
            per_seed.append({"seed": s, "step_train_acc_1": step1, "test_acc_final": float(te[-1]), "test_acc_max": float(te.max()), "plateau_step": plateau,
                             "entropy_at_plateau": float(en[plateau - 1]), "entropy_final": float(en[-1]), "early_stop_step": es, "test_acc_at_early_stop": float(te[es - 1])})
        s1 = [q["step_train_acc_1"] for q in per_seed if q["step_train_acc_1"]]
        summary[f"{ds}/{enc}"] = {
            "n_seeds_reaching_train_1": len(s1), "step_train_acc_1_median": float(np.median(s1)) if s1 else None,
            "step_train_acc_1_iqr": [float(np.percentile(s1, 25)), float(np.percentile(s1, 75))] if s1 else None,
            "test_acc_final_mean": float(np.mean([q["test_acc_final"] for q in per_seed])),
            "test_acc_at_early_stop_mean": float(np.mean([q["test_acc_at_early_stop"] for q in per_seed])),
            "early_stop_step_median": float(np.median([q["early_stop_step"] for q in per_seed])),
            "entropy_plateau_mean": float(np.mean([q["entropy_at_plateau"] for q in per_seed])),
            "entropy_final_mean": float(np.mean([q["entropy_final"] for q in per_seed])),
            "seeds_entropy_falls_after_plateau": int(sum(q["entropy_final"] < q["entropy_at_plateau"] for q in per_seed)),
            "per_seed": per_seed}
        s_ = summary[f"{ds}/{enc}"]
        print(ds, enc, f"train1@{s_['step_train_acc_1_median']} test_final={s_['test_acc_final_mean']:.3f} test@ES={s_['test_acc_at_early_stop_mean']:.3f} ent {s_['entropy_plateau_mean']:.2f}->{s_['entropy_final_mean']:.2f} ({s_['seeds_entropy_falls_after_plateau']}/10) {time.time()-t:.0f}s", flush=True)

with open(OUT / "curves.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

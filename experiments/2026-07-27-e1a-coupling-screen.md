# E1a — semantic-view coupling screen: does align-vs-preserve flip with homophily?

| | |
| --- | --- |
| **Date** | 2026-07-27 |
| **Chain link** | a (redundancy), and the draft-v1 central claim |
| **Code** | grid commits [`1141f06`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/1141f06) and [`5302e0e`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/5302e0e) (comment-only diff; rows pool safely) · `conf/experiment/e1a_crossover.yaml` · redundancy scoring [`46ed3ad`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/46ed3ad) (`scripts/score_datasets.py`) |
| **Status** | done — **pre-registered negative** |

> The branch the vault log names, `exp/pre-registration-scoring`, is not on GitHub. The commits above are.

## Hypothesis
The draft's central claim: on heterophilic graphs it pays to *preserve* the semantic view's independence from structure,
on homophilic graphs to *align* it — so the sign of (align − preserve) tracks homophily.

## Setup
- **Data:** 8 text-attributed graphs, h_adj −0.29 → +0.77 — texas, cornell, wisconsin, washington, actor_hetgb, and the
  Planetoid trio as TAG bundles.
- **Semantic view:** `sbert` = `sentence-transformers/all-MiniLM-L6-v2`, 384-d, frozen, raw node text, no prompt.
  (Earlier write-ups called this an "LLM view"; it is a text-encoder view.)
- **Arms:** `loss ∈ {structural, +align, +disparity(CKA), +hsic}` × `attach ∈ {wide, class}`.
- **Protocol:** 3 labels/class, `val_per_class=1`, 10 seeds, patience 200, early stopping on, `select_best_by=val_acc`,
  `n_sample=0`. 14 cells across 8 graphs.

## Result
1. **Redundancy scales with homophily (no training).** Spearman(h_adj, semantic–structural redundancy) = **+0.933, p < 0.001**
   over 9 staged graphs; marginal semantic gain over X runs the other way (−0.717, p = 0.030).
   *Confound:* h_adj and X-informativeness correlate +0.817, so report this as a joint gradient.
2. **The attach point decides whether a penalty is viable.**

   | penalty | attach = wide (256-d) | attach = class (C-d) | p |
   | --- | --- | --- | --- |
   | unnormalised HSIC | **−7.40 pts** (n=6) | −0.09 pts (n=5) | 0.00007 |
   | CKA disparity | −0.61 pts (n=8) | −2.98 pts (n=6) | 0.065 |

   Mechanism: CKA is scale-invariant; unnormalised HSIC falls by 10⁴ under a 100× rescale, so at wide attach the
   semantic block can shrink to satisfy the penalty without decorrelating.
3. **The crossover is falsified — flat, not underpowered.** Spearman(h_adj, align − preserve) = +0.108 (p = 0.71);
   OLS slope +0.51 pts per unit h_adj (p = 0.80), R² = 0.005, n = 14 cells.
4. **No coupling beats no coupling.**

   | arm | mean gain vs structural-only | p | n |
   | --- | --- | --- | --- |
   | align | −1.32 pts | 0.075 | 14 |
   | CKA disparity | −1.63 pts | 0.025 | 14 |
   | unnormalised HSIC | −4.08 pts | 0.009 | 11 |

   Near-miss worth remembering: actor_hetgb first read +5.16 pts; splitting its seeds gave +5.16 (0–3) vs +0.89 (4–7),
   and its control never trained (`best_epoch = 1`).

## What it changes
The draft-v1 claim is dead. Redundancy is a reliable *negative* filter only: high redundancy predicts no gain; low
redundancy does not predict that a coupling term will route the signal. Motivated E3 and the move to 20 labels/class.

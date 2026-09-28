# Exp 1 — coupling-penalty screen: does "align vs keep separate" flip with homophily?

| | |
| --- | --- |
| **Date** | 2026-07-27 |
| **Tests assumption** | 1, and the central claim of the first paper draft |
| **Code** | grid commits [`1141f06`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/1141f06) and [`5302e0e`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/5302e0e) (they differ only in comments, so results pool) · config `conf/experiment/e1a_crossover.yaml` · overlap scoring [`46ed3ad`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/46ed3ad) (`scripts/score_datasets.py`) |
| **Status** | done: **a preregistered negative result** |

## Hypothesis
The first draft claimed that on heterophilic graphs it pays to keep the semantic view *separate* from the structural
view, and on homophilic graphs to *align* the two. If so, the sign of (align − keep separate) should follow homophily.

## Setup
- **Data:** 8 text-attributed graphs spanning homophily −0.29 to +0.77: texas, cornell, wisconsin, washington,
  actor, and Cora / CiteSeer / PubMed with their raw text.
- **Semantic view:** sbert (all-MiniLM-L6-v2, 384-d, frozen) on each node's raw text, with no prompt and no LLM
  generation.
- **Arms:** four coupling losses (none, align, CKA-based disparity penalty, HSIC penalty) × two places to attach the
  semantic view (a wide 256-d hidden layer, or the C-d class layer).
- **Protocol:** 3 labels per class, 1 validation label per class, 10 seeds, patience 200, early stopping on,
  model selection by validation accuracy (CG3's published choice). 14 dataset × setting cells.

## Result
1. **The semantic view's overlap with structure rises with homophily (no training needed).** Spearman(homophily,
   overlap) = **+0.933, p < 0.001** over 9 graphs; the semantic view's extra gain over node features runs the other way
   (−0.717, p = 0.030). *Confound:* homophily and node-feature informativeness correlate at +0.817, so treat this as one
   joint trend.
2. **Where the view attaches decides whether a penalty is usable.**

   | penalty | wide attach (256-d) | class attach (C-d) | p |
   | --- | --- | --- | --- |
   | unnormalised HSIC | **−7.40 pts** (n=6) | −0.09 pts (n=5) | 0.00007 |
   | CKA disparity | −0.61 pts (n=8) | −2.98 pts (n=6) | 0.065 |

   Why: CKA does not change when the embeddings are rescaled, but unnormalised HSIC falls by 10⁴ under a 100× rescale.
   At the wide attach point the semantic block can simply shrink to satisfy the HSIC penalty without becoming any more
   independent.
3. **The flip is not there, and this is a flat result, not an underpowered one.** Spearman(homophily, align − keep
   separate) = +0.108 (p = 0.71); regression slope +0.51 pts per unit homophily (p = 0.80), R² = 0.005, n = 14.
4. **No coupling penalty beats no coupling.**

   | coupling | mean gain vs structure-only | p | n |
   | --- | --- | --- | --- |
   | align | −1.32 pts | 0.075 | 14 |
   | CKA disparity | −1.63 pts | 0.025 | 14 |
   | unnormalised HSIC | −4.08 pts | 0.009 | 11 |

   A near-miss worth remembering: actor first read +5.16 pts. Splitting its seeds into two halves gave +5.16 (seeds
   0–3) vs +0.89 (seeds 4–7), and its structure-only baseline never trained (best epoch = 1).

## What it changes
The first draft's central claim is dead. High overlap with structure reliably predicts *no* gain; low overlap does
not guarantee that a coupling penalty will make use of the extra information. This led to Exp 2 and to the move to
20 labels per class.

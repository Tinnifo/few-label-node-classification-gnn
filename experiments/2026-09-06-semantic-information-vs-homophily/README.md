# Exp 5 — does the semantic view carry label information that structure lacks, and where?

| | |
| --- | --- |
| **Date** | 2026-09-06 (preregistered and run the same day) |
| **Tests assumption** | 1 |
| **Code** | [`e5_probe_cka.py`](e5_probe_cka.py) — runs against `scaffold/data-and-experiment` at commit [`cb48a3a`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/cb48a3ab9a128554478b3825febc3ee9d2e94f0a) |
| **Compute** | cloud CPU container (8 vCPU, 16 GB), about 1 minute |
| **Raw outputs** | [`probe.csv`](probe.csv) · [`cka.csv`](cka.csv) · [`summary.json`](summary.json) · [`run_modal.log`](run_modal.log) |
| **Status** | done: **supported** |

## Hypothesis
- **Claim:** on low-homophily graphs a probe on the semantic view predicts the label better than a probe on structure,
  and the two views overlap least there.
- **Supported if:** (i) the semantic probe beats the best structural probe on all four WebKB graphs at 20 labels per
  class, over 10 paired seeds with the CI excluding 0; and (ii) Spearman(CKA(semantic, structure), homophily) > 0 across
  nine datasets.
- **What this does not test:** CG3's *trained* representations. The structural probes here use the model's inputs
  (node features X, and X averaged over 2-hop neighbourhoods, Â²X).

## Prediction (written before the run)
The semantic probe beats structure on heterophilic graphs, and overlap (CKA) is **lowest** there. On high-homophily
graphs the views overlap and the semantic view adds nothing.

## Setup
Logistic regression (C = 1.0, max_iter = 2000), 20 labelled nodes per class drawn from the non-test pool, 10 seeds.
Views: node features X, 2-hop-averaged features Â²X, sbert (384-d), gpt3l (3072-d; citation graphs + WebKB only).
CKA computed on all nodes. Nine datasets, homophily from −0.29 (texas) to +0.77 (cora).

## Result

| dataset | homophily | best structural probe | sbert probe (Δ ± CI) | gpt3l probe (Δ ± CI) | CKA sbert·Â²X | CKA gpt3l·Â²X |
| --- | --- | --- | --- | --- | --- | --- |
| texas | -0.29 | 2-hop 50.3 | 64.4 (+14.1 ± 1.1*) | 78.9 (+28.6 ± 2.0*) | 0.05 | 0.07 |
| cornell | -0.22 | features 54.8 | 64.0 (+9.2 ± 1.2*) | 78.9 (+24.0 ± 1.2*) | 0.09 | 0.12 |
| washington | -0.19 | 2-hop 62.9 | 72.9 (+10.0 ± 2.2*) | 81.3 (+18.4 ± 1.5*) | 0.15 | 0.18 |
| wisconsin | -0.17 | 2-hop 42.5 | 71.6 (+29.1 ± 1.4*) | 85.8 (+43.4 ± 1.8*) | 0.10 | 0.11 |
| amazon_ratings | +0.14 | 2-hop 21.4 | 23.0 (+1.6 ± 2.9) | — | 0.12 | — |
| actor | +0.33 | features 38.7 | 51.1 (+12.4 ± 3.5*) | — | 0.09 | — |
| citeseer | +0.68 | 2-hop 71.0 | 70.2 (-0.9 ± 0.7*) | 71.0 (-0.1 ± 0.7) | 0.60 | 0.63 |
| pubmed | +0.69 | 2-hop 72.9 | 75.0 (+2.1 ± 1.9*) | 83.3 (+10.4 ± 2.4*) | 0.58 | 0.61 |
| cora | +0.77 | 2-hop 80.5 | 74.7 (-5.7 ± 0.7*) | 77.7 (-2.8 ± 0.5*) | 0.52 | 0.56 |

\* CI excludes zero. Accuracy in %, Δ in points; "2-hop" = the Â²X probe. Spearman(CKA, homophily) = **+0.80 (p = 0.010, n = 9)** for sbert, +0.75 (p = 0.052, n = 7) for gpt3l.

![Left: semantic gain vs homophily. Right: overlap (CKA) vs homophily.](figure.png)

## Predicted vs actual

| | Predicted | Actual |
| --- | --- | --- |
| semantic − structural probe on WebKB | > 0, CI excludes 0 | **+9 to +43 pts, all four graphs** |
| overlap (CKA) vs homophily | lowest on WebKB; Spearman > 0 | **WebKB 0.05–0.18, citation graphs 0.52–0.63; +0.80 / +0.75** |
| high homophily: nothing to add | no gain | Cora loses (−5.7 / −2.8); CiteSeer level; **PubMed gains +10.4 (gpt3l)** |

## What it changes
Assumption 1 is partly supported and splits in two. **1a:** the semantic view carries label information that structure
lacks, and more of it where homophily is low (supported). **1b:** the *size* of the gain also depends on how weak the
structural signal is, as PubMed shows (open).
Caveats: this compares against structural *inputs*, not CG3's trained representations. We have not yet checked whether
the text encoders saw these datasets during pre-training, and that check should come before the large WebKB gpt3l
numbers are quoted.

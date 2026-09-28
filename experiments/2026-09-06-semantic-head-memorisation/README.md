# Exp 6 — does the semantic classifier memorise its few training labels?

| | |
| --- | --- |
| **Date** | 2026-09-06 (preregistered and run the same day) |
| **Tests assumption** | 3 |
| **Code** | [`e6_head_alone.py`](e6_head_alone.py), [`modal_run.sh`](modal_run.sh) — against `scaffold/data-and-experiment` at commit [`cb48a3a`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/cb48a3ab9a128554478b3825febc3ee9d2e94f0a) |
| **Compute** | cloud CPU container (8 vCPU, 16 GB), about 24 minutes |
| **Raw outputs** | [`summary.json`](summary.json) · [`run_modal.log`](run_modal.log) (per-step curves, 8 MB, not committed) |
| **Status** | done: **supported**, so assumption 3 does not hold as stated |

## Hypothesis
- **Claim:** the pipeline's semantic classifier, a small MLP (Linear(d,256) – ReLU – Linear(256,128) – Linear(128,C);
  Adam, lr 0.01; no weight decay, dropout or early stopping), reaches 100 % training accuracy within tens of steps.
  After that, test accuracy stops improving while the model keeps growing more confident (test entropy keeps falling).
- **Supported if:** the median step to 100 % training accuracy is ≤ 100 for every dataset × encoder, and test entropy at
  step 400 is lower than at the plateau on at least 7 of 10 seeds.

## Prediction (written before the run)
100 % training accuracy within about 40–50 steps.

## Setup
The semantic classifier trained alone (no graph): 400 full-batch steps on the 20-per-class training nodes, logging
train / test / validation accuracy and mean test entropy every step. 10 seeds; nine datasets; sbert on all, gpt3l on
the citation graphs + WebKB, which gives 16 settings.

## Result

| dataset | encoder | step when train acc hits 1.0, median [IQR] | best-validation step | test acc at step 400 | test acc at best-validation step (Δ) | test entropy: plateau → step 400 |
| --- | --- | --- | --- | --- | --- | --- |
| texas | sbert | 12 [11–14] | 4 | 63.4 | 63.1 (-0.3) | 0.56 → 0.13 |
| texas | gpt3l | 9 [8–11] | 9 | 83.2 | 87.8 (+4.6) | 0.69 → 0.05 |
| cornell | sbert | 8 [8–9] | 5 | 57.8 | 60.1 (+2.3) | 1.28 → 0.08 |
| cornell | gpt3l | 7 [7–8] | 8 | 81.0 | 83.3 (+2.3) | 0.52 → 0.05 |
| washington | sbert | never (max 0.98) | 6 | 67.1 | 70.0 (+2.9) | 0.94 → 0.12 |
| washington | gpt3l | never (max 0.98) | 8 | 79.6 | 83.8 (+4.2) | 0.83 → 0.08 |
| wisconsin | sbert | 9 [9–10] | 9 | 64.3 | 69.3 (+5.0) | 0.93 → 0.10 |
| wisconsin | gpt3l | 10 [9–11] | 10 | 84.7 | 87.7 (+3.0) | 0.65 → 0.07 |
| amazon_ratings | sbert | 13 [12–16] | 6 | 22.7 | 30.7 (+8.1) | 1.46 → 0.23 |
| actor | sbert | 14 [14–16] | 8 | 43.6 | 56.7 (+13.1) | 1.25 → 0.17 |
| citeseer | sbert | 10 [9–10] | 4 | 61.2 | 70.9 (+9.7) | 1.64 → 0.07 |
| citeseer | gpt3l | 7 [7–7] | 3 | 62.5 | 71.1 (+8.7) | 1.50 → 0.06 |
| pubmed | sbert | 6 [5–7] | 7 | 74.7 | 76.3 (+1.6) | 0.79 → 0.03 |
| pubmed | gpt3l | 5 [5–6] | 6 | 79.2 | 84.7 (+5.5) | 0.54 → 0.03 |
| cora | sbert | 10 [9–10] | 6 | 68.4 | 72.1 (+3.8) | 1.29 → 0.07 |
| cora | gpt3l | 8 [7–8] | 6 | 72.7 | 77.5 (+4.8) | 1.28 → 0.06 |

Test entropy falls after the plateau on 10 of 10 seeds in every setting. (Lower entropy = more confident.)

![Left: training curves on Cora. Right: test accuracy at step 400 vs at the best-validation step, all 16 settings.](figure.png)

## Predicted vs actual

| | Predicted | Actual |
| --- | --- | --- |
| step when training accuracy hits 100 % (Cora, sbert) | about 40–50 | **10** [9–10]; gpt3l 8 |
| test entropy after the plateau | keeps falling | **1.29 → 0.07**, 10 of 10 seeds, same shape in all 16 settings |

## What it changes
Assumption 3 does not hold: the semantic classifier memorises within about 10 steps, so every model we have trained
with it so far combined CG3 with a fully memorised semantic view. Stopping at the best-validation step is worth +2 to
+13 points in 15 of 16 settings.
Restated assumption: *the semantic classifier is stopped or calibrated using validation labels, and those labels count
towards the label budget.* Assumption 4 (fusion trusting the confident view) cannot be tested until this is fixed.

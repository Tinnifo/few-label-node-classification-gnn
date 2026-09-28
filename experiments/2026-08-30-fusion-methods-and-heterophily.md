# Exp 4 — which fusion method works best, and does the gain track heterophily?

| | |
| --- | --- |
| **Date** | 2026-08-30 |
| **Tests assumptions** | 2 (shared space), 5 |
| **Preregistration** | [`docs/hypotheses_few31.md`](../docs/hypotheses_few31.md) |
| **Full report** | [`docs/report_2026-08-30.md`](../docs/report_2026-08-30.md), sections 3–6 |
| **Code** | [`71b3c583`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/71b3c583) |
| **Compute** | cloud, 4 jobs, 10 seeds |
| **Status** | done |

## Result

**Fusion methods (gpt3l view; accuracy change vs CG3 alone, points):**

| fusion method | Cora | CiteSeer | PubMed |
| --- | --- | --- | --- |
| concatenation | −0.5 | +1.4 | **+3.9** |
| confidence-weighted attention | −2.0 | +0.1 | +2.7 |
| shared/private spaces (AM-GCN style, CKA penalty) | −7.8 ± 3.0 | −3.3 | +0.3 |

- **Concatenation is the strongest fusion method.** Attention does not rescue Cora. The extra HSIC penalty between the
  views does nothing (+0.08 ± 0.24 pts). The shared/private design is worst everywhere, with 2–3× the seed variance.
- **Probes (logistic regression, 20 labels per class, no graph):** on PubMed, a probe on the stripped LLM explanations
  scores 0.8768. That is about as high as the full trained model with that view (0.8955), which means the explanation
  carries the label without the graph and confirms the leakage. More generally, frozen text + logistic regression comes
  close to the full pipeline at these label budgets (Cora: gpt3l probe 0.777 vs CG3 0.789).
- **Heterophilic graphs: the CG3 baseline fails its sanity check on all six.** CG3 scores below always guessing the
  most common class on every one (texas 0.454 vs 0.541; actor 0.380 vs 0.488). Gains there come from rescuing a
  failing baseline, so they are provisional: concatenation improves all 6 (cornell +8.9 ± 3.5, wisconsin +9.8 ± 6.1,
  actor +8.0 ± 8.3, …). Spearman(homophily, gain) = −0.550, p = 0.125, n = 9. Attention beat concatenation by the
  preregistered 0.5 points on only 1 of the 4 most heterophilic graphs.

## What it changes
On heterophilic graphs the bottleneck is the backbone, not the view, so we need a backbone designed for heterophily.
Open question: why the shared/private design, the only one where the views share a space (assumption 2), did worst.

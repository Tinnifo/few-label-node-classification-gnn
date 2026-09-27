# FEW-31 — fusion mechanism, heterophily axis, and FEW-30 anomaly probes

| | |
| --- | --- |
| **Date** | 2026-08-30 |
| **Chain link** | b (fusion / shared space), e |
| **Linear** | FEW-31 |
| **Preregistration** | [`docs/hypotheses_few31.md`](../docs/hypotheses_few31.md) |
| **Full report** | [`docs/report_2026-08-30.md`](../docs/report_2026-08-30.md) §3–§6 |
| **Code** | [`71b3c583`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/71b3c583) |
| **Compute** | Modal, 4 jobs (probe / fusion trio / heterophilic small / amazon_ratings), 10 seeds |
| **Status** | done |

## Result

**Fusion (gpt3l view, Δ vs A0, pts):**

| fusion | cora | citeseer_tag | pubmed |
| --- | --- | --- | --- |
| concat | −0.5 | +1.4 | **+3.9** |
| entropy attention | −2.0 | +0.1 | +2.7 |
| shared_private v1 (AM-GCN placement, CKA) | −7.8 ± 3.0 | −3.3 | +0.3 |

- **Concat is the strongest fusion.** F1 refuted (attention does not rescue cora); F2 confirmed (cross-view HSIC
  penalty is a no-op: +0.08 ± 0.24 pts). shared_private v1 is worst everywhere with 2–3× seed variance (S1, S2 refuted).
- **Probes (graph-free logistic, 20/class):** pubmed `tape_stripped` probe 0.8768 ≈ trained A3 0.8955 → the explanation
  carries the label without the graph (leakage confirmed). Frozen text + logistic regression is close to the full
  pipeline at these budgets (cora gpt3l probe 0.777 vs A0 0.789).
- **Heterophilic graphs — validity check V1 fails on all six:** CG3 (A0) is below the majority-class rate everywhere
  (texas 0.454 vs 0.541; actor 0.380 vs 0.488). Gains there rescue a failing baseline and are provisional:
  concat Δ > 0 on 6/6 (cornell +8.9 ± 3.5, wisconsin +9.8 ± 6.1, actor +8.0 ± 8.3, …);
  Spearman(h_adj, gain) = −0.550, p = 0.125, n = 9. G3 (attention ≥ concat + 0.5 on heterophilic graphs) refuted, 1/4.

## What it changes
The backbone, not the view, is the bottleneck on heterophilic graphs → a heterophily-appropriate backbone is needed.
Open: why shared_private (the only arm with one shared space, link b) did worst — see PROGRESS.md.

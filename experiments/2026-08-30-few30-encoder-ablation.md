# FEW-30 — semantic-view encoder ablation on CG3 at 20 labels/class

| | |
| --- | --- |
| **Date** | 2026-08-30 |
| **Chain link** | e (partial — encoder level only) |
| **Linear** | FEW-30 |
| **Preregistration** | [`docs/hypotheses.md`](../docs/hypotheses.md) (§1–3 frozen before the run; results appended in §4) |
| **Full report** | [`docs/report_2026-08-30.md`](../docs/report_2026-08-30.md) |
| **Code** | [`3358678b`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/3358678b), configs [`130e724a`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/130e724a); report code [`71b3c583`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/71b3c583) |
| **Compute** | Modal A10G + 8 CPU, seeds 0–9 |
| **Status** | done |

## Setup
Arms: A0 = CG3 (`semantic=none`), A1 = `sbert`, A2 = `e5`, A2′ = `gpt3l`, A3 = `tape_stripped`, A4 = `tape_full`.
20 labels/class, paired per-seed Δ with 95 % CI; *real* if the CI excludes 0, *interesting* if |Δ| ≥ 0.5 pts.
Validity check V1 passed: new A0 0.7890 ± 0.0122 ≡ faithful CG3 port 0.7853 on cora.

## Result — Δ accuracy vs A0 (pts, paired 95 % CI)

| arm (concat fusion) | cora | citeseer_tag | pubmed |
| --- | --- | --- | --- |
| sbert (MiniLM 384d) | **−3.6** [−4.2, −3.0] | +1.2 [+0.1, +2.4] | −1.6 [−2.2, −0.9] |
| e5-large (1024d) | **−7.1** [−7.8, −6.3] | +0.5 [−0.4, +1.5] | −0.0 [−0.8, +0.8] |
| gpt3l (3072d) | −0.5 [−1.6, +0.5] | +1.4 [+0.4, +2.4] | **+3.9** [+3.4, +4.5] |
| tape_stripped (LLM explanations) | −5.8 | +2.4 | **+10.4** (suspected leakage) |
| tape_full (unstripped) | −4.4 | +2.2 | +10.9 (suspected leakage) |

A0 = cora 0.789 / citeseer_tag 0.694 / pubmed 0.792.

- The view **hurts where structure is strong** (cora), mildly helps citeseer_tag, and only the strongest encoder helps pubmed.
- Most encoder-ordering predictions were **refuted**; H3 (declarative leak, A4 − A3 on cora) **confirmed**: +1.32 vs prior +1.36.
- **pubmed tape (+10.4):** treat as semantic label leakage — the LLM answers the 3-way task inside the explanation. Do not headline.
- **V2 gate:** the HSIC gate never opened (`fused_by_concat = 1.0` everywhere), so every result above is concat fusion → FEW-31.

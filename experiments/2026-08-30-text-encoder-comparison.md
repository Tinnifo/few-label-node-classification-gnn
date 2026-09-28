# Exp 3 — which text encoder helps CG3 at 20 labels per class?

| | |
| --- | --- |
| **Date** | 2026-08-30 |
| **Tests assumption** | 5 (partly; compares encoders only) |
| **Preregistration** | [`docs/hypotheses.md`](../docs/hypotheses.md) (sections 1–3 frozen before the run; results appended in section 4) |
| **Full report** | [`docs/report_2026-08-30.md`](../docs/report_2026-08-30.md) |
| **Code** | [`3358678b`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/3358678b), configs [`130e724a`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/130e724a); report code [`71b3c583`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/71b3c583) |
| **Compute** | cloud GPU (A10G), seeds 0–9 |
| **Status** | done |

## Setup
Arms: **A0** = CG3 with no semantic view; **A1** = + sbert; **A2** = + e5; **A2′** = + gpt3l; **A3** = + tape (stripped);
**A4** = + tape (full). 20 labels per class, paired per-seed Δ with 95 % CI. An effect is *real* if the CI excludes 0 and
*interesting* if it is at least 0.5 points.
Sanity check passed: our CG3 (A0) reproduces an independent faithful CG3 implementation on Cora (0.7890 ± 0.0122 vs 0.7853).

## Result: accuracy change vs CG3 alone (points, paired 95 % CI)

| added view (concatenated) | Cora | CiteSeer | PubMed |
| --- | --- | --- | --- |
| sbert (384-d) | **−3.6** [−4.2, −3.0] | +1.2 [+0.1, +2.4] | −1.6 [−2.2, −0.9] |
| e5-large (1024-d) | **−7.1** [−7.8, −6.3] | +0.5 [−0.4, +1.5] | −0.0 [−0.8, +0.8] |
| gpt3l (3072-d) | −0.5 [−1.6, +0.5] | +1.4 [+0.4, +2.4] | **+3.9** [+3.4, +4.5] |
| tape, stripped (LLM explanations) | −5.8 | +2.4 | **+10.4** (suspected leakage) |
| tape, full | −4.4 | +2.2 | +10.9 (suspected leakage) |

CG3 alone: Cora 0.789 · CiteSeer 0.694 · PubMed 0.792.

- The semantic view **hurts where structure is strong** (Cora), helps CiteSeer a little, and only the largest encoder
  helps PubMed.
- Most predictions about encoder size were **refuted**. The label-leak prediction (full − stripped explanations on Cora)
  was **confirmed**: +1.32 vs the predicted +1.36.
- **PubMed with LLM explanations (+10.4):** treat this as label leakage. The LLM effectively answers PubMed's 3-class
  task inside its explanation. Do not headline this number.
- The model's built-in switch between fusion methods never switched: every result above is plain concatenation.
  Exp 4 tests the fusion methods directly.

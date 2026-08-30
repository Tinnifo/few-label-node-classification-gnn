# Preregistered hypotheses — FEW-31: fusion mechanism, heterophily axis, and FEW-30 anomaly probes

Same rules as `hypotheses.md`: FROZEN at first submit; results append to §4;
predictions never edited. Yardstick unchanged (paired seeds 0–9, paired 95%
CIs, minimum effect of interest 0.5 pts) except where a dataset's size forces
a smaller budget (table below).

**Priors:** homophilous-graph priors are now TRAINED per-dataset numbers
(FEW-30, 2026-08-30). Heterophilic priors are theory only (LLMNodeBed / HeTGB
lineage: LLM text helps more as graph signal weakens).

## 1. Design

**Views:** `gpt3l` on the homophilous trio (FEW-30's least-harmful/best arm);
`sbert_pre` on the heterophilic six (the only view shipped for all six).
**Fusion factor:** `concat` vs `attention` (forced paths — FEW-30 showed the
HSIC gate is degenerate: realized HSIC ≈ 0.01 ≪ τ = 0.1, entropy attention
never executed). `auto` (the gate) is retained only as the deprecated control.
**Loss placement (AM-GCN critique):** the cross-view HSIC penalty
decorrelates two label-bearing outputs — wrong placement in principle, but at
w = 0.1 × HSIC ≈ 0.009 it contributes ~0.001 to the loss. Arm F2 (w = 0)
verifies it is a no-op. **`fusion=shared_private`** implements the fix
(added pre-freeze, at Tinni's direction): each view splits into a shared
channel (consistency loss pulls the pair together — the label lives here)
and a private channel (linear-CKA disparity from its OWN shared channel —
scale-invariant, so no α²-shrink gaming); classifier on
[mean shared, p_struct, p_sem] (d = 64 each); the cross-view HSIC penalty is
REPLACED by consistency + disparity, both at fixed weight 1.0 (no tuning,
shared across all arms and datasets). Runs as a third fusion arm everywhere.

**Heterophilic split rule (no shipped splits):** class-stratified, RNG(0):
per class 25% train pool / 25% val / 50% test (≥1 node per class in pool and
val). Budgets = min(20, smallest class's pool):

| dataset | N | budget/class | mean node homophily (nanmean) |
| --- | --- | --- | --- |
| texas | 187 | 1 | 0.067 |
| cornell | 191 | 4 | 0.116 |
| wisconsin | 265 | 2 | 0.163 |
| washington | 229 | 2 | 0.171 |
| amazon_ratings | 24,492 | 20 | 0.376 |
| actor | 4,416 | 7 | 0.607 |

## 2. Validity checks (fail → debug, not conclude)

| # | check | pass condition |
| --- | --- | --- |
| V1 | hetero A0 sanity | A0 beats the majority-class rate on every hetero graph |
| V2 | budget honesty | achieved = requested budget everywhere (guard refuses otherwise) |
| V3 | fusion factor real | `fused_by_concat` logs 1.0 on concat arms, 0.0 on attention arms |

## 3. Hypotheses

| # | claim | prediction | falsified if |
| --- | --- | --- | --- |
| P1 | e5's −7.1 on cora is a fusion problem, not an embedding problem | cora e5 probe within 2 pts of sbert probe | e5 probe ≫ worse → encoder/prefix problem (check `query:` prefix) |
| P2 | pubmed tape views leak the label semantically | graph-free probe on tape_stripped ≥ 0.85 (≈ A3's trained 0.895), ≫ gpt3l-of-raw-text probe | probe ≈ raw-text probe → +11.9 needs another explanation |
| F1 | attention rescues cora | cora Δ(gpt3l·attention − A0) > Δ(gpt3l·concat − A0) = −0.5; attention ≥ −0.5 pts of A0 | attention worse than concat on cora |
| F1b | concat's pubmed win is feature-level | pubmed gpt3l·attention < gpt3l·concat (+3.9) — parameter-free logit mixing can't exploit complementary features | attention ≥ concat there |
| F2 | HSIC penalty is a no-op | cora gpt3l, w=0 vs w=0.1: |Δ| < 0.5 pts (CI includes 0) | Δ ≥ 0.5 → FEW-30 results are confounded by the penalty; escalate the AM-GCN redesign |
| G1 | semantic view helps low-homophily graphs | Δ(sbert_pre·concat − A0) > 0 on ≥ 4/6 hetero graphs | ≤ 2/6 positive |
| G2 | gain grows as homophily falls | Spearman ρ(Δ_concat, homophily) ≤ −0.5 across all 9 graphs (6 hetero + FEW-30's cora/citeseer_tag/pubmed sbert Δs) | ρ > 0, or |ρ| < 0.3 |
| G3 | (Tinni's expectation) attention beats concat where homophily is low | Δ(attention) − Δ(concat) ≥ +0.5 pts on ≥ 3 of the 4 lowest-homophily graphs | positive on ≤ 1 |
| S1 | shared_private rescues cora | cora Δ(gpt3l·sp − A0) > Δ(concat) = −0.5 and within noise of 0 (CI touches 0) | sp worse than concat |
| S2 | shared_private keeps concat's feature-level win | pubmed Δ(gpt3l·sp) ≥ +2.0 (vs concat's +3.9) | sp < +2.0 → the split costs more than the placement fix buys |
| S3 | shared_private ≥ concat on low-homophily graphs | Δ(sp) ≥ Δ(concat) on ≥ 4/6 hetero graphs | ≤ 2/6 |

**Caveats preregistered:** tiny-graph budgets (1–4/class) make single-graph
CIs wide — G1/G3 are counted across graphs, not per-graph claims; actor's
homophily (0.61) overlaps the homophilous trio, which strengthens G2's axis;
no tape views exist for hetero graphs, so leakage questions stay on pubmed.

## 4. Results (append-only)

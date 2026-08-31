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

**Runs of 2026-08-30** (Modal, code `71b3c583`; 4 jobs: probe / fusion trio /
het small / amazon_ratings; logs, MLflow dbs, probe CSV saved as artifacts).

**Probes (graph-free logistic, 20/class, 10 seeds):**
- **P1 borderline-confirmed.** cora e5 probe 0.7274 vs sbert 0.7474 — gap 2.0
  pts, exactly at the boundary. Embeddings mildly worse; cannot explain the
  in-model −7.1 → the fused head amplifies the difference.
- **P2 confirmed.** pubmed tape_stripped probe 0.8768 (≥ 0.85), +4.4 over the
  raw-text gpt3l probe (0.8327), ≈ the trained A3 (0.8955): the explanation
  view carries the label without the graph. Semantic leakage.
- Unregistered observation: probes nearly match trained CG3 arms everywhere
  (cora gpt3l probe 0.777 vs A0 0.789; pubmed gpt3l probe 0.833 vs trained
  0.831) — at these budgets frozen text + logistic regression is close to
  competitive with the full pipeline.

**Fusion trio (gpt3l view; Δ vs A0):** concat/attention/shared_private =
cora −0.5 / −2.0 / −7.8±3.0; citeseer_tag +1.4 / +0.1 / −3.3;
pubmed +3.9 / +2.7 / +0.3.
- **F1 refuted** (attention does not rescue cora). **F1b confirmed**
  (attention < concat on pubmed). **F2 confirmed** (w=0 vs w=0.1 on cora
  gpt3l: +0.08 ± 0.24 pts — the cross-view HSIC penalty is a no-op; FEW-30
  unconfounded).
- **S1, S2 refuted.** shared_private v1 is the worst fusion everywhere, with
  2–3× seed variance (loss_consist/loss_disp logged for the post-mortem;
  suspects: 7-dim→64-dim structural projection, consistency weight 1.0).

**Heterophilic graphs — V1 FAILS on all six.** A0 (CG3) is below the
majority-class rate on every hetero graph (e.g. texas 0.454 vs 0.541, actor
0.380 vs 0.488). Per §2: debug, not conclude — CG3's homophily assumptions
make it the wrong backbone there, so G-verdicts below are PROVISIONAL
(gains rescue a failing baseline; semantic arms cross the majority bar on
3/6: cornell, wisconsin, washington).
- G1 (provisional): concat Δ > 0 on 6/6 — cornell +8.9±3.5, wisconsin
  +9.8±6.1, actor +8.0±8.3, washington +4.4±6.9, amazon +1.1±2.5, texas
  +0.6±11.2.
- G2 (directional, underpowered): Spearman ρ = **−0.550** across 9 graphs —
  the point estimate lands exactly on the preregistered threshold — but
  p = 0.125 at n = 9. Figure: fig_homophily_gain.
- **G3 refuted** (Tinni's expectation): attention − concat ≥ +0.5 on 1/4
  lowest-homophily graphs (cornell +1.5; texas −1.1, washington −2.9,
  wisconsin −0.3). **S3 refuted** (sp ≥ concat on 2/6, and its texas "win"
  has the lowest macro-F1 of any arm — majority-class drift, not learning).

**§5 review notes:** the surprise inventory is (1) V1's hetero failure —
the baseline, not the view, is the bottleneck on heterophilic graphs;
(2) attention never beating concat anywhere; (3) shared_private v1
underperforming with high variance. Next experiments licensed: a
heterophily-appropriate backbone (H2GCN-class) under the same arms; a
shared_private v2 post-mortem (projection dims, consistency weight); a
leakage-robust explanation prompt. Confirmation rule stands: any hit here
re-runs on fresh seeds before it is believed.

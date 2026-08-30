# Preregistered hypotheses — semantic-view encoder ablation

**Rules of the game.** This file is FROZEN the moment the first job is
submitted: results get appended to §4, predictions in §2–§3 are never edited.
A prediction that turns out wrong is a result, not a mistake — the only way to
ruin this document is to change it after seeing data.

Priors marked `(probe)` come from the July E3 measurement (2026-07-28, job
7402608c): a **training-free linear probe** over the four staged views,
pooled across cora/citeseer/pubmed — not a trained CG3 model, and not
per-dataset. Trust their *direction and ordering*; the magnitudes may not
transfer to a trained model at all. The 10-seed paired CIs decide.

## 1. The yardstick (decided before any run)

- Every arm runs the **same seeds** `0–9`. Same seed = same label draw, so arm
  differences are **paired per seed**: compute Δ_s = acc_arm(s) − acc_ref(s)
  for each seed s, then mean(Δ) with its 95% CI = mean ± 1.96·sd(Δ)/√10.
- An effect is **real** if the paired CI excludes zero.
- An effect is **interesting** if |mean(Δ)| ≥ **0.5 pts** accuracy
  (minimum effect of interest — edit before freezing if you disagree).
- Primary metric: test accuracy. Secondary: macro-F1 (report both; if they
  disagree, say so rather than picking the better one).
- Protocol constants (identical on every arm, from `conf/`): early stopping
  on, patience 200, budget 20/class, `hsic.sigma`/`hsic.max_samples` fixed.

## 2. Validity checks — failure means DEBUG, not conclude

| # | check | pass condition | on failure |
| --- | --- | --- | --- |
| V1 | A0 is CG3 | A0 reproduces the faithful-port CG3 numbers **seed-for-seed** | stop; nothing downstream is interpretable |
| V2 | gate stability | `fused_by_concat` constant within a run after warm-up; same path across seeds of one arm | freeze/adjust `hsic.sigma`, report flip rate; do not interpret accuracy |
| V3 | stripper sanity | eyeball 20 Granite descriptors vs stripped versions: no label declarations survive, no explanation body destroyed | fix regexes before trusting A3/A4 |
| V4 | budget honesty | every run logs exactly 20 labels/class (the guard refuses more) | the guard has a bug |

## 3. Science hypotheses

Reference arm for Δ is named per row. Arms: A0 = `semantic=none`,
A1 = `sbert`, A2 = `e5`, A2′ = `gpt3l`, A3 = `tape`, A4 = `tape_leak`.

| # | claim | prediction (prior) | falsified if | action per outcome |
| --- | --- | --- | --- | --- |
| H1 | Encoder capacity is the dominant axis | Δ(A2′−A0) > Δ(A1−A0) > 0; priors: gpt3l +0.067, sbert +0.017, pooled over the three graphs `(probe)` | ordering flips, or both CIs include 0 | holds → capacity is the headline; both null → semantic view doesn't help CG3 at few labels, fusion diagnostics become the story |
| H1b | The open mid-size encoder sits between | Δ(A1−A0) ≤ Δ(A2−A0) ≤ Δ(A2′−A0) | A2 outside the bracket | either way, feeds the width-confound discussion (384 vs 1024 vs 3072 d) |
| H2 | Stripped explanations add nothing over own text | Δ(A3−A1) ≈ 0; prior: explanation +0.0048 vs sbert +0.0170, diff −0.012 at p=0.75 — indistinguishable `(probe)` | CI excludes 0 AND ≥ 0.5 pts | holds → explanation view stays secondary; violated → Granite-TAPE prompt differs from the LLMNodeBed dumps — inspect descriptors before believing it |
| H3 | Label leakage inflates the unstripped arm | Δ(A4−A3) ≈ +0.010 to +0.015; prior +0.0136 `(probe)` | A4 ≈ A3 (CI includes 0) | holds → stripper validated, A4 is the cautionary row; null → check whether Granite declares labels at all (V3), or the stripper over-strips |
| H4 | Datasets do not agree in magnitude | pubmed Δs smaller than cora Δs (denser text already in BoW features; neighbor-structure evidence from Chen Obs. 15–16 suggests dataset-dependence) | pubmed ≥ cora | either way: report per-dataset, never pooled |

**citeseer_tag:** within-arm comparisons only — its absolute numbers are not
comparable to any published CiteSeer result (3,186-node TAG-native graph).
Predictions H1–H3 apply directionally; no magnitude priors exist.

## 4. Results (append after runs — never edit above this line)

For each dataset × arm: MLflow run name, commit SHA, mean acc ± std,
and for each H: predicted vs observed, paired CI, verdict —
**confirmed / refuted / underpowered** (CI includes both 0 and the MEI —
i.e. the data cannot distinguish "no effect" from "interesting effect").

**Runs of 2026-08-30** (Modal A10G+8cpu; code `3358678b`, configs `130e724a`;
seeds 0–9; MLflow dbs + logs saved as artifacts; V1 passed earlier the same
day: new A0 0.7890±0.0122 ≡ faithful port 0.7853±0.0127 ≡ A0@1000ep
0.7871±0.0127). A0 means: cora 0.7887, citeseer_tag 0.6936, pubmed 0.7916.
Amendment (logged on FEW-30 pre-run): A3/A4 = precomputed LLMNodeBed views
(`tape_pre`/`tape_pre_leak`), Granite arms deferred.

| dataset | comparison | predicted | observed mean Δ [95% CI] | verdict |
| --- | --- | --- | --- | --- |
| cora | A1−A0 | > 0 (~+0.017) | −0.0357 [−0.0416, −0.0298] | **refuted** (sign flipped) |
| cora | A2′−A0 | > A1−A0 (~+0.067) | −0.0054 [−0.0161, +0.0053] | refuted (null; least harmful arm) |
| cora | A2 bracket A1≤A2≤A2′ | in bracket | −0.0708 [−0.0784, −0.0632] | **refuted** (e5 worst, below bracket) |
| cora | A3−A1 | ≈ 0 | −0.0218 [−0.0262, −0.0174] | refuted (explanations < own text) |
| cora | A4−A3 | +0.010–0.015 | +0.0132 [+0.0068, +0.0196] | **confirmed** (prior +0.0136) |
| citeseer_tag | A1−A0 | > 0 (directional) | +0.0124 [+0.0005, +0.0243] | confirmed (barely clears 0) |
| citeseer_tag | A2′−A0 | > A1−A0 | +0.0139 [+0.0035, +0.0242] | confirmed (weakly; A2′>A1>0) |
| citeseer_tag | A2 bracket | in bracket | +0.0054 [−0.0041, +0.0149] | refuted (e5 below A1) |
| citeseer_tag | A3−A1 | ≈ 0 | +0.0120 [+0.0045, +0.0196] | refuted (positive) |
| citeseer_tag | A4−A3 | +0.010–0.015 | −0.0019 [−0.0042, +0.0004] | refuted (null) |
| pubmed | A1−A0 | > 0 | −0.0155 [−0.0217, −0.0093] | refuted (negative) |
| pubmed | A2′−A0 | > A1−A0 | +0.0392 [+0.0337, +0.0447] | partially (ordering yes; A1>0 no) |
| pubmed | A2 bracket | in bracket | −0.0002 [−0.0081, +0.0077] | holds on this dataset |
| pubmed | A3−A1 | ≈ 0 | **+0.1194** [+0.1139, +0.1249] | **refuted, spectacularly** — see note |
| pubmed | A4−A3 | +0.010–0.015 | +0.0052 [+0.0013, +0.0091] | real but below prior range |
| H4 | pubmed Δs < cora Δs | smaller | pubmed has the LARGEST Δs | **refuted as stated**; "datasets differ" overwhelmingly true |

**V2 (gate):** `fused_by_concat = 1.0` on every semantic arm, all datasets
(realized HSIC ≈ 0.008–0.010 ≪ τ = 0.1). Stable but **degenerate** — the
entropy-attention path never executed; every semantic result above is concat
fusion. Fusion mechanism becomes an explicit factor in the follow-up (FEW-31).

**Note on pubmed A3 (+11.9 over own-text, +10.4 over A0): treat as suspected
label leakage, not semantic gain.** PubMed's 3-way diabetes-type task is
near-solvable by an LLM zero-shot, so a gpt-4o-mini explanation carries the
label *semantically* even after declaration stripping ("this study of type 2
diabetes …"). Consistent with A4−A3 being tiny (+0.5): the declaration is
marginal because the meaning already leaks. The stripped/unstripped contrast
controls declarative leakage only — semantic leakage needs a different
control (e.g. explanation generated without the category list in the prompt,
or a post-cutoff dataset). Do not headline this number.

## 5. Review protocol (the part where the learning happens)

After results land, answer in writing, in this order:

1. Did every validity check pass? If not, stop — fix, rerun, only then read on.
2. For each H: verdict. For every *refuted* row: was the prior wrong, or the
   setup different from the prior's setup? (E3 priors came from a
   training-free probe pooled over three graphs, not a trained CG3 —
   "refuted" may mean "the prior didn't transfer".)
3. What surprised you? (If nothing surprised you, the predictions were too
   safe — next preregistration, tighten the ranges.)
4. What is the single next experiment this result licenses — and what would
   its H-table look like?

The skill being practiced: **write the number down before you see it.** The
predictions above are deliberately concrete enough to be wrong.

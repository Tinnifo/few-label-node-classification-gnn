# Progress — few-label node classification with semantic views

_Last updated: 2026-09-27 · Maintainer: Tinni · Next supervision meeting: see Linear (FEW)._

## The question

> We're studying **semantic views of graph node features** because we want to find out **whether a semantic view
> improves node-classification accuracy when labels are scarce and structure-based GNNs or contrastive methods fall
> short**, so that **people doing node classification with few labels know when adding a semantic view helps and when
> it will not matter**. _(target as revised 2026-09-06)_

**Where we stand, in one line:** the semantic view carries real label information exactly where structure is weak
(E5), but every way we have fused it into CG3 so far either loses to plain concatenation or fuses a head that has
already memorised its 140 labels (E6). Nothing yet shows the fused model beating the better single view.

## The argument, link by link

The target is true only if every link below holds. Status: `open` · `alive` · `dead` · `split`.
When a link dies it is re-stated from the target, not patched.

| Link | Must be true | Status | Evidence | Next test |
| --- | --- | --- | --- | --- |
| **a** | The semantic view carries label-relevant information the structural views lack, and the amount is measurable | **split** — a₁ supported, a₂ open | [E5](experiments/2026-09-06-e5-probe-cka/README.md): WebKB +9…+43 pts, CKA 0.05–0.18 vs 0.52–0.63 on the Planetoid trio; [E1a](experiments/2026-07-27-e1a-coupling-screen.md): redundancy tracks homophily, ρ = +0.933 | e5b: same test against CG3's *trained* views; a′: encoder-contamination audit |
| **b** | The semantic view joins CG3 in one training *and* one shared embedding space | **split** — one training alive; one space dead in current code | code audit 2026-09-06 (`cb48a3a`); [FEW-31](experiments/2026-08-30-few31-fusion-heterophily.md): shared_private (only one-space arm) ran worst | why shared_private failed — hypothesis: `loss_consist` only pulls, so it can collapse (2026-09-24) |
| **c** | The semantic MLP head does not memorise the 140 labels | **dead as stated → re-stated** | [E6](experiments/2026-09-06-e6-head-memorization/README.md): train acc 1.0 by step 5–15; stopping at best-val +2…+13 pts on 15/16 heads | FEW-35: head stopped on validation, validation labels counted in the budget |
| **d** | The fusion does not collapse onto whichever view is most confident | open | FEW-31: entropy-fusion degeneracy | blocked on c; then branch entropy-vs-error AUROC (> 0.65) |
| **e** | With a–d holding, the fused model beats the **better** of CG3-alone and semantic-alone at 20 labels/class | open | [FEW-30](experiments/2026-08-30-few30-encoder-ablation.md) (partial): helps pubmed (+3.9 gpt3l), hurts cora (−3.6 sbert) | preregistered arms A0–A4 on datasets spanning a's number |

## Hypothesis ledger

Every prediction was written before its run. Verdicts are from paired per-seed 95 % CIs.

| # | Hypothesis | Predicted | Observed | Verdict | Experiment |
| --- | --- | --- | --- | --- | --- |
| H1 | Align-vs-preserve the semantic view flips with homophily | slope > 0 | slope +0.51 pts/unit h_adj, p = 0.80, R² = 0.005 | **refuted** | [E1a](experiments/2026-07-27-e1a-coupling-screen.md) |
| H2 | A dependence-penalty coupling improves on no coupling (3 labels/class) | gain > 0 | align −1.32, CKA −1.63, HSIC −4.08 pts | **refuted** | [E1a](experiments/2026-07-27-e1a-coupling-screen.md) |
| H3 | LLM explanations are a more useful view than own text | gain(tape) > gain(sbert) | +0.005 vs +0.017, p = 0.75 (less redundant, not more useful) | **refuted** | [E3](experiments/2026-07-28-e3-view-probe.md) |
| H4 | Adding the sbert view helps CG3 at 20/class | Δ > 0 | cora −3.6, citeseer +1.2, pubmed −1.6 | **depends on the graph** | [FEW-30](experiments/2026-08-30-few30-encoder-ablation.md) |
| H5 | A larger encoder helps more (sbert ≤ e5 ≤ gpt3l) | in bracket | e5 worst on cora (−7.1) | **refuted** | [FEW-30](experiments/2026-08-30-few30-encoder-ablation.md) |
| H6 | The LLM's answer line leaks the label (tape_full − tape_stripped) | +1.0…+1.5 pts | cora +1.32 | **confirmed** | [FEW-30](experiments/2026-08-30-few30-encoder-ablation.md) |
| H7 | Entropy attention beats concat fusion | attention > concat | concat best on all three homophilous graphs | **refuted** | [FEW-31](experiments/2026-08-30-few31-fusion-heterophily.md) |
| H8 | Shared/private (AM-GCN placement) beats concat | sp ≥ concat | worst arm everywhere, 2–3× seed variance | **refuted** | [FEW-31](experiments/2026-08-30-few31-fusion-heterophily.md) |
| H9 | Semantic gain grows as homophily falls (trained CG3) | ρ ≤ −0.55 | ρ = −0.550, p = 0.125, n = 9; CG3 below majority on all 6 heterophilic graphs | **provisional** (baseline fails V1) | [FEW-31](experiments/2026-08-30-few31-fusion-heterophily.md) |
| H10 | Semantic probe beats structural probe where h_adj < 0, and CKA tracks homophily | Δ > 0; ρ > 0 | +9…+43 pts on WebKB; ρ = +0.80 | **supported** | [E5](experiments/2026-09-06-e5-probe-cka/README.md) |
| H11 | The semantic head memorises within tens of steps | ≈ 40–50 steps | 5–15 steps; test entropy → 0.03–0.23 | **supported** (faster than predicted) | [E6](experiments/2026-09-06-e6-head-memorization/README.md) |

## Latest results (newest first)

- **2026-09-06 — E6.** The semantic head memorises its labels in about ten steps and then grows confident about wrong
  answers. Early stopping on validation recovers +2…+13 pts. → link c re-stated.
- **2026-09-06 — E5.** Without any training, the semantic view beats structure by +9…+43 pts on the four heterophilic
  WebKB graphs and is nearly independent of it there (CKA < 0.2). Pubmed is the exception that matters: redundant views,
  yet +10.4 pts. → link a splits.
- **2026-08-30 — FEW-30 / FEW-31.** Semantic view helps pubmed, hurts cora; concat beats every smarter fusion; CG3
  itself fails on heterophilic graphs. Full report: [`docs/report_2026-08-30.md`](docs/report_2026-08-30.md).
- **2026-07-28 — E3.** LLM explanations are less redundant than the node's own text, but not more useful.
- **2026-07-27 — E1a.** The draft's central crossover claim is falsified; no coupling term beats no coupling.

## Open directions

- **LLM-feedback loop (from the 2026-09-08 meeting).** Feed confident false positives back to a frozen LLM, let it
  rewrite the node descriptions, re-encode and re-train. Literature check: the exact loop is not published; nearest
  neighbours are VGRL (Ji et al. 2024) and LOGIN (Qiao et al. 2024) — see [`papers/`](papers/README.md).
- **Uncertainty / confident-but-wrong penalty** (FEW-36) — Amir's line.
- **Queued, not run:** e5b (trained views), a′ contamination audit, FEW-35 (head early-stopping protocol), a
  heterophily-appropriate backbone, shared_private v2 post-mortem.

## Where things live

| What | Where |
| --- | --- |
| One write-up per experiment | [`experiments/`](experiments/README.md) (template: [`experiments/_template.md`](experiments/_template.md)) |
| Preregistrations (frozen before runs) | [`docs/hypotheses.md`](docs/hypotheses.md) · [`docs/hypotheses_few31.md`](docs/hypotheses_few31.md) |
| Papers we have discussed | [`papers/README.md`](papers/README.md) — links only; PDFs are not committed (public repo) |
| Code for the experiments | branch [`scaffold/data-and-experiment`](https://github.com/Tinnifo/few-label-node-classification-gnn/tree/scaffold/data-and-experiment) |
| Data | release [`data-v1`](https://github.com/Tinnifo/few-label-node-classification-gnn/releases/tag/data-v1) |
| Tasks and preregistration issues | Linear team FEW (project flnc-Arguments) |

### How to update this page
After each experiment: add its write-up under `experiments/`, add or update its row in the **hypothesis ledger**,
update the **link status**, and add one line to **Latest results**. Predictions are never edited after a run.

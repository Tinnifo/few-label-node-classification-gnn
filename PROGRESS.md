# Progress — few-label node classification with semantic views

_Last updated: 2026-09-27._

## The question

We want to know **whether adding a semantic view of each node — an embedding of the node's text from a language
model — improves node-classification accuracy when only a few nodes are labelled**, and **when it helps versus when
it makes no difference**. The base model throughout is **CG3** (Wan et al., 2021), a graph contrastive model that
uses two structural views of the graph.

**Where we stand, in one line:** the semantic view carries real label information exactly where the graph structure
is weak ([Exp 5](experiments/2026-09-06-semantic-information-vs-homophily/README.md)), but every way we have combined it with CG3 so far either loses to simple concatenation or uses a
semantic classifier that has already memorised its few training labels ([Exp 6](experiments/2026-09-06-semantic-head-memorisation/README.md)). Nothing yet shows the combined model
beating the better of its two parts.

## Terms used on this page

| Term | Meaning |
| --- | --- |
| **Semantic view** | A fixed text embedding of each node's text (e.g. a paper's abstract), fed to the model alongside the graph |
| **Structural view** | What the model learns from the graph itself: node features and edges (CG3's two views) |
| **Few labels / 20 per class** | Only 20 labelled training nodes per class (140 on Cora); 3 per class in [Exp 1](experiments/2026-07-27-coupling-penalty-screen.md) |
| **Homophily (h_adj)** | How often linked nodes share a label, adjusted for chance: ≈ +0.8 on Cora (neighbours agree), below 0 on the WebKB graphs (neighbours tend to disagree) |
| **Heterophilic graph** | A graph with low or negative homophily (texas, cornell, washington, wisconsin, actor, amazon_ratings) |
| **Probe** | A plain logistic-regression classifier on a fixed embedding, with no graph and no training of the embedding — measures how much label information the embedding holds |
| **CKA** | A 0-to-1 similarity score between two sets of embeddings: near 1 = they carry the same information, near 0 = independent |
| **Paired 95 % CI** | Every arm uses the same 10 random seeds (same labelled nodes), so we compare arms seed by seed; an effect is *real* if the interval excludes 0 |
| **Preregistered** | The hypothesis and predicted outcome were written down and committed *before* the run, and never edited afterwards |

Text embeddings used: **sbert** = all-MiniLM-L6-v2 sentence encoder (384-d) · **e5** = e5-large (1024-d) ·
**gpt3l** = OpenAI text-embedding-3-large (3072-d) · **tape** = explanations of each node written by an LLM
(gpt-4o-mini, from the TAPE paper) and then embedded; *stripped* removes the LLM's predicted-label line, *full* keeps it.

## What must be true for the claim to hold

Our claim holds only if all five assumptions below hold. When one fails, we restate it rather than quietly patching it.

| # | Assumption | Status | Evidence | Next step |
| --- | --- | --- | --- | --- |
| **1** | The semantic view holds label information the structural view lacks, and the amount can be measured | **Partly supported.** 1a (it holds extra information where homophily is low) is supported; 1b (the size of the gain also depends on how weak the structure is) is open | [Exp 5](experiments/2026-09-06-semantic-information-vs-homophily/README.md): +9 to +43 points on the four WebKB graphs, CKA 0.05–0.18 there vs 0.52–0.63 on Cora/CiteSeer/PubMed; [Exp 1](experiments/2026-07-27-coupling-penalty-screen.md): overlap with structure rises with homophily (ρ = +0.933) | Repeat against CG3's *trained* representations, not just its inputs; check whether the text encoders saw these datasets during pre-training |
| **2** | The semantic view is trained together with CG3 *and* lives in the same embedding space | **Half holds.** Trained together: yes. Same space: no, not in the current code | Code review (commit `cb48a3a`); [Exp 4](experiments/2026-08-30-fusion-methods-and-heterophily.md): the one design that shares a space ran worst | Find out why the shared-space design failed — current hypothesis: its consistency loss only pulls views together, so it can collapse |
| **3** | The semantic classifier does not simply memorise the few training labels | **Does not hold as stated → restated** | [Exp 6](experiments/2026-09-06-semantic-head-memorisation/README.md): it reaches 100 % training accuracy within 5–15 steps; stopping early on validation adds +2 to +13 points on 15 of 16 settings | Stop the semantic classifier early using validation labels, and count those labels in the label budget |
| **4** | When combining views, the model does not simply trust whichever view is most confident | **Untested** | Earlier signs that confidence-weighted fusion degenerates | Waits on #3, because the current classifier's confidence is an artefact of memorisation |
| **5** | With 1–4 in place, the combined model beats the **better** of CG3 alone and the semantic view alone, at 20 labels per class | **Untested — this is the claim** | [Exp 3](experiments/2026-08-30-text-encoder-comparison.md) (partial): helps PubMed (+3.9 with gpt3l), hurts Cora (−3.6 with sbert) | The preregistered comparison on datasets spanning the homophily range |

## Hypothesis ledger

Every prediction was written before its run. Verdicts come from paired per-seed 95 % CIs.

| # | Hypothesis | Predicted | Observed | Verdict | Experiment |
| --- | --- | --- | --- | --- | --- |
| H1 | Whether to align the semantic view with structure or keep it separate flips with homophily | slope > 0 | slope +0.51 pts per unit homophily, p = 0.80, R² = 0.005 | **refuted** | [Exp 1](experiments/2026-07-27-coupling-penalty-screen.md) |
| H2 | A penalty coupling the semantic and structural views beats no coupling (3 labels/class) | gain > 0 | align −1.32, CKA penalty −1.63, HSIC penalty −4.08 pts | **refuted** | [Exp 1](experiments/2026-07-27-coupling-penalty-screen.md) |
| H3 | LLM-written explanations are a more useful view than the node's own text | gain(tape) > gain(sbert) | +0.005 vs +0.017, p = 0.75 (less overlap with structure, but not more useful) | **refuted** | [Exp 2](experiments/2026-07-28-llm-explanations-vs-own-text.md) |
| H4 | Adding the sbert view helps CG3 at 20 labels/class | Δ > 0 | Cora −3.6, CiteSeer +1.2, PubMed −1.6 | **depends on the graph** | [Exp 3](experiments/2026-08-30-text-encoder-comparison.md) |
| H5 | A larger text encoder helps more (sbert ≤ e5 ≤ gpt3l) | in that order | e5 is worst on Cora (−7.1) | **refuted** | [Exp 3](experiments/2026-08-30-text-encoder-comparison.md) |
| H6 | Keeping the LLM's predicted-label line leaks the label (tape full − stripped) | +1.0 to +1.5 pts | Cora +1.32 | **confirmed** | [Exp 3](experiments/2026-08-30-text-encoder-comparison.md) |
| H7 | Confidence-weighted attention fusion beats concatenation | attention > concat | concatenation best on all three citation graphs | **refuted** | [Exp 4](experiments/2026-08-30-fusion-methods-and-heterophily.md) |
| H8 | A shared/private design (as in AM-GCN) beats concatenation | shared/private ≥ concat | worst design everywhere, 2–3× the seed variance | **refuted** | [Exp 4](experiments/2026-08-30-fusion-methods-and-heterophily.md) |
| H9 | The semantic view's gain grows as homophily falls (trained CG3) | ρ ≤ −0.55 | ρ = −0.550, p = 0.125, n = 9; but CG3 itself scores below always-guess-the-majority-class on all 6 heterophilic graphs | **provisional** (baseline fails its sanity check) | [Exp 4](experiments/2026-08-30-fusion-methods-and-heterophily.md) |
| H10 | Where homophily is negative, a probe on the semantic view beats one on structure, and the views' overlap tracks homophily | Δ > 0; ρ > 0 | +9 to +43 pts on WebKB; ρ = +0.80 | **supported** | [Exp 5](experiments/2026-09-06-semantic-information-vs-homophily/README.md) |
| H11 | The semantic classifier memorises its training labels within tens of steps | ≈ 40–50 steps | 5–15 steps; its confidence keeps rising while test accuracy falls | **supported** (faster than predicted) | [Exp 6](experiments/2026-09-06-semantic-head-memorisation/README.md) |

## Latest results (newest first)

- **2026-09-06 — Exp 6, semantic classifier memorisation.** The semantic classifier memorises its labels in about ten
  steps and then becomes confident about wrong answers. Early stopping on validation recovers +2 to +13 points.
- **2026-09-06 — Exp 5, semantic information vs homophily.** Without any training, the semantic view beats structure by
  +9 to +43 points on the four heterophilic WebKB graphs and is nearly independent of it there (CKA < 0.2). PubMed is
  the exception that matters: the views overlap heavily, yet the semantic view still gains +10.4 points.
- **2026-08-30 — Exp 3 and 4, text encoders and fusion methods.** The semantic view helps PubMed and hurts Cora;
  plain concatenation beats every more elaborate fusion method; CG3 itself fails on heterophilic graphs. Full report:
  [`docs/report_2026-08-30.md`](docs/report_2026-08-30.md).
- **2026-07-28 — Exp 2, LLM explanations vs own text.** LLM explanations overlap less with structure than the node's own
  text does, but they are not more useful.
- **2026-07-27 — Exp 1, coupling-penalty screen.** The original central claim (align vs keep separate flips with
  homophily) is falsified; no coupling penalty beats no coupling.

## Open directions

- **LLM feedback loop** (proposed at the 2026-09-08 group meeting). Send the nodes the model gets confidently wrong back
  to a frozen LLM, let it rewrite those nodes' descriptions, then re-embed and retrain. A literature check found no
  published version of this exact loop. The closest work is VGRL (Ji et al., 2024) and LOGIN (Qiao et al., 2024); see
  [`papers/`](papers/README.md).
- **Penalising confident-but-wrong predictions** — Amir is leading this line.
- **Planned, not yet run:** repeat Exp 5 on CG3's trained representations; check the text encoders for pre-training
  contamination; early-stopping protocol for the semantic classifier (assumption 3); a backbone designed for
  heterophilic graphs; a post-mortem of the shared-space design.

## Where things live

| What | Where |
| --- | --- |
| One write-up per experiment | [`experiments/`](experiments/README.md) (template: [`experiments/_template.md`](experiments/_template.md)) |
| Preregistered hypotheses (frozen before the runs) | [`docs/`](docs/README.md) |
| Papers we have discussed | [`papers/README.md`](papers/README.md) — links only; PDFs are not stored in this public repo |
| Code for the experiments | branch [`scaffold/data-and-experiment`](https://github.com/Tinnifo/few-label-node-classification-gnn/tree/scaffold/data-and-experiment) |
| Data | release [`data-v1`](https://github.com/Tinnifo/few-label-node-classification-gnn/releases/tag/data-v1) |

### How to update this page
After each experiment: add its write-up under `experiments/`, add or update its row in the **hypothesis ledger**,
update the **assumption status**, and add one line to **Latest results**. Never edit a prediction after its run.

# Experiments

One entry per experiment, in date order. The one-page overview, including a glossary of terms, is
[`PROGRESS.md`](../PROGRESS.md).

| # | Date | Question | Verdict | Write-up |
| --- | --- | --- | --- | --- |
| 1 | 2026-07-27 | Does "align the semantic view with structure vs keep it separate" flip with homophily? | refuted | [coupling-penalty screen](2026-07-27-coupling-penalty-screen.md) |
| 2 | 2026-07-28 | Are LLM-written explanations a more useful view than the node's own text? | refuted | [LLM explanations vs own text](2026-07-28-llm-explanations-vs-own-text.md) |
| 3 | 2026-08-30 | Which text encoder helps CG3 at 20 labels per class? | depends on the graph | [text-encoder comparison](2026-08-30-text-encoder-comparison.md) |
| 4 | 2026-08-30 | Which fusion method works best, and does the gain track heterophily? | concatenation wins; CG3 fails on heterophilic graphs | [fusion methods and heterophily](2026-08-30-fusion-methods-and-heterophily.md) |
| 5 | 2026-09-06 | Does the semantic view carry label information that structure lacks, and where? | supported | [semantic information vs homophily](2026-09-06-semantic-information-vs-homophily/README.md) |
| 6 | 2026-09-06 | Does the semantic classifier memorise its few training labels? | yes | [semantic classifier memorisation](2026-09-06-semantic-head-memorisation/README.md) |

## Adding an experiment

1. **Before the run:** copy [`_template.md`](_template.md) to `YYYY-MM-DD-<short-name>.md` (or to a folder with a
   `README.md` when the run ships scripts or data). Fill in *Hypothesis* and *Prediction*, and commit. The commit
   time is the preregistration.
2. **After the run:** fill in *Result*, *Predicted vs actual* and *What it changes*. Never edit the prediction.
3. Add one row to the table above and update the ledger in [`PROGRESS.md`](../PROGRESS.md).

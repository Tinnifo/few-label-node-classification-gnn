# Experiments

One entry per experiment, newest last. The one-page overview is [`PROGRESS.md`](../PROGRESS.md).

| ID | Date | Question | Verdict | Write-up |
| --- | --- | --- | --- | --- |
| E1a | 2026-07-27 | Does aligning vs. preserving the semantic view flip with homophily? | refuted | [2026-07-27-e1a-coupling-screen.md](2026-07-27-e1a-coupling-screen.md) |
| E3 | 2026-07-28 | Are LLM explanations a more useful view than the node's own text? | refuted | [2026-07-28-e3-view-probe.md](2026-07-28-e3-view-probe.md) |
| FEW-30 | 2026-08-30 | Which text encoder / view helps CG3 at 20 labels/class? | depends on the graph | [2026-08-30-few30-encoder-ablation.md](2026-08-30-few30-encoder-ablation.md) |
| FEW-31 | 2026-08-30 | Which fusion mechanism, and does the gain track heterophily? | concat wins; CG3 fails on heterophilic graphs | [2026-08-30-few31-fusion-heterophily.md](2026-08-30-few31-fusion-heterophily.md) |
| E5 | 2026-09-06 | Does the semantic view carry label information structure lacks, and where? | supported (link a splits) | [2026-09-06-e5-probe-cka/](2026-09-06-e5-probe-cka/README.md) |
| E6 | 2026-09-06 | Does the semantic MLP head memorise the 140 labels? | yes (link c dead as stated) | [2026-09-06-e6-head-memorization/](2026-09-06-e6-head-memorization/README.md) |

## Adding an experiment

1. **Before the run:** copy [`_template.md`](_template.md) to `YYYY-MM-DD-<id>-<slug>.md` (or a folder with a `README.md`
   when the run ships scripts or data), fill in *Hypothesis* and *Prediction*, and commit it. The commit timestamp
   is the preregistration.
2. **After the run:** fill in *Result*, *Predicted vs actual* and *What it changes*. Never edit the prediction.
3. Add one row to the table above and update the ledger in [`PROGRESS.md`](../PROGRESS.md).

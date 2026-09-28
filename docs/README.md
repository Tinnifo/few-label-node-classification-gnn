# Preregistrations and reports

These files are the original records, kept word for word. The hypotheses in them were frozen before each run, so we
do not edit them afterwards. Some of them use internal tracking labels. This table maps each file to the plain-language
write-ups:

| File | What it is | Internal label used inside | Write-up |
| --- | --- | --- | --- |
| [`hypotheses.md`](hypotheses.md) | Preregistered hypotheses and results for the text-encoder comparison | FEW-30 | [Exp 3](../experiments/2026-08-30-text-encoder-comparison.md) |
| [`hypotheses_few31.md`](hypotheses_few31.md) | Preregistered hypotheses and results for fusion methods × heterophily | FEW-31 | [Exp 4](../experiments/2026-08-30-fusion-methods-and-heterophily.md) |
| [`report_2026-08-30.md`](report_2026-08-30.md) | Combined report for Exp 3 and Exp 4 | FEW-30, FEW-31 | — |
| [`experiment_scaffold.md`](experiment_scaffold.md) | Notes on the experiment code layout | — | — |

In these files, **A0** is CG3 without a semantic view and **A1–A4** are the arms that add one (sbert, e5 / gpt3l, and
LLM explanations stripped / full). **V1, V2, …** are sanity checks: if one fails, the run is debugged rather than
interpreted. Other `FEW-<n>` labels refer to entries in our internal task tracker. Terms are defined in the glossary in
[`PROGRESS.md`](../PROGRESS.md).

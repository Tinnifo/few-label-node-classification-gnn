# Exp 2 — are LLM-written explanations a more useful view than the node's own text?

| | |
| --- | --- |
| **Date** | 2026-07-28 |
| **Tests assumption** | 1 |
| **Code** | commit [`5302e0e`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/5302e0e) · probe script `scripts/probe_views.py` on `scaffold/data-and-experiment` |
| **Status** | done (no model training) |

## Question
The standing objection to Exp 1: a frozen encoder on the node's own text overlaps with structure almost by
construction, because the node features come from the same text. Is an LLM-written *explanation* of the node a less
overlapping and more useful view?

## Setup
A probe on each of four text views, on Cora / CiteSeer / PubMed; overlap = CKA with the structural view.

## Result

| view | source | dim | overlap with structure | gain over node features |
| --- | --- | --- | --- | --- |
| sbert | own text, small encoder | 384 | 0.565 | +0.0170 |
| gpt3l | own text, large encoder | 3072 | 0.601 | +0.0672 |
| tape (stripped) | gpt-4o-mini explanation | 384 | **0.500** | +0.0048 |
| tape (full) | same, with the LLM's predicted-label line kept | 384 | 0.497 | +0.0184 |

- LLM explanations **do** overlap less with structure (−0.065 CKA, paired p = 0.039) but are **not** more useful
  (+0.005 vs +0.017, p = 0.75). Being different from structure is necessary for a view to help, but not sufficient.
- A bigger encoder matters more than LLM reasoning: the same own text through the 3072-d encoder gains +0.067.
- Keeping the LLM's predicted-label line adds +0.014 with almost no change in overlap. That pattern points to label
  leakage, not to extra information about the node.

**Limits:** three graphs, all homophilic; a probe, not a trained model.

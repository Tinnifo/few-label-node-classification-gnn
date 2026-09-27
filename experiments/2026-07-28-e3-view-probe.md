# E3 — are LLM explanations a more useful view than the node's own text?

| | |
| --- | --- |
| **Date** | 2026-07-28 |
| **Chain link** | a |
| **Code** | commit [`5302e0e`](https://github.com/Tinnifo/few-label-node-classification-gnn/commit/5302e0e) · probe script `scripts/probe_views.py` on `scaffold/data-and-experiment` |
| **Status** | done — training-free |

## Question
E1a's standing objection: a frozen encoder on the node's own text is redundant with structure by construction.
Does an LLM *explanation* of the node give a less redundant, more useful view?

## Setup
Graph-free linear probe of four staged views on cora / citeseer / pubmed; redundancy = CKA with the structural channel.

## Result

| view | source | d | redundancy | gain over X |
| --- | --- | --- | --- | --- |
| `sbert` | own text, small encoder | 384 | 0.565 | +0.0170 |
| `gpt3l` | own text, large encoder (text-embedding-3-large) | 3072 | 0.601 | +0.0672 |
| `tape_stripped` | gpt-4o-mini explanation (TAPE/LLMNodeBed) | 384 | **0.500** | +0.0048 |
| `tape_full` | + leaked answer line | 384 | 0.497 | +0.0184 |

- LLM explanations **are** less redundant (−0.065 CKA, paired p = 0.039) but **not** more useful
  (+0.005 vs +0.017, p = 0.75). Being different from structure is necessary, not sufficient.
- Encoder capacity beats LLM reasoning: the same text through the 3072-d encoder gains +0.067.
- Restoring the LLM's answer line adds +0.014 for almost no redundancy change — the signature of label leakage.

**Limits:** three homophilous graphs; probe, not a trained model.

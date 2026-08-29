"""Case study: concatenate LLM embeddings with original node features."""

from __future__ import annotations

from evals.methods.placeholder import PlaceholderMethod


class LLMConcatMethod(PlaceholderMethod):
    method_name = "llm_concat"
    todo = "Load LLM node embeddings and concatenate with data.x before a GCN classifier."

"""Case study: build a k-NN graph from LLM embeddings for message passing."""

from __future__ import annotations

from evals.methods.placeholder import PlaceholderMethod


class KNNLLMMethod(PlaceholderMethod):
    method_name = "knn_llm"
    todo = "Build a k-NN graph from LLM embeddings and train a GNN on that graph."

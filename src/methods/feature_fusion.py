"""Case study: standard feature fusion of structural features + LLM embeddings."""

from __future__ import annotations

from src.methods.placeholder import PlaceholderMethod


class FeatureFusionMethod(PlaceholderMethod):
    method_name = "feature_fusion"
    todo = "Fuse LLM and bag-of-words / structural features (e.g. gated / attention fusion)."

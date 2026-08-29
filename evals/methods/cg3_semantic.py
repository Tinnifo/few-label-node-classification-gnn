"""Direction-2 method: CG3 structural views + LLM semantic view + disparity/HSIC.

Keeps CG3's structural contrastive loss between local/global views, and will
add a disparity or HSIC term so the LLM semantic view stays complementary.
"""

from __future__ import annotations

from src.methods.placeholder import PlaceholderMethod


class CG3SemanticMethod(PlaceholderMethod):
    method_name = "cg3_semantic"
    todo = (
        "Wire an LLM semantic encoder into CG3 and apply conf/loss "
        "(structural_plus_hsic / structural_plus_disparity)."
    )

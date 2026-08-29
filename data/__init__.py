"""Dataset stubs. AG and TAG graphs land here later.

`main.py` imports these today so the call sites exist; Planetoid loading
stays in `evals.loader` until the real files are dropped in this folder.
"""

from __future__ import annotations


def load_ag(*args, **kwargs):
    raise NotImplementedError(
        "AG dataset loader is not wired yet. Drop graphs under data/ and implement load_ag."
    )


def load_tag(*args, **kwargs):
    raise NotImplementedError(
        "TAG dataset loader is not wired yet. Drop graphs under data/ and implement load_tag."
    )

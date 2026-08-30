#!/usr/bin/env python3
"""Write the `--texts` file for src/cg3_semantic.py from a TAG bundle.

Reads datasets/tag/<name>/<name>.npz (from scripts/download_data.py) and
writes datasets/tag/<name>/<name>_texts.txt with one whitespace-normalized
line per node. The cora and pubmed bundles are already aligned to the PyG
Planetoid node order (edge sets match exactly), so the output can be passed
straight to `--texts`.

CiteSeer is refused by default: the TAG release has 3,186 nodes while PyG
Planetoid has 3,327 — `load_texts` would (correctly) reject the file. Use
--force only if you are loading the TAG-native graph instead of Planetoid.

Usage:
    python scripts/make_texts.py cora pubmed
    python scripts/make_texts.py citeseer --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PYG_NUM_NODES = {"cora": 2708, "citeseer": 3327, "pubmed": 19717}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+", choices=sorted(PYG_NUM_NODES))
    ap.add_argument("--root", default="datasets", help="data root (download_data.py output)")
    ap.add_argument("--force", action="store_true",
                    help="write the file even if the node count does not match PyG Planetoid")
    args = ap.parse_args()

    for name in args.datasets:
        npz = Path(args.root) / "tag" / name / f"{name}.npz"
        if not npz.exists():
            sys.exit(f"{npz} not found — run scripts/download_data.py first")
        texts = np.load(npz, allow_pickle=True)["node_texts"]

        n, expected = len(texts), PYG_NUM_NODES[name]
        if n != expected and not args.force:
            sys.exit(
                f"{name}: bundle has {n} nodes but PyG Planetoid has {expected} — "
                f"this text file cannot be used with the Planetoid graph. "
                f"(Known for citeseer: the TAG release is a 3,186-node subset.) "
                f"Pass --force only for a TAG-native graph path."
            )

        lines = []
        for i, t in enumerate(texts):
            line = " ".join(str(t).split())  # strip newlines/tabs, collapse runs
            if not line:
                sys.exit(f"{name}: node {i} has empty text — refusing to write a silent gap")
            lines.append(line)

        out = npz.with_name(f"{name}_texts.txt")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{name}: wrote {len(lines)} lines -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download the frozen FLNC datasets (release `data-v1`) into ./datasets/.

Fetches the classic Planetoid files (Yang et al. 2016) and the text-attributed
(TAG) versions of Cora/CiteSeer/PubMed (raw text: Chen et al. 2024,
arXiv:2307.03393), sha256-verifies every asset, and unpacks:

    datasets/planetoid/ind.{cora,citeseer,pubmed}.*   # classic ind.* files
    datasets/tag/{cora,citeseer,pubmed}/              # .npz + semantic views

Stdlib only — no gdown, no credentials. Usage:

    python scripts/download_data.py                # everything
    python scripts/download_data.py --only tag_cora planetoid_trio

NOTE: the TAG CiteSeer is the 3,186-node subset (Planetoid has 3,327);
numbers on it are NOT comparable to published Planetoid-CiteSeer results.
See the data-v1 release notes for full provenance and citations.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

RELEASE_URL = (
    "https://github.com/Tinnifo/few-label-node-classification-gnn"
    "/releases/download/data-v1"
)

# sha256 of each release asset (pinned at release time — do not edit).
ASSETS = {
    "planetoid_trio.tar.gz": "9f27be589fe4ecd6ea93bad7ee7503ff096736346e3fc80356f3c4c46a6abfe5",
    "tag_cora.tar.gz": "591c64b81f1d676b5649d539211c1f73818cfd4a3c2fb365fbfbf38e7bf22c17",
    "tag_citeseer.tar.gz": "33d5ca5c038f6d98c326fa6345e7253abe664637713d14f04025706132d937ce",
    "tag_pubmed.tar.gz": "e5d8db2fcb79b61fce1e72a32ad6355dcee1b62b2b6f76a4576d118987c7bad2",
    "DATA_MANIFEST.json": "0135508e8fd78b86dfa52ea4fd18bd12b69a1dae63a3ec1d015039e1c05b87b6",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, expected: str, cache: Path) -> Path:
    dest = cache / name
    if dest.exists() and sha256(dest) == expected:
        print(f"  {name}: cached, checksum ok")
        return dest
    print(f"  {name}: downloading ...")
    urllib.request.urlretrieve(f"{RELEASE_URL}/{name}", dest)
    got = sha256(dest)
    if got != expected:
        dest.unlink()
        sys.exit(f"CHECKSUM MISMATCH for {name}: got {got}, expected {expected}")
    print(f"  {name}: checksum ok")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="datasets", help="output directory")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="asset stems to fetch (e.g. tag_cora planetoid_trio); default: all",
    )
    args = ap.parse_args()

    root = Path(args.root)
    cache = root / "_downloads"
    cache.mkdir(parents=True, exist_ok=True)

    wanted = {
        name: digest
        for name, digest in ASSETS.items()
        if args.only is None or name.split(".")[0] in args.only
    }
    for name, digest in wanted.items():
        path = fetch(name, digest, cache)
        if name.endswith(".tar.gz"):
            # planetoid_trio -> datasets/planetoid/, tag_* -> datasets/tag/<ds>/
            with tarfile.open(path) as tar:
                tar.extractall(root)
            print(f"  {name}: extracted into {root}/")

    print(f"\ndone — data under {root}/")


if __name__ == "__main__":
    main()

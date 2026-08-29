"""Cluster / shell launch helpers.

`.sh` scripts in this folder are for SLURM; this module is what `main.py`
imports. Fill in later.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def script_path(name: str) -> Path:
    """Return the path of a shell script in `sh/`."""
    return ROOT / name


def run_cg3(*args, **kwargs):
    raise NotImplementedError("Wire SLURM / local launch scripts in sh/.")

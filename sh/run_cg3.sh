#!/bin/bash
# Placeholder SLURM / local launcher. Fill in when cluster jobs are wired.
# Example: python main.py --method cg3 --dataset cora --budget 20
set -euo pipefail
cd "$(dirname "$0")/.."
python main.py "$@"

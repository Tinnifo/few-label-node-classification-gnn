# few-label-node-classification-gnn

Hydra-driven pipeline for few-label node classification with the `cg3` method
(contrastive graph-to-graph multi-task) on homophilic Planetoid graphs, with
scaffolding for Direction-2 (LLM semantic view + disparity/HSIC) and local
MLflow tracking. Metrics: accuracy and macro F1.

See [PLAN.md](PLAN.md) for the Direction-2 experiment design.

## 1. Setup

Managed with [uv](https://docs.astral.sh/uv/). The interpreter is pinned in `.python-version` (3.12) and dependencies are locked in `uv.lock`.

```bash
uv sync                 # creates .venv from uv.lock (incl. notebook deps)
uv sync --no-dev        # training only — skips matplotlib/igraph/networkx
```

Run anything with `uv run <cmd>`. Before the first dataset load:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

## 2. How experiments are configured

Hydra composes one experiment from config groups under `conf/`:

| Group | Files | Purpose |
|---|---|---|
| `model/` | `gcn`, `cg3` | Backbone label (`cg3` is a placeholder; CG3 builds its own model) |
| `method/` | `cg3`, `cg3_semantic`, `llm_concat`, `knn_llm`, `feature_fusion` | Training recipe (`cg3` works; others are stubs) |
| `loss/` | `structural`, `disparity`, `hsic`, `structural_plus_*` | Pluggable view loss (structural = original CG3) |
| `dataset/` | `cora`, `citeseer`, `pubmed`, `roman_empire`, `amazon_ratings` | Homo Planetoid + hetero placeholders |
| `label_strategy/` | `per_class`, `percentage` | How the labeled set is sampled |
| `metrics/` | `default` | accuracy + macro F1 |

## 3. Running

```bash
# Baseline CG3
uv run python src/train.py method=cg3 loss=structural dataset=cora label_strategy.budget=20

# Swap loss without touching CG3 source
uv run python src/train.py method=cg3 loss=structural_plus_hsic dataset=cora

# Bundled experiment recipe
uv run python src/train.py --multirun +experiment=cg3_combinations
```

### MLflow (local)

Tracking is on by default (`mlflow.enable=true`, SQLite store at `mlflow.db`).

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Toggle off with `mlflow.enable=false`. TensorBoard remains available via `tensorboard.enable`.

### Where results go

- Master CSV: `master_csv` knob (default `all_experiments.csv`)
- Per-config CSVs + TB events under `runs/<dataset>/budget_<X>/<model>_<method>/`
- MLflow: `mlflow.db` (+ artifacts under `mlartifacts/` if created)

## 4. Adding a loss / method / dataset

**New loss**: implement `BaseViewLoss` in `src/losses/`, register in `build_loss`, add `conf/loss/foo.yaml`. Use with `loss=foo`.

**New method**: subclass `BaseMethod`, add to `METHOD_REGISTRY` in `src/train.py`, write `conf/method/foo.yaml`.

**New dataset**: add `conf/dataset/foo.yaml` with `kind` (`planetoid` | `placeholder` | future kinds) and wire the loader in `src/data/loader.py`.

## 5. Project structure

```
PLAN.md                   # Direction-2 experiment plan
conf/                     # Hydra config groups
src/
├── train.py              # Hydra entry + MLflow / TensorBoard logging
├── losses/               # Pluggable structural / disparity / HSIC losses
├── eval/                 # accuracy + macro F1
├── tracking/             # local MLflow helper
├── models/               # BaseGNN + GCN
├── methods/
│   ├── cg3.py            # working baseline
│   ├── cg3_semantic.py   # Direction-2 stub
│   ├── llm_concat.py / knn_llm.py / feature_fusion.py  # case-study stubs
│   └── _cg3/             # CG3 architecture
└── data/
```

## 6. TensorBoard

```bash
uv run tensorboard --logdir runs
```

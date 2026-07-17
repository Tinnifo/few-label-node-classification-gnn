# few-label-node-classification-gnn

Hydra-driven pipeline for benchmarking GNN backbones (GCN, GAT, GIN, SAGE, GT, Diff) under two label-budget regimes (per-class N, global %) on Cora / CiteSeer / PubMed, with three training methods plug-in: `vanilla` (CE), `iceberg` (debiased self-training: pseudo-label + balanced softmax), `cg3` (contrastive graph-to-graph multi-task). All metrics log to TensorBoard.

## 1. Setup

Managed with [uv](https://docs.astral.sh/uv/). The interpreter is pinned in `.python-version` (3.12) and dependencies are locked in `uv.lock`.

```bash
uv sync                 # creates .venv from uv.lock (incl. notebook deps)
uv sync --no-dev        # training only — skips matplotlib/igraph/networkx
```

Run anything with `uv run <cmd>` (no manual venv activation needed), e.g. `uv run python src/train.py ...`. The HPC/RunPod scripts under `sh/` still `source .venv/bin/activate`, which works because `uv sync` produces a standard `.venv`.

> **Note:** torch ≥ 2.6 flipped `torch.load` to `weights_only=True`, which rejects PyG's pickled Planetoid cache. The `sh/` scripts export `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; set it yourself before the first run if you invoke `src/train.py` directly:
> ```bash
> export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
> ```

## 2. How experiments are configured

Hydra composes one experiment from four config groups under `conf/`:

| Group | Files | Purpose |
|---|---|---|
| `model/` | `gcn`, `gat`, `gin`, `sage`, `gt`, `diff`, `cg3` | Backbone architecture |
| `method/` | `vanilla`, `iceberg`, `cg3` | Training recipe (loss, optional pseudo-labeling) |
| `dataset/` | `cora`, `citeseer`, `pubmed` | Planetoid loader + dataset-specific percentage budgets |
| `label_strategy/` | `per_class`, `percentage` | How the labeled set is sampled (uses `src/data/labels.py`) |

The top-level `conf/config.yaml` sets shared knobs (`seeds=[0..4]`, TensorBoard log dir, `master_csv`). Per-method training length lives in `conf/method/*.yaml` (`epochs`, `patience`).

## 3. Running

### Single experiment
```bash
# vanilla GCN on Cora, 20 labels per class
uv run python src/train.py model=gcn method=vanilla dataset=cora label_strategy=per_class label_strategy.budget=20

# GCN with the IceBerg trick — clean A/B test against the line above
uv run python src/train.py model=gcn method=iceberg dataset=cora label_strategy=per_class label_strategy.budget=20

# Diff backbone + IceBerg (the paper's headline recipe)
uv run python src/train.py model=diff method=iceberg dataset=cora label_strategy.budget=20

# CG3 — model knob ignored (the method bundles its own architecture)
uv run python src/train.py model=cg3 method=cg3 dataset=cora label_strategy.budget=20

# Percentage label strategy
uv run python src/train.py model=gcn method=iceberg dataset=cora label_strategy=percentage label_strategy.budget=0.01

# Disable TB logging if you just want stdout
uv run python src/train.py model=gcn method=vanilla dataset=cora tensorboard.enable=false
```

> In zsh, quote any bracketed override so the shell doesn't glob it: `'seeds=[0,1,2]'`.

### Sweeps (Hydra `--multirun`)
```bash
# All standard backbones × {vanilla, iceberg} on Cora @ budgets {1,3,5,10,20}
uv run python src/train.py --multirun \
    model=gcn,gat,gin,sage,gt,diff method=vanilla,iceberg \
    dataset=cora label_strategy=per_class label_strategy.budget=1,3,5,10,20

# Use a bundled experiment recipe (see conf/experiment/)
uv run python src/train.py --multirun +experiment=full_grid
```

### Where results go

Every run appends a one-row summary to a master CSV in the launch directory. The filename is the `master_csv` config knob (default `all_experiments.csv`) — route a sweep's results without editing source:

```bash
uv run python src/train.py --multirun +experiment=cg3_pct_cora master_csv=all_experimentsCG3Percentage.csv
```

Per-config CSVs (`summary_results.csv`, `seed_results.csv`) are also written next to the TensorBoard events under `runs/<dataset>/budget_<X>/<model>_<method>/`.

### HPC (Slurm) / RunPod
```bash
sbatch sh/run.sh          # full grid on Slurm
sh/runpod.sh              # full grid, no Slurm (single-GPU box)
sh/runpod.sh full_grid    # a single experiment yaml
```

## 4. Adding a new model or method

**New model**: drop a `src/models/foo.py` that subclasses `BaseGNN` (`forward(x, edge_index) → logits`), re-export it in `src/models/__init__.py`, and add `conf/model/foo.yaml` with `_target_: src.models.foo.Foo`. It now works with `vanilla` and `iceberg`.

**New method**: drop a `src/methods/foo.py` that subclasses `BaseMethod` (implementing `build_model` and `train_step`; optionally `prepare`, `validate`, `predict_logits`). Add it to `METHOD_REGISTRY` in `src/train.py` and write `conf/method/foo.yaml` with at least `name: foo`.

## 5. Notebooks

`notebooks/datasets.ipynb` explores the Planetoid datasets (Cora/CiteSeer/PubMed) and visualizes graphs with igraph/networkx. It loads data through the same Hydra config and `src/data/loader.load_dataset` used by training, so notebook and experiments stay in sync. The visualization libraries come from the `dev` dependency group (`uv sync` installs them by default).

## 6. Project structure

```
conf/                     # Hydra config groups (model, method, dataset, label_strategy, experiment)
src/
├── train.py              # Hydra entry point — single run or `--multirun` sweep
├── models/               # BaseGNN + GCN, GAT, GIN, SAGE, GT, Diff
├── methods/
│   ├── base.py           # BaseMethod (build_model, prepare, train_step, evaluate, validate)
│   ├── vanilla.py
│   ├── iceberg.py
│   ├── cg3.py            # wraps the bundled CG3 code below
│   └── _cg3/             # CG3's bundled architecture (CG3Model, HGCN, hierarchy build)
└── data/
    ├── loader.py         # Planetoid loading + label-strategy dispatch
    └── labels.py         # set_few_label_mask, set_budget_percent, set_seed
docs/                     # Dataset / method notes
notebooks/                # Dataset exploration + visualization
sh/                       # Slurm + RunPod run scripts
data/                     # Auto-created by PyG (Planetoid downloads)
outputs/, multirun/       # Auto-created by Hydra (single-run / sweep working dirs)
runs/                     # TensorBoard event files + per-config CSVs
```

## 7. TensorBoard output

Each Hydra run = one `(model, method, dataset, label_strategy, budget)` config evaluated across all seeds, writing to `runs/<dataset>/budget_<X>/<model>_<method>/`:
- Per-epoch scalars under `seed_{seed}/...`; per-seed test metrics; aggregate across seeds under `agg/...`
- An HParams entry (model/method/dataset/budget vs. final metrics) for the **HParams** tab
- `best_state_seed{seed}.pt` — best-val checkpoint per seed (toggle via `save_checkpoints` in `conf/config.yaml`)

```bash
uv run tensorboard --logdir runs
```

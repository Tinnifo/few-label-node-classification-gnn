"""Train one (method, dataset, label strategy) run and log it to MLflow.

Usage:
  python main.py
  python main.py --method cg3 --dataset cora --budget 20 --seeds 0,1,2
  python main.py --method cg3_semantic --loss structural_plus_hsic
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

import data  # AG / TAG stubs — fill data/ later
import sh  # cluster launch stubs — fill sh/ later
from evals.labels import set_seed
from evals.loader import apply_label_strategy, format_budget, load_dataset
from evals.methods import (
    CG3Method,
    CG3SemanticMethod,
    FeatureFusionMethod,
    KNNLLMMethod,
    LLMConcatMethod,
)
from evals.methods.base import BaseMethod
from src.method import SemanticGNNModel
from src.model import GCN, HGAT, HGCN
from utils.graph import GNNModel, build_cg3_artifacts
from utils.losses import build_loss
from utils.mlflow_logger import start_run

log = logging.getLogger(__name__)

METHOD_REGISTRY = {
    "cg3": CG3Method,
    "cg3_semantic": CG3SemanticMethod,
    "llm_concat": LLMConcatMethod,
    "knn_llm": KNNLLMMethod,
    "feature_fusion": FeatureFusionMethod,
}


def _ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def build_cfg(args: argparse.Namespace) -> SimpleNamespace:
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    return _ns(
        device=args.device,
        data_root=args.data_root,
        seeds=seeds,
        epoch_log_every=args.epoch_log_every,
        save_checkpoints=args.save_checkpoints,
        dataset=_ns(name=args.dataset, kind=args.dataset_kind, normalize_features=False),
        label_strategy=_ns(name=args.label_strategy, budget=args.budget),
        method=_ns(
            name=args.method,
            local_model=args.local_model,
            global_model=args.global_model,
            lr=args.lr,
            epochs=args.epochs,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden_local=args.hidden_local,
            hidden_global=args.hidden_global,
            coarsen_level=args.coarsen_level,
            max_node_wgt=args.max_node_wgt,
            channel_num=args.channel_num,
            node_wgt_embed_dim=args.node_wgt_embed_dim,
            use_early_stopping=args.early_stopping,
            patience=args.patience,
            semantic_channel=None,
        ),
        loss=_ns(name=args.loss, temperature=0.5, hp1=0.9, lambda_reg=1.0, weight=1.0),
        model=_ns(name=args.backbone, hidden_channels=args.hidden_local, dropout=args.dropout),
        mlflow=_ns(
            enable=not args.no_mlflow,
            tracking_uri=args.mlflow_uri,
            experiment_name=args.mlflow_experiment,
        ),
    )


def build_method(cfg) -> BaseMethod:
    name = cfg.method.name
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Choose from {list(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name](cfg)


def run_one_seed(cfg, method: BaseMethod, base_data, in_channels: int,
                 num_classes: int, seed: int, device: torch.device,
                 checkpoint_path: str | None = None):
    set_seed(seed)
    data_obj = base_data.clone().to(device)
    data_obj = apply_label_strategy(
        data_obj, cfg.label_strategy.name, cfg.label_strategy.budget, seed
    )

    model = method.build_model(in_channels, num_classes, data=data_obj).to(device)
    data_obj = method.prepare(model, data_obj)
    optimizer = method.build_optimizer(model)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    best_metric = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    counter = 0
    epoch_log = []
    stopped_at = 0

    for epoch in range(1, cfg.method.epochs + 1):
        train_out = method.train_step(model, data_obj, optimizer, epoch)
        val_out = method.validate(model, data_obj)
        epoch_log.append({"epoch": epoch, **train_out, **val_out})
        stopped_at = epoch

        early = val_out.get("early_stop_metric")
        if early is not None:
            if early > best_metric:
                best_metric = float(early)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
            if cfg.method.use_early_stopping and counter >= cfg.method.patience:
                break

    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(best_state, checkpoint_path)

    metrics = method.evaluate(model, data_obj)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_sec = time.perf_counter() - start

    return {
        "metrics": metrics,
        "epoch_log": epoch_log,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "stopped_at_epoch": stopped_at,
        "runtime_sec": runtime_sec,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Few-label node classification")
    p.add_argument("--method", default="cg3", choices=list(METHOD_REGISTRY))
    p.add_argument("--dataset", default="cora")
    p.add_argument("--dataset-kind", default=None, help="planetoid | placeholder")
    p.add_argument("--label-strategy", default="per_class", choices=("per_class", "percentage"))
    p.add_argument("--budget", type=float, default=20)
    p.add_argument("--loss", default="structural")
    p.add_argument("--backbone", default="gcn", help="PyG backbone name (vanilla GCN path)")
    p.add_argument("--local-model", default="gcn", choices=("gcn", "gat"))
    p.add_argument("--global-model", default="hgcn", choices=("hgcn", "hgat"))
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--hidden-local", type=int, default=1024)
    p.add_argument("--hidden-global", type=int, default=32)
    p.add_argument("--coarsen-level", type=int, default=4)
    p.add_argument("--max-node-wgt", type=int, default=50)
    p.add_argument("--channel-num", type=int, default=4)
    p.add_argument("--node-wgt-embed-dim", type=int, default=5)
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seeds", default="0")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--epoch-log-every", type=int, default=10)
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    p.add_argument("--mlflow-experiment", default="few-label-gnn")
    return p.parse_args()


def main() -> float:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = build_cfg(args)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    log.info("device=%s", device)

    loaded = load_dataset(
        cfg.dataset.name,
        root=cfg.data_root,
        normalize_features=cfg.dataset.normalize_features,
        kind=cfg.dataset.kind,
    )
    method = build_method(cfg)

    run_name = (
        f"{cfg.dataset.name}/{cfg.method.name}/"
        f"{cfg.method.global_model}/b{format_budget(cfg.label_strategy.budget)}"
    )
    mlf = start_run(
        enable=cfg.mlflow.enable,
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        run_name=run_name,
        params={
            "method": cfg.method.name,
            "dataset": cfg.dataset.name,
            "loss": cfg.loss.name,
            "label_strategy": cfg.label_strategy.name,
            "budget": cfg.label_strategy.budget,
            "local_model": cfg.method.local_model,
            "global_model": cfg.method.global_model,
            "epochs": cfg.method.epochs,
            "seeds": cfg.seeds,
        },
    )

    log_dir = os.path.join(
        "runs",
        cfg.dataset.name,
        f"budget_{format_budget(cfg.label_strategy.budget)}",
        f"{cfg.method.name}_{cfg.method.global_model}",
    )

    all_metrics = []
    all_runtimes = []
    every = max(1, int(cfg.epoch_log_every))

    try:
        for seed in cfg.seeds:
            log.info("[seed=%s] training...", seed)
            ckpt_path = (
                os.path.join(log_dir, f"best_state_seed{seed}.pt")
                if cfg.save_checkpoints
                else None
            )
            result = run_one_seed(
                cfg,
                method,
                loaded.data,
                loaded.in_channels,
                loaded.num_classes,
                int(seed),
                device,
                checkpoint_path=ckpt_path,
            )
            m = result["metrics"]
            all_metrics.append(m)
            all_runtimes.append(result["runtime_sec"])
            log.info(
                "[seed=%s] stopped@%s best@%s acc=%.4f macroF1=%.4f",
                seed,
                result["stopped_at_epoch"],
                result["best_epoch"],
                m["accuracy"],
                m["macro_f1"],
            )

            for entry in result["epoch_log"]:
                if (
                    entry["epoch"] % every != 0
                    and entry["epoch"] != result["stopped_at_epoch"]
                ):
                    continue
                epoch_metrics = {
                    f"seed_{seed}/{k}": float(v)
                    for k, v in entry.items()
                    if k != "epoch" and isinstance(v, (int, float))
                }
                mlf.log_metrics(epoch_metrics, step=int(entry["epoch"]))
            mlf.log_metrics(
                {
                    f"seed_{seed}/test_accuracy": float(m["accuracy"]),
                    f"seed_{seed}/test_macro_f1": float(m["macro_f1"]),
                    f"seed_{seed}/best_epoch": float(result["best_epoch"]),
                    f"seed_{seed}/runtime_sec": float(result["runtime_sec"]),
                },
                step=0,
            )

        accs = np.array([m["accuracy"] for m in all_metrics], dtype=float)
        f1s = np.array([m["macro_f1"] for m in all_metrics], dtype=float)
        mean_acc, std_acc = float(accs.mean()), float(accs.std())
        mean_f1, std_f1 = float(f1s.mean()), float(f1s.std())
        n = max(len(cfg.seeds), 1)
        moe_acc = 1.96 * std_acc / np.sqrt(n)
        moe_f1 = 1.96 * std_f1 / np.sqrt(n)

        log.info(
            "agg acc=%.4f ± %.4f (moe=%.4f)  macroF1=%.4f ± %.4f (moe=%.4f)  "
            "runtime=%.1fs",
            mean_acc,
            std_acc,
            moe_acc,
            mean_f1,
            std_f1,
            moe_f1,
            float(np.mean(all_runtimes)),
        )
        mlf.log_metrics(
            {
                "agg/mean_accuracy": mean_acc,
                "agg/std_accuracy": std_acc,
                "agg/moe_accuracy": moe_acc,
                "agg/mean_macro_f1": mean_f1,
                "agg/std_macro_f1": std_f1,
                "agg/moe_macro_f1": moe_f1,
                "agg/mean_runtime_sec": float(np.mean(all_runtimes)),
            }
        )
        return mean_acc
    finally:
        mlf.end()


if __name__ == "__main__":
    main()

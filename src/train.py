"""Hydra entry point: trains one (model, method, dataset, label_strategy, loss)
configuration over a list of seeds and reports aggregated test metrics.

CG3:
  python src/train.py method=cg3 loss=structural dataset=cora label_strategy.budget=20

Composite loss (HSIC/disparity stubs return 0 until implemented):
  python src/train.py method=cg3 loss=structural_plus_hsic dataset=cora

MLflow UI (local):
  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import pandas as pd

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Ensure project root is on the path so `src/` resolves.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.loader import apply_label_strategy, format_budget, load_dataset
from src.methods import (
    CG3Method,
    CG3SemanticMethod,
    FeatureFusionMethod,
    KNNLLMMethod,
    LLMConcatMethod,
)
from src.methods.base import BaseMethod
from src.tracking.mlflow_logger import start_run

METHOD_REGISTRY = {
    "cg3": CG3Method,
    "cg3_semantic": CG3SemanticMethod,
    "llm_concat": LLMConcatMethod,
    "knn_llm": KNNLLMMethod,
    "feature_fusion": FeatureFusionMethod,
}


def build_method(cfg: DictConfig) -> BaseMethod:
    name = cfg.method.name
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Add it to METHOD_REGISTRY in src/train.py.")
    return METHOD_REGISTRY[name](cfg)


def run_log_dir(cfg: DictConfig) -> str:
    """`runs/<dataset>/budget_<X>/<model>_<method>/`."""
    return os.path.join(
        cfg.tensorboard.log_dir,
        cfg.dataset.name,
        f"budget_{format_budget(cfg.label_strategy.budget)}",
        f"{cfg.model.name}_{cfg.method.name}",
    )


def init_tensorboard(cfg: DictConfig):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=run_log_dir(cfg))


def _mean_or_none(vals):
    return float(np.mean(vals)) if vals else None


def run_one_seed(cfg: DictConfig, method: BaseMethod, base_data, in_channels: int,
                 num_classes: int, seed: int, device: torch.device,
                 checkpoint_path: str | None = None):
    import time

    from src.data.labels import set_seed
    set_seed(seed)
    data = base_data.clone().to(device)
    data = apply_label_strategy(data, cfg.label_strategy.name, cfg.label_strategy.budget, seed)

    model = method.build_model(in_channels, num_classes, data=data).to(device)
    data = method.prepare(model, data)
    if cfg.compile_model:
        try:
            model = torch.compile(model)
        except Exception as e:
            log.warning(f"torch.compile failed ({e}); falling back to eager")
    optimizer = method.build_optimizer(model)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    best_metric = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    counter = 0
    epoch_log = []

    for epoch in range(1, cfg.method.epochs + 1):
        train_out = method.train_step(model, data, optimizer, epoch)
        val_out = method.validate(model, data)
        epoch_log.append({"epoch": epoch, **train_out, **val_out})

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

    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path is not None:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            clean_state = {k.removeprefix("_orig_mod."): v for k, v in best_state.items()}
            torch.save(clean_state, checkpoint_path)

    metrics = method.evaluate(model, data)

    loss_ce_vals, loss_gen_vals, loss_contrastive_vals = [], [], []
    loss_reg_vals, loss_total_vals = [], []
    for e in epoch_log:
        if e.get("loss_ce") is not None:
            loss_ce_vals.append(e["loss_ce"])
        if e.get("loss_gen") is not None:
            loss_gen_vals.append(e["loss_gen"])
        if e.get("loss_contrastive") is not None:
            loss_contrastive_vals.append(e["loss_contrastive"])
        if e.get("loss_reg") is not None:
            loss_reg_vals.append(e["loss_reg"])
        if e.get("loss_total") is not None:
            loss_total_vals.append(e["loss_total"])

    loss_stats = {
        "loss_ce": _mean_or_none(loss_ce_vals),
        "loss_gen": _mean_or_none(loss_gen_vals),
        "loss_contrastive": _mean_or_none(loss_contrastive_vals),
        "loss_reg": _mean_or_none(loss_reg_vals),
        "loss_total": _mean_or_none(loss_total_vals),
    }

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_sec = time.perf_counter() - start_time

    return {
        "metrics": metrics,
        "epoch_log": epoch_log,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "stopped_at_epoch": epoch_log[-1]["epoch"] if epoch_log else 0,
        "loss_stats": loss_stats,
        "runtime_sec": runtime_sec,
    }


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    log.info(f"device={device}")

    loaded = load_dataset(
        cfg.dataset.name,
        root=cfg.data_root,
        normalize_features=cfg.dataset.normalize_features,
        kind=cfg.dataset.get("kind", None),
    )
    base_data = loaded.data
    method = build_method(cfg)

    writer = None
    if cfg.tensorboard.enable:
        writer = init_tensorboard(cfg)

    mlf = start_run(cfg)

    seeds = list(cfg.seeds)
    all_metrics = []
    all_loss_stats = []
    all_runtimes = []
    best_epochs = []
    every = max(1, int(cfg.epoch_log_every))

    log_dir = run_log_dir(cfg)
    try:
        for seed in seeds:
            log.info(f"[seed={seed}] training...")
            ckpt_path = (
                os.path.join(log_dir, f"best_state_seed{seed}.pt")
                if cfg.save_checkpoints else None
            )
            result = run_one_seed(
                cfg, method, base_data, loaded.in_channels,
                loaded.num_classes, int(seed), device,
                checkpoint_path=ckpt_path,
            )
            m = result["metrics"]
            all_metrics.append(m)
            all_loss_stats.append(result["loss_stats"])
            all_runtimes.append(result["runtime_sec"])
            best_epochs.append(result["best_epoch"])
            log.info(
                f"[seed={seed}] stopped@{result['stopped_at_epoch']} "
                f"best@{result['best_epoch']} "
                f"acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}"
            )

            seed_result = {
                "seed": seed,
                "test_acc": float(m["accuracy"]),
                "macro_f1": float(m["macro_f1"]),
            }
            seed_df = pd.DataFrame([seed_result])
            seed_csv = os.path.join(log_dir, "seed_results.csv")
            os.makedirs(log_dir, exist_ok=True)
            seed_df.to_csv(
                seed_csv,
                mode="a",
                header=not os.path.exists(seed_csv),
                index=False,
            )

            if writer is not None:
                for entry in result["epoch_log"]:
                    if entry["epoch"] % every != 0 and entry["epoch"] != result["stopped_at_epoch"]:
                        continue
                    step = int(entry["epoch"])
                    for k, v in entry.items():
                        if k == "epoch":
                            continue
                        if isinstance(v, (int, float)):
                            writer.add_scalar(f"seed_{seed}/{k}", float(v), step)
                writer.add_scalar(f"seed_{seed}/test_accuracy", float(m["accuracy"]), 0)
                writer.add_scalar(f"seed_{seed}/test_macro_f1", float(m["macro_f1"]), 0)
                writer.add_scalar(f"seed_{seed}/best_early_stop_metric", float(result["best_metric"]), 0)

            # MLflow: epoch curves + per-seed test metrics
            for entry in result["epoch_log"]:
                if entry["epoch"] % every != 0 and entry["epoch"] != result["stopped_at_epoch"]:
                    continue
                step = int(entry["epoch"])
                epoch_metrics = {
                    f"seed_{seed}/{k}": float(v)
                    for k, v in entry.items()
                    if k != "epoch" and isinstance(v, (int, float))
                }
                mlf.log_metrics(epoch_metrics, step=step)
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
        n = len(seeds)
        moe_acc = 1.96 * std_acc / np.sqrt(n)
        moe_f1 = 1.96 * std_f1 / np.sqrt(n)

        loss_ce_mean = _mean_or_none([x["loss_ce"] for x in all_loss_stats if x["loss_ce"] is not None])
        loss_gen_mean = _mean_or_none([x["loss_gen"] for x in all_loss_stats if x["loss_gen"] is not None])
        loss_contrastive_mean = _mean_or_none(
            [x["loss_contrastive"] for x in all_loss_stats if x["loss_contrastive"] is not None]
        )
        loss_reg_mean = _mean_or_none([x["loss_reg"] for x in all_loss_stats if x["loss_reg"] is not None])
        loss_total_mean = _mean_or_none(
            [x["loss_total"] for x in all_loss_stats if x["loss_total"] is not None]
        )

        runtime_mean = float(np.mean(all_runtimes))
        runtime_std = float(np.std(all_runtimes))

        model_name = cfg.model.name
        if cfg.method.name in ("cg3", "cg3_semantic"):
            model_name = f"{cfg.method.local_model}_{cfg.method.global_model}"

        def _round_or_none(v):
            return round(float(v), 4) if v is not None else None

        results_data = {
            "model": model_name,
            "method": cfg.method.name,
            "dataset": cfg.dataset.name,
            "loss": cfg.loss.name,
            "budget": cfg.label_strategy.budget,
            "mean_acc": round(mean_acc, 4),
            "std_acc": round(std_acc, 4),
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
            "l_ce": _round_or_none(loss_ce_mean),
            "l_gen": _round_or_none(loss_gen_mean),
            "l_con": _round_or_none(loss_contrastive_mean),
            "l_reg": _round_or_none(loss_reg_mean),
            "l_total": _round_or_none(loss_total_mean),
            "rt_sec_mean": round(runtime_mean, 4),
            "best_epoch_mean": int(np.mean(best_epochs)) if best_epochs else None,
            "rt_sec_std": round(runtime_std, 4),
            "epochs": cfg.method.epochs,
            "patience": cfg.method.patience,
            "lr": cfg.method.lr,
            "weight_decay": cfg.method.weight_decay,
            "use_early_stopping": cfg.method.use_early_stopping,
            "seeds": str(list(cfg.seeds)),
        }

        df_run = pd.DataFrame([results_data])
        os.makedirs(log_dir, exist_ok=True)
        run_csv_path = os.path.join(log_dir, "summary_results.csv")
        df_run.to_csv(
            run_csv_path,
            mode="a",
            header=not os.path.exists(run_csv_path),
            index=False,
        )
        log.info(f"Run results saved to {run_csv_path}")

        from hydra.utils import get_original_cwd
        master_csv_path = os.path.join(get_original_cwd(), cfg.master_csv)
        df_run.to_csv(
            master_csv_path,
            mode="a",
            header=not os.path.exists(master_csv_path),
            index=False,
        )
        log.info(f"Appended results to {master_csv_path}")

        log.info(
            f"[summary {cfg.model.name}/{cfg.method.name} {cfg.dataset.name} "
            f"loss={cfg.loss.name} b={format_budget(cfg.label_strategy.budget)}] "
            f"acc={mean_acc:.4f}+-{moe_acc:.4f}  macroF1={mean_f1:.4f}+-{moe_f1:.4f}"
        )

        mlf.log_metrics(
            {
                "agg/mean_accuracy": mean_acc,
                "agg/std_accuracy": std_acc,
                "agg/moe_accuracy": moe_acc,
                "agg/mean_macro_f1": mean_f1,
                "agg/std_macro_f1": std_f1,
                "agg/moe_macro_f1": moe_f1,
                "agg/best_epoch_mean": float(np.mean(best_epochs)) if best_epochs else 0.0,
                "agg/runtime_sec_mean": runtime_mean,
            }
        )
        if loss_ce_mean is not None:
            mlf.log_metrics({"agg/loss_ce": loss_ce_mean})
        if loss_gen_mean is not None:
            mlf.log_metrics({"agg/loss_gen": loss_gen_mean})
        if loss_contrastive_mean is not None:
            mlf.log_metrics({"agg/loss_contrastive": loss_contrastive_mean})
        if loss_reg_mean is not None:
            mlf.log_metrics({"agg/loss_reg": loss_reg_mean})
        if loss_total_mean is not None:
            mlf.log_metrics({"agg/loss_total": loss_total_mean})

        if writer is not None:
            writer.add_scalar("agg/mean_accuracy", mean_acc, 0)
            writer.add_scalar("agg/mean_macro_f1", mean_f1, 0)
            writer.add_scalar("agg/std_accuracy", std_acc, 0)
            writer.add_scalar("agg/std_macro_f1", std_f1, 0)
            writer.add_scalar("agg/moe_accuracy", moe_acc, 0)
            writer.add_scalar("agg/moe_macro_f1", moe_f1, 0)
            writer.add_hparams(
                {
                    "model": str(cfg.model.name),
                    "method": str(cfg.method.name),
                    "dataset": str(cfg.dataset.name),
                    "loss": str(cfg.loss.name),
                    "label_strategy": str(cfg.label_strategy.name),
                    "budget": float(cfg.label_strategy.budget),
                    "epochs": int(cfg.method.epochs),
                    "patience": int(cfg.method.patience),
                    "seeds": str(list(cfg.seeds)),
                },
                {
                    "hparam/mean_accuracy": mean_acc,
                    "hparam/mean_macro_f1": mean_f1,
                    "hparam/moe_accuracy": moe_acc,
                },
                run_name=".",
            )
            writer.flush()
            writer.close()

        return mean_acc
    finally:
        mlf.end()


if __name__ == "__main__":
    main()

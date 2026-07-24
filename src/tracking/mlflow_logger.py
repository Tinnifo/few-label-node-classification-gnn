"""Local MLflow helpers for Hydra training runs."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _flatten_params(cfg: DictConfig) -> Dict[str, Any]:
    """Flatten selected Hydra knobs into MLflow-friendly scalar params."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    flat: Dict[str, Any] = {}

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                walk(key, v)
        elif isinstance(obj, (list, tuple)):
            flat[prefix] = str(list(obj))
        elif obj is None or isinstance(obj, (bool, int, float, str)):
            flat[prefix] = obj
        else:
            flat[prefix] = str(obj)

    for key in ("model", "method", "dataset", "label_strategy", "loss", "metrics",
                "device", "seeds", "data_root", "master_csv", "epoch_log_every",
                "save_checkpoints", "compile_model"):
        if key in raw:
            walk(key, raw[key])
    return flat


def _resolve_tracking_uri(tracking_uri: str) -> str:
    """Make relative sqlite/file URIs absolute so Hydra CWD changes are safe."""
    if tracking_uri.startswith("sqlite:///"):
        path = tracking_uri[len("sqlite:///"):]
        if path and not os.path.isabs(path):
            return "sqlite:///" + os.path.abspath(path)
        return tracking_uri
    if "://" not in tracking_uri and not os.path.isabs(tracking_uri):
        return os.path.abspath(tracking_uri)
    return tracking_uri


class MLflowRun:
    """Thin wrapper so train.py can no-op when MLflow is disabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._active = False

    def __enter__(self) -> "MLflowRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end()

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        import mlflow
        cleaned = {}
        for k, v in params.items():
            s = str(v)
            cleaned[k] = s if len(s) <= 500 else s[:497] + "..."
        mlflow.log_params(cleaned)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        import mlflow
        payload = {k: float(v) for k, v in metrics.items() if v is not None}
        if payload:
            mlflow.log_metrics(payload, step=step)

    def end(self) -> None:
        if not self.enabled or not self._active:
            return
        import mlflow
        mlflow.end_run()
        self._active = False


def start_run(cfg: DictConfig, run_name: Optional[str] = None) -> MLflowRun:
    """Start an MLflow run if `cfg.mlflow.enable`, else return a no-op wrapper."""
    ml_cfg = cfg.get("mlflow", {})
    enabled = bool(ml_cfg.get("enable", False))
    wrapper = MLflowRun(enabled=enabled)
    if not enabled:
        return wrapper

    import mlflow

    tracking_uri = _resolve_tracking_uri(str(ml_cfg.get("tracking_uri", "sqlite:///mlflow.db")))
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = str(ml_cfg.get("experiment_name", "few-label-gnn"))
    mlflow.set_experiment(experiment_name)

    if run_name is None:
        run_name = (
            f"{cfg.dataset.name}/{cfg.method.name}/"
            f"{cfg.model.name}/b{cfg.label_strategy.budget}"
        )
    mlflow.start_run(run_name=run_name)
    wrapper._active = True
    wrapper.log_params(_flatten_params(cfg))
    log.info("MLflow tracking at %s (experiment=%s)", tracking_uri, experiment_name)
    return wrapper

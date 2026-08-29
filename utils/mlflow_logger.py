"""Local MLflow helpers for training runs."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional

log = logging.getLogger(__name__)


def _flatten_params(params: Mapping[str, Any]) -> Dict[str, Any]:
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

    walk("", dict(params))
    return {k.lstrip("."): v for k, v in flat.items()}


def _resolve_tracking_uri(tracking_uri: str) -> str:
    if tracking_uri.startswith("sqlite:///"):
        path = tracking_uri[len("sqlite:///") :]
        if path and not os.path.isabs(path):
            return "sqlite:///" + os.path.abspath(path)
        return tracking_uri
    if "://" not in tracking_uri and not os.path.isabs(tracking_uri):
        return os.path.abspath(tracking_uri)
    return tracking_uri


class MLflowRun:
    """Thin wrapper so training can no-op when MLflow is disabled."""

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
        for k, v in _flatten_params(params).items():
            s = str(v)
            cleaned[k] = s if len(s) <= 500 else s[:497] + "..."
        mlflow.log_params(cleaned)

    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
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


def start_run(
    *,
    enable: bool = True,
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "few-label-gnn",
    run_name: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> MLflowRun:
    wrapper = MLflowRun(enabled=enable)
    if not enable:
        return wrapper

    import mlflow

    uri = _resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    wrapper._active = True
    if params:
        wrapper.log_params(params)
    log.info("MLflow tracking at %s (experiment=%s)", uri, experiment_name)
    return wrapper

"""Metric helpers shared by the optimizer, legacy wrapper, and API."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


def _positive_scores(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if np.ndim(proba) == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return np.asarray(proba).ravel()
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    return np.asarray(model.predict(X)).ravel()


def prediction_output(model: Any, task: str, X: Any) -> dict[str, np.ndarray]:
    predictions = np.asarray(model.predict(X)).ravel()
    output = {"predictions": predictions}
    if task == "binary classification":
        output["scores"] = _positive_scores(model, X)
    return output


def metric_value(model: Any, task: str, metric: str, X: Any, y: Any) -> float:
    """Return the human metric value for a fitted model."""

    metric = {"acc": "accuracy", "auc": "roc_auc"}.get(metric, metric)
    y_true = np.asarray(y).ravel()
    if metric == "rmse":
        preds = np.asarray(model.predict(X)).ravel()
        return float(np.sqrt(mean_squared_error(y_true, preds)))
    if task != "binary classification":
        raise ValueError(f"Metric {metric!r} is only supported for classification.")
    if metric == "accuracy":
        preds = np.asarray(model.predict(X)).ravel()
        return float(accuracy_score(y_true, preds))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, _positive_scores(model, X)))
    raise ValueError(f"Unsupported metric: {metric!r}")


def cost_from_metric(task: str, metric: str, value: float) -> float:
    """Convert a metric into the minimization cost used by optimizers."""

    metric = {"acc": "accuracy", "auc": "roc_auc"}.get(metric, metric)
    if metric == "rmse":
        return float(value)
    if task == "binary classification" and metric in {"accuracy", "roc_auc"}:
        return float(1 - value)
    raise ValueError(f"Unsupported metric for cost conversion: {metric!r}")


def evaluate_model(model: Any, task: str, metric: str, X: Any, y: Any) -> tuple[float, float]:
    value = metric_value(model, task, metric, X, y)
    return cost_from_metric(task, metric, value), value


__all__ = ["cost_from_metric", "evaluate_model", "metric_value", "prediction_output"]

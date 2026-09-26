"""Configuration and result objects for pspso."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

TaskName = Literal["regression", "binary_classification", "multiclass_classification"]
MetricName = Literal["rmse", "mae", "r2", "accuracy", "roc_auc", "pr_auc", "log_loss", "f1_macro"]
StrategyName = Literal["pso", "grid", "random"]
EvaluationProtocol = Literal["holdout", "cross_validation"]


@dataclass(frozen=True)
class OptimizationConfig:
    """Runtime settings for an optimization run."""

    task: TaskName = "regression"
    metric: MetricName = "rmse"
    strategy: StrategyName = "pso"
    validation_size: float = 0.2
    test_size: float = 0.0
    random_state: int | None = None
    n_particles: int = 5
    n_iterations: int = 10
    pso_options: Mapping[str, float] = field(
        default_factory=lambda: {"c1": 1.49618, "c2": 1.49618, "w": 0.7298}
    )
    pso_topology: Literal["global", "local"] = "global"
    max_trials: int | None = None
    timeout_seconds: float | None = None
    early_stopping_rounds: int | None = None
    evaluation_protocol: EvaluationProtocol = "holdout"
    cv_folds: int = 5
    shuffle_folds: bool = True
    time_series: bool = False
    time_gap: int = 0
    stratify: bool = True
    positive_label: str | int | float | bool | None = None
    decision_threshold: float = 0.5
    refit_best: bool = True
    trial_workers: int = 1
    verbose: int = 0

    def normalized_metric(self) -> str:
        return self.metric

    def validate(self) -> None:
        if self.task not in {"regression", "binary_classification", "multiclass_classification"}:
            raise ValueError("Unsupported task.")
        metric = self.normalized_metric()
        valid_metrics = {
            "regression": {"rmse", "mae", "r2"},
            "binary_classification": {"accuracy", "roc_auc", "pr_auc", "log_loss", "f1_macro"},
            "multiclass_classification": {"accuracy", "log_loss", "f1_macro"},
        }
        if metric not in valid_metrics[self.task]:
            raise ValueError(f"Metric {metric!r} is not valid for {self.task!r}.")
        if self.strategy not in {"pso", "grid", "random"}:
            raise ValueError("strategy must be one of pso, grid, or random.")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        if self.test_size < 0 or self.test_size >= 1:
            raise ValueError("test_size must be in [0, 1).")
        if self.validation_size + self.test_size >= 1:
            raise ValueError("validation_size and test_size must leave at least one training row.")
        if self.n_particles < 1:
            raise ValueError("n_particles must be at least 1.")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be at least 1.")
        if self.max_trials is not None and self.max_trials < 1:
            raise ValueError("max_trials must be at least 1 when provided.")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided.")
        if self.early_stopping_rounds is not None and self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive when provided.")
        if self.evaluation_protocol not in {"holdout", "cross_validation"}:
            raise ValueError("evaluation_protocol must be holdout or cross_validation.")
        if not 2 <= self.cv_folds <= 20:
            raise ValueError("cv_folds must be between 2 and 20.")
        if self.time_gap < 0:
            raise ValueError("time_gap must be non-negative.")
        if not 0 <= self.decision_threshold <= 1:
            raise ValueError("decision_threshold must be between 0 and 1.")
        if self.trial_workers < 1:
            raise ValueError("trial_workers must be at least 1.")


@dataclass(frozen=True)
class EstimatorConfig:
    """Estimator name/factory plus parameters fixed across all trials."""

    name: str | None = None
    factory: Callable[..., Any] | None = None
    fixed_params: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.name is None and self.factory is None:
            raise ValueError("EstimatorConfig requires a name or a factory.")


@dataclass(frozen=True)
class ProgressEvent:
    """Structured event emitted while an optimization run executes."""

    type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }


@dataclass
class TrialResult:
    """A single evaluated parameter set."""

    trial_id: int
    params: Mapping[str, Any]
    cost: float | None
    metric: float | None
    train_metric: float | None
    status: Literal["completed", "failed"]
    duration: float
    error: str | None = None
    iteration: int | None = None
    particle_index: int | None = None
    strategy_slot: int | None = None
    worker_slot: int | None = None
    validation_metrics: Mapping[str, float | None] = field(default_factory=dict)
    train_metrics: Mapping[str, float | None] = field(default_factory=dict)
    metric_std: float | None = None
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    failure_category: str | None = None
    model: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": dict(self.params),
            "cost": self.cost,
            "metric": self.metric,
            "train_metric": self.train_metric,
            "status": self.status,
            "duration": self.duration,
            "error": self.error,
            "iteration": self.iteration,
            "particle_index": self.particle_index,
            "strategy_slot": self.strategy_slot,
            "worker_slot": self.worker_slot,
            "validation_metrics": dict(self.validation_metrics),
            "train_metrics": dict(self.train_metrics),
            "metric_std": self.metric_std,
            "fold_metrics": [dict(item) for item in self.fold_metrics],
            "failure_category": self.failure_category,
        }


@dataclass
class OptimizationResult:
    """Final result returned by the modern optimizer."""

    best_params: Mapping[str, Any] | None
    best_cost: float | None
    best_metric: float | None
    model: Any
    duration: float
    trials: list[TrialResult]
    strategy: str
    task: str
    metric_name: str
    failures: list[str] = field(default_factory=list)
    best_position: list[float] | None = None
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    experiment_id: str | None = None
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a compact human-readable summary of the optimization."""

        metric = "n/a" if self.best_metric is None else f"{self.best_metric:.6g}"
        cost = "n/a" if self.best_cost is None else f"{self.best_cost:.6g}"
        return (
            f"{self.strategy.upper()} {self.task}: {self.metric_name}={metric}, "
            f"cost={cost}, trials={len(self.trials)}, failures={len(self.failures)}, "
            f"duration={self.duration:.3f}s, best_params={dict(self.best_params or {})}"
        )

    def _repr_html_(self) -> str:
        """Render a concise result card in Jupyter-compatible notebooks."""

        from html import escape

        rows = {
            "Strategy": self.strategy,
            "Task": self.task,
            "Metric": self.metric_name,
            "Best metric": self.best_metric,
            "Optimization cost": self.best_cost,
            "Trials": len(self.trials),
            "Failures": len(self.failures),
            "Duration (seconds)": round(self.duration, 3),
            "Best parameters": dict(self.best_params or {}),
        }
        body = "".join(
            f"<tr><th style='text-align:left;padding:4px 12px 4px 0'>{escape(str(key))}</th>"
            f"<td>{escape(str(value))}</td></tr>"
            for key, value in rows.items()
        )
        return f"<div><strong>PSPSO optimization result</strong><table>{body}</table></div>"

    def trials_frame(self):
        """Return trial results as a pandas DataFrame."""

        import pandas as pd

        rows = []
        for trial in self.trials:
            row = trial.to_dict()
            params = row.pop("params")
            row.update({f"param_{name}": value for name, value in params.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def predict(self, X: Any) -> Any:
        """Predict with the fitted best model."""

        if self.model is None:
            raise RuntimeError("No successful fitted model is available.")
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Return probabilities when the fitted best model supports them."""

        if self.model is None:
            raise RuntimeError("No successful fitted model is available.")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("The fitted best model does not support predict_proba().")
        return self.model.predict_proba(X)

    def evaluate(self, X: Any, y: Any) -> dict[str, Any]:
        """Evaluate the fitted best model with the run's task and metric."""

        if self.model is None:
            raise RuntimeError("No successful fitted model is available.")
        from .metrics import evaluation_report

        return evaluation_report(
            self.model,
            self.task,
            self.metric_name,
            X,
            y,
            positive_label=self.optimizer_state.get("positive_label"),
            decision_threshold=self.optimizer_state.get("decision_threshold", 0.5),
        )

    def to_dict(self, include_model: bool = False) -> dict[str, Any]:
        data = {
            "best_params": dict(self.best_params) if self.best_params else None,
            "best_cost": self.best_cost,
            "best_metric": self.best_metric,
            "duration": self.duration,
            "trials": [trial.to_dict() for trial in self.trials],
            "strategy": self.strategy,
            "task": self.task,
            "metric_name": self.metric_name,
            "failures": list(self.failures),
            "best_position": self.best_position,
            "optimizer_state": dict(self.optimizer_state),
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "artifacts": dict(self.artifacts),
        }
        if include_model:
            data["model"] = self.model
        return data

    def trials_as_rows(self) -> list[dict[str, Any]]:
        """Return JSON-serializable trial dictionaries."""

        return [trial.to_dict() for trial in self.trials]


ProgressCallback = Callable[[ProgressEvent], None]


__all__ = [
    "EstimatorConfig",
    "EvaluationProtocol",
    "MetricName",
    "OptimizationConfig",
    "OptimizationResult",
    "ProgressCallback",
    "ProgressEvent",
    "StrategyName",
    "TaskName",
    "TrialResult",
]

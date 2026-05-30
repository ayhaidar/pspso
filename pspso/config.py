"""Configuration and result objects for pspso."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping


TaskName = Literal["regression", "binary classification"]
MetricName = Literal["rmse", "accuracy", "acc", "roc_auc", "auc"]
StrategyName = Literal["pso", "grid", "random"]


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
    verbose: int = 0

    def normalized_metric(self) -> str:
        if self.metric == "acc":
            return "accuracy"
        if self.metric == "auc":
            return "roc_auc"
        return self.metric

    def validate(self) -> None:
        if self.task not in {"regression", "binary classification"}:
            raise ValueError("task must be 'regression' or 'binary classification'.")
        metric = self.normalized_metric()
        if self.task == "regression" and metric != "rmse":
            raise ValueError("regression currently supports only the rmse metric.")
        if self.task == "binary classification" and metric not in {"accuracy", "roc_auc"}:
            raise ValueError(
                "binary classification supports only accuracy or roc_auc metrics."
            )
        if self.strategy not in {"pso", "grid", "random"}:
            raise ValueError("strategy must be one of pso, grid, or random.")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        if self.test_size < 0 or self.test_size >= 1:
            raise ValueError("test_size must be in [0, 1).")
        if self.n_particles < 1:
            raise ValueError("n_particles must be at least 1.")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be at least 1.")
        if self.max_trials is not None and self.max_trials < 1:
            raise ValueError("max_trials must be at least 1 when provided.")


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
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    optimizer_state: Mapping[str, Any] = field(default_factory=dict)

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
        }
        if include_model:
            data["model"] = self.model
        return data

    def trials_as_rows(self) -> list[dict[str, Any]]:
        return [trial.to_dict() for trial in self.trials]


ProgressCallback = Callable[[ProgressEvent], None]


__all__ = [
    "EstimatorConfig",
    "MetricName",
    "OptimizationConfig",
    "OptimizationResult",
    "ProgressCallback",
    "ProgressEvent",
    "StrategyName",
    "TaskName",
    "TrialResult",
]

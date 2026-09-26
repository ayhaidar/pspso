"""Notebook-friendly entry points for modern PSPSO optimization."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

from .config import (
    EstimatorConfig,
    OptimizationConfig,
    OptimizationResult,
    ProgressCallback,
    ProgressEvent,
)
from .dashboard.artifacts import ArtifactStore
from .dashboard.tracking import TrackingRepository
from .optimizer import OptimizationCancelled, OptimizationTimedOut, PSPSOOptimizer
from .search_space import ParameterSpec, SearchSpace


@dataclass(frozen=True)
class TrackingConfig:
    """Opt-in persistence settings for an in-process Python optimization.

    Args:
        workspace: Directory containing the 1.0 tracking database and artifacts.
        experiment_name: Name used when creating a new experiment.
        experiment_id: Existing experiment to receive the run.
        run_name: Optional human-readable label stored in the run specification.
        tags: Tags applied when a new experiment is created.
        snapshot_data: Save the supplied training and validation objects with the run.
    """

    workspace: str | Path | None = None
    experiment_name: str = "Notebook experiment"
    experiment_id: str | None = None
    run_name: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    snapshot_data: bool = False


def optimize(
    X: Any,
    y: Any,
    *,
    estimator: str | EstimatorConfig | Callable[..., Any] = "svm",
    search_space: SearchSpace | Mapping[str, ParameterSpec] | None = None,
    config: OptimizationConfig | None = None,
    X_validation: Any | None = None,
    y_validation: Any | None = None,
    progress_callback: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    tracking: TrackingConfig | None = None,
) -> OptimizationResult:
    """Optimize an estimator directly from Python or a notebook.

    Tracking is disabled unless ``tracking`` is supplied. Tracked calls write to
    the same versioned workspace used by the dashboard and CLI, but execute in
    the current Python process and do not require FastAPI.
    """

    optimizer = PSPSOOptimizer(estimator, search_space, config)
    if tracking is None:
        return optimizer.optimize(
            X,
            y,
            X_validation,
            y_validation,
            progress_callback,
            should_cancel=should_cancel,
        )

    repository = TrackingRepository(_tracking_database(tracking.workspace))
    experiment = _resolve_experiment(repository, tracking)
    spec = _python_run_spec(optimizer, X, y, X_validation, y_validation, tracking)
    snapshot = repository.create_run(spec, experiment_id=experiment["experiment_id"])
    run_id = snapshot["run_id"]
    attempt = repository.create_attempt(run_id)
    repository.set_attempt_running(attempt["attempt_id"], os.getpid())

    def tracked_callback(event: ProgressEvent) -> None:
        repository.record_event(run_id, event.to_dict())
        if progress_callback is not None:
            progress_callback(event)

    try:
        result = optimizer.optimize(
            X,
            y,
            X_validation,
            y_validation,
            tracked_callback,
            should_cancel=should_cancel,
            defer_terminal_event=True,
        )
        result.run_id = run_id
        result.experiment_id = experiment["experiment_id"]
        artifact_store = ArtifactStore(repository.workspace)
        artifacts = artifact_store.save_run(
            run_id,
            spec,
            result,
            provenance={
                "dataset_fingerprint": spec["dataset"]["fingerprint"],
                "random_seed": optimizer.config.random_state,
                "seeds": {"optimization": optimizer.config.random_state},
            },
        )
        for warning in artifact_store.last_warnings:
            tracked_callback(
                ProgressEvent(
                    "run_warning",
                    {
                        "reason": "artifact_serialization",
                        "message": warning,
                    },
                )
            )
        if tracking.snapshot_data:
            data_path = artifact_store.run_path(run_id) / "dataset.joblib"
            joblib.dump(
                {
                    "X": X,
                    "y": y,
                    "X_validation": X_validation,
                    "y_validation": y_validation,
                },
                data_path,
            )
            artifacts["dataset_snapshot"] = str(data_path)
            manifest = artifact_store.run_path(run_id) / "manifest.json"
            manifest.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
        result.artifacts = artifacts
        artifact_store.save_result(run_id, result)
        repository.save_result(run_id, result)
        repository.save_artifacts(run_id, artifacts)
        tracked_callback(optimizer.terminal_event)
        status = "completed" if optimizer.terminal_event.type == "run_completed" else "failed"
        repository.finish_attempt(attempt["attempt_id"], status)
        return result
    except (Exception, KeyboardInterrupt) as exc:
        repository.save_error(run_id, str(exc))
        status = (
            "cancelled" if isinstance(exc, (OptimizationCancelled, KeyboardInterrupt)) else "failed"
        )
        category = (
            "cancellation"
            if status == "cancelled"
            else "timeout"
            if isinstance(exc, OptimizationTimedOut)
            else optimizer._failure_category(exc)
            if isinstance(exc, Exception)
            else "cancellation"
        )
        repository.record_event(
            run_id,
            ProgressEvent(f"run_{status}", {"error": str(exc), "reason": category}).to_dict(),
        )
        repository.finish_attempt(attempt["attempt_id"], status)
        raise


def _tracking_database(workspace: str | Path | None) -> Path | None:
    if workspace is None:
        return None
    return Path(workspace) / "tracking.sqlite3"


def _resolve_experiment(
    repository: TrackingRepository,
    tracking: TrackingConfig,
) -> dict[str, Any]:
    if tracking.experiment_id is not None:
        experiment = repository.get_experiment(tracking.experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment {tracking.experiment_id!r} was not found.")
        return experiment
    return repository.create_experiment(
        tracking.experiment_name,
        tags=list(tracking.tags),
    )


def _python_run_spec(
    optimizer: PSPSOOptimizer,
    X: Any,
    y: Any,
    X_validation: Any | None,
    y_validation: Any | None,
    tracking: TrackingConfig,
) -> dict[str, Any]:
    estimator = optimizer.estimator_config
    estimator_name = estimator.name or getattr(
        estimator.factory, "__name__", repr(estimator.factory)
    )
    return {
        "schema_version": 1,
        "source": "python",
        "run_name": tracking.run_name,
        "dataset": {
            "fingerprint": joblib.hash((X, y, X_validation, y_validation)),
            "training_shape": _shape(X),
            "validation_shape": _shape(X_validation) if X_validation is not None else None,
            "snapshot_data": tracking.snapshot_data,
        },
        "task": optimizer.config.task,
        "metric": optimizer.config.metric,
        "estimator": estimator_name,
        "fixed_params": dict(estimator.fixed_params),
        "search_space": optimizer.search_space.to_schema(),
        "strategy": optimizer.config.strategy,
        "optimization": asdict(optimizer.config),
    }


def _shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    return [int(item) for item in shape] if shape is not None else [len(value)]


__all__ = ["TrackingConfig", "optimize"]

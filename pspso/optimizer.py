"""Modern optimizer core for PSPSO."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from queue import SimpleQueue
from threading import Lock
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    EstimatorConfig,
    OptimizationConfig,
    OptimizationResult,
    ProgressCallback,
    ProgressEvent,
    TrialResult,
)
from .ensemble import CrossValidationEnsemble
from .estimators import create_estimator, default_fixed_params, default_search_space
from .metrics import evaluation_report
from .search_space import ParameterSpec, SearchSpace, count_grid


class OptimizationCancelled(RuntimeError):
    """Raised when a managed run is cancelled between trials."""


class OptimizationTimedOut(RuntimeError):
    """Raised when a managed run exceeds its configured wall-clock budget."""


class PSPSOOptimizer:
    """Hyperparameter optimizer with PSO, grid, and random strategies."""

    def __init__(
        self,
        estimator: str | EstimatorConfig | Callable[..., Any] = "svm",
        search_space: SearchSpace | Mapping[str, ParameterSpec] | None = None,
        config: OptimizationConfig | None = None,
    ) -> None:
        self.config = config or OptimizationConfig()
        if self.config.random_state is None:
            self.config = replace(self.config, random_state=secrets.randbits(32))
        self.config.validate()
        if isinstance(estimator, EstimatorConfig):
            self.estimator_config = estimator
        elif isinstance(estimator, str):
            self.estimator_config = EstimatorConfig(
                name=estimator,
                fixed_params=default_fixed_params(estimator, self.config.task),
            )
        else:
            self.estimator_config = EstimatorConfig(factory=estimator)
        self.estimator_config.validate()
        if search_space is None:
            if self.estimator_config.name is None:
                raise ValueError("search_space is required for custom estimator factories.")
            search_space = default_search_space(self.estimator_config.name, self.config.task)
        self.search_space = (
            search_space if isinstance(search_space, SearchSpace) else SearchSpace(search_space)
        )
        self.result: OptimizationResult | None = None
        self.best_params: dict[str, Any] | None = None
        self.best_position: list[float] | None = None
        self.best_cost: float | None = None
        self.best_metric: float | None = None
        self.best_model: Any = None
        self.trials: list[TrialResult] = []
        self.failures: list[str] = []
        self.optimizer_state: dict[str, Any] = {}
        self._state_lock = Lock()
        self._next_trial_id = 1
        self._best_trial: TrialResult | None = None
        self.prepared_bundle: dict[str, Any] | None = None
        self.dataset_frame: Any = None
        self._worker_slots: SimpleQueue[int] = SimpleQueue()

    def optimize(
        self,
        X: Any,
        y: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
        progress_callback: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
        defer_terminal_event: bool = False,
    ) -> OptimizationResult:
        """Run the configured hyperparameter optimization."""

        self._reset()
        X_train, X_validation, y_train, y_validation = self._prepare_data(X, y, X_val, y_val)
        start = time.time()
        self._started_at = start
        self._should_cancel = should_cancel
        self._best_trial_index = 0
        planned_trials = self._planned_trials()
        candidate_fits = (planned_trials or 0) * (
            self.config.cv_folds if self.config.evaluation_protocol == "cross_validation" else 1
        )
        self.optimizer_state["planned_fits"] = candidate_fits + int(
            self.config.refit_best
            and (
                self.config.evaluation_protocol == "cross_validation"
                or self.prepared_bundle is not None
            )
        )
        self._emit(
            progress_callback,
            "run_started",
            {
                "strategy": self.config.strategy,
                "task": self.config.task,
                "metric": self.config.normalized_metric(),
                "dimensions": self.search_space.dimensions,
                "planned_trials": planned_trials,
                "planned_fits": self.optimizer_state["planned_fits"],
                "planned_candidate_fits": candidate_fits,
                "evaluation_protocol": self.config.evaluation_protocol,
                "cv_folds": self.config.cv_folds,
                "trial_workers": self.config.trial_workers,
                "random_seed": self.config.random_state,
                "positive_label": self.config.positive_label,
                "decision_threshold": self.config.decision_threshold,
                "execution_mode": ("parallel" if self.config.trial_workers > 1 else "sequential"),
            },
        )
        if self.config.strategy == "grid":
            self._run_grid(X_train, y_train, X_validation, y_validation, progress_callback)
        elif self.config.strategy == "random":
            self._run_random(X_train, y_train, X_validation, y_validation, progress_callback)
        elif self.config.strategy == "pso":
            self._run_pso(X_train, y_train, X_validation, y_validation, progress_callback)
        else:
            raise ValueError(f"Unsupported strategy: {self.config.strategy!r}")
        refit_error = None
        if (
            self.best_params is not None
            and self.config.evaluation_protocol == "cross_validation"
            and X_validation is None
            and self.config.refit_best
        ):
            self._emit(progress_callback, "final_refit_started", {"best_params": self.best_params})
            try:
                self._check_runtime()
                self.best_model = create_estimator(
                    self.estimator_config, self.config.task, self.best_params
                )
                self._fit_model(
                    self.best_model,
                    X_train,
                    y_train,
                    progress_callback,
                    phase="final_refit",
                )
                self._emit(
                    progress_callback,
                    "final_refit_completed",
                    {"rows": int(len(y_train)), "best_params": self.best_params},
                )
            except (OptimizationCancelled, OptimizationTimedOut):
                raise
            except Exception as exc:
                refit_error = str(exc)
                self.failures.append(refit_error)
                self.best_model = None
                self._emit(
                    progress_callback,
                    "run_warning",
                    {"reason": "final_refit_failed", "error": refit_error},
                )
        duration = time.time() - start
        self.optimizer_state.update(
            {
                "evaluation_protocol": self.config.evaluation_protocol,
                "cv_folds": (
                    self.config.cv_folds
                    if self.config.evaluation_protocol == "cross_validation"
                    else 1
                ),
                "trial_workers": self.config.trial_workers,
            }
        )
        self.optimizer_state.update(
            {
                "random_state": self.config.random_state,
                "positive_label": self.config.positive_label,
                "decision_threshold": self.config.decision_threshold,
            }
        )
        self.result = OptimizationResult(
            best_params=self.best_params,
            best_cost=self.best_cost,
            best_metric=self.best_metric,
            model=self.best_model,
            duration=duration,
            trials=list(self.trials),
            strategy=self.config.strategy,
            task=self.config.task,
            metric_name=self.config.normalized_metric(),
            failures=list(self.failures),
            best_position=self.best_position,
            optimizer_state=dict(self.optimizer_state),
        )
        event_type = (
            "run_completed"
            if self.best_params is not None
            and (not self.config.refit_best or self.best_model is not None)
            else "run_failed"
        )
        terminal_payload = {
            "duration": duration,
            "best_cost": self.best_cost,
            "best_metric": self.best_metric,
            "best_params": self.best_params,
            "n_trials": len(self.trials),
            "n_failures": len(self.failures),
            "refit_error": refit_error,
        }
        self.terminal_event = ProgressEvent(event_type, terminal_payload)
        if not defer_terminal_event:
            self._emit(progress_callback, event_type, terminal_payload)
        return self.result

    def _planned_trials(self) -> int | None:
        """Return the exact fit budget when it is known before execution."""

        if self.config.strategy == "pso":
            planned = self.config.n_particles * self.config.n_iterations
            if self.config.max_trials is not None:
                return min(planned, self.config.max_trials)
            return planned
        if self.config.strategy == "random":
            return self.config.max_trials or 20
        planned = count_grid(self.search_space)
        return (
            min(planned, self.config.max_trials) if self.config.max_trials is not None else planned
        )

    def _reset(self) -> None:
        self.result = None
        self.best_params = None
        self.best_position = None
        self.best_cost = None
        self.best_metric = None
        self.best_model = None
        self.trials = []
        self.failures = []
        self.optimizer_state = {}
        self._next_trial_id = 1
        self._best_trial = None
        self._worker_slots = SimpleQueue()
        for slot in range(self.config.trial_workers):
            self._worker_slots.put(slot)
        self.optimizer_state.update({"completed_fits": 0, "failed_fits": 0, "started_fits": 0})

    def _prepare_data(
        self,
        X: Any,
        y: Any,
        X_val: Any | None,
        y_val: Any | None,
    ) -> tuple[Any, Any | None, Any, Any | None]:
        if X_val is not None and y_val is not None:
            return X, X_val, np.asarray(y).ravel(), np.asarray(y_val).ravel()
        if self.config.evaluation_protocol == "cross_validation":
            return X, None, np.asarray(y).ravel(), None
        if self.config.time_series:
            boundary = len(y) - int(np.ceil(len(y) * self.config.validation_size))
            if boundary - self.config.time_gap < 1:
                raise ValueError("The validation share and time gap leave no training rows.")
            return (
                self._take(X, np.arange(boundary - self.config.time_gap)),
                self._take(X, np.arange(boundary, len(y))),
                np.asarray(y).ravel()[: boundary - self.config.time_gap],
                np.asarray(y).ravel()[boundary:],
            )
        stratify = (
            y
            if self.config.stratify
            and self.config.task in {"binary_classification", "multiclass_classification"}
            else None
        )
        return train_test_split(
            X,
            np.asarray(y).ravel(),
            test_size=self.config.validation_size,
            random_state=self.config.random_state,
            stratify=stratify,
        )

    def _run_grid(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None,
        y_val: Any | None,
        callback: ProgressCallback | None,
    ) -> None:
        candidates: list[tuple[Sequence[float], Mapping[str, Any], Mapping[str, int | None]]] = []
        for grid_index, encoded in enumerate(self.search_space.iter_grid_encoded()):
            if self.config.max_trials is not None and len(candidates) >= self.config.max_trials:
                break
            candidates.append(
                (encoded, self.search_space.decode(encoded), {"strategy_slot": grid_index})
            )
        for trial, candidate_position in self._evaluate_candidates(
            candidates, X_train, y_train, X_val, y_val, callback
        ):
            self._consider_best(trial, candidate_position, callback)
            if self._should_stop_early():
                self._emit(
                    callback,
                    "run_warning",
                    {"reason": "early_stopping", "trial_id": trial.trial_id},
                )
                break

    def _run_random(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None,
        y_val: Any | None,
        callback: ProgressCallback | None,
    ) -> None:
        rng = np.random.default_rng(self.config.random_state)
        attempts = self.config.max_trials or 20
        candidates: list[tuple[Sequence[float], Mapping[str, Any], Mapping[str, int | None]]] = []
        for random_index in range(attempts):
            encoded = self.search_space.random_encoded(rng)
            candidates.append(
                (encoded, self.search_space.decode(encoded), {"strategy_slot": random_index})
            )
        for trial, candidate_position in self._evaluate_candidates(
            candidates, X_train, y_train, X_val, y_val, callback
        ):
            self._consider_best(trial, candidate_position, callback)
            if self._should_stop_early():
                self._emit(
                    callback,
                    "run_warning",
                    {"reason": "early_stopping", "trial_id": trial.trial_id},
                )
                break

    def _run_pso(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None,
        y_val: Any | None,
        callback: ProgressCallback | None,
    ) -> None:
        rng = np.random.default_rng(self.config.random_state)
        lower, upper = self.search_space.bounds
        span = np.maximum(upper - lower, 1e-12)
        positions = rng.uniform(
            lower, upper, size=(self.config.n_particles, self.search_space.dimensions)
        )
        velocities = rng.uniform(-span, span, size=positions.shape) * 0.1
        personal_positions = positions.copy()
        personal_costs = np.full(self.config.n_particles, np.inf)
        global_position = positions[0].copy()
        global_cost = np.inf
        cost_history: list[float | None] = []
        position_history: list[list[float]] = []
        c1 = float(self.config.pso_options.get("c1", 1.49618))
        c2 = float(self.config.pso_options.get("c2", 1.49618))
        w = float(self.config.pso_options.get("w", 0.7298))

        for iteration in range(self.config.n_iterations):
            remaining = (
                self.config.n_particles
                if self.config.max_trials is None
                else min(self.config.n_particles, self.config.max_trials - len(self.trials))
            )
            candidates = []
            for particle_index in range(max(0, remaining)):
                encoded = positions[particle_index].tolist()
                candidates.append(
                    (
                        encoded,
                        self.search_space.decode(encoded),
                        {
                            "iteration": iteration,
                            "particle_index": particle_index,
                            "strategy_slot": particle_index,
                        },
                    )
                )
            for trial, encoded in self._evaluate_candidates(
                candidates, X_train, y_train, X_val, y_val, callback
            ):
                particle_index = int(trial.particle_index or 0)
                self._consider_best(trial, encoded, callback)
                if trial.cost is not None and trial.cost < personal_costs[particle_index]:
                    personal_costs[particle_index] = trial.cost
                    personal_positions[particle_index] = positions[particle_index]
                if trial.cost is not None and trial.cost < global_cost:
                    global_cost = trial.cost
                    global_position = positions[particle_index].copy()
            cost_history.append(None if self.best_cost is None else float(self.best_cost))
            position_history.append(global_position.tolist())
            self._emit(
                callback,
                "iteration_completed",
                {
                    "iteration": iteration,
                    "best_cost": self.best_cost,
                    "best_metric": self.best_metric,
                    "best_params": self.best_params,
                    "n_trials": len(self.trials),
                },
            )
            if self._max_trials_reached():
                break
            if self._should_stop_early():
                self._emit(
                    callback, "run_warning", {"reason": "early_stopping", "iteration": iteration}
                )
                break
            r1 = rng.random(size=positions.shape)
            r2 = rng.random(size=positions.shape)
            social_positions = np.tile(global_position, (self.config.n_particles, 1))
            if self.config.pso_topology == "local":
                # Each particle follows the best position in its ring neighbourhood.
                for particle_index in range(self.config.n_particles):
                    neighbours = [
                        (particle_index - 1) % self.config.n_particles,
                        particle_index,
                        (particle_index + 1) % self.config.n_particles,
                    ]
                    best_neighbour = min(neighbours, key=lambda index: personal_costs[index])
                    social_positions[particle_index] = personal_positions[best_neighbour]
            velocities = (
                w * velocities
                + c1 * r1 * (personal_positions - positions)
                + c2 * r2 * (social_positions - positions)
            )
            positions = np.clip(positions + velocities, lower, upper)

        self.optimizer_state.update(
            {
                "cost_history": cost_history,
                "position_history": position_history,
                "bounds": [lower.tolist(), upper.tolist()],
                "n_particles": self.config.n_particles,
                "n_iterations": self.config.n_iterations,
                "options": dict(self.config.pso_options),
                "topology": self.config.pso_topology,
            }
        )

    def _evaluate(
        self,
        encoded: Sequence[float],
        params: Mapping[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any | None,
        y_val: Any | None,
        callback: ProgressCallback | None,
        iteration: int | None = None,
        particle_index: int | None = None,
        strategy_slot: int | None = None,
    ) -> TrialResult:
        worker_slot = self._worker_slots.get()
        try:
            return self._evaluate_in_slot(
                encoded,
                params,
                X_train,
                y_train,
                X_val,
                y_val,
                callback,
                iteration,
                particle_index,
                strategy_slot,
                worker_slot,
            )
        finally:
            self._worker_slots.put(worker_slot)

    def _evaluate_in_slot(
        self,
        encoded: Sequence[float],
        params: Mapping[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any | None,
        y_val: Any | None,
        callback: ProgressCallback | None,
        iteration: int | None = None,
        particle_index: int | None = None,
        strategy_slot: int | None = None,
        worker_slot: int = 0,
    ) -> TrialResult:
        self._check_runtime()
        with self._state_lock:
            trial_id = self._next_trial_id
            self._next_trial_id += 1
        self._emit(
            callback,
            "trial_started",
            {
                "trial_id": trial_id,
                "iteration": iteration,
                "particle_index": particle_index,
                "strategy_slot": strategy_slot,
                "worker_slot": worker_slot,
                "params": dict(params),
                "position": [float(value) for value in encoded],
            },
        )
        start = time.time()
        try:
            if X_val is None and self.config.evaluation_protocol == "cross_validation":
                trial, trial_model = self._evaluate_cross_validation(
                    trial_id,
                    params,
                    X_train,
                    y_train,
                    callback,
                    iteration,
                    worker_slot,
                    start,
                )
                trial.particle_index = particle_index
                trial.strategy_slot = strategy_slot
            else:
                model = create_estimator(self.estimator_config, self.config.task, params)
                self._fit_model(
                    model,
                    X_train,
                    y_train,
                    callback,
                    X_validation=X_val,
                    y_validation=y_val,
                    trial_id=trial_id,
                    iteration=iteration,
                    worker_slot=worker_slot,
                )
                validation_report = evaluation_report(
                    model,
                    self.config.task,
                    self.config.normalized_metric(),
                    X_val,
                    y_val,
                    positive_label=self.config.positive_label,
                    decision_threshold=self.config.decision_threshold,
                )
                train_report = evaluation_report(
                    model,
                    self.config.task,
                    self.config.normalized_metric(),
                    X_train,
                    y_train,
                    positive_label=self.config.positive_label,
                    decision_threshold=self.config.decision_threshold,
                )
                if not np.isfinite([validation_report["cost"], validation_report["value"]]).all():
                    raise FloatingPointError("Candidate evaluation produced a nonfinite metric.")
                trial = TrialResult(
                    trial_id=trial_id,
                    params=dict(params),
                    cost=validation_report["cost"],
                    metric=validation_report["value"],
                    train_metric=train_report["value"],
                    status="completed",
                    duration=time.time() - start,
                    iteration=iteration,
                    particle_index=particle_index,
                    strategy_slot=strategy_slot,
                    worker_slot=worker_slot,
                    validation_metrics=validation_report["metrics"],
                    train_metrics=train_report["metrics"],
                    model=model,
                )
                trial_model = model
        except (OptimizationCancelled, OptimizationTimedOut):
            raise
        except Exception as exc:  # pragma: no cover - exact backend errors vary.
            message = str(exc)
            category = self._failure_category(exc)
            with self._state_lock:
                self.failures.append(message)
            trial = TrialResult(
                trial_id=trial_id,
                params=dict(params),
                cost=None,
                metric=None,
                train_metric=None,
                status="failed",
                duration=time.time() - start,
                error=message,
                iteration=iteration,
                particle_index=particle_index,
                strategy_slot=strategy_slot,
                worker_slot=worker_slot,
                failure_category=category,
            )
            trial_model = None
        with self._state_lock:
            self.trials.append(trial)
        payload = trial.to_dict()
        payload["position"] = [float(value) for value in encoded]
        payload["particle_index"] = particle_index
        payload["strategy_slot"] = strategy_slot
        payload["worker_slot"] = worker_slot
        self._emit(
            callback,
            "trial_completed" if trial.status == "completed" else "trial_failed",
            payload,
        )
        if trial.status == "completed":
            trial.model = trial_model
        return trial

    def _evaluate_cross_validation(
        self,
        trial_id: int,
        params: Mapping[str, Any],
        X: Any,
        y: Any,
        callback: ProgressCallback | None,
        iteration: int | None,
        worker_slot: int,
        started_at: float,
    ) -> tuple[TrialResult, CrossValidationEnsemble]:
        target = np.asarray(y).ravel()
        splitter = (
            TimeSeriesSplit(n_splits=self.config.cv_folds, gap=self.config.time_gap)
            if self.config.time_series
            else StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=self.config.shuffle_folds,
                random_state=self.config.random_state if self.config.shuffle_folds else None,
            )
            if self.config.stratify
            and self.config.task in {"binary_classification", "multiclass_classification"}
            else KFold(
                n_splits=self.config.cv_folds,
                shuffle=self.config.shuffle_folds,
                random_state=self.config.random_state if self.config.shuffle_folds else None,
            )
        )
        fold_rows: list[dict[str, Any]] = []
        fold_models: list[Any] = []
        folds = list(splitter.split(X, target))
        with self._state_lock:
            if "fold_indices" not in self.optimizer_state:
                self.optimizer_state["fold_indices"] = [
                    {
                        "fold": fold,
                        "train": [int(value) for value in train_indexes],
                        "validation": [int(value) for value in validation_indexes],
                    }
                    for fold, (train_indexes, validation_indexes) in enumerate(folds, start=1)
                ]
        for fold, (train_indexes, validation_indexes) in enumerate(folds, start=1):
            self._check_runtime()
            self._emit(
                callback,
                "fold_started",
                {
                    "trial_id": trial_id,
                    "iteration": iteration,
                    "fold": fold,
                    "total_folds": self.config.cv_folds,
                    "worker_slot": worker_slot,
                },
            )
            model = create_estimator(self.estimator_config, self.config.task, params)
            X_fold_train = self._take(X, train_indexes)
            y_fold_train = target[train_indexes]
            X_fold_validation = self._take(X, validation_indexes)
            y_fold_validation = target[validation_indexes]
            self._fit_model(
                model,
                X_fold_train,
                y_fold_train,
                callback,
                X_validation=X_fold_validation,
                y_validation=y_fold_validation,
                trial_id=trial_id,
                fold=fold,
                iteration=iteration,
                worker_slot=worker_slot,
            )
            validation = evaluation_report(
                model,
                self.config.task,
                self.config.normalized_metric(),
                X_fold_validation,
                y_fold_validation,
                positive_label=self.config.positive_label,
                decision_threshold=self.config.decision_threshold,
            )
            train = evaluation_report(
                model,
                self.config.task,
                self.config.normalized_metric(),
                X_fold_train,
                y_fold_train,
                positive_label=self.config.positive_label,
                decision_threshold=self.config.decision_threshold,
            )
            if not np.isfinite([validation["cost"], validation["value"]]).all():
                raise FloatingPointError("Fold evaluation produced a nonfinite metric.")
            row = {
                "fold": fold,
                "metric": float(validation["value"]),
                "cost": float(validation["cost"]),
                "train_metric": float(train["value"]),
                "validation_metrics": validation["metrics"],
                "train_metrics": train["metrics"],
            }
            fold_rows.append(row)
            fold_models.append(model)
            self._emit(
                callback,
                "fold_completed",
                {
                    "trial_id": trial_id,
                    "iteration": iteration,
                    "fold": fold,
                    "total_folds": self.config.cv_folds,
                    "worker_slot": worker_slot,
                    "metric": row["metric"],
                    "cost": row["cost"],
                },
            )
        metrics = np.asarray([row["metric"] for row in fold_rows], dtype=float)
        costs = np.asarray([row["cost"] for row in fold_rows], dtype=float)
        train_metrics = np.asarray([row["train_metric"] for row in fold_rows], dtype=float)
        return (
            TrialResult(
                trial_id=trial_id,
                params=dict(params),
                cost=float(np.mean(costs)),
                metric=float(np.mean(metrics)),
                train_metric=float(np.mean(train_metrics)),
                status="completed",
                duration=time.time() - started_at,
                iteration=iteration,
                worker_slot=worker_slot,
                validation_metrics={self.config.normalized_metric(): float(np.mean(metrics))},
                train_metrics={self.config.normalized_metric(): float(np.mean(train_metrics))},
                metric_std=float(np.std(metrics)),
                fold_metrics=list(fold_rows),
            ),
            CrossValidationEnsemble(fold_models, self.config.task),
        )

    def _fit_model(
        self,
        model: Any,
        X: Any,
        y: Any,
        callback: ProgressCallback | None,
        *,
        X_validation: Any = None,
        y_validation: Any = None,
        trial_id: int | None = None,
        fold: int | None = None,
        iteration: int | None = None,
        worker_slot: int | None = None,
        phase: str = "candidate",
    ) -> None:
        self._check_runtime()
        with self._state_lock:
            self.optimizer_state["started_fits"] += 1
            fit_id = self.optimizer_state["started_fits"]
        context = {
            "fit_id": fit_id,
            "trial_id": trial_id,
            "fold": fold,
            "iteration": iteration,
            "worker_slot": worker_slot,
            "phase": phase,
            "total_fits": self.optimizer_state["planned_fits"],
        }
        self._emit(callback, "model_fit_started", context)
        estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
        try:
            if hasattr(estimator, "set_progress_callback"):

                def on_epoch(payload: Mapping[str, Any]) -> None:
                    self._check_runtime()
                    self._emit(callback, "training_epoch", {**context, **payload})

                estimator.set_progress_callback(on_epoch)
                train = X
                validation = X_validation
                if isinstance(model, Pipeline):
                    preprocessing = model[:-1]
                    train = preprocessing.fit_transform(X, y)
                    if validation is not None:
                        validation = preprocessing.transform(validation)
                if validation is not None and hasattr(estimator, "set_validation_data"):
                    estimator.set_validation_data(validation, y_validation)
                estimator.fit(train, y)
            else:
                model.fit(X, y)
            self._check_runtime()
        except Exception:
            with self._state_lock:
                self.optimizer_state["failed_fits"] += 1
                counts = {
                    key: self.optimizer_state[key] for key in ("completed_fits", "failed_fits")
                }
            self._emit(callback, "model_fit_failed", {**context, **counts})
            raise
        finally:
            if hasattr(estimator, "set_progress_callback"):
                estimator.set_progress_callback(None)
            if hasattr(estimator, "_validation_data"):
                estimator._validation_data = None
        with self._state_lock:
            self.optimizer_state["completed_fits"] += 1
            counts = {key: self.optimizer_state[key] for key in ("completed_fits", "failed_fits")}
        self._emit(callback, "model_fit_completed", {**context, **counts})

    def _evaluate_candidates(
        self,
        candidates: Sequence[tuple[Sequence[float], Mapping[str, Any], Mapping[str, int | None]]],
        X_train: Any,
        y_train: Any,
        X_val: Any | None,
        y_val: Any | None,
        callback: ProgressCallback | None,
    ) -> list[tuple[TrialResult, Sequence[float]]]:
        self._check_runtime()
        if self.config.trial_workers == 1 or len(candidates) <= 1:
            return [
                (
                    self._evaluate(
                        encoded,
                        params,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        callback,
                        **metadata,
                    ),
                    encoded,
                )
                for encoded, params, metadata in candidates
            ]
        completed: list[tuple[TrialResult, Sequence[float]]] = []
        with ThreadPoolExecutor(max_workers=self.config.trial_workers) as executor:
            futures = {
                executor.submit(
                    self._evaluate,
                    encoded,
                    params,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    callback,
                    **metadata,
                ): encoded
                for encoded, params, metadata in candidates
            }
            for future in as_completed(futures):
                completed.append((future.result(), futures[future]))
        return completed

    @staticmethod
    def _take(values: Any, indexes: np.ndarray) -> Any:
        if hasattr(values, "iloc"):
            return values.iloc[indexes]
        return values[indexes]

    @staticmethod
    def _failure_category(exc: Exception) -> str:
        if isinstance(exc, OptimizationCancelled):
            return "cancellation"
        if isinstance(exc, (OptimizationTimedOut, TimeoutError)):
            return "timeout"
        if isinstance(exc, ImportError):
            return "missing_dependency"
        message = str(exc).lower()
        if isinstance(exc, (FloatingPointError, OverflowError)) or any(
            word in message for word in ("infinity", "infinite", "overflow", "nan", "singular")
        ):
            return "numerical_failure"
        if any(
            word in message
            for word in ("shape", "samples", "classes", "target", "features", "input contains")
        ):
            return "incompatible_data"
        if isinstance(exc, (TypeError, ValueError)):
            return "invalid_parameters"
        return "training_failure"

    def _consider_best(
        self,
        trial: TrialResult,
        encoded: Sequence[float],
        callback: ProgressCallback | None,
    ) -> None:
        if trial.cost is None:
            trial.model = None
            return
        if self.best_cost is None or trial.cost < self.best_cost:
            if self._best_trial is not None and self._best_trial is not trial:
                self._best_trial.model = None
            self.best_cost = trial.cost
            self.best_metric = trial.metric
            self.best_params = dict(trial.params)
            self.best_position = [float(value) for value in encoded]
            self.best_model = trial.model
            self._best_trial = trial
            self.optimizer_state["best_trial_id"] = trial.trial_id
            self._best_trial_index = len(self.trials)
            self._emit(
                callback,
                "best_updated",
                {
                    "trial_id": trial.trial_id,
                    "best_cost": self.best_cost,
                    "best_metric": self.best_metric,
                    "best_params": self.best_params,
                    "best_position": self.best_position,
                },
            )
        else:
            trial.model = None

    def _max_trials_reached(self) -> bool:
        return self.config.max_trials is not None and len(self.trials) >= self.config.max_trials

    def _check_runtime(self) -> None:
        should_cancel = getattr(self, "_should_cancel", None)
        if should_cancel is not None and should_cancel():
            raise OptimizationCancelled("Run was cancelled by the user.")
        timeout = self.config.timeout_seconds
        if (
            timeout is not None
            and time.time() - getattr(self, "_started_at", time.time()) >= timeout
        ):
            raise OptimizationTimedOut(f"Run exceeded its {timeout:g} second timeout.")

    def _should_stop_early(self) -> bool:
        rounds = self.config.early_stopping_rounds
        return (
            rounds is not None
            and self.best_params is not None
            and len(self.trials) - self._best_trial_index >= rounds
        )

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        if callback is not None:
            callback(ProgressEvent(event_type, payload))


__all__ = ["OptimizationCancelled", "OptimizationTimedOut", "PSPSOOptimizer"]

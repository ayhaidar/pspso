"""Modern optimizer core for PSPSO."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from .config import (
    EstimatorConfig,
    OptimizationConfig,
    OptimizationResult,
    ProgressCallback,
    ProgressEvent,
    TrialResult,
)
from .estimators import create_estimator, default_fixed_params, default_search_space
from .metrics import evaluate_model
from .search_space import SearchSpace, coerce_search_space


class PSPSOOptimizer:
    """Hyperparameter optimizer with PSO, grid, and random strategies."""

    def __init__(
        self,
        estimator: str | EstimatorConfig | Callable[..., Any] = "svm",
        search_space: SearchSpace | Mapping[str, Any] | None = None,
        config: OptimizationConfig | None = None,
    ) -> None:
        self.config = config or OptimizationConfig()
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
        self.search_space = coerce_search_space(search_space)
        self.result: OptimizationResult | None = None
        self.best_params: dict[str, Any] | None = None
        self.best_position: list[float] | None = None
        self.best_cost: float | None = None
        self.best_metric: float | None = None
        self.best_model: Any = None
        self.trials: list[TrialResult] = []
        self.failures: list[str] = []
        self.optimizer_state: dict[str, Any] = {}

    def optimize(
        self,
        X: Any,
        y: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> OptimizationResult:
        """Run the configured hyperparameter optimization."""

        self._reset()
        X_train, X_validation, y_train, y_validation = self._prepare_data(X, y, X_val, y_val)
        start = time.time()
        self._emit(
            progress_callback,
            "run_started",
            {
                "strategy": self.config.strategy,
                "task": self.config.task,
                "metric": self.config.normalized_metric(),
                "dimensions": self.search_space.dimensions,
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
        duration = time.time() - start
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
        event_type = "run_completed" if self.best_params is not None else "run_failed"
        self._emit(
            progress_callback,
            event_type,
            {
                "duration": duration,
                "best_cost": self.best_cost,
                "best_metric": self.best_metric,
                "best_params": self.best_params,
                "n_trials": len(self.trials),
                "n_failures": len(self.failures),
            },
        )
        return self.result

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

    def _prepare_data(
        self,
        X: Any,
        y: Any,
        X_val: Any | None,
        y_val: Any | None,
    ) -> tuple[Any, Any, Any, Any]:
        if X_val is not None and y_val is not None:
            return X, X_val, np.asarray(y).ravel(), np.asarray(y_val).ravel()
        stratify = y if self.config.task == "binary classification" else None
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
        X_val: Any,
        y_val: Any,
        callback: ProgressCallback | None,
    ) -> None:
        for grid_index, encoded in enumerate(self.search_space.iter_grid_encoded()):
            if self._max_trials_reached():
                break
            params = self.search_space.decode(encoded)
            trial = self._evaluate(
                encoded,
                params,
                X_train,
                y_train,
                X_val,
                y_val,
                callback,
                strategy_slot=grid_index,
            )
            self._consider_best(trial, encoded, callback)

    def _run_random(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        callback: ProgressCallback | None,
    ) -> None:
        rng = np.random.default_rng(self.config.random_state)
        attempts = self.config.max_trials or 20
        for random_index in range(attempts):
            encoded = self.search_space.random_encoded(rng)
            params = self.search_space.decode(encoded)
            trial = self._evaluate(
                encoded,
                params,
                X_train,
                y_train,
                X_val,
                y_val,
                callback,
                strategy_slot=random_index,
            )
            self._consider_best(trial, encoded, callback)

    def _run_pso(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        callback: ProgressCallback | None,
    ) -> None:
        rng = np.random.default_rng(self.config.random_state)
        lower, upper = self.search_space.bounds
        span = np.maximum(upper - lower, 1e-12)
        positions = rng.uniform(lower, upper, size=(self.config.n_particles, self.search_space.dimensions))
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
            for particle_index in range(self.config.n_particles):
                if self._max_trials_reached():
                    break
                encoded = positions[particle_index].tolist()
                params = self.search_space.decode(encoded)
                trial = self._evaluate(
                    encoded,
                    params,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    callback,
                    iteration=iteration,
                    particle_index=particle_index,
                    strategy_slot=particle_index,
                )
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
            r1 = rng.random(size=positions.shape)
            r2 = rng.random(size=positions.shape)
            velocities = (
                w * velocities
                + c1 * r1 * (personal_positions - positions)
                + c2 * r2 * (global_position - positions)
            )
            positions = np.clip(positions + velocities, lower, upper)

        self.optimizer_state = {
            "cost_history": cost_history,
            "position_history": position_history,
            "bounds": [lower.tolist(), upper.tolist()],
            "n_particles": self.config.n_particles,
            "n_iterations": self.config.n_iterations,
            "options": dict(self.config.pso_options),
            "topology": self.config.pso_topology,
        }

    def _evaluate(
        self,
        encoded: Sequence[float],
        params: Mapping[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        callback: ProgressCallback | None,
        iteration: int | None = None,
        particle_index: int | None = None,
        strategy_slot: int | None = None,
    ) -> TrialResult:
        trial_id = len(self.trials) + 1
        self._emit(
            callback,
            "trial_started",
            {
                "trial_id": trial_id,
                "iteration": iteration,
                "particle_index": particle_index,
                "strategy_slot": strategy_slot,
                "params": dict(params),
                "position": [float(value) for value in encoded],
            },
        )
        start = time.time()
        try:
            model = create_estimator(self.estimator_config, self.config.task, params)
            model.fit(X_train, y_train)
            cost, metric = evaluate_model(
                model,
                self.config.task,
                self.config.normalized_metric(),
                X_val,
                y_val,
            )
            _, train_metric = evaluate_model(
                model,
                self.config.task,
                self.config.normalized_metric(),
                X_train,
                y_train,
            )
            trial = TrialResult(
                trial_id=trial_id,
                params=dict(params),
                cost=cost,
                metric=metric,
                train_metric=train_metric,
                status="completed",
                duration=time.time() - start,
                iteration=iteration,
            )
            trial_model = model
        except Exception as exc:  # pragma: no cover - exact backend errors vary.
            message = str(exc)
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
            )
            trial_model = None
        self.trials.append(trial)
        payload = trial.to_dict()
        payload["position"] = [float(value) for value in encoded]
        payload["particle_index"] = particle_index
        payload["strategy_slot"] = strategy_slot
        self._emit(
            callback,
            "trial_completed" if trial.status == "completed" else "trial_failed",
            payload,
        )
        if trial.status == "completed":
            trial._model = trial_model  # type: ignore[attr-defined]
        return trial

    def _consider_best(
        self,
        trial: TrialResult,
        encoded: Sequence[float],
        callback: ProgressCallback | None,
    ) -> None:
        if trial.cost is None:
            return
        if self.best_cost is None or trial.cost < self.best_cost:
            self.best_cost = trial.cost
            self.best_metric = trial.metric
            self.best_params = dict(trial.params)
            self.best_position = [float(value) for value in encoded]
            self.best_model = getattr(trial, "_model", None)
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

    def _max_trials_reached(self) -> bool:
        return self.config.max_trials is not None and len(self.trials) >= self.config.max_trials

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        if callback is not None:
            callback(ProgressEvent(event_type, payload))


__all__ = ["PSPSOOptimizer"]

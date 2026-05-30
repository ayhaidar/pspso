"""Backward-compatible wrapper around the modern PSPSO optimizer."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .config import EstimatorConfig, OptimizationConfig
from .estimators import default_fixed_params, default_search_space
from .metrics import cost_from_metric, metric_value
from .optimizer import PSPSOOptimizer
from .search_space import SearchSpace, count_grid


class pspso:
    """Legacy pspso API preserved for existing users.

    New projects should prefer :class:`pspso.PSPSOOptimizer`, but the original
    import and method names continue to work:

    ``from pspso import pspso``
    """

    _last_search_space: SearchSpace | None = None

    verbose = 0
    early_stopping = 20

    def __init__(
        self,
        estimator: str = "xgboost",
        params: Mapping[str, Sequence[Any]] | None = None,
        task: str = "regression",
        score: str = "rmse",
    ) -> None:
        self.estimator = estimator
        self.task = task
        self.score = score
        self.selectiontype: str | None = None
        self.cost: float | None = None
        self.met: float | None = None
        self.pos: list[float] | None = None
        self.model: Any = None
        self.duration: float | None = None
        self.optimizer: PSPSOOptimizer | None = None
        self.results: list[float | None] = []
        self.combinations: list[list[float]] = []
        self.history: Any = None
        self.miniopt: dict[str, Any] = {}

        if params is None:
            params = self.get_default_search_space(estimator, task)
        self.paramdetails = dict(params)
        self.search_space = SearchSpace.from_legacy(self.paramdetails)
        self.parameters = list(self.search_space.names)
        self.x_min = self.search_space.bounds[0].tolist()
        self.x_max = self.search_space.bounds[1].tolist()
        self.rounding = [
            0 if not hasattr(self.search_space.params[name], "precision") else self.search_space.params[name].precision
            for name in self.parameters
        ]
        self.bounds = self.search_space.bounds
        self.dimensions = self.search_space.dimensions
        self.defaultparams = self.get_default_params(estimator, task)
        pspso._last_search_space = self.search_space

    @staticmethod
    def get_default_search_space(estimator: str, task: str) -> dict[str, list[Any]]:
        return default_search_space(estimator, task)

    @staticmethod
    def get_default_params(estimator: str, task: str) -> dict[str, Any]:
        return default_fixed_params(estimator, task)

    @staticmethod
    def read_parameters(
        params: Mapping[str, Sequence[Any]] | None = None,
        estimator: str | None = None,
        task: str | None = None,
    ) -> tuple[list[str], dict[str, Any], list[float], list[float], list[int], tuple[np.ndarray, np.ndarray], int, Mapping[str, Sequence[Any]]]:
        if params is None:
            if estimator is None or task is None:
                raise ValueError("estimator and task are required when params is None.")
            params = pspso.get_default_search_space(estimator, task)
        space = SearchSpace.from_legacy(params)
        lower, upper = space.bounds
        rounding = [
            0 if not hasattr(space.params[name], "precision") else space.params[name].precision
            for name in space.names
        ]
        defaultparams = pspso.get_default_params(estimator or "svm", task or "regression")
        pspso._last_search_space = space
        return (
            list(space.names),
            defaultparams,
            lower.tolist(),
            upper.tolist(),
            rounding,
            (lower, upper),
            space.dimensions,
            params,
        )

    @staticmethod
    def decode_parameters(particle: Sequence[float]) -> dict[str, Any]:
        if pspso._last_search_space is None:
            raise ValueError("No pspso instance has initialized a search space yet.")
        return pspso._last_search_space.decode(particle)

    def decode_position(self, particle: Sequence[float]) -> dict[str, Any]:
        """Decode a position using this instance's search space."""

        return self.search_space.decode(particle)

    def fitpspso(
        self,
        X_train: Any = None,
        Y_train: Any = None,
        X_val: Any = None,
        Y_val: Any = None,
        psotype: str = "global",
        number_of_particles: int = 5,
        number_of_iterations: int = 10,
        options: Mapping[str, float] | None = None,
    ) -> tuple[list[float] | None, float | None, float | None, Any, PSPSOOptimizer]:
        self.selectiontype = "PSO"
        self.number_of_particles = number_of_particles
        self.number_of_iterations = number_of_iterations
        self.psotype = psotype
        self.options = dict(options or {"c1": 1.49618, "c2": 1.49618, "w": 0.7298})
        config = OptimizationConfig(
            task=self.task,  # type: ignore[arg-type]
            metric=self.score,  # type: ignore[arg-type]
            strategy="pso",
            n_particles=number_of_particles,
            n_iterations=number_of_iterations,
            pso_options=self.options,
            pso_topology="local" if psotype == "local" else "global",
            early_stopping_rounds=self.early_stopping,
            verbose=self.verbose,
        )
        result = self._run_modern_optimizer(X_train, Y_train, X_val, Y_val, config)
        self.number_of_attempts = len(result.trials)
        self.totalnbofcombinations = count_grid(self.search_space)
        return self.pos, self.cost, self.duration, self.model, self.optimizer  # type: ignore[return-value]

    def fitpsgrid(
        self,
        X_train: Any = None,
        Y_train: Any = None,
        X_val: Any = None,
        Y_val: Any = None,
    ) -> tuple[list[float] | None, float | None, float | None, Any, list[list[float]], list[float | None]]:
        self.selectiontype = "Grid"
        config = OptimizationConfig(
            task=self.task,  # type: ignore[arg-type]
            metric=self.score,  # type: ignore[arg-type]
            strategy="grid",
            verbose=self.verbose,
        )
        result = self._run_modern_optimizer(X_train, Y_train, X_val, Y_val, config)
        self.combinations = self.calculatecombinations()
        self.results = [trial.cost for trial in result.trials]
        self.number_of_attempts = len(result.trials)
        self.totalnbofcombinations = len(self.combinations)
        return self.pos, self.cost, self.duration, self.model, self.combinations, self.results

    def fitpsrandom(
        self,
        X_train: Any = None,
        Y_train: Any = None,
        X_val: Any = None,
        Y_val: Any = None,
        number_of_attempts: int = 20,
    ) -> tuple[list[float] | None, float | None, float | None, Any, list[list[float]], list[float | None]]:
        self.selectiontype = "Random"
        config = OptimizationConfig(
            task=self.task,  # type: ignore[arg-type]
            metric=self.score,  # type: ignore[arg-type]
            strategy="random",
            max_trials=number_of_attempts,
            verbose=self.verbose,
        )
        result = self._run_modern_optimizer(X_train, Y_train, X_val, Y_val, config)
        self.combinations = self.calculatecombinations()
        self.results = [trial.cost for trial in result.trials]
        self.number_of_attempts = len(result.trials)
        self.totalnbofcombinations = len(self.combinations)
        return self.pos, self.cost, self.duration, self.model, self.combinations, self.results

    def _run_modern_optimizer(
        self,
        X_train: Any,
        Y_train: Any,
        X_val: Any,
        Y_val: Any,
        config: OptimizationConfig,
    ):
        estimator = EstimatorConfig(name=self.estimator, fixed_params=dict(self.defaultparams))
        self.optimizer = PSPSOOptimizer(estimator, self.search_space, config)
        result = self.optimizer.optimize(X_train, Y_train, X_val, Y_val)
        self.cost = result.best_cost
        self.met = result.best_metric
        self.pos = result.best_position
        self.model = result.model
        self.duration = result.duration
        self.miniopt = dict(result.optimizer_state)
        return result

    def calculatecombinations(self) -> list[list[float]]:
        return list(self.search_space.iter_grid_encoded())

    def print_results(self) -> None:
        print(f"Estimator: {self.estimator}")
        print(f"Task: {self.task}")
        print(f"Selection type: {self.selectiontype}")
        print(f"Number of attempts:{getattr(self, 'number_of_attempts', 0)}")
        print(f"Total number of combinations: {getattr(self, 'totalnbofcombinations', 0)}")
        print("Parameters:")
        print(self.decode_position(self.pos or []))
        print(f"Global best position: {self.pos}")
        print(f"Global best cost: {None if self.cost is None else round(self.cost, 4)}")
        print(f"Time taken to find the set of parameters: {self.duration}")
        if self.selectiontype == "PSO":
            print(f"Number of particles: {self.number_of_particles}")
            print(f"Number of iterations: {self.number_of_iterations}")

    @staticmethod
    def predict(model: Any, estimator: str, task: str, score: str, X_val: Any, Y_val: Any) -> float:
        value = metric_value(model, task, score, X_val, Y_val)
        return cost_from_metric(task, score, value)


__all__ = ["pspso"]

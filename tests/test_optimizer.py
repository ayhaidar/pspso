import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pspso import Choice, FloatRange, IntRange, OptimizationConfig, PSPSOOptimizer, SearchSpace


def test_random_search_binary_classification_emits_progress_events():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X[:120], y[:120], test_size=0.25, random_state=42, stratify=y[:120]
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    events = []
    optimizer = PSPSOOptimizer(
        estimator="svm",
        search_space=SearchSpace(
            {
                "kernel": Choice(["linear"]),
                "C": FloatRange(0.1, 0.2, precision=1),
                "gamma": FloatRange(0.1, 0.2, precision=1),
            }
        ),
        config=OptimizationConfig(
            task="binary_classification",
            metric="roc_auc",
            strategy="random",
            max_trials=2,
            random_state=7,
        ),
    )

    result = optimizer.optimize(X_train, y_train, X_val, y_val, events.append)

    assert result.best_params is not None
    assert result.best_metric is not None
    assert len(result.trials) == 2
    assert {event.type for event in events} >= {
        "run_started",
        "trial_completed",
        "best_updated",
        "run_completed",
    }


def test_generic_estimator_factory_regression():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X[:90], y[:90], test_size=0.25, random_state=42
    )

    optimizer = PSPSOOptimizer(
        estimator=lambda **params: RandomForestRegressor(random_state=0, **params),
        search_space=SearchSpace({"max_depth": IntRange(2, 3), "n_estimators": IntRange(5, 6)}),
        config=OptimizationConfig(task="regression", metric="rmse", strategy="grid"),
    )

    result = optimizer.optimize(X_train, y_train, X_val, y_val)

    assert result.best_params is not None
    assert result.best_cost is not None
    assert len(result.trials) == 4


def test_pso_progress_events_include_particle_metadata():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X[:80], y[:80], test_size=0.25, random_state=42
    )
    events = []
    optimizer = PSPSOOptimizer(
        estimator="random_forest",
        search_space=SearchSpace({"max_depth": IntRange(2, 3), "n_estimators": IntRange(5, 6)}),
        config=OptimizationConfig(
            task="regression",
            metric="rmse",
            strategy="pso",
            n_particles=2,
            n_iterations=1,
            max_trials=2,
            random_state=3,
        ),
    )

    result = optimizer.optimize(X_train, y_train, X_val, y_val, events.append)

    assert result.best_params is not None
    started = [event for event in events if event.type == "trial_started"]
    completed = [event for event in events if event.type == "trial_completed"]
    assert len(started) == 2
    assert all("particle_index" in event.payload for event in started)
    assert all("strategy_slot" in event.payload for event in completed)


def test_pso_runs_every_particle_in_every_iteration_without_a_trial_cap():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X[:80], y[:80], test_size=0.25, random_state=42
    )
    events = []
    optimizer = PSPSOOptimizer(
        estimator="random_forest",
        search_space=SearchSpace({"max_depth": IntRange(2, 3), "n_estimators": IntRange(5, 6)}),
        config=OptimizationConfig(
            task="regression",
            metric="rmse",
            strategy="pso",
            n_particles=3,
            n_iterations=2,
            max_trials=None,
            random_state=3,
        ),
    )

    result = optimizer.optimize(X_train, y_train, X_val, y_val, events.append)

    started = [event for event in events if event.type == "trial_started"]
    completed = [event for event in events if event.type == "trial_completed"]
    iterations = [event for event in events if event.type == "iteration_completed"]
    run_started = next(event for event in events if event.type == "run_started")
    assert len(result.trials) == 6
    assert len(started) == 6
    assert len(completed) == 6
    assert len(iterations) == 2
    assert run_started.payload["planned_trials"] == 6
    assert run_started.payload["execution_mode"] == "sequential"


def test_cross_validation_reports_fold_aggregation_and_refits_the_best_model():
    X, y = load_breast_cancer(return_X_y=True)
    events = []
    optimizer = PSPSOOptimizer(
        estimator="random_forest",
        search_space=SearchSpace({"max_depth": IntRange(2, 2), "n_estimators": IntRange(8, 8)}),
        config=OptimizationConfig(
            task="binary_classification",
            metric="roc_auc",
            strategy="random",
            max_trials=2,
            evaluation_protocol="cross_validation",
            cv_folds=3,
            random_state=11,
            trial_workers=2,
        ),
    )

    result = optimizer.optimize(X[:180], y[:180], progress_callback=events.append)

    completed = [event for event in events if event.type == "fold_completed"]
    started = next(event for event in events if event.type == "run_started")
    assert len(completed) == 6
    assert started.payload["planned_fits"] == 7
    assert started.payload["planned_candidate_fits"] == 6
    assert result.model is not None
    assert all(len(trial.fold_metrics) == 3 for trial in result.trials)
    assert all(trial.metric_std is not None for trial in result.trials)
    assert len(result.optimizer_state["fold_indices"]) == 3
    assert {event.type for event in events} >= {"final_refit_started", "final_refit_completed"}
    for trial in result.trials:
        values = [fold["metric"] for fold in trial.fold_metrics]
        np.testing.assert_allclose(trial.metric, np.mean(values))
        np.testing.assert_allclose(trial.metric_std, np.std(values))


def test_cross_validation_fits_preprocessing_only_on_each_training_fold():
    import pandas as pd

    observed = []

    class RecordingScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            observed.append(tuple(X.index))
            return super().fit(X, y, sample_weight)

    X = pd.DataFrame({"value": np.arange(30, dtype=float) ** 3})
    y = np.arange(30, dtype=float)

    def factory(**params):
        return Pipeline([("scale", RecordingScaler()), ("model", DummyRegressor())])

    optimizer = PSPSOOptimizer(
        factory,
        SearchSpace({"dummy": Choice([1])}),
        OptimizationConfig(
            task="regression",
            metric="rmse",
            strategy="random",
            max_trials=1,
            evaluation_protocol="cross_validation",
            cv_folds=3,
            random_state=5,
        ),
    )
    result = optimizer.optimize(X, y)
    folds = result.optimizer_state["fold_indices"]
    assert [set(rows) for rows in observed[:-1]] == [set(fold["train"]) for fold in folds]
    assert all(
        set(rows).isdisjoint(fold["validation"])
        for rows, fold in zip(observed[:-1], folds, strict=True)
    )
    assert set(observed[-1]) == set(X.index)


def test_parallel_candidates_never_share_a_worker_slot_and_respect_pso_barriers():
    events = []
    X, y = load_diabetes(return_X_y=True)
    optimizer = PSPSOOptimizer(
        "random_forest",
        SearchSpace({"n_estimators": Choice([2, 3]), "max_depth": Choice([2])}),
        OptimizationConfig(
            task="regression",
            metric="rmse",
            strategy="pso",
            n_particles=5,
            n_iterations=2,
            trial_workers=2,
            evaluation_protocol="cross_validation",
            cv_folds=2,
            random_state=7,
        ),
    )
    optimizer.optimize(X[:80], y[:80], progress_callback=events.append)
    active = {}
    finished_by_iteration = {}
    for event in events:
        payload = event.payload
        if event.type == "trial_started":
            assert payload["worker_slot"] not in active
            active[payload["worker_slot"]] = payload["trial_id"]
            assert len(active) <= 2
        elif event.type in {"trial_completed", "trial_failed"}:
            assert active.pop(payload["worker_slot"]) == payload["trial_id"]
            iteration = payload["iteration"]
            finished_by_iteration[iteration] = finished_by_iteration.get(iteration, 0) + 1
        elif event.type == "iteration_completed":
            assert not active
            assert finished_by_iteration[payload["iteration"]] == 5
    assert len([event for event in events if event.type == "model_fit_completed"]) == 21

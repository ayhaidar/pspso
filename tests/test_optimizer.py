from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
            task="binary classification",
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
    assert {event.type for event in events} >= {"run_started", "trial_completed", "best_updated", "run_completed"}


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

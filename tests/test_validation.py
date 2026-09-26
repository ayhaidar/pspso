import numpy as np
import pytest

from pspso.dashboard.app import _validate_request, create_app
from pspso.dashboard.schemas import ExperimentSpec
from pspso.dashboard.validation import validate_objectives
from pspso.optimizer import OptimizationCancelled, OptimizationTimedOut, PSPSOOptimizer


@pytest.mark.parametrize(
    "estimator,objective,target",
    [
        ("xgboost", "reg:gamma", [0, 1]),
        ("xgboost", "reg:tweedie", [-1, 2]),
        ("xgboost", "count:poisson", [0, 0]),
        ("xgboost", "reg:squaredlogerror", [-1, 2]),
        ("xgboost", "reg:logistic", [0, 2]),
        ("xgboost", "binary:logistic", [1, 2]),
        ("lightgbm", "gamma", [0, 1]),
        ("lightgbm", "poisson", [-1, 2]),
        ("lightgbm", "tweedie", [0, 0]),
        ("lightgbm", "binary", [1, 2]),
        ("xgboost", "reg:squarederror", [np.inf, 2]),
    ],
)
def test_objectives_reject_incompatible_targets(estimator, objective, target):
    assert validate_objectives(estimator, "regression", "rmse", [objective], target)


def test_valid_objective_domains_and_probability_requirements():
    assert not validate_objectives(
        "xgboost", "regression", "rmse", ["reg:gamma", "reg:tweedie"], [1, 2]
    )
    assert not validate_objectives("lightgbm", "regression", "rmse", ["poisson", "tweedie"], [0, 2])
    assert validate_objectives(
        "xgboost", "binary_classification", "roc_auc", ["binary:hinge"], [0, 1]
    )
    assert not validate_objectives(
        "xgboost", "binary_classification", "accuracy", ["binary:hinge"], [0, 1]
    )


@pytest.mark.parametrize(
    "error,category",
    [
        (ValueError("Invalid parameter depth"), "invalid_parameters"),
        (ValueError("Inconsistent number of samples"), "incompatible_data"),
        (FloatingPointError("overflow"), "numerical_failure"),
        (ImportError("torch"), "missing_dependency"),
        (OptimizationCancelled("cancel"), "cancellation"),
        (OptimizationTimedOut("deadline"), "timeout"),
        (RuntimeError("unknown library failure"), "training_failure"),
    ],
)
def test_failure_categories(error, category):
    assert PSPSOOptimizer._failure_category(error) == category


@pytest.mark.parametrize("protocol", ["holdout", "cross_validation"])
def test_nonfinite_scores_are_failed_candidates(protocol):
    from sklearn.dummy import DummyRegressor

    from pspso import Choice, EstimatorConfig, OptimizationConfig

    config = OptimizationConfig(
        task="regression",
        metric="r2",
        strategy="random",
        max_trials=1,
        evaluation_protocol=protocol,
        cv_folds=2,
        refit_best=False,
    )
    optimizer = PSPSOOptimizer(
        EstimatorConfig(factory=DummyRegressor), {"strategy": Choice(["mean"])}, config
    )
    with pytest.warns(UserWarning):
        result = optimizer.optimize(np.array([[0], [1]]), np.array([0, 1]))
    assert result.trials[0].status == "failed"
    assert result.trials[0].failure_category == "numerical_failure"


def test_fold_validation_uses_development_class_counts_and_positive_label():
    spec = ExperimentSpec(
        dataset={
            "source": "csv",
            "csv_text": "x,target\n" + "\n".join(f"{i},{i % 2}" for i in range(12)),
        },
        estimator="logistic_regression",
        evaluation={"protocol": "cross_validation", "folds": 5, "test_size": 0.5},
    )
    assert _validate_request(spec)["dataset"]
    spec = ExperimentSpec(evaluation={"positive_label": "absent"})
    assert "Positive label" in " ".join(_validate_request(spec)["dataset"])


def test_missing_feature_values_are_json_null_and_evaluation_aliases_agree(tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(create_app(tmp_path / "tracking.sqlite3", start_manager=False)) as client:
        response = client.post("/api/v1/datasets/preview", json={"csv_text": "x,target\n,0\n1,1\n"})
        assert response.status_code == 200
        assert response.json()["preview"][0]["x"] is None
    spec = ExperimentSpec(
        split={"random_state": 42},
        evaluation={"random_state": 7, "stratify": False, "test_size": 0.3},
    )
    assert spec.split.random_state == 7
    assert spec.split.stratify is False
    assert spec.split.test_size == 0.3


def test_tournament_rejects_mixed_budgets_without_creating_runs(tmp_path, client_factory):
    client = client_factory(tmp_path / "tracking.sqlite3")
    experiment = client.post("/api/v1/experiments", json={"name": "Frozen comparison"}).json()
    base = {"estimator": "logistic_regression", "runtime": {"max_trials": 1}}
    result = client.post(
        f"/api/v1/experiments/{experiment['experiment_id']}/tournament",
        json={
            "runs": [base, {**base, "runtime": {"max_trials": 2}}],
        },
    )
    assert result.status_code == 400
    assert "must share" in result.json()["detail"]
    assert client.get("/api/v1/runs").json() == []


def test_tournament_freezes_one_unspecified_seed_and_cv_inspection(tmp_path, client_factory):
    client = client_factory(tmp_path / "tracking.sqlite3")
    experiment = client.post("/api/v1/experiments", json={"name": "Shared seed"}).json()
    base = {
        "estimator": "logistic_regression",
        "split": {"random_state": None},
        "runtime": {"max_trials": 1},
    }
    result = client.post(
        f"/api/v1/experiments/{experiment['experiment_id']}/tournament", json={"runs": [base, base]}
    )
    assert result.status_code == 200
    runs = result.json()["runs"]
    seed = runs[0]["request"]["split"]["random_state"]
    assert isinstance(seed, int)
    assert runs[1]["request"]["evaluation"]["random_state"] == seed
    profile = client.post(
        "/api/v1/datasets/inspect",
        json={
            "source": "example",
            "name": "diabetes",
            "task": "regression",
            "split": {"test_size": 0.2},
            "evaluation": {"protocol": "cross_validation"},
        },
    ).json()
    assert set(profile["split_summary"]["partitions"]) == {"development", "test"}

import json
import time
from pathlib import Path

import pytest
from sklearn.datasets import load_diabetes

from pspso import IntRange, OptimizationConfig, PSPSOOptimizer, SearchSpace
from pspso.optimizer import OptimizationTimedOut


def test_local_pso_topology_and_early_stopping_are_recorded():
    X, y = load_diabetes(return_X_y=True)
    optimizer = PSPSOOptimizer(
        "random_forest",
        SearchSpace({"n_estimators": IntRange(5, 5), "max_depth": IntRange(2, 2)}),
        OptimizationConfig(
            task="regression",
            metric="rmse",
            strategy="pso",
            n_particles=3,
            n_iterations=4,
            pso_topology="local",
            early_stopping_rounds=1,
            random_state=42,
        ),
    )
    result = optimizer.optimize(X, y)
    assert result.optimizer_state["topology"] == "local"
    assert len(result.trials) < 12


def test_optimizer_enforces_timeout():
    X, y = load_diabetes(return_X_y=True)
    optimizer = PSPSOOptimizer(
        "random_forest",
        SearchSpace({"n_estimators": IntRange(400, 400), "max_depth": IntRange(10, 10)}),
        OptimizationConfig(
            task="regression",
            metric="rmse",
            strategy="random",
            max_trials=2,
            timeout_seconds=0.000001,
        ),
    )
    with pytest.raises(OptimizationTimedOut):
        optimizer.optimize(X, y)


def test_saved_dataset_and_worker_artifacts_persist(tmp_path, client_factory):
    client = client_factory(tmp_path / "tracking.sqlite3")
    saved = client.post(
        "/api/v1/datasets",
        json={
            "name": "Small binary dataset",
            "csv_text": "feature,target\n1,0\n2,0\n3,1\n4,1\n5,1\n6,0\n",
        },
    )
    assert saved.status_code == 200
    dataset_id = saved.json()["dataset_id"]
    payload = {
        "dataset": {"source": "stored", "dataset_id": dataset_id, "target_column": "target"},
        "task": "binary_classification",
        "metric": "accuracy",
        "estimator": "logistic_regression",
        "fixed_params": {"max_iter": 100},
        "search_space": {"C": {"type": "float", "low": 1, "high": 1, "precision": 1}},
        "strategy": "random",
        "split": {"validation_size": 0.33, "random_state": 42, "stratify": True},
        "runtime": {"max_trials": 1},
    }
    created = client.post("/api/v1/runs", json=payload)
    assert created.status_code == 200
    run_id = created.json()["run_id"]
    for _ in range(80):
        snapshot = client.get(f"/api/v1/runs/{run_id}").json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert snapshot["status"] == "completed", snapshot
    assert snapshot["artifacts"].get("spec")
    assert snapshot["artifacts"].get("result")
    result_payload = json.loads(Path(snapshot["artifacts"]["result"]).read_text(encoding="utf-8"))
    assert result_payload["run_id"] == run_id
    assert result_payload["experiment_id"] == snapshot["experiment_id"]
    assert result_payload["artifacts"]["manifest"] == snapshot["artifacts"]["manifest"]


def test_tournament_queues_multiple_recipes_under_one_experiment(tmp_path, client_factory):
    client = client_factory(tmp_path / "tracking.sqlite3")
    experiment = client.post("/api/v1/experiments", json={"name": "Model comparison"}).json()
    base = {
        "dataset": {"source": "example", "name": "breast_cancer", "target_column": "target"},
        "task": "binary_classification",
        "metric": "accuracy",
        "strategy": "random",
        "split": {"validation_size": 0.2, "random_state": 42, "stratify": True},
        "runtime": {"max_trials": 1},
    }
    response = client.post(
        f"/api/v1/experiments/{experiment['experiment_id']}/tournament",
        json={
            "runs": [
                {**base, "estimator": "logistic_regression", "search_space": None},
                {**base, "estimator": "random_forest", "search_space": None},
            ]
        },
    )
    assert response.status_code == 200
    assert len(response.json()["runs"]) == 2

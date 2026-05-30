import importlib.util
import time

import pytest
from fastapi.testclient import TestClient

from pspso.dashboard.app import create_app


def _payload():
    return {
        "experiment_id": None,
        "dataset": {
            "source": "example",
            "name": "breast_cancer",
            "target_column": "target",
        },
        "task": "binary classification",
        "metric": "roc_auc",
        "estimator": "svm",
        "fixed_params": {},
        "search_space": {
            "kernel": {"type": "choice", "values": ["linear"]},
            "C": {"type": "float", "low": 0.1, "high": 0.2, "precision": 1},
            "gamma": {"type": "float", "low": 0.1, "high": 0.1, "precision": 1},
        },
        "strategy": "random",
        "split": {"validation_size": 0.2, "random_state": 42, "stratify": True},
        "runtime": {"max_trials": 1},
    }


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(tmp_path / "tracking.sqlite3"))


def _wait_for_terminal_run(client: TestClient, run_id: str) -> dict:
    snapshot = None
    for _ in range(50):
        snapshot = client.get(f"/api/runs/{run_id}").json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert snapshot is not None
    return snapshot


def test_dashboard_metadata_and_validation_endpoints(tmp_path):
    client = _client(tmp_path)

    datasets = client.get("/api/datasets/examples")
    estimators = client.get("/api/estimators")
    validation = client.post("/api/runs/validate", json=_payload())
    dataset_preview = client.post(
        "/api/datasets/inspect",
        json={"source": "example", "name": "breast_cancer", "target_column": "target"},
    )

    assert datasets.status_code == 200
    assert any(row["name"] == "breast_cancer" for row in datasets.json())
    assert dataset_preview.status_code == 200
    assert dataset_preview.json()["row_count"] == 569
    assert estimators.status_code == 200
    assert "svm" in estimators.json()["estimators"]
    assert "tasks" in estimators.json()
    assert "metrics" in estimators.json()
    assert validation.status_code == 200
    assert validation.json()["valid"] is True
    assert validation.json()["errors"] == {
        "dataset": [],
        "task": [],
        "estimator": [],
        "fixed_params": [],
        "search_space": [],
        "strategy": [],
    }


def test_validation_rejects_stale_search_space_for_estimator(tmp_path):
    client = _client(tmp_path)
    payload = _payload()
    payload["estimator"] = "random_forest"

    validation = client.post("/api/runs/validate", json=payload)

    assert validation.status_code == 200
    body = validation.json()
    assert body["valid"] is False
    assert "kernel" in " ".join(body["errors"]["search_space"])


def test_validation_rejects_wrong_metric_for_task(tmp_path):
    client = _client(tmp_path)
    payload = _payload()
    payload["dataset"] = {
        "source": "example",
        "name": "diabetes",
        "target_column": "target",
    }
    payload["task"] = "regression"
    payload["metric"] = "roc_auc"

    validation = client.post("/api/runs/validate", json=payload)

    assert validation.status_code == 200
    body = validation.json()
    assert body["valid"] is False
    assert "Metric" in " ".join(body["errors"]["task"])


def test_create_run_rejects_invalid_config_with_grouped_errors(tmp_path):
    client = _client(tmp_path)
    payload = _payload()
    payload["estimator"] = "random_forest"

    created = client.post("/api/runs", json=payload)

    assert created.status_code == 400
    assert created.json()["detail"]["valid"] is False
    assert "search_space" in created.json()["detail"]["errors"]


def test_dashboard_run_lifecycle_and_event_stream(tmp_path):
    client = _client(tmp_path)
    created = client.post("/api/runs", json=_payload())
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    snapshot = _wait_for_terminal_run(client, run_id)
    assert snapshot["status"] == "completed"
    result = client.get(f"/api/runs/{run_id}/result")
    assert result.status_code == 200
    assert result.json()["best_params"] is not None
    predictions = client.get(f"/api/runs/{run_id}/predictions?split=validation&limit=5")
    assert predictions.status_code == 200
    assert predictions.json()["split"] == "validation"
    assert len(predictions.json()["preview_rows"]) == 5
    history = client.get(f"/api/runs/{run_id}/history")
    assert history.status_code == 200
    assert history.json()["events"][0]["type"] == "validation_passed"
    assert any(event["type"] == "dataset_prepared" for event in history.json()["events"])

    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        text = "".join(response.iter_text())

    assert "run_started" in text
    assert "trial_completed" in text
    assert "run_completed" in text


def test_experiment_creation_and_persisted_run_history(tmp_path):
    client = _client(tmp_path)
    experiment = client.post(
        "/api/experiments",
        json={"name": "Regression experiments", "description": "Compare retries", "tags": ["local"]},
    )
    assert experiment.status_code == 200
    experiment_id = experiment.json()["experiment_id"]

    payload = _payload()
    payload["experiment_id"] = experiment_id
    created = client.post("/api/runs", json=payload)
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    snapshot = _wait_for_terminal_run(client, run_id)
    assert snapshot["experiment_id"] == experiment_id
    assert snapshot["status"] == "completed"

    history = client.get(f"/api/runs/{run_id}/history")
    assert history.status_code == 200
    sequence_numbers = [event["sequence_number"] for event in history.json()["events"]]
    assert sequence_numbers == sorted(sequence_numbers)

    detail = client.get(f"/api/experiments/{experiment_id}")
    assert detail.status_code == 200
    assert any(item["run_id"] == run_id for item in detail.json()["runs"])

    restarted = TestClient(create_app(tmp_path / "tracking.sqlite3"))
    reloaded = restarted.get(f"/api/runs/{run_id}")
    assert reloaded.status_code == 200
    assert reloaded.json()["status"] == "completed"
    reloaded_history = restarted.get(f"/api/runs/{run_id}/history")
    assert reloaded_history.status_code == 200
    assert len(reloaded_history.json()["events"]) == len(history.json()["events"])


def test_runs_without_experiment_use_ad_hoc_experiment(tmp_path):
    client = _client(tmp_path)
    created = client.post("/api/runs", json=_payload())
    assert created.status_code == 200
    run = created.json()
    assert run["experiment_id"]

    detail = client.get(f"/api/experiments/{run['experiment_id']}")
    assert detail.status_code == 200
    assert detail.json()["is_ad_hoc"] is True
    assert detail.json()["run_count"] >= 1


def test_failed_run_persists_terminal_error_and_artifacts(tmp_path):
    client = _client(tmp_path)
    experiment = client.post("/api/experiments", json={"name": "Failure runs", "description": "", "tags": []})
    failing = _payload()
    failing["experiment_id"] = experiment.json()["experiment_id"]
    failing["search_space"]["gamma"] = {"type": "float", "low": -1.0, "high": -1.0, "precision": 1}
    created = client.post("/api/runs", json=failing)
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    snapshot = _wait_for_terminal_run(client, run_id)
    assert snapshot["status"] == "failed"
    assert snapshot["error"] or snapshot["n_failures"] >= 1

    history = client.get(f"/api/runs/{run_id}/history")
    assert history.status_code == 200
    assert any(event["type"] == "trial_failed" for event in history.json()["events"])
    assert history.json()["events"][-1]["type"] == "run_failed"

    artifacts = client.get(f"/api/runs/{run_id}/artifacts")
    assert artifacts.status_code == 200
    assert artifacts.json()["run"]["run_id"] == run_id
    assert len(artifacts.json()["events"]) >= 2


def test_validation_rejects_xgboost_gamma_regression_for_nonpositive_targets(tmp_path):
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed in this environment.")
    client = _client(tmp_path)
    payload = _payload()
    payload["dataset"] = {
        "source": "csv",
        "csv_text": "feature,target\n1,-1\n2,0\n3,2\n4,4\n",
        "target_column": "target",
    }
    payload["task"] = "regression"
    payload["metric"] = "rmse"
    payload["estimator"] = "xgboost"
    payload["fixed_params"] = {"objective": "reg:gamma"}
    payload["search_space"] = {
        "learning_rate": {"type": "float", "low": 0.1, "high": 0.2, "precision": 1},
        "max_depth": {"type": "int", "low": 2, "high": 3},
        "n_estimators": {"type": "int", "low": 5, "high": 6},
        "subsample": {"type": "float", "low": 0.8, "high": 1.0, "precision": 1},
    }

    validation = client.post("/api/runs/validate", json=payload)

    assert validation.status_code == 200
    body = validation.json()
    assert body["valid"] is False
    assert "reg:gamma" in " ".join(body["errors"]["estimator"])

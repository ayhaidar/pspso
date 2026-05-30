import importlib.util

from fastapi.testclient import TestClient

from pspso.dashboard.app import create_app
from pspso.estimators import estimator_presets


def _base_payload(task: str, estimator: str) -> dict:
    if task == "binary classification":
        return {
            "experiment_id": None,
            "dataset": {
                "source": "example",
                "name": "breast_cancer",
                "target_column": "target",
            },
            "task": task,
            "metric": "roc_auc",
            "estimator": estimator,
            "fixed_params": {},
            "search_space": None,
            "strategy": "random",
            "split": {"validation_size": 0.2, "random_state": 42, "stratify": True},
            "runtime": {"max_trials": 1},
        }
    return {
        "experiment_id": None,
        "dataset": {
            "source": "example",
            "name": "diabetes",
            "target_column": "target",
        },
        "task": task,
        "metric": "rmse",
        "estimator": estimator,
        "fixed_params": {},
        "search_space": None,
        "strategy": "random",
        "split": {"validation_size": 0.2, "random_state": 42, "stratify": False},
        "runtime": {"max_trials": 1},
    }


def test_builtin_option_matrix_validates_with_supported_dependencies(tmp_path):
    client = TestClient(create_app(tmp_path / "matrix.sqlite3"))
    presets = estimator_presets()

    for estimator, meta in presets.items():
        dependency = meta.get("optional_dependency")
        if dependency and importlib.util.find_spec(dependency) is None:
            continue
        for task in meta["tasks"]:
            payload = _base_payload(task, estimator)
            response = client.post("/api/runs/validate", json=payload)
            assert response.status_code == 200, (estimator, task)
            body = response.json()
            assert body["valid"] is True, (estimator, task, body["errors"])

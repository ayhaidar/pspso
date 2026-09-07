import importlib.util

import pytest
from sklearn.datasets import load_wine

from pspso.estimators import allowed_estimator_params, create_estimator, estimator_presets


def _base_payload(task: str, estimator: str) -> dict:
    if task == "multiclass_classification":
        return {
            "experiment_id": None,
            "dataset": {"source": "example", "name": "wine", "target_column": "target"},
            "task": task,
            "metric": "f1_macro",
            "estimator": estimator,
            "fixed_params": {},
            "search_space": None,
            "strategy": "random",
            "split": {"validation_size": 0.2, "random_state": 42, "stratify": True},
            "runtime": {"max_trials": 1},
        }
    if task == "binary_classification":
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


def test_builtin_option_matrix_validates_with_supported_dependencies(tmp_path, client_factory):
    client = client_factory(tmp_path / "matrix.sqlite3")
    presets = estimator_presets()

    for estimator, meta in presets.items():
        dependency = meta.get("optional_dependency")
        if dependency and importlib.util.find_spec(dependency) is None:
            continue
        for task in meta["tasks"]:
            payload = _base_payload(task, estimator)
            response = client.post("/api/v1/runs/validate", json=payload)
            assert response.status_code == 200, (estimator, task)
            body = response.json()
            assert body["valid"] is True, (estimator, task, body["errors"])


@pytest.mark.filterwarnings("ignore:X does not have valid feature names:UserWarning")
def test_booster_recipes_accept_runtime_settings_and_use_a_multiclass_classifier():
    assert {"random_state", "n_jobs"}.issubset(
        allowed_estimator_params("xgboost", "binary_classification")
    )
    assert {"random_state", "n_jobs"}.issubset(
        allowed_estimator_params("lightgbm", "multiclass_classification")
    )
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("LightGBM optional dependency is not installed.")
    model = create_estimator("lightgbm", "multiclass_classification", {"n_estimators": 5})
    assert type(model).__name__ == "LGBMClassifier"
    X, y = load_wine(return_X_y=True)
    model.fit(X, y)
    assert set(model.predict(X[:3], validate_features=False)).issubset(set(y))

import importlib.util

import joblib
import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pspso.estimators import create_estimator, default_fixed_params, estimator_presets

CASES = [(name, task) for name, meta in estimator_presets().items() for task in meta["tasks"]]


@pytest.mark.parametrize("name,task", CASES)
@pytest.mark.filterwarnings("ignore:.*converged.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*valid feature names.*:UserWarning")
def test_every_builtin_family_trains_predicts_and_serializes(name, task, tmp_path):
    dependency = estimator_presets()[name].get("optional_dependency")
    if dependency and importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed")
    if task == "regression":
        X, y = make_regression(n_samples=90, n_features=6, random_state=17)
        y = y / 100
    else:
        X, y = make_classification(
            n_samples=90,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            n_classes=3 if task == "multiclass_classification" else 2,
            n_clusters_per_class=1,
            random_state=17,
        )
    params = {
        "random_forest": {"n_estimators": 4, "max_depth": 2, "n_jobs": 1},
        "xgboost": {"n_estimators": 4, "max_depth": 2, "n_jobs": 1},
        "lightgbm": {"n_estimators": 4, "max_depth": 2, "n_jobs": 1, "verbosity": -1},
        "sklearn_mlp": {"hidden_layer_sizes": (4,), "max_iter": 5},
        "pytorch_mlp": {"neurons": 4, "epochs": 2, "batch_size": 32, "device": "cpu"},
    }.get(name, {})
    model = Pipeline(
        [("scale", StandardScaler()), ("estimator", create_estimator(name, task, params))]
    )
    model = clone(model)
    model.fit(X, y)
    assert is_classifier(model) == (task != "regression")
    predictions = model.predict(X[:8])
    assert predictions.shape == (8,)
    assert np.isfinite(predictions).all()
    if task != "regression":
        assert set(predictions).issubset(set(y))
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X[:8])
            assert probabilities.shape == (8, len(np.unique(y)))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(X[:8]), predictions)


def test_pytorch_mlp_is_seeded_and_restores_its_best_epoch():
    pytest.importorskip("torch")
    X, y = make_classification(
        n_samples=64,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=17,
    )
    params = {
        "neurons": 6,
        "epochs": 4,
        "batch_size": 16,
        "learning_rate": 0.01,
        "device": "cpu",
        "patience": 2,
        "random_state": 23,
    }
    first = create_estimator("pytorch_mlp", "binary_classification", params)
    second = create_estimator("pytorch_mlp", "binary_classification", params)
    first.fit(X, y)
    second.fit(X, y)
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X), atol=1e-7)
    assert 1 <= first.best_epoch_ <= first.n_iter_ <= params["epochs"]
    assert np.isfinite(first.best_loss_)
    assert default_fixed_params("pytorch_mlp", "binary_classification")["random_state"] == 42

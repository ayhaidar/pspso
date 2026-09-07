import importlib.util
import json

import pytest
from sklearn.datasets import load_diabetes

import pspso
from pspso import IntRange, OptimizationConfig, SearchSpace, TrackingConfig, optimize
from pspso.dashboard.tracking import TrackingRepository


def test_notebook_helper_returns_rich_result():
    X, y = load_diabetes(return_X_y=True)
    result = optimize(
        X[:100],
        y[:100],
        estimator="random_forest",
        search_space=SearchSpace({"n_estimators": IntRange(5, 6), "max_depth": IntRange(2, 3)}),
        config=OptimizationConfig(
            task="regression",
            metric="rmse",
            strategy="random",
            max_trials=2,
            random_state=42,
        ),
    )

    assert result.best_params is not None
    assert len(result.trials_frame()) == 2
    assert "RANDOM regression" in result.summary()
    assert "PSPSO optimization result" in result._repr_html_()
    assert result.predict(X[:2]).shape == (2,)
    assert result.evaluate(X[:10], y[:10])["selected_metric"] == "rmse"


def test_notebook_helper_can_track_without_fastapi(tmp_path):
    X, y = load_diabetes(return_X_y=True)
    result = optimize(
        X[:80],
        y[:80],
        estimator="random_forest",
        search_space=SearchSpace({"n_estimators": IntRange(5, 5), "max_depth": IntRange(2, 2)}),
        config=OptimizationConfig(
            task="regression",
            metric="rmse",
            strategy="random",
            max_trials=1,
            random_state=42,
        ),
        tracking=TrackingConfig(
            workspace=tmp_path,
            experiment_name="Notebook test",
            snapshot_data=True,
        ),
    )

    assert result.run_id
    assert result.experiment_id
    assert (tmp_path / "tracking.sqlite3").exists()
    assert "dataset_snapshot" in result.artifacts
    saved_result = json.loads(
        (tmp_path / "artifacts" / "runs" / result.run_id / "result.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_result["run_id"] == result.run_id
    assert saved_result["experiment_id"] == result.experiment_id
    assert saved_result["artifacts"]["dataset_snapshot"] == result.artifacts["dataset_snapshot"]


def test_removed_legacy_surfaces_are_unavailable():
    assert not hasattr(pspso, "pspso")
    assert importlib.util.find_spec("pspso.pspso") is None
    with pytest.raises(TypeError):
        SearchSpace({"C": [0.1, 1.0, 1]})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OptimizationConfig(task="regression", metric="auc").validate()  # type: ignore[arg-type]


def test_default_workspace_is_versioned_and_configurable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert TrackingRepository().db_path == tmp_path / ".pspso" / "v1" / "tracking.sqlite3"

    alternate = tmp_path / "workspace"
    monkeypatch.setenv("PSPSO_HOME", str(alternate))
    assert TrackingRepository().db_path == alternate / "v1" / "tracking.sqlite3"

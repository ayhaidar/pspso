import json
import os
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from pspso.dashboard.app import RunStore, create_app
from pspso.dashboard.results import load_saved_inputs, saved_positive_label
from pspso.dashboard.schemas import ExperimentSpec
from pspso.dashboard.tracking import TrackingRepository
from pspso.dashboard.worker import execute_run
from pspso.ensemble import CrossValidationEnsemble
from pspso.metrics import metric_value
from pspso.optimizer import OptimizationCancelled, OptimizationTimedOut, PSPSOOptimizer


def recorded(tmp_path, **updates):
    task = updates.pop("task", "regression")
    classification = task != "regression"
    payload = {
        "dataset": {
            "source": "example",
            "name": {
                "regression": "diabetes",
                "binary_classification": "breast_cancer",
                "multiclass_classification": "wine",
            }[task],
        },
        "task": task,
        "metric": "accuracy" if classification else "rmse",
        "estimator": "logistic_regression" if classification else "linear_regression",
        "fixed_params": {"max_iter": 300} if classification else {},
        "search_space": {"fit_intercept": {"type": "choice", "values": [True]}},
        "runtime": {"max_trials": 1},
        "split": {"test_size": 0.2, "random_state": None},
        **updates,
    }
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    run = RunStore(repository).create(ExperimentSpec(**payload))
    attempt = repository.create_attempt(run.run_id)
    repository.set_attempt_running(attempt["attempt_id"], os.getpid())
    return repository, run.run_id, attempt["attempt_id"]


def test_cv_without_refit_saves_winning_fold_ensemble(tmp_path):
    repository, run_id, attempt = recorded(
        tmp_path, evaluation={"protocol": "cross_validation", "folds": 2, "refit_best": False}
    )
    assert execute_run(repository.db_path, run_id, attempt) == 0
    run = repository.get_run(run_id)
    assert run["status"] == "completed"
    model = joblib.load(run["artifacts"]["model"])
    assert isinstance(model, CrossValidationEnsemble)
    assert len(model.models_) == 2
    analysis = json.loads(Path(run["artifacts"]["analysis"]).read_text())
    assert len(analysis["validation"]["cross_validation"]["folds"]) == 2
    assert "test" in analysis
    assert not analysis["feature_importance"]["available"]
    assert "permutation importance" in analysis["feature_importance"]["reason"]
    assert run["result"]["optimizer_state"]["completed_fits"] == 2
    with TestClient(create_app(repository.db_path, start_manager=False)) as client:
        response = client.get(f"/api/v1/runs/{run_id}/predictions?split=test")
        assert response.status_code == 200, response.text
        assert response.json()["preview_rows"]


@pytest.mark.parametrize("protocol", ["holdout", "cross_validation"])
@pytest.mark.parametrize(
    "task", ["regression", "binary_classification", "multiclass_classification"]
)
def test_worker_evaluates_saved_splits_refits_and_downloads_without_training(
    tmp_path, monkeypatch, protocol, task
):
    repository, run_id, attempt = recorded(
        tmp_path,
        task=task,
        evaluation={
            "protocol": protocol,
            "folds": 3,
            "decision_threshold": 0.7,
            "positive_label": 0 if task == "binary_classification" else None,
        },
    )
    assert execute_run(repository.db_path, run_id, attempt) == 0
    run = repository.get_run(run_id)
    assert run["status"] == "completed", run
    request = ExperimentSpec(**run["request"])
    assert request.split.random_state is not None
    artifacts = run["artifacts"]
    saved = joblib.load(artifacts["dataset"])
    splits = json.loads(Path(artifacts["split_indices"]).read_text())
    partitions = splits["partitions"]
    assert set(partitions["development"]).isdisjoint(partitions["test"])
    assert set(partitions["development"]) | set(partitions["test"]) == set(saved.index)
    model = joblib.load(artifacts["model"])
    scaler = model.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"]
    assert np.all(scaler.n_samples_seen_ == len(partitions["development"]))
    if protocol == "cross_validation":
        assert len(splits["folds"]) == 3
        assert sorted(i for fold in splits["folds"] for i in fold["validation"]) == list(
            range(len(partitions["development"]))
        )
    else:
        selection = joblib.load(artifacts["selection_model"])
        scaler = (
            selection.named_steps["preprocessor"]
            .named_transformers_["numeric"]
            .named_steps["scaler"]
        )
        assert np.all(scaler.n_samples_seen_ == len(partitions["train"]))
    analysis = json.loads(Path(artifacts["analysis"]).read_text())
    model, features, _, target, mapping = load_saved_inputs(request, artifacts, "test")
    expected = metric_value(
        model,
        task,
        request.metric,
        features,
        target,
        positive_label=saved_positive_label(request, mapping),
        decision_threshold=request.evaluation.decision_threshold,
    )
    assert analysis["test"]["metrics"][request.metric] == pytest.approx(expected)
    environment = json.loads(Path(artifacts["environment"]).read_text())
    assert environment["dataset_fingerprint"] == joblib.hash(saved)
    assert environment["random_seed"] == request.split.random_state
    terminal = [
        event for event in repository.get_run_history(run_id) if event["type"] == "run_completed"
    ]
    assert len(terminal) == 1

    def no_fitting(*args, **kwargs):
        raise AssertionError("Result downloads must never fit a model.")

    monkeypatch.setattr(Pipeline, "fit", no_fitting)
    with TestClient(create_app(repository.db_path, start_manager=False)) as client:
        response = client.get(f"/api/v1/runs/{run_id}/predictions?split=test&limit=1000")
        assert response.status_code == 200, response.text
        prediction = response.json()
        assert [row["row_index"] for row in prediction["preview_rows"]] == partitions["test"]
        for kind in ("spec", "result", "analysis", "events", "predictions"):
            response = client.get(f"/api/v1/runs/{run_id}/exports/{kind}?split=test")
            assert response.status_code == 200, response.text
            assert "attachment" in response.headers["content-disposition"]
        if protocol == "holdout":
            response = client.get(f"/api/v1/runs/{run_id}/predictions?split=validation")
            assert response.status_code == 200, response.text
        else:
            assert (
                client.get(f"/api/v1/runs/{run_id}/predictions?split=validation").status_code == 400
            )


@pytest.mark.parametrize(
    "exception, expected",
    [
        (OptimizationCancelled("cancelled"), "cancelled"),
        (OptimizationTimedOut("timeout"), "failed"),
        (RuntimeError("worker failure"), "failed"),
    ],
)
def test_worker_records_one_terminal_event_on_errors(tmp_path, monkeypatch, exception, expected):
    repository, run_id, attempt = recorded(tmp_path)

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(PSPSOOptimizer, "optimize", fail)
    assert execute_run(repository.db_path, run_id, attempt) == (0 if expected == "cancelled" else 1)
    assert repository.get_run(run_id)["status"] == expected
    terminals = [
        e
        for e in repository.get_run_history(run_id)
        if e["type"] in {"run_failed", "run_cancelled"}
    ]
    assert len(terminals) == 1


def test_worker_cancel_before_loading_and_unknown_run(tmp_path):
    repository, run_id, attempt = recorded(tmp_path)
    repository.request_cancel(run_id)
    assert execute_run(repository.db_path, run_id, attempt) == 0
    assert repository.get_run(run_id)["status"] == "cancelled"
    assert execute_run(repository.db_path, "missing", "missing") == 2


def test_model_serialization_failure_is_visible_and_cannot_trigger_retraining(
    tmp_path, monkeypatch
):
    repository, run_id, attempt = recorded(tmp_path)
    original_dump = joblib.dump

    def dump(value, filename, *args, **kwargs):
        if str(filename).endswith("model.joblib"):
            raise TypeError("model cannot be serialized")
        return original_dump(value, filename, *args, **kwargs)

    monkeypatch.setattr(joblib, "dump", dump)
    assert execute_run(repository.db_path, run_id, attempt) == 0
    run = repository.get_run(run_id)
    assert "model" not in run["artifacts"]
    assert Path(run["artifacts"]["warnings"]).is_file()
    assert any(
        e["type"] == "run_warning" and e["payload"]["reason"] == "artifact_serialization"
        for e in repository.get_run_history(run_id)
    )
    with pytest.raises(ValueError, match="Saved artifacts are unavailable"):
        load_saved_inputs(ExperimentSpec(**run["request"]), run["artifacts"], "test")


def test_pytorch_pipeline_reports_epochs_for_each_fold_and_serializes(tmp_path):
    pytest.importorskip("torch")
    repository, run_id, attempt = recorded(
        tmp_path,
        task="binary_classification",
        estimator="pytorch_mlp",
        fixed_params={"neurons": 4, "epochs": 2, "batch_size": 128, "device": "cpu"},
        search_space={"learning_rate": {"type": "choice", "values": [0.001]}},
        runtime={"max_trials": 2, "trial_workers": 2},
        evaluation={"protocol": "cross_validation", "folds": 2},
    )
    assert execute_run(repository.db_path, run_id, attempt) == 0
    run = repository.get_run(run_id)
    assert run["status"] == "completed", run
    assert Path(run["artifacts"]["model"]).is_file()
    events = repository.get_run_history(run_id)
    epochs = [e for e in events if e["type"] == "training_epoch"]
    assert len(epochs) == 10
    assert {e["payload"]["fold"] for e in epochs if e["payload"]["phase"] == "candidate"} == {1, 2}
    assert any(e["payload"]["phase"] == "final_refit" for e in epochs)
    assert run["result"]["optimizer_state"]["completed_fits"] == 5


def test_pytorch_cv_without_refit_saves_fold_ensemble(tmp_path):
    pytest.importorskip("torch")
    repository, run_id, attempt = recorded(
        tmp_path,
        task="binary_classification",
        estimator="pytorch_mlp",
        fixed_params={
            "neurons": 4,
            "epochs": 2,
            "batch_size": 128,
            "device": "cpu",
            "random_state": 17,
        },
        search_space={"learning_rate": {"type": "choice", "values": [0.001]}},
        runtime={"max_trials": 1},
        evaluation={"protocol": "cross_validation", "folds": 2, "refit_best": False},
    )
    assert execute_run(repository.db_path, run_id, attempt) == 0
    run = repository.get_run(run_id)
    model = joblib.load(run["artifacts"]["model"])
    assert isinstance(model, CrossValidationEnsemble)
    assert len(model.models_) == 2
    assert all(member.named_steps["estimator"].random_state == 17 for member in model.models_)
    assert run["result"]["optimizer_state"]["completed_fits"] == 2
    assert not any(
        event["type"].startswith("final_refit") for event in repository.get_run_history(run_id)
    )

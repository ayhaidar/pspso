"""FastAPI application for the pspso dashboard."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from pspso.config import EstimatorConfig, OptimizationConfig, OptimizationResult, ProgressEvent
from pspso.dashboard.data import (
    DatasetSelection,
    PreprocessingConfig,
    SplitConfig,
    example_datasets,
    load_dataset,
    prepare_prediction_bundle,
    prepare_tabular_data,
    preview_csv,
    preview_frame,
)
from pspso.dashboard.tracking import TrackingRepository
from pspso.estimators import default_fixed_params, default_search_space, estimator_presets
from pspso.estimators import (
    allowed_estimator_params,
    create_estimator,
    metric_metadata,
    normalize_estimator_name,
    optional_dependency_status,
    task_metadata,
)
from pspso.metrics import prediction_output
from pspso.optimizer import PSPSOOptimizer
from pspso.search_space import Choice, FloatRange, IntRange, SearchSpace, coerce_search_space


class CsvPreviewRequest(BaseModel):
    csv_text: str


class DatasetRequest(BaseModel):
    source: Literal["example", "csv"] = "example"
    name: str | None = "breast_cancer"
    csv_text: str | None = None
    target_column: str = "target"


class SplitRequest(BaseModel):
    validation_size: float = 0.2
    test_size: float = 0.0
    stratify: bool = True
    random_state: int | None = 42


class PreprocessingRequest(BaseModel):
    scale_numeric: bool = True
    encode_categorical: bool = True
    ignored_columns: list[str] = Field(default_factory=list)


class PsoRequest(BaseModel):
    particles: int = 5
    iterations: int = 10
    c1: float = 1.49618
    c2: float = 1.49618
    w: float = 0.7298
    topology: Literal["global", "local"] = "global"


class RuntimeRequest(BaseModel):
    max_trials: int | None = None
    timeout_seconds: float | None = None
    early_stopping_rounds: int | None = None
    verbose: int = 0


class RunRequest(BaseModel):
    experiment_id: str | None = None
    dataset: DatasetRequest = Field(default_factory=DatasetRequest)
    task: Literal["regression", "binary classification"] = "binary classification"
    metric: Literal["rmse", "accuracy", "acc", "roc_auc", "auc"] = "roc_auc"
    estimator: str = "svm"
    fixed_params: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, Any] | None = None
    strategy: Literal["pso", "grid", "random"] = "random"
    split: SplitRequest = Field(default_factory=SplitRequest)
    preprocessing: PreprocessingRequest = Field(default_factory=PreprocessingRequest)
    pso: PsoRequest = Field(default_factory=PsoRequest)
    runtime: RuntimeRequest = Field(default_factory=RuntimeRequest)


class ExperimentCreateRequest(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)


@dataclass
class RunState:
    run_id: str
    experiment_id: str
    experiment_name: str
    request: RunRequest
    status: str = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self.updated_at = event["timestamp"]
        if event["type"] in {
            "dataset_prepared",
            "run_started",
            "trial_started",
            "trial_completed",
            "trial_failed",
            "iteration_completed",
            "best_updated",
            "run_warning",
        }:
            self.status = "running"
        if event["type"] == "run_completed":
            self.status = "completed"
        if event["type"] in {"run_failed", "validation_failed"}:
            self.status = "failed"

    def snapshot(self) -> dict[str, Any]:
        best = None
        for event in reversed(self.events):
            if event["type"] == "best_updated":
                best = event["payload"]
                break
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "event_count": len(self.events),
            "best": best,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "duration": self.result["duration"] if self.result else None,
            "n_trials": len(
                [event for event in self.events if event["type"] in {"trial_completed", "trial_failed"}]
            ),
            "n_failures": len([event for event in self.events if event["type"] == "trial_failed"]),
            "strategy": self.request.strategy,
            "task": self.request.task,
            "metric": self.request.metric,
            "estimator": self.request.estimator,
            "request": self.request.model_dump(),
        }


class RunStore:
    def __init__(self, repository: TrackingRepository) -> None:
        self.repository = repository
        self._runs: dict[str, RunState] = {}
        self._lock = threading.Lock()

    def create(self, request: RunRequest) -> RunState:
        experiment = (
            self.repository.get_experiment(request.experiment_id)
            if request.experiment_id
            else self.repository.create_ad_hoc_experiment()
        )
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment was not found.")
        snapshot = self.repository.create_run(
            request.model_dump(),
            experiment_id=experiment["experiment_id"],
        )
        run = RunState(
            run_id=snapshot["run_id"],
            experiment_id=snapshot["experiment_id"],
            experiment_name=snapshot["experiment_name"],
            request=request,
            status=snapshot["status"],
            result=snapshot["result"],
            error=snapshot["error"],
            created_at=snapshot["created_at"],
            updated_at=snapshot["updated_at"],
        )
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> RunState | None:
        with self._lock:
            return self._runs.get(run_id)

    def get_snapshot(self, run_id: str) -> dict[str, Any]:
        run = self.get(run_id)
        if run is not None:
            return run.snapshot()
        snapshot = self.repository.get_run(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Run was not found.")
        return snapshot

    def record_event(self, run_id: str, event: ProgressEvent) -> dict[str, Any]:
        persisted = self.repository.record_event(run_id, event.to_dict())
        run = self.get(run_id)
        if run is not None:
            run.add_event(persisted)
        return persisted

    def save_result(self, run_id: str, result: OptimizationResult) -> None:
        self.repository.save_result(run_id, result)
        run = self.get(run_id)
        if run is not None:
            run.result = result.to_dict()
            run.status = "completed" if result.best_params is not None else "failed"
            run.error = None

    def save_error(self, run_id: str, error: str) -> None:
        self.repository.save_error(run_id, error)
        run = self.get(run_id)
        if run is not None:
            run.error = error
            run.status = "failed"

    def create_experiment(self, request: ExperimentCreateRequest) -> dict[str, Any]:
        return self.repository.create_experiment(request.name, request.description, request.tags)

    def list_experiments(self) -> list[dict[str, Any]]:
        return self.repository.list_experiments()

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.repository.get_experiment(experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment was not found.")
        return experiment

    def get_history(self, run_id: str) -> list[dict[str, Any]]:
        if self.repository.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run was not found.")
        return self.repository.get_run_history(run_id)

    def get_result(self, run_id: str) -> dict[str, Any]:
        result = self.repository.get_run_result(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run result is not ready yet.")
        return result

    def get_artifacts(self, run_id: str) -> dict[str, Any]:
        artifacts = self.repository.get_artifacts(run_id)
        if artifacts is None:
            raise HTTPException(status_code=404, detail="Run was not found.")
        return artifacts


def create_app(db_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="pspso dashboard API", version="0.2.0")
    store = RunStore(TrackingRepository(db_path))
    app.state.store = store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        static_index = Path(__file__).with_name("static").joinpath("index.html")
        if static_index.exists():
            return static_index.read_text(encoding="utf-8")
        return "<h1>pspso dashboard API</h1><p>Open the React frontend or visit /docs.</p>"

    @app.get("/api/datasets/examples")
    def datasets_examples() -> list[dict[str, Any]]:
        return example_datasets()

    @app.post("/api/datasets/preview")
    def datasets_preview(request: CsvPreviewRequest) -> dict[str, Any]:
        try:
            return preview_csv(request.csv_text)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/datasets/inspect")
    def datasets_inspect(request: DatasetRequest) -> dict[str, Any]:
        try:
            frame = load_dataset(
                DatasetSelection(
                    source=request.source,
                    name=request.name,
                    csv_text=request.csv_text,
                    target_column=request.target_column,
                )
            )
            return preview_frame(frame, rows=8)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/experiments")
    def list_experiments() -> list[dict[str, Any]]:
        return store.list_experiments()

    @app.post("/api/experiments")
    def create_experiment(request: ExperimentCreateRequest) -> dict[str, Any]:
        return store.create_experiment(request)

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        return store.get_experiment(experiment_id)

    @app.get("/api/estimators")
    def estimators() -> dict[str, Any]:
        presets = estimator_presets()
        for name, preset in presets.items():
            preset["dependency"] = optional_dependency_status(preset.get("optional_dependency"))
            preset["defaults"] = {}
            for task in preset["tasks"]:
                preset["defaults"][task] = {
                    "fixed_params": default_fixed_params(name, task),
                    "search_space": _legacy_search_space_to_api(default_search_space(name, task)),
                    "allowed_params": sorted(allowed_estimator_params(name, task)),
                }
        return {
            "estimators": presets,
            "tasks": task_metadata(),
            "metrics": metric_metadata(),
        }

    @app.post("/api/runs/validate")
    def validate_run(request: RunRequest) -> dict[str, Any]:
        errors = _validate_request(request)
        return {"valid": not any(errors.values()), "errors": errors}

    @app.post("/api/runs")
    def create_run(request: RunRequest) -> dict[str, Any]:
        errors = _validate_request(request)
        if any(errors.values()):
            raise HTTPException(status_code=400, detail={"valid": False, "errors": errors})
        run = store.create(request)
        store.record_event(
            run.run_id,
            ProgressEvent(
                "validation_passed",
                {
                    "experiment_id": run.experiment_id,
                    "experiment_name": run.experiment_name,
                    "estimator": request.estimator,
                    "strategy": request.strategy,
                },
            ),
        )
        thread = threading.Thread(target=_execute_run, args=(store, run), daemon=True)
        thread.start()
        return store.get_snapshot(run.run_id)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return store.get_snapshot(run_id)

    @app.get("/api/runs/{run_id}/result")
    def get_result(run_id: str) -> dict[str, Any]:
        return store.get_result(run_id)

    @app.get("/api/runs/{run_id}/history")
    def get_history(run_id: str) -> dict[str, Any]:
        return {"run_id": run_id, "events": store.get_history(run_id)}

    @app.get("/api/runs/{run_id}/artifacts")
    def get_artifacts(run_id: str) -> dict[str, Any]:
        return store.get_artifacts(run_id)

    @app.get("/api/runs/{run_id}/predictions")
    def get_predictions(run_id: str, split: Literal["validation", "train"] = "validation", limit: int = 50) -> dict[str, Any]:
        snapshot = store.get_snapshot(run_id)
        request_payload = snapshot.get("request")
        result_payload = snapshot.get("result")
        if request_payload is None:
            raise HTTPException(status_code=404, detail="Run configuration is not available.")
        if result_payload is None or result_payload.get("best_params") is None:
            raise HTTPException(status_code=404, detail="Predictions are only available for completed runs with a best model.")
        request = RunRequest(**request_payload)
        try:
            preview = _build_prediction_preview(request, result_payload["best_params"], split=split, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        preview["run_id"] = run_id
        return preview

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str) -> StreamingResponse:
        if store.repository.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run was not found.")
        run = store.get(run_id)
        history = store.get_history(run_id)

        async def generator():
            index = 0
            while True:
                current_events = run.events if run is not None else history
                while index < len(current_events):
                    event = current_events[index]
                    index += 1
                    yield _format_sse(event)
                current_status = store.get_snapshot(run_id)["status"]
                if current_status in {"completed", "failed"}:
                    if run is not None and index < len(run.events):
                        continue
                    break
                await asyncio.sleep(0.25)

        return StreamingResponse(generator(), media_type="text/event-stream")

    return app


def _format_sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def _execute_run(store: RunStore, run: RunState) -> None:
    try:
        X_train, y_train, X_val, y_val, optimizer = _build_optimizer_inputs(
            run.request,
            validate_dataset=False,
        )
        store.record_event(
            run.run_id,
            ProgressEvent(
                "dataset_prepared",
                {
                    "train_rows": int(getattr(X_train, "shape", [len(X_train)])[0]),
                    "validation_rows": int(getattr(X_val, "shape", [len(X_val)])[0]),
                    "features": int(getattr(X_train, "shape", [0, 0])[1]),
                },
            ),
        )

        def callback(event: ProgressEvent) -> None:
            store.record_event(run.run_id, event)

        result: OptimizationResult = optimizer.optimize(X_train, y_train, X_val, y_val, callback)
        store.save_result(run.run_id, result)
    except Exception as exc:
        store.save_error(run.run_id, str(exc))
        store.record_event(run.run_id, ProgressEvent("run_failed", {"error": str(exc)}))


def _build_prediction_preview(
    request: RunRequest,
    best_params: dict[str, Any],
    *,
    split: Literal["validation", "train"],
    limit: int,
) -> dict[str, Any]:
    dataset = DatasetSelection(
        source=request.dataset.source,
        name=request.dataset.name,
        csv_text=request.dataset.csv_text,
        target_column=request.dataset.target_column,
    )
    frame = load_dataset(dataset)
    bundle = prepare_prediction_bundle(
        frame,
        request.dataset.target_column,
        request.task,
        SplitConfig(**request.split.model_dump()),
        PreprocessingConfig(**request.preprocessing.model_dump()),
    )
    fixed_params = {
        **default_fixed_params(request.estimator, request.task),
        **request.fixed_params,
    }
    model = create_estimator(
        EstimatorConfig(name=request.estimator, fixed_params=fixed_params),
        request.task,
        best_params,
    )
    model.fit(bundle["X_train"], bundle["y_train"])
    feature_frame = bundle["X_validation_frame"] if split == "validation" else bundle["X_train_frame"]
    raw_target = bundle["y_validation_raw"] if split == "validation" else bundle["y_train_raw"]
    transformed = bundle["X_validation"] if split == "validation" else bundle["X_train"]
    predictions = prediction_output(model, request.task, transformed)
    rows: list[dict[str, Any]] = []
    target_mapping = bundle["target_mapping"]
    predicted_labels = predictions["predictions"]
    score_values = predictions.get("scores")
    limit = max(1, min(limit, len(feature_frame)))
    for offset, (row_index, feature_row) in enumerate(feature_frame.head(limit).iterrows()):
        actual_value = raw_target.loc[row_index]
        predicted_value = predicted_labels[offset]
        row = {
            "row_index": int(row_index) if isinstance(row_index, (int, np.integer)) else str(row_index),
            "actual": _json_scalar(actual_value),
            "predicted": _decode_prediction(predicted_value, target_mapping),
        }
        if request.task == "regression":
            actual_number = float(actual_value)
            predicted_number = float(predicted_value)
            row["residual"] = float(predicted_number - actual_number)
        else:
            row["score"] = float(score_values[offset]) if score_values is not None else None
        row["features"] = {key: _json_scalar(value) for key, value in feature_row.to_dict().items()}
        rows.append(row)
    return {
        "split": split,
        "task": request.task,
        "metric": request.metric,
        "estimator": request.estimator,
        "feature_columns": bundle["feature_columns"],
        "target_mapping": target_mapping,
        "row_count": int(len(feature_frame)),
        "preview_rows": rows,
    }


def _decode_prediction(value: Any, mapping: dict[int, Any] | None) -> Any:
    if mapping is None:
        return _json_scalar(value)
    try:
        return _json_scalar(mapping[int(value)])
    except Exception:
        return _json_scalar(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_request(request: RunRequest) -> dict[str, list[str]]:
    errors: dict[str, list[str]] = {
        "dataset": [],
        "task": [],
        "estimator": [],
        "fixed_params": [],
        "search_space": [],
        "strategy": [],
    }
    normalized_metric = {"acc": "accuracy", "auc": "roc_auc"}.get(request.metric, request.metric)
    tasks = task_metadata()
    if request.task not in tasks:
        errors["task"].append(f"Unsupported task: {request.task!r}.")
    elif normalized_metric not in tasks[request.task]["metrics"]:
        errors["task"].append(
            f"Metric {request.metric!r} is not valid for {request.task!r}. "
            f"Use one of: {', '.join(tasks[request.task]['metrics'])}."
        )

    estimator_name = normalize_estimator_name(request.estimator)
    presets = estimator_presets()
    preset = presets.get(estimator_name)
    if preset is None:
        errors["estimator"].append(f"Unknown estimator: {request.estimator!r}.")
    else:
        if request.task not in preset["tasks"]:
            errors["estimator"].append(
                f"{preset['label']} does not support {request.task!r}."
            )
        dependency = optional_dependency_status(preset.get("optional_dependency"))
        if not dependency["installed"]:
            errors["estimator"].append(
                f"{preset['label']} requires {dependency['required']}. "
                f"Install it with `{dependency['install']}`."
            )

    frame = None
    try:
        frame = load_dataset(
            DatasetSelection(
                source=request.dataset.source,
                name=request.dataset.name,
                csv_text=request.dataset.csv_text,
                target_column=request.dataset.target_column,
            )
        )
        if request.dataset.target_column not in frame.columns:
            errors["dataset"].append(
                f"Target column {request.dataset.target_column!r} was not found."
            )
        elif request.task == "binary classification":
            unique_targets = frame[request.dataset.target_column].dropna().nunique()
            if unique_targets != 2:
                errors["dataset"].append(
                    "Binary classification requires exactly two target classes."
                )
    except Exception as exc:
        errors["dataset"].append(str(exc))

    try:
        config = OptimizationConfig(
            task=request.task,
            metric=request.metric,
            strategy=request.strategy,
            validation_size=request.split.validation_size,
            test_size=request.split.test_size,
            random_state=request.split.random_state,
            n_particles=request.pso.particles,
            n_iterations=request.pso.iterations,
            pso_options={"c1": request.pso.c1, "c2": request.pso.c2, "w": request.pso.w},
            pso_topology=request.pso.topology,
            max_trials=request.runtime.max_trials,
            timeout_seconds=request.runtime.timeout_seconds,
            early_stopping_rounds=request.runtime.early_stopping_rounds,
            verbose=request.runtime.verbose,
        )
        config.validate()
    except Exception as exc:
        errors["strategy"].append(str(exc))

    space: SearchSpace | None = None
    try:
        space = _build_search_space(request)
    except Exception as exc:
        errors["search_space"].append(str(exc))

    if preset is not None and space is not None:
        allowed = allowed_estimator_params(estimator_name, request.task)
        fixed_names = set(request.fixed_params)
        search_names = set(space.names)
        invalid_fixed = sorted(fixed_names - allowed)
        invalid_search = sorted(search_names - allowed)
        if invalid_fixed:
            errors["fixed_params"].append(
                f"Unsupported fixed parameter(s) for {estimator_name}: "
                f"{', '.join(invalid_fixed)}."
            )
        if invalid_search:
            errors["search_space"].append(
                f"Unsupported tunable parameter(s) for {estimator_name}: "
                f"{', '.join(invalid_search)}."
            )
        if not invalid_fixed and not invalid_search and not errors["estimator"]:
            if (
                estimator_name == "xgboost"
                and request.task == "regression"
                and frame is not None
                and request.dataset.target_column in frame.columns
            ):
                gamma_requested = request.fixed_params.get("objective") == "reg:gamma"
                if "objective" in search_names:
                    objective_spec = request.search_space.get("objective") if request.search_space else None
                    if (
                        isinstance(objective_spec, dict)
                        and objective_spec.get("type") == "choice"
                        and "reg:gamma" in objective_spec.get("values", [])
                    ):
                        gamma_requested = True
                if gamma_requested and (frame[request.dataset.target_column] <= 0).any():
                    errors["estimator"].append(
                        "XGBoost objective 'reg:gamma' requires all target values to be strictly positive. "
                        "Use 'reg:squarederror' for general regression data."
                    )
            if not errors["estimator"]:
                try:
                    lower, _ = space.bounds
                    sample_params = space.decode(lower)
                    fixed_params = {
                        **default_fixed_params(estimator_name, request.task),
                        **request.fixed_params,
                    }
                    create_estimator(
                        EstimatorConfig(name=estimator_name, fixed_params=fixed_params),
                        request.task,
                        sample_params,
                    )
                except Exception as exc:
                    errors["estimator"].append(str(exc))

    if frame is not None and not errors["dataset"]:
        try:
            prepare_tabular_data(
                frame,
                request.dataset.target_column,
                request.task,
                SplitConfig(**request.split.model_dump()),
                PreprocessingConfig(**request.preprocessing.model_dump()),
            )
        except Exception as exc:
            errors["dataset"].append(str(exc))

    return errors


def _build_optimizer_inputs(
    request: RunRequest,
    validate_dataset: bool,
) -> tuple[Any, Any, Any, Any, PSPSOOptimizer]:
    dataset = DatasetSelection(
        source=request.dataset.source,
        name=request.dataset.name,
        csv_text=request.dataset.csv_text,
        target_column=request.dataset.target_column,
    )
    frame = load_dataset(dataset)
    split = SplitConfig(**request.split.model_dump())
    preprocessing = PreprocessingConfig(**request.preprocessing.model_dump())
    if validate_dataset:
        # Forces validation before returning to the caller.
        frame.head(1)
    X_train, y_train, X_val, y_val, _ = prepare_tabular_data(
        frame,
        request.dataset.target_column,
        request.task,
        split,
        preprocessing,
    )
    search_space = _build_search_space(request)
    fixed_params = {
        **default_fixed_params(request.estimator, request.task),
        **request.fixed_params,
    }
    config = OptimizationConfig(
        task=request.task,
        metric=request.metric,
        strategy=request.strategy,
        validation_size=request.split.validation_size,
        test_size=request.split.test_size,
        random_state=request.split.random_state,
        n_particles=request.pso.particles,
        n_iterations=request.pso.iterations,
        pso_options={"c1": request.pso.c1, "c2": request.pso.c2, "w": request.pso.w},
        pso_topology=request.pso.topology,
        max_trials=request.runtime.max_trials,
        timeout_seconds=request.runtime.timeout_seconds,
        early_stopping_rounds=request.runtime.early_stopping_rounds,
        verbose=request.runtime.verbose,
    )
    estimator = EstimatorConfig(name=request.estimator, fixed_params=fixed_params)
    optimizer = PSPSOOptimizer(estimator, search_space, config)
    return X_train, y_train, X_val, y_val, optimizer


def _build_search_space(request: RunRequest) -> SearchSpace:
    if not request.search_space:
        return coerce_search_space(default_search_space(request.estimator, request.task))
    converted: dict[str, Any] = {}
    for name, spec in request.search_space.items():
        if isinstance(spec, dict) and "type" in spec:
            spec_type = spec["type"]
            if spec_type == "choice":
                converted[name] = Choice(spec["values"])
            elif spec_type == "int":
                converted[name] = IntRange(int(spec["low"]), int(spec["high"]))
            elif spec_type == "float":
                converted[name] = FloatRange(
                    float(spec["low"]),
                    float(spec["high"]),
                    int(spec.get("precision", 2)),
                )
            else:
                raise ValueError(f"Unsupported search-space spec type: {spec_type!r}")
        else:
            converted[name] = spec
    return coerce_search_space(converted)


def _legacy_search_space_to_api(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    converted: dict[str, dict[str, Any]] = {}
    for name, spec in params.items():
        if all(isinstance(value, str) for value in spec):
            converted[name] = {"type": "choice", "values": list(spec)}
        else:
            low, high, precision = spec
            if int(precision) == 0:
                converted[name] = {"type": "int", "low": int(low), "high": int(high)}
            else:
                converted[name] = {
                    "type": "float",
                    "low": float(low),
                    "high": float(high),
                    "precision": int(precision),
                }
    return converted


app = create_app()


__all__ = ["RunRequest", "app", "create_app"]

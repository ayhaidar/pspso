"""FastAPI application for the pspso dashboard."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import os
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from pspso.config import EstimatorConfig, OptimizationConfig, OptimizationResult, ProgressEvent
from pspso.dashboard.data import (
    DatasetSelection,
    PreprocessingConfig,
    SplitConfig,
    encoded_positive_label,
    example_datasets,
    load_dataset,
    prepare_cross_validation_bundle,
    prepare_tabular_data,
    preview_csv,
    preview_frame,
)
from pspso.dashboard.manager import LocalRunManager
from pspso.dashboard.responses import (
    AnalysisResponse,
    DatasetPreviewResponse,
    ExampleDatasetResponse,
    ImportanceResponse,
    SavedDatasetResponse,
)
from pspso.dashboard.results import load_saved_inputs, saved_positive_label
from pspso.dashboard.runtime import (
    build_search_space as _build_search_space,
)
from pspso.dashboard.schemas import (
    ArtifactResponse,
    CsvPreviewRequest,
    DatasetCreateRequest,
    DatasetInspectRequest,
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentSpec,
    MetadataResponse,
    PredictionResponse,
    ResultLayoutRequest,
    RunHistoryResponse,
    RunResponse,
    TournamentRequest,
    TournamentResponse,
    ValidationResponse,
    WorkflowValidationRequest,
)
from pspso.dashboard.tracking import TrackingRepository
from pspso.dashboard.validation import validate_objectives
from pspso.estimators import (
    allowed_estimator_params,
    canonical_estimator_name,
    create_estimator,
    default_fixed_params,
    default_search_space,
    estimator_presets,
    metric_metadata,
    optional_dependency_status,
    task_metadata,
)
from pspso.metrics import evaluation_report, prediction_output
from pspso.search_space import (
    SearchSpace,
)


@dataclass
class RunState:
    run_id: str
    experiment_id: str
    experiment_name: str
    request: ExperimentSpec
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
                [
                    event
                    for event in self.events
                    if event["type"] in {"trial_completed", "trial_failed"}
                ]
            ),
            "n_failures": len([event for event in self.events if event["type"] == "trial_failed"]),
            "strategy": self.request.strategy,
            "task": self.request.task,
            "metric": self.request.metric,
            "estimator": self.request.estimator,
            "request": self.request.model_dump(),
        }


class RunStore:
    def __init__(self, repository: TrackingRepository, *, start_manager: bool = False) -> None:
        self.repository = repository
        self._runs: dict[str, RunState] = {}
        self._lock = threading.Lock()
        self.manager = LocalRunManager(repository) if start_manager else None

    def create(self, request: ExperimentSpec, *, parent_run_id: str | None = None) -> RunState:
        experiment = (
            self.repository.get_experiment(request.experiment_id)
            if request.experiment_id
            else self.repository.create_ad_hoc_experiment()
        )
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment was not found.")
        payload = request.model_dump()
        if payload["split"]["random_state"] is None:
            payload["split"]["random_state"] = secrets.randbits(32)
        payload["evaluation"]["random_state"] = payload["split"]["random_state"]
        if request.dataset.source == "stored":
            if not request.dataset.dataset_id:
                raise HTTPException(
                    status_code=400, detail="A saved dataset must include dataset_id."
                )
            dataset = self.repository.get_dataset(request.dataset.dataset_id)
            if dataset is None or not dataset.get("source_path"):
                raise HTTPException(status_code=404, detail="Saved dataset was not found.")
            payload["dataset"]["csv_text"] = Path(dataset["source_path"]).read_text(
                encoding="utf-8"
            )
        snapshot = self.repository.create_run(
            payload,
            experiment_id=experiment["experiment_id"],
            parent_run_id=parent_run_id,
        )
        run = RunState(
            run_id=snapshot["run_id"],
            experiment_id=snapshot["experiment_id"],
            experiment_name=snapshot["experiment_name"],
            request=ExperimentSpec(**payload),
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

    def get_history(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        if self.repository.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run was not found.")
        return self.repository.get_run_history(run_id, after=after)

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

    def cancel(self, run_id: str) -> bool:
        if self.repository.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run was not found.")
        return self.manager.cancel(run_id) if self.manager is not None else False

    def retry(self, run_id: str) -> RunState:
        source = self.repository.get_run(run_id)
        if source is None or source.get("request") is None:
            raise HTTPException(status_code=404, detail="Run configuration was not found.")
        request = ExperimentSpec(**source["request"])
        request.experiment_id = source["experiment_id"]
        return self.create(request, parent_run_id=run_id)


def create_app(db_path: str | Path | None = None, *, start_manager: bool = True) -> FastAPI:
    store = RunStore(TrackingRepository(db_path), start_manager=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_manager:
            store.manager = LocalRunManager(store.repository)
        try:
            yield
        finally:
            if store.manager is not None:
                store.manager.shutdown()
                store.manager = None

    app = FastAPI(
        title="PSPSO API",
        version=package_version("pspso"),
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.store = store
    configured_origins = os.environ.get(
        "PSPSO_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip() for origin in configured_origins.split(",") if origin.strip()
        ],
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )
    static_root = Path(__file__).with_name("static")
    if static_root.joinpath("assets").exists():
        app.mount("/assets", StaticFiles(directory=static_root / "assets"), name="assets")
    if static_root.joinpath("brand").exists():
        app.mount("/brand", StaticFiles(directory=static_root / "brand"), name="brand")

    @app.get("/", response_class=HTMLResponse, response_model=None)
    def root() -> Response:
        static_index = static_root / "index.html"
        if static_index.exists():
            return FileResponse(static_index)
        return HTMLResponse(
            "<h1>PSPSO API</h1><p>Open the React frontend or visit /api/v1/docs.</p>"
        )

    @app.get("/api/v1/datasets/examples", response_model=list[ExampleDatasetResponse])
    def datasets_examples() -> list[dict[str, Any]]:
        return example_datasets()

    @app.get("/api/v1/datasets", response_model=list[SavedDatasetResponse])
    def list_saved_datasets() -> list[dict[str, Any]]:
        return store.repository.list_datasets()

    @app.post("/api/v1/datasets", response_model=SavedDatasetResponse)
    def create_saved_dataset(request: DatasetCreateRequest) -> dict[str, Any]:
        try:
            profile = preview_csv(request.csv_text, rows=8)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        fingerprint = hashlib.sha256(request.csv_text.encode("utf-8")).hexdigest()
        dataset_dir = store.repository.workspace / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        source_path = dataset_dir / f"{fingerprint}.csv"
        if not source_path.exists():
            source_path.write_text(request.csv_text, encoding="utf-8")
        return store.repository.create_dataset(
            request.name,
            "csv",
            str(source_path),
            fingerprint,
            profile,
        )

    @app.post("/api/v1/datasets/preview", response_model=DatasetPreviewResponse)
    def datasets_preview(request: CsvPreviewRequest) -> dict[str, Any]:
        try:
            return preview_csv(request.csv_text)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/datasets/inspect", response_model=DatasetPreviewResponse)
    def datasets_inspect(request: DatasetInspectRequest) -> dict[str, Any]:
        try:
            csv_text = request.csv_text
            if request.source == "stored":
                stored = (
                    store.repository.get_dataset(request.dataset_id) if request.dataset_id else None
                )
                if stored is None or not stored.get("source_path"):
                    raise ValueError("Saved dataset was not found.")
                csv_text = Path(stored["source_path"]).read_text(encoding="utf-8")
            frame = load_dataset(
                DatasetSelection(
                    source=request.source,
                    name=request.name,
                    csv_text=csv_text,
                    target_column=request.target_column,
                )
            )
            return preview_frame(
                frame,
                rows=8,
                target_column=request.target_column,
                task=request.task,
                split=SplitConfig(**request.split.model_dump()),
                evaluation_protocol=request.evaluation.protocol
                if request.evaluation
                else "holdout",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/experiments", response_model=list[ExperimentResponse])
    def list_experiments() -> list[dict[str, Any]]:
        return store.list_experiments()

    @app.post("/api/v1/experiments", response_model=ExperimentResponse)
    def create_experiment(request: ExperimentCreateRequest) -> dict[str, Any]:
        return store.create_experiment(request)

    @app.get("/api/v1/experiments/{experiment_id}", response_model=ExperimentResponse)
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        return store.get_experiment(experiment_id)

    @app.put("/api/v1/experiments/{experiment_id}/result-layout", response_model=ExperimentResponse)
    def save_result_layout(experiment_id: str, request: ResultLayoutRequest) -> dict[str, Any]:
        if store.repository.get_experiment(experiment_id) is None:
            raise HTTPException(status_code=404, detail="Experiment was not found.")
        updated = store.repository.save_result_layout(experiment_id, request.tools)
        if updated is None:
            raise HTTPException(status_code=404, detail="Experiment was not found.")
        return updated

    @app.post("/api/v1/experiments/{experiment_id}/tournament", response_model=TournamentResponse)
    def create_tournament(experiment_id: str, request: TournamentRequest) -> dict[str, Any]:
        store.get_experiment(experiment_id)
        shared_fields = (
            "dataset",
            "task",
            "metric",
            "split",
            "preprocessing",
            "strategy",
            "pso",
            "evaluation",
            "runtime",
        )
        baseline = request.runs[0].model_dump(include=set(shared_fields))
        for spec in request.runs[1:]:
            if spec.model_dump(include=set(shared_fields)) != baseline:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Tournament runs must share the same dataset, task, metric, "
                        "preprocessing, evaluation, split, strategy and budget."
                    ),
                )
        seed = request.runs[0].split.random_state
        if seed is None:
            seed = secrets.randbits(32)
        prepared: list[ExperimentSpec] = []
        errors: list[dict[str, Any]] = []
        for index, spec in enumerate(request.runs):
            spec = spec.model_copy(deep=True)
            spec.split.random_state = seed
            spec.evaluation.random_state = seed
            spec.experiment_id = experiment_id
            validation_errors = _validate_request(spec, store.repository)
            if any(validation_errors.values()):
                errors.append({"index": index, "errors": validation_errors})
            prepared.append(spec)
        if errors:
            raise HTTPException(status_code=400, detail={"valid": False, "runs": errors})
        if store.manager is None:
            raise HTTPException(
                status_code=503,
                detail="The local run manager is unavailable in worker mode.",
            )
        snapshots = []
        for spec in prepared:
            run = store.create(spec)
            store.record_event(
                run.run_id,
                ProgressEvent(
                    "validation_passed",
                    {"source": "tournament", "experiment_id": experiment_id},
                ),
            )
            store.manager.submit(run.run_id, spec.runtime.timeout_seconds)
            snapshots.append(store.get_snapshot(run.run_id))
        return {"experiment_id": experiment_id, "runs": snapshots}

    @app.get("/api/v1/estimators", response_model=MetadataResponse)
    def estimators() -> dict[str, Any]:
        presets = estimator_presets()
        for name, preset in presets.items():
            preset["dependency"] = optional_dependency_status(preset.get("optional_dependency"))
            preset["defaults"] = {}
            for task in preset["tasks"]:
                preset["defaults"][task] = {
                    "fixed_params": default_fixed_params(name, task),
                    "search_space": default_search_space(name, task).to_schema(),
                    "allowed_params": sorted(allowed_estimator_params(name, task)),
                }
            preset["group"] = _estimator_group(name)
            preset["capabilities"] = _estimator_capabilities(name)
        return {
            "estimators": presets,
            "tasks": task_metadata(),
            "metrics": metric_metadata(),
        }

    @app.post("/api/v1/runs/validate", response_model=ValidationResponse)
    def validate_run(request: ExperimentSpec) -> dict[str, Any]:
        errors = _validate_request(request, store.repository)
        return {"valid": not any(errors.values()), "errors": errors}

    @app.post("/api/v1/workflow/validate", response_model=ValidationResponse)
    def validate_workflow(request: WorkflowValidationRequest) -> dict[str, Any]:
        errors = _validate_request(request.request, store.repository)
        sections = {
            "data": {"dataset", "task"},
            "model": {"dataset", "task", "estimator", "fixed_params"},
            "search": set(errors),
            "full": set(errors),
        }[request.stage]
        scoped = {name: values for name, values in errors.items() if name in sections}
        return {"stage": request.stage, "valid": not any(scoped.values()), "errors": scoped}

    @app.get("/api/v1/runs", response_model=list[RunResponse])
    def list_runs() -> list[dict[str, Any]]:
        return store.repository.list_runs()

    @app.post("/api/v1/runs", response_model=RunResponse)
    def create_run(request: ExperimentSpec) -> dict[str, Any]:
        if store.manager is None:
            raise HTTPException(status_code=503, detail="The local run manager is unavailable.")
        errors = _validate_request(request, store.repository)
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
        if store.manager is None:
            raise HTTPException(
                status_code=503, detail="The local run manager is unavailable in worker mode."
            )
        store.manager.submit(run.run_id, request.runtime.timeout_seconds)
        return store.get_snapshot(run.run_id)

    @app.post("/api/v1/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        cancelled = store.cancel(run_id)
        return {"run_id": run_id, "cancel_requested": cancelled}

    @app.post("/api/v1/runs/{run_id}/retry", response_model=RunResponse)
    def retry_run(run_id: str) -> dict[str, Any]:
        if store.manager is None:
            raise HTTPException(status_code=503, detail="The local run manager is unavailable.")
        retry = store.retry(run_id)
        store.record_event(
            retry.run_id,
            ProgressEvent("retry_created", {"parent_run_id": run_id}),
        )
        if store.manager is None:
            raise HTTPException(
                status_code=503, detail="The local run manager is unavailable in worker mode."
            )
        store.manager.submit(
            retry.run_id, retry.request.runtime.timeout_seconds, parent_run_id=run_id
        )
        return store.get_snapshot(retry.run_id)

    @app.get("/api/v1/runs/{run_id}", response_model=RunResponse)
    def get_run(run_id: str) -> dict[str, Any]:
        return store.get_snapshot(run_id)

    @app.get("/api/v1/runs/{run_id}/result")
    def get_result(run_id: str) -> dict[str, Any]:
        return store.get_result(run_id)

    @app.get("/api/v1/runs/{run_id}/history", response_model=RunHistoryResponse)
    def get_history(run_id: str, after: int = Query(default=0, ge=0)) -> dict[str, Any]:
        return {"run_id": run_id, "events": store.get_history(run_id, after=after)}

    @app.get("/api/v1/runs/{run_id}/artifacts", response_model=ArtifactResponse)
    def get_artifacts(run_id: str) -> dict[str, Any]:
        return store.get_artifacts(run_id)

    @app.get("/api/v1/runs/{run_id}/analysis", response_model=AnalysisResponse)
    def get_analysis(run_id: str) -> dict[str, Any]:
        snapshot = store.get_snapshot(run_id)
        analysis_path = (snapshot.get("artifacts") or {}).get("analysis")
        if not analysis_path or not Path(analysis_path).exists():
            raise HTTPException(
                status_code=404, detail="Analysis is available after a successful completed run."
            )
        analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
        request = snapshot.get("request")
        if "dataset" not in analysis and request:
            try:
                dataset = request["dataset"]
                frame = load_dataset(
                    DatasetSelection(
                        source=dataset["source"],
                        name=dataset.get("name"),
                        csv_text=dataset.get("csv_text"),
                        target_column=dataset["target_column"],
                    )
                )
                analysis["dataset"] = preview_frame(
                    frame,
                    rows=0,
                    target_column=dataset["target_column"],
                    task=request["task"],
                    split=SplitConfig(**request["split"]),
                )
            except Exception:
                # The saved metrics remain useful if an old external dataset is no longer available.
                pass
        return analysis

    @app.post("/api/v1/runs/{run_id}/feature-importance", response_model=ImportanceResponse)
    def calculate_feature_importance(run_id: str) -> dict[str, Any]:
        snapshot = store.get_snapshot(run_id)
        artifacts = snapshot.get("artifacts") or {}
        request_payload = snapshot.get("request")
        required = ("model", "dataset", "split_indices")
        if not request_payload or not all(
            Path(artifacts.get(key, "")).is_file() for key in required
        ):
            raise HTTPException(
                status_code=404, detail="A saved model and preprocessor are required."
            )
        request = ExperimentSpec(**request_payload)
        try:
            saved_indices = json.loads(Path(artifacts["split_indices"]).read_text(encoding="utf-8"))
            partitions = saved_indices["partitions"]
            split = "test" if partitions.get("test") else "validation"
            if split == "validation" and request.evaluation.protocol == "cross_validation":
                split = "train"
            model, features, _, target, mapping = load_saved_inputs(request, artifacts, split)
            names = list(features.columns)

            def scoring(estimator, X, y):
                report = evaluation_report(
                    estimator,
                    request.task,
                    request.metric,
                    X,
                    y,
                    positive_label=saved_positive_label(request, mapping),
                    decision_threshold=request.evaluation.decision_threshold,
                )
                return -report["cost"]

            result = permutation_importance(
                model,
                features,
                target,
                scoring=scoring,
                n_repeats=5,
                random_state=request.split.random_state,
            )
            items = sorted(
                [
                    {"feature": name, "importance": float(value), "std": float(std)}
                    for name, value, std in zip(
                        names,
                        result.importances_mean,
                        result.importances_std,
                        strict=True,
                    )
                ],
                key=lambda item: abs(float(str(item["importance"]))),
                reverse=True,
            )
            return {"available": True, "source": "permutation", "split": split, "items": items}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_id}/predictions", response_model=PredictionResponse)
    def get_predictions(
        run_id: str, split: Literal["validation", "train", "test"] = "validation", limit: int = 50
    ) -> dict[str, Any]:
        snapshot = store.get_snapshot(run_id)
        request_payload = snapshot.get("request")
        result_payload = snapshot.get("result")
        if request_payload is None:
            raise HTTPException(status_code=404, detail="Run configuration is not available.")
        if result_payload is None or result_payload.get("best_params") is None:
            raise HTTPException(
                status_code=404,
                detail="Predictions are only available for completed runs with a best model.",
            )
        request = ExperimentSpec(**request_payload)
        try:
            preview = _build_prediction_preview(
                request,
                result_payload["best_params"],
                split=split,
                limit=limit,
                artifacts=snapshot.get("artifacts") or {},
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        preview["run_id"] = run_id
        return preview

    @app.get("/api/v1/runs/{run_id}/exports/{kind}")
    def export_run(
        run_id: str,
        kind: Literal[
            "spec",
            "result",
            "analysis",
            "events",
            "predictions",
            "metrics",
            "manifest",
            "model",
            "selection_model",
            "preprocessor",
            "environment",
            "split_indices",
            "log",
        ],
        split: Literal["validation", "train", "test"] | None = None,
    ) -> Response:
        snapshot = store.get_snapshot(run_id)
        filename = f"pspso-{run_id[:8]}-{kind}"
        if kind in {
            "manifest",
            "model",
            "selection_model",
            "preprocessor",
            "environment",
            "split_indices",
            "log",
        }:
            artifact_path = (snapshot.get("artifacts") or {}).get(kind)
            if not artifact_path or not Path(artifact_path).is_file():
                raise HTTPException(status_code=404, detail=f"The {kind} artifact is unavailable.")
            path = Path(artifact_path)
            return FileResponse(path, filename=f"{filename}{path.suffix}")
        if kind == "spec":
            payload = snapshot.get("request")
        elif kind == "result":
            payload = snapshot.get("result")
        elif kind == "events":
            payload = store.get_history(run_id)
        elif kind in {"analysis", "metrics"}:
            analysis_path = (snapshot.get("artifacts") or {}).get("analysis")
            payload = (
                json.loads(Path(analysis_path).read_text(encoding="utf-8"))
                if analysis_path and Path(analysis_path).exists()
                else None
            )
            if kind == "metrics" and payload is not None:
                payload = {
                    "task": payload["task"],
                    "metric": payload["metric"],
                    "splits": {
                        name: value["metrics"]
                        for name, value in payload.items()
                        if name in {"train", "validation", "test"}
                    },
                    "cross_validation": payload.get("validation", {}).get("cross_validation"),
                }
        else:
            request_payload = snapshot.get("request")
            result_payload = snapshot.get("result")
            if not request_payload or not result_payload or not result_payload.get("best_params"):
                raise HTTPException(status_code=404, detail="Predictions are not ready.")
            request = ExperimentSpec(**request_payload)
            split_name: Literal["validation", "train", "test"] = (
                "test"
                if request.split.test_size > 0
                else "train"
                if request.evaluation.protocol == "cross_validation"
                else "validation"
            )
            try:
                preview = _build_prediction_preview(
                    request,
                    result_payload["best_params"],
                    split=split or split_name,
                    limit=1_000_000_000,
                    artifacts=snapshot.get("artifacts") or {},
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            rows = [
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"features", "probabilities"}
                    },
                    **{f"feature.{key}": value for key, value in row.get("features", {}).items()},
                    **{
                        f"probability.{key}": value
                        for key, value in row.get("probabilities", {}).items()
                    },
                }
                for row in preview["preview_rows"]
            ]
            columns = list(dict.fromkeys(key for row in rows for key in row))
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            return Response(
                buffer.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'},
            )
        if payload is None:
            raise HTTPException(status_code=404, detail=f"{kind.title()} is not ready.")
        return Response(
            json.dumps(payload, indent=2, default=str),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}.json"'},
        )

    @app.get("/api/v1/runs/{run_id}/events")
    async def stream_events(
        run_id: str,
        last_event_id: int | None = Header(default=None, alias="Last-Event-ID", ge=0),
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        if store.repository.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run was not found.")

        async def generator():
            cursor = max(last_event_id or 0, after)
            last_heartbeat = asyncio.get_running_loop().time()
            while True:
                current_events = store.get_history(run_id, after=cursor)
                for event in current_events:
                    cursor = int(event["sequence_number"])
                    yield _format_sse(event)
                    last_heartbeat = asyncio.get_running_loop().time()
                current_status = store.get_snapshot(run_id)["status"]
                if current_status in {"completed", "failed", "cancelled", "interrupted"}:
                    break
                if asyncio.get_running_loop().time() - last_heartbeat >= 10:
                    yield ": heartbeat\n\n"
                    last_heartbeat = asyncio.get_running_loop().time()
                await asyncio.sleep(0.25)

        return StreamingResponse(generator(), media_type="text/event-stream")

    if static_root.joinpath("index.html").exists():

        @app.get("/{frontend_path:path}", include_in_schema=False)
        def frontend_fallback(frontend_path: str) -> FileResponse:
            if frontend_path == "api" or frontend_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found.")
            requested = static_root / frontend_path
            if requested.is_file() and static_root in requested.resolve().parents:
                return FileResponse(requested)
            return FileResponse(static_root / "index.html")

    return app


def _format_sse(event: dict[str, Any]) -> str:
    return f"id: {event['sequence_number']}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"


def _build_prediction_preview(
    request: ExperimentSpec,
    best_params: dict[str, Any],
    *,
    split: Literal["validation", "train", "test"],
    limit: int,
    artifacts: dict[str, str],
) -> dict[str, Any]:
    model, feature_frame, raw_target, _, target_mapping = load_saved_inputs(
        request, artifacts, split
    )
    predictions = prediction_output(
        model,
        request.task,
        feature_frame,
        positive_label=saved_positive_label(request, target_mapping),
        decision_threshold=request.evaluation.decision_threshold,
    )
    rows: list[dict[str, Any]] = []
    predicted_labels = predictions["predictions"]
    score_values = predictions.get("scores")
    probability_values = predictions.get("probabilities")
    limit = max(1, min(limit, len(feature_frame)))
    for offset, (row_index, feature_row) in enumerate(feature_frame.head(limit).iterrows()):
        actual_value = raw_target.loc[row_index]
        predicted_value = predicted_labels[offset]
        row = {
            "row_index": int(row_index)
            if isinstance(row_index, (int, np.integer))
            else str(row_index),
            "actual": _json_scalar(actual_value),
            "predicted": _decode_prediction(predicted_value, target_mapping),
        }
        if request.task == "regression":
            actual_number = float(actual_value)
            predicted_number = float(predicted_value)
            row["residual"] = float(predicted_number - actual_number)
        else:
            row["score"] = float(score_values[offset]) if score_values is not None else None
            if probability_values is not None:
                row["probabilities"] = {
                    str(_decode_prediction(index, target_mapping)): float(value)
                    for index, value in enumerate(probability_values[offset])
                }
        row["features"] = {key: _json_scalar(value) for key, value in feature_row.to_dict().items()}
        rows.append(row)
    return {
        "split": split,
        "task": request.task,
        "metric": request.metric,
        "estimator": request.estimator,
        "feature_columns": list(feature_frame.columns),
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
    if value is None or isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_request(
    request: ExperimentSpec,
    repository: TrackingRepository | None = None,
) -> dict[str, list[str]]:
    errors: dict[str, list[str]] = {
        "dataset": [],
        "task": [],
        "estimator": [],
        "fixed_params": [],
        "search_space": [],
        "strategy": [],
    }
    normalized_metric = request.metric
    tasks = task_metadata()
    if request.task not in tasks:
        errors["task"].append(f"Unsupported task: {request.task!r}.")
    elif normalized_metric not in tasks[request.task]["metrics"]:
        errors["task"].append(
            f"Metric {request.metric!r} is not valid for {request.task!r}. "
            f"Use one of: {', '.join(tasks[request.task]['metrics'])}."
        )

    estimator_name = canonical_estimator_name(request.estimator)
    presets = estimator_presets()
    preset = presets.get(estimator_name)
    if preset is None:
        errors["estimator"].append(f"Unknown estimator: {request.estimator!r}.")
    else:
        if request.task not in preset["tasks"]:
            errors["estimator"].append(f"{preset['label']} does not support {request.task!r}.")
        dependency = optional_dependency_status(preset.get("optional_dependency"))
        if not dependency["installed"]:
            errors["estimator"].append(
                f"{preset['label']} requires {dependency['required']}. "
                f"Install it with `{dependency['install']}`."
            )

    frame = None
    try:
        csv_text = request.dataset.csv_text
        if request.dataset.source == "stored":
            stored = (
                repository.get_dataset(request.dataset.dataset_id)
                if repository and request.dataset.dataset_id
                else None
            )
            if stored is None or not stored.get("source_path"):
                raise ValueError("Saved dataset was not found.")
            csv_text = Path(stored["source_path"]).read_text(encoding="utf-8")
        frame = load_dataset(
            DatasetSelection(
                source=request.dataset.source,
                name=request.dataset.name,
                csv_text=csv_text,
                target_column=request.dataset.target_column,
            )
        )
        if request.dataset.target_column not in frame.columns:
            errors["dataset"].append(
                f"Target column {request.dataset.target_column!r} was not found."
            )
        elif request.task == "binary_classification":
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
            evaluation_protocol=request.evaluation.protocol,
            cv_folds=request.evaluation.folds,
            shuffle_folds=request.evaluation.shuffle,
            time_series=request.split.method == "chronological",
            time_gap=request.split.gap,
            stratify=request.split.stratify,
            positive_label=request.evaluation.positive_label,
            decision_threshold=request.evaluation.decision_threshold,
            refit_best=request.evaluation.refit_best,
            trial_workers=request.runtime.trial_workers,
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
                f"Unsupported fixed parameter(s) for {estimator_name}: {', '.join(invalid_fixed)}."
            )
        if invalid_search:
            errors["search_space"].append(
                f"Unsupported tunable parameter(s) for {estimator_name}: "
                f"{', '.join(invalid_search)}."
            )
        if not invalid_fixed and not invalid_search and not errors["estimator"]:
            if frame is not None and request.dataset.target_column in frame.columns:
                objectives = [request.fixed_params.get("objective")]
                objective_spec = (
                    request.search_space.get("objective") if request.search_space else None
                )
                if objective_spec is not None:
                    if objective_spec.type == "choice":
                        objectives.extend(objective_spec.values)
                    else:
                        errors["search_space"].append("Objective must use a choice search domain.")
                errors["estimator"].extend(
                    validate_objectives(
                        estimator_name,
                        request.task,
                        request.metric,
                        objectives,
                        frame[request.dataset.target_column],
                    )
                )
            if not errors["estimator"]:
                try:
                    lower, _ = space.bounds
                    sample_params = space.decode(lower.tolist())
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
            if request.evaluation.protocol == "cross_validation":
                bundle = prepare_cross_validation_bundle(
                    frame,
                    request.dataset.target_column,
                    request.task,
                    SplitConfig(**request.split.model_dump()),
                    PreprocessingConfig(**request.preprocessing.model_dump()),
                )
                target = frame.loc[bundle["X_development"].index, request.dataset.target_column]
                if request.evaluation.folds > len(target):
                    raise ValueError("Cross-validation folds exceed the development row count.")
                if request.split.method == "chronological":
                    folds = list(
                        TimeSeriesSplit(
                            n_splits=request.evaluation.folds, gap=request.split.gap
                        ).split(target)
                    )
                    if request.task != "regression":
                        classes = set(target)
                        if any(
                            set(target.iloc[train]) != classes
                            or set(target.iloc[validation]) != classes
                            for train, validation in folds
                        ):
                            raise ValueError(
                                "Every chronological training and validation window must contain "
                                "all target classes. Use fewer folds or more history."
                            )
                if request.split.stratify and request.task != "regression":
                    smallest_class = int(target.value_counts().min())
                    if request.evaluation.folds > smallest_class:
                        raise ValueError(
                            f"Cross-validation folds ({request.evaluation.folds}) exceed the "
                            f"smallest class count ({smallest_class})."
                        )
            else:
                prepare_tabular_data(
                    frame,
                    request.dataset.target_column,
                    request.task,
                    SplitConfig(**request.split.model_dump()),
                    PreprocessingConfig(**request.preprocessing.model_dump()),
                )
            if request.task == "binary_classification":
                encoded_positive_label(
                    frame, request.dataset.target_column, request.evaluation.positive_label
                )
        except Exception as exc:
            errors["dataset"].append(str(exc))

    return errors


def _estimator_group(name: str) -> str:
    if name in {"linear_regression", "logistic_regression", "elastic_net"}:
        return "Baselines"
    if name in {"random_forest", "extra_trees"}:
        return "Ensembles"
    if name in {"xgboost", "lightgbm", "hist_gradient_boosting"}:
        return "Boosting"
    if name in {"sklearn_mlp", "pytorch_mlp"}:
        return "Neural networks"
    return "Kernel methods"


def _estimator_capabilities(name: str) -> dict[str, Any]:
    classification_probability = name not in {"linear_regression", "elastic_net"}
    importance = {
        "random_forest": "native tree importance",
        "extra_trees": "native tree importance",
        "xgboost": "native tree importance",
        "lightgbm": "native tree importance",
        "logistic_regression": "model coefficients",
        "linear_regression": "model coefficients",
        "elastic_net": "model coefficients",
    }.get(name)
    return {
        "probability_output": classification_probability,
        "feature_importance": importance,
        "permutation_importance": True,
        "scaling_recommended": name
        in {
            "svm",
            "sklearn_mlp",
            "pytorch_mlp",
            "logistic_regression",
            "linear_regression",
            "elastic_net",
        },
        "epoch_progress": name == "pytorch_mlp",
    }


__all__ = ["ExperimentSpec", "create_app"]

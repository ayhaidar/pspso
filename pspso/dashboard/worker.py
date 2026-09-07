"""Child-process entry point for one recorded optimization run."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.base import clone

from pspso.config import ProgressEvent
from pspso.dashboard.artifacts import ArtifactStore
from pspso.dashboard.data import (
    SplitConfig,
    encoded_positive_label,
    preview_frame,
)
from pspso.dashboard.runtime import build_optimizer_inputs
from pspso.dashboard.tracking import TrackingRepository
from pspso.ensemble import CrossValidationEnsemble
from pspso.interpretability import native_feature_importance, transformed_feature_names
from pspso.metrics import evaluation_report
from pspso.optimizer import OptimizationCancelled, OptimizationTimedOut, PSPSOOptimizer


def execute_run(database: str | Path, run_id: str, attempt_id: str) -> int:
    repository = TrackingRepository(database)
    snapshot = repository.get_run(run_id)
    if snapshot is None or snapshot.get("request") is None:
        return 2
    request = snapshot["request"]
    try:
        if repository.cancel_requested(run_id):
            repository.record_event(
                run_id,
                ProgressEvent("run_cancelled", {"reason": "cancelled before start"}).to_dict(),
            )
            return 0
        X_train, y_train, X_val, y_val, optimizer = build_optimizer_inputs(request)
        repository.record_event(
            run_id,
            ProgressEvent(
                "dataset_prepared",
                {
                    "train_rows": int(getattr(X_train, "shape", [len(X_train)])[0]),
                    "validation_rows": 0 if X_val is None else len(X_val),
                    "features": int(getattr(X_train, "shape", [0, 0])[1]),
                    "attempt_id": attempt_id,
                },
            ).to_dict(),
        )

        def callback(event: ProgressEvent) -> None:
            repository.record_event(run_id, event.to_dict())

        result = optimizer.optimize(
            X_train,
            y_train,
            X_val,
            y_val,
            callback,
            should_cancel=lambda: (
                repository.cancel_requested(run_id)
                or repository.termination_reason(attempt_id) is not None
            ),
            defer_terminal_event=True,
        )
        result.run_id = run_id
        result.experiment_id = snapshot.get("experiment_id")
        frame = optimizer.dataset_frame
        bundle = optimizer.prepared_bundle
        if bundle is None:
            raise RuntimeError("The worker did not preserve its prepared data.")
        evaluation = request.get("evaluation", {})
        is_cross_validation = evaluation.get("protocol") == "cross_validation"
        result.optimizer_state["split_indices"] = {
            "partitions": bundle["split_indices"],
            "folds": result.optimizer_state.get("fold_indices", []),
        }
        analysis = None
        selection_model = None
        if is_cross_validation and result.model is None and result.best_params is not None:
            winner = min(
                (trial for trial in result.trials if trial.status == "completed"),
                key=lambda trial: float("inf") if trial.cost is None else trial.cost,
            )
            analysis = {
                "task": request["task"],
                "metric": request["metric"],
                "train": {"metrics": winner.train_metrics},
                "validation": {
                    "metrics": winner.validation_metrics,
                    "cross_validation": {
                        "folds": winner.fold_metrics,
                        "mean": winner.metric,
                        "standard_deviation": winner.metric_std,
                    },
                },
                "feature_importance": {
                    "available": False,
                    "items": [],
                    "reason": (
                        "The winning configuration was selected, but its final model could not "
                        "be fitted."
                    ),
                },
            }
        if result.model is not None:
            positive_label = (
                encoded_positive_label(
                    frame,
                    request["dataset"]["target_column"],
                    evaluation.get("positive_label"),
                )
                if request["task"] == "binary_classification"
                else None
            )
            decision_threshold = evaluation.get("decision_threshold", 0.5)
            if is_cross_validation:
                model = result.model
                X_train_analysis = bundle["X_development"]
                y_train_analysis = bundle["y_development"]
                best_trial = min(
                    (trial for trial in result.trials if trial.status == "completed"),
                    key=lambda trial: float("inf") if trial.cost is None else trial.cost,
                    default=None,
                )
                validation_analysis = {
                    "selected_metric": request["metric"],
                    "value": result.best_metric,
                    "cost": result.best_cost,
                    "metrics": {request["metric"]: result.best_metric},
                    "cross_validation": {
                        "folds": [] if best_trial is None else best_trial.fold_metrics,
                        "mean": result.best_metric,
                        "standard_deviation": None if best_trial is None else best_trial.metric_std,
                    },
                }
                if isinstance(model, CrossValidationEnsemble):
                    fitted_preprocessor = None
                    fitted_estimator = None
                else:
                    fitted_preprocessor = model.named_steps.get("preprocessor")
                    fitted_estimator = model.named_steps.get("estimator")
            else:
                model = result.model
                X_train_analysis = bundle["X_train_frame"]
                y_train_analysis = bundle["y_train"]
                validation_analysis = evaluation_report(
                    model,
                    request["task"],
                    request["metric"],
                    bundle["X_validation_frame"],
                    bundle["y_validation"],
                    positive_label=positive_label,
                    decision_threshold=decision_threshold,
                )
                if evaluation.get("refit_best", True):
                    selection_model = model
                    callback(
                        ProgressEvent("final_refit_started", {"best_params": result.best_params})
                    )
                    optimizer._check_runtime()
                    model = clone(model)
                    optimizer._fit_model(
                        model,
                        bundle["X_development_frame"],
                        bundle["y_development"],
                        callback,
                        phase="final_refit",
                    )
                    optimizer._check_runtime()
                    result.model = model
                    callback(
                        ProgressEvent(
                            "final_refit_completed",
                            {
                                "rows": len(bundle["y_development"]),
                                "best_params": result.best_params,
                            },
                        )
                    )
                    X_train_analysis = bundle["X_development_frame"]
                    y_train_analysis = bundle["y_development"]
                fitted_preprocessor = model.named_steps.get("preprocessor")
                fitted_estimator = model.named_steps.get("estimator")
            analysis = {
                "task": request["task"],
                "metric": request["metric"],
                "dataset": preview_frame(
                    frame,
                    rows=0,
                    target_column=request["dataset"]["target_column"],
                    task=request["task"],
                    split=SplitConfig(**request["split"]),
                ),
                "train": evaluation_report(
                    model,
                    request["task"],
                    request["metric"],
                    X_train_analysis,
                    y_train_analysis,
                    positive_label=positive_label,
                    decision_threshold=decision_threshold,
                ),
                "validation": validation_analysis,
                "feature_importance": (
                    {
                        "available": False,
                        "items": [],
                        "reason": (
                            "This saved cross-validation ensemble supports permutation importance."
                        ),
                    }
                    if isinstance(model, CrossValidationEnsemble)
                    else native_feature_importance(
                        fitted_estimator,
                        transformed_feature_names(fitted_preprocessor, bundle["feature_columns"]),
                    )
                ),
            }
            if is_cross_validation and bundle.get("X_test_frame") is not None:
                analysis["test"] = evaluation_report(
                    model,
                    request["task"],
                    request["metric"],
                    bundle["X_test_frame"],
                    bundle["y_test"],
                    positive_label=positive_label,
                    decision_threshold=decision_threshold,
                )
            elif not is_cross_validation and "X_test" in bundle:
                analysis["test"] = evaluation_report(
                    model,
                    request["task"],
                    request["metric"],
                    bundle["X_test_frame"],
                    bundle["y_test"],
                    positive_label=positive_label,
                    decision_threshold=decision_threshold,
                )
        if analysis is not None:
            profile = preview_frame(
                frame,
                rows=0,
                target_column=request["dataset"]["target_column"],
                task=request["task"],
            )
            partitions = bundle["split_indices"]
            profile["split_summary"] = {
                "available": True,
                "stratified": request["split"]["stratify"],
                "random_state": request["split"]["random_state"],
                "partitions": {
                    name: {
                        "rows": len(indices),
                        "percentage": 100 * len(indices) / len(frame),
                        "target": preview_frame(
                            frame.loc[indices],
                            rows=0,
                            target_column=request["dataset"]["target_column"],
                            task=request["task"],
                        )["target_summary"],
                    }
                    for name, indices in partitions.items()
                    if indices and (name != "development" or is_cross_validation)
                },
            }
            analysis["dataset"] = profile
        artifact_store = ArtifactStore(repository.workspace)
        transformer = (
            None
            if isinstance(result.model, CrossValidationEnsemble)
            else result.model.named_steps.get("preprocessor")
            if result.model is not None
            else bundle.get("transformer")
        )
        result.optimizer_state.update(optimizer.optimizer_state)
        artifacts = artifact_store.save_run(
            run_id,
            request,
            result,
            transformer=transformer,
            analysis=analysis,
            dataset=frame,
            selection_model=selection_model,
            provenance={
                "dataset_fingerprint": joblib.hash(frame),
                "device": str(
                    getattr(
                        (
                            result.model.models_[0].named_steps["estimator"]
                            if isinstance(result.model, CrossValidationEnsemble)
                            else result.model.named_steps["estimator"]
                        ),
                        "device_",
                        request.get("fixed_params", {}).get("device", "cpu"),
                    )
                )
                if result.model is not None
                else None,
                "seeds": {
                    "split": request["split"]["random_state"],
                    "optimization": optimizer.config.random_state,
                    "estimator": getattr(
                        (
                            result.model.models_[0].named_steps["estimator"]
                            if isinstance(result.model, CrossValidationEnsemble)
                            else result.model.named_steps["estimator"]
                        ),
                        "random_state",
                        None,
                    )
                    if result.model is not None
                    else None,
                },
            },
        )
        for warning in artifact_store.last_warnings:
            repository.record_event(
                run_id,
                ProgressEvent(
                    "run_warning", {"reason": "artifact_serialization", "message": warning}
                ).to_dict(),
            )
        repository.save_result(run_id, result)
        repository.save_artifacts(run_id, artifacts)
        repository.record_event(run_id, optimizer.terminal_event.to_dict())
        return 0 if optimizer.terminal_event.type == "run_completed" else 1
    except OptimizationCancelled as exc:
        repository.record_event(
            run_id, ProgressEvent("run_cancelled", {"reason": str(exc)}).to_dict()
        )
        return 0
    except OptimizationTimedOut as exc:
        repository.save_error(run_id, str(exc))
        repository.record_event(
            run_id, ProgressEvent("run_failed", {"error": str(exc), "reason": "timeout"}).to_dict()
        )
        return 1
    except Exception as exc:
        repository.save_error(run_id, str(exc))
        repository.record_event(
            run_id,
            ProgressEvent(
                "run_failed",
                {
                    "error": str(exc),
                    "reason": PSPSOOptimizer._failure_category(exc),
                },
            ).to_dict(),
        )
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a recorded pspso run in a local worker.")
    parser.add_argument("--database", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    raise SystemExit(execute_run(args.database, args.run_id, args.attempt_id))


if __name__ == "__main__":
    main()

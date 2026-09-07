"""Shared, non-web runtime construction for dashboard and CLI workers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from pspso.config import EstimatorConfig, OptimizationConfig
from pspso.dashboard.data import (
    DatasetSelection,
    PreprocessingConfig,
    SplitConfig,
    encoded_positive_label,
    load_dataset,
    prepare_cross_validation_bundle,
    prepare_prediction_bundle,
)
from pspso.estimators import create_estimator, default_fixed_params, default_search_space
from pspso.optimizer import PSPSOOptimizer
from pspso.search_space import Choice, FloatRange, IntRange, LogFloatRange, SearchSpace


def _value(source: Any, name: str) -> Any:
    return source[name] if isinstance(source, Mapping) else getattr(source, name)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Expected a mapping-like configuration value, got {type(value).__name__}.")


def build_search_space(request: Any) -> SearchSpace:
    """Translate the public search-space payload into optimizer domains."""

    requested_space = _value(request, "search_space")
    estimator = _value(request, "estimator")
    task = _value(request, "task")
    if not requested_space:
        return default_search_space(estimator, task)
    converted: dict[str, Any] = {}
    for name, spec in requested_space.items():
        if hasattr(spec, "model_dump"):
            spec = spec.model_dump()
        if isinstance(spec, Mapping) and "type" in spec:
            spec_type = spec["type"]
            if spec_type == "choice":
                converted[name] = Choice(spec["values"])
            elif spec_type == "int":
                converted[name] = IntRange(int(spec["low"]), int(spec["high"]))
            elif spec_type == "float":
                converted[name] = FloatRange(
                    float(spec["low"]), float(spec["high"]), int(spec.get("precision", 2))
                )
            elif spec_type == "log_float":
                converted[name] = LogFloatRange(
                    float(spec["low"]), float(spec["high"]), int(spec.get("precision", 4))
                )
            else:
                raise ValueError(f"Unsupported search-space spec type: {spec_type!r}")
        else:
            converted[name] = spec
    return SearchSpace(converted)


def build_optimizer_inputs(
    request: Any,
    *,
    validate_dataset: bool = False,
) -> tuple[Any, Any, Any, Any, PSPSOOptimizer]:
    """Load data and construct an optimizer without importing FastAPI."""

    dataset_payload = _as_dict(_value(request, "dataset"))
    split_payload = _as_dict(_value(request, "split"))
    preprocessing_payload = _as_dict(_value(request, "preprocessing"))
    pso_payload = _as_dict(_value(request, "pso"))
    runtime_payload = _as_dict(_value(request, "runtime"))
    task = _value(request, "task")
    estimator_name = _value(request, "estimator")
    dataset = DatasetSelection(
        source=dataset_payload["source"],
        name=dataset_payload.get("name"),
        csv_text=dataset_payload.get("csv_text"),
        target_column=dataset_payload["target_column"],
    )
    frame = load_dataset(dataset)
    split = SplitConfig(**split_payload)
    preprocessing = PreprocessingConfig(**preprocessing_payload)
    if validate_dataset:
        frame.head(1)
    fixed_params = {
        **default_fixed_params(estimator_name, task),
        **dict(_value(request, "fixed_params")),
    }
    if int(runtime_payload.get("trial_workers", 1)) > 1 and "n_jobs" in fixed_params:
        fixed_params["n_jobs"] = 1
    evaluation_payload = _as_dict(_value(request, "evaluation"))
    positive_label = evaluation_payload.get("positive_label")
    if task == "binary_classification":
        positive_label = encoded_positive_label(frame, dataset.target_column, positive_label)
    config = OptimizationConfig(
        task=task,
        metric=_value(request, "metric"),
        strategy=_value(request, "strategy"),
        validation_size=split.validation_size,
        test_size=split.test_size,
        random_state=split.random_state,
        n_particles=pso_payload["particles"],
        n_iterations=pso_payload["iterations"],
        pso_options={key: pso_payload[key] for key in ("c1", "c2", "w")},
        pso_topology=pso_payload["topology"],
        max_trials=runtime_payload.get("max_trials"),
        timeout_seconds=runtime_payload.get("timeout_seconds"),
        early_stopping_rounds=runtime_payload.get("early_stopping_rounds"),
        evaluation_protocol=evaluation_payload.get("protocol", "holdout"),
        cv_folds=evaluation_payload.get("folds", 5),
        shuffle_folds=evaluation_payload.get("shuffle", True),
        time_series=split.method == "chronological",
        time_gap=split.gap,
        stratify=split.stratify,
        positive_label=positive_label,
        decision_threshold=evaluation_payload.get("decision_threshold", 0.5),
        refit_best=evaluation_payload.get("refit_best", True),
        trial_workers=runtime_payload.get("trial_workers", 1),
        verbose=runtime_payload.get("verbose", 0),
    )
    if config.evaluation_protocol == "cross_validation":
        bundle = prepare_cross_validation_bundle(
            frame, dataset.target_column, task, split, preprocessing
        )
        preprocessor = bundle["preprocessor"]
        X_train = bundle["X_development"]
        y_train = bundle["y_development"]
        X_validation = y_validation = None
    else:
        bundle = prepare_prediction_bundle(frame, dataset.target_column, task, split, preprocessing)
        preprocessor = bundle["transformer"]
        X_train, y_train = bundle["X_train_frame"], bundle["y_train"]
        X_validation, y_validation = bundle["X_validation_frame"], bundle["y_validation"]

    def pipeline_factory(**params: Any) -> Pipeline:
        if config.trial_workers > 1 and "n_jobs" in params:
            params["n_jobs"] = 1
        model = create_estimator(estimator_name, task, params)
        return Pipeline([("preprocessor", clone(preprocessor)), ("estimator", model)])

    optimizer = PSPSOOptimizer(
        EstimatorConfig(factory=pipeline_factory, fixed_params=fixed_params),
        build_search_space(request),
        config,
    )
    optimizer.prepared_bundle = bundle
    optimizer.dataset_frame = frame
    return X_train, y_train, X_validation, y_validation, optimizer


__all__ = ["build_optimizer_inputs", "build_search_space"]

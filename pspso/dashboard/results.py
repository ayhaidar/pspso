"""Read saved model inputs without rerunning splits or fitting any estimator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from pspso.dashboard.data import PreprocessingConfig, SplitConfig, feature_frame
from pspso.dashboard.schemas import ExperimentSpec


def load_saved_inputs(
    request: ExperimentSpec,
    artifacts: dict[str, str],
    split: str,
) -> tuple[Any, pd.DataFrame, pd.Series, pd.Series, dict[int, Any] | None]:
    model_key = (
        "selection_model" if split == "validation" and request.evaluation.refit_best else "model"
    )
    required = (model_key, "dataset", "split_indices")
    missing = [name for name in required if not Path(artifacts.get(name, "")).is_file()]
    if missing:
        raise ValueError(
            f"Saved artifacts are unavailable: {', '.join(missing)}. "
            "Retry this run to produce reproducible results."
        )
    frame = joblib.load(artifacts["dataset"])
    indices = json.loads(Path(artifacts["split_indices"]).read_text(encoding="utf-8"))
    partitions = indices["partitions"]
    if split == "train" and (
        request.evaluation.protocol == "cross_validation" or request.evaluation.refit_best
    ):
        partition = "development"
    else:
        partition = split
    if partition not in partitions or not partitions[partition]:
        raise ValueError(f"This run has no saved {split} partition.")
    target_column = request.dataset.target_column
    features = feature_frame(
        frame,
        target_column,
        PreprocessingConfig(**request.preprocessing.model_dump()),
        SplitConfig(**request.split.model_dump()),
    ).loc[partitions[partition]]
    raw = frame[target_column].loc[partitions[partition]]
    mapping = None
    if request.task != "regression":
        categorical = pd.Categorical(frame[target_column])
        mapping = {
            index: value.item() if hasattr(value, "item") else value
            for index, value in enumerate(categorical.categories)
        }
        encoded = pd.Series(categorical.codes, index=frame.index).loc[partitions[partition]]
    else:
        encoded = raw
    return joblib.load(artifacts[model_key]), features, raw, encoded, mapping


def saved_positive_label(request: ExperimentSpec, mapping: dict[int, Any] | None) -> int | None:
    if request.task != "binary_classification" or request.evaluation.positive_label is None:
        return None
    for index, label in (mapping or {}).items():
        if str(label) == str(request.evaluation.positive_label):
            return index
    raise ValueError("The chosen positive class is not in the saved target labels.")

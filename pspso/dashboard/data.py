"""Dataset and preprocessing helpers for the dashboard API."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from pspso.dashboard.preprocessing import build_transformer

_DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"

_EXAMPLE_DATASETS: tuple[dict[str, Any], ...] = (
    {
        "name": "breast_cancer",
        "label": "Breast Cancer Wisconsin (Diagnostic)",
        "task": "binary_classification",
        "default_metric": "roc_auc",
        "target_column": "target",
        "target_description": "Malignant or benign diagnosis",
        "rows": 569,
        "features": 30,
        "source": "UCI via scikit-learn",
        "source_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "description": "Diagnostic measurements from digitized breast-mass images.",
    },
    {
        "name": "diabetes",
        "label": "Diabetes Progression",
        "task": "regression",
        "default_metric": "rmse",
        "target_column": "target",
        "target_description": "Disease progression one year after baseline",
        "rows": 442,
        "features": 10,
        "source": "Efron et al. via scikit-learn",
        "source_url": "https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset",
        "license": "See source",
        "license_url": "https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset",
        "description": (
            "Baseline clinical variables used to predict a continuous progression measure."
        ),
    },
    {
        "name": "wine",
        "label": "Wine Recognition",
        "task": "multiclass_classification",
        "default_metric": "accuracy",
        "target_column": "target",
        "target_description": "One of three wine cultivars",
        "rows": 178,
        "features": 13,
        "source": "UCI via scikit-learn",
        "source_url": "https://archive.ics.uci.edu/dataset/109/wine",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "description": "Chemical measurements used to identify three Italian wine cultivars.",
    },
    {
        "name": "banknote_authentication",
        "label": "Banknote Authentication",
        "task": "binary_classification",
        "default_metric": "roc_auc",
        "target_column": "class",
        "target_description": "Genuine or forged banknote",
        "rows": 1372,
        "features": 4,
        "source": "UCI Machine Learning Repository",
        "source_url": "https://archive.ics.uci.edu/dataset/267/banknote+authentication",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "description": "Wavelet features extracted from images of genuine and forged banknotes.",
    },
    {
        "name": "auto_mpg",
        "label": "Auto MPG",
        "task": "regression",
        "default_metric": "rmse",
        "target_column": "mpg",
        "target_description": "City-cycle fuel consumption in miles per gallon",
        "rows": 398,
        "features": 7,
        "source": "CMU StatLib via UCI",
        "source_url": "https://archive.ics.uci.edu/dataset/9/auto+mpg",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "description": "Vehicle attributes used for the standard fuel-economy regression task.",
    },
    {
        "name": "palmer_penguins",
        "label": "Palmer Penguins",
        "task": "multiclass_classification",
        "default_metric": "accuracy",
        "target_column": "species",
        "target_description": "Adelie, Chinstrap, or Gentoo species",
        "rows": 344,
        "features": 7,
        "source": "Palmer Station LTER",
        "source_url": "https://allisonhorst.github.io/palmerpenguins/",
        "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "description": "Field measurements with categorical features and realistic missing values.",
    },
)


@dataclass(frozen=True)
class DatasetSelection:
    source: str
    name: str | None = None
    csv_text: str | None = None
    target_column: str = "target"


@dataclass(frozen=True)
class SplitConfig:
    validation_size: float = 0.2
    test_size: float = 0.0
    stratify: bool = True
    random_state: int | None = 42
    method: str = "random"
    time_column: str | None = None
    gap: int = 0


@dataclass(frozen=True)
class PreprocessingConfig:
    scale_numeric: bool = True
    encode_categorical: bool = True
    ignored_columns: list[str] = field(default_factory=list)
    numeric_imputation: str = "median"
    categorical_imputation: str = "most_frequent"
    numeric_fill_value: float = 0.0
    categorical_fill_value: str = "Missing"
    add_missing_indicators: bool = False
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    outlier_method: str = "none"
    outlier_iqr_multiplier: float = 1.5
    outlier_lower_quantile: float = 0.01
    outlier_upper_quantile: float = 0.99


def example_datasets() -> list[dict[str, Any]]:
    """Return the bundled examples and their standard supervised-learning task."""

    return [dict(dataset) for dataset in _EXAMPLE_DATASETS]


def load_dataset(selection: DatasetSelection) -> pd.DataFrame:
    if selection.source == "example":
        if selection.name == "breast_cancer":
            data = datasets.load_breast_cancer(as_frame=True)
            frame = data.frame.copy()
            frame["target"] = data.target
            return frame
        if selection.name == "diabetes":
            data = datasets.load_diabetes(as_frame=True)
            frame = data.frame.copy()
            frame["target"] = data.target
            return frame
        if selection.name == "wine":
            data = datasets.load_wine(as_frame=True)
            frame = data.frame.copy()
            frame["target"] = data.target
            return frame
        if selection.name == "banknote_authentication":
            return pd.read_csv(_DATASET_DIR / "banknote_authentication.csv")
        if selection.name == "auto_mpg":
            frame = pd.read_csv(_DATASET_DIR / "auto_mpg.csv").drop(columns=["car_name"])
            frame["origin"] = frame["origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
            return frame
        if selection.name == "palmer_penguins":
            return pd.read_csv(_DATASET_DIR / "palmer_penguins.csv", na_values=["NA"])
        raise ValueError(f"Unknown example dataset: {selection.name!r}")
    if selection.source in {"csv", "stored"}:
        if not selection.csv_text:
            raise ValueError("csv_text is required for a CSV dataset.")
        return pd.read_csv(StringIO(selection.csv_text))
    raise ValueError("dataset source must be 'example' or 'csv'.")


def preview_csv(csv_text: str, rows: int = 5) -> dict[str, Any]:
    frame = pd.read_csv(StringIO(csv_text))
    return preview_frame(frame, rows=rows)


def preview_frame(
    frame: pd.DataFrame,
    rows: int = 5,
    target_column: str | None = None,
    task: str | None = None,
    split: SplitConfig | None = None,
    evaluation_protocol: str = "holdout",
) -> dict[str, Any]:
    """Return a JSON-safe tabular profile, target summary, and optional split preview."""

    numeric_columns = list(frame.select_dtypes(include=["number", "bool"]).columns)
    profile = {
        "columns": [
            {
                "name": column,
                "dtype": str(frame[column].dtype),
                "missing": int(frame[column].isna().sum()),
                "unique": int(frame[column].nunique(dropna=True)),
                "examples": [_scalar(value) for value in frame[column].dropna().unique()[:12]],
            }
            for column in frame.columns
        ],
        "row_count": int(len(frame)),
        "preview": [
            {key: _scalar(value) for key, value in row.items()}
            for row in frame.head(rows).to_dict(orient="records")
        ],
        "target_candidates": list(frame.columns),
        "summary": {
            "column_count": int(len(frame.columns)),
            "numeric_columns": int(len(numeric_columns)),
            "categorical_columns": int(len(frame.columns) - len(numeric_columns)),
            "total_missing": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        },
        "numeric_summary": [_numeric_summary(frame[column], column) for column in numeric_columns],
    }
    if target_column and target_column in frame.columns:
        profile["target_summary"] = _target_summary(frame[target_column], task)
        if split is not None and task is not None:
            profile["split_summary"] = _split_summary(
                ordered_frame(frame, split, target_column)[target_column],
                task,
                split,
                evaluation_protocol,
            )
    return profile


def _numeric_summary(series: pd.Series, name: str | None = None) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    return {
        "column": name,
        "count": int(values.count()),
        "missing": int(values.isna().sum()),
        "mean": _finite(values.mean()),
        "std": _finite(values.std()),
        "min": _finite(values.min()),
        "q25": _finite(values.quantile(0.25)),
        "median": _finite(values.median()),
        "q75": _finite(values.quantile(0.75)),
        "max": _finite(values.max()),
        "outliers_iqr": int(
            ((values < q1 - 1.5 * (q3 - q1)) | (values > q3 + 1.5 * (q3 - q1))).sum()
        ),
    }


def _target_summary(series: pd.Series, task: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique(dropna=True)),
    }
    if task in {"binary_classification", "multiclass_classification"}:
        counts = series.dropna().value_counts(sort=True)
        denominator = max(1, int(counts.sum()))
        result["distribution"] = [
            {
                "label": _scalar(label),
                "count": int(count),
                "percentage": round(float(count) / denominator * 100, 2),
            }
            for label, count in counts.items()
        ]
    elif task == "regression":
        result["statistics"] = _numeric_summary(series)
    return result


def ordered_frame(frame: pd.DataFrame, split: SplitConfig, target_column: str) -> pd.DataFrame:
    """Preserve source row identifiers while ordering a single chronological series."""
    if split.method != "chronological" or not split.time_column:
        return frame
    if split.time_column == target_column:
        raise ValueError("The time column must be different from the prediction target.")
    if split.time_column not in frame.columns:
        raise ValueError(f"Time column {split.time_column!r} was not found.")
    values = frame[split.time_column]
    order = (
        values
        if pd.api.types.is_numeric_dtype(values)
        else pd.to_datetime(values, errors="raise", utc=True)
    )
    if order.isna().any():
        raise ValueError("The time column contains missing timestamps.")
    if order.duplicated().any():
        raise ValueError(
            "Chronological evaluation requires one row per timestamp. "
            "Aggregate duplicate times first."
        )
    return frame.loc[order.sort_values(kind="stable").index]


def partition_positions(
    series: pd.Series, task: str, split: SplitConfig, protocol="holdout"
) -> dict[str, list[int]]:
    if split.validation_size + split.test_size >= 1:
        raise ValueError("Validation and test shares must leave training rows.")
    rows = len(series)
    indices = list(range(rows))
    chronological = split.method == "chronological"
    stratify = series if task != "regression" and split.stratify and not chronological else None
    if chronological:
        test_rows = int(np.ceil(rows * split.test_size))
        test_start = rows - test_rows
        development_end = test_start - (split.gap if test_rows else 0)
        development, test = indices[: max(0, development_end)], indices[test_start:]
        gaps = indices[max(0, development_end) : test_start]
        if not development:
            raise ValueError("The test share and gap leave no development rows.")
    elif split.test_size > 0:
        development, test = train_test_split(
            indices, test_size=split.test_size, random_state=split.random_state, stratify=stratify
        )
        gaps = []
    else:
        development, test, gaps = indices, [], []
    result = {"development": list(development), "test": list(test)}
    if protocol == "holdout":
        if chronological:
            validation_start = len(development) - int(np.ceil(rows * split.validation_size))
            train_end = validation_start - split.gap
            if train_end < 1 or validation_start < 1:
                raise ValueError("Validation, test and gap settings leave no training rows.")
            train, validation = development[:train_end], development[validation_start:]
            gaps += development[train_end:validation_start]
        else:
            train, validation = train_test_split(
                development,
                test_size=split.validation_size / (1 - split.test_size),
                random_state=split.random_state,
                stratify=series.iloc[np.asarray(development)] if stratify is not None else None,
            )
        result.update(train=list(train), validation=list(validation))
    if gaps:
        result["gap"] = sorted(gaps)
    return result


def _split_summary(
    series: pd.Series, task: str, split: SplitConfig, protocol="holdout"
) -> dict[str, Any]:
    try:
        positions = partition_positions(series, task, split, protocol)
        names = (
            ["development", "test", "gap"]
            if protocol == "cross_validation"
            else ["train", "validation", "test", "gap"]
        )
        return {
            "available": True,
            "stratified": task != "regression"
            and split.stratify
            and split.method != "chronological",
            "random_state": split.random_state,
            "method": split.method,
            "time_column": split.time_column,
            "gap": split.gap if split.method == "chronological" else 0,
            "partitions": {
                name: _split_partition(series.iloc[np.asarray(positions[name])], task, len(series))
                for name in names
                if positions.get(name)
            },
        }
    except Exception as exc:
        return {"available": False, "error": str(exc), "partitions": {}}


def _split_partition(series: pd.Series, task: str, total_rows: int) -> dict[str, Any]:
    return {
        "rows": int(len(series)),
        "percentage": round(len(series) / max(1, total_rows) * 100, 2),
        "target": _target_summary(series, task),
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if pd.notna(number) and number not in {float("inf"), float("-inf")} else None
    except (TypeError, ValueError):
        return None


def _scalar(value: Any) -> Any:
    converted = value.item() if hasattr(value, "item") else value
    if pd.isna(converted) or isinstance(converted, float) and not np.isfinite(converted):
        return None
    return (
        converted
        if isinstance(converted, (str, int, float, bool)) or converted is None
        else str(converted)
    )


def build_preprocessor(
    features: pd.DataFrame, preprocessing: PreprocessingConfig
) -> ColumnTransformer:
    """Build an unfitted transformer suitable for holdout or fold-local fitting."""

    return build_transformer(features, preprocessing)


def feature_frame(
    frame: pd.DataFrame, target_column: str, preprocessing: PreprocessingConfig, split: SplitConfig
) -> pd.DataFrame:
    ignored = set(preprocessing.ignored_columns)
    if split.method == "chronological" and split.time_column:
        ignored.add(split.time_column)
    return frame.drop(
        columns=[
            target_column,
            *[column for column in ignored if column in frame.columns and column != target_column],
        ]
    )


def encoded_positive_label(
    frame: pd.DataFrame, target_column: str, requested: Any | None
) -> int | None:
    """Translate a user-facing class label to the stable encoded target value."""

    if requested is None:
        return None
    categories = list(pd.Categorical(frame[target_column]).categories)
    for index, value in enumerate(categories):
        scalar = _scalar(value)
        if scalar == requested or str(scalar) == str(requested):
            return index
    raise ValueError(f"Positive label {requested!r} is not present in the target column.")


def _raw_bundle(frame, target_column, task, split, preprocessing, protocol):
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    if frame[target_column].isna().any():
        raise ValueError(
            "The target column contains missing values. Remove or label those rows before training."
        )
    frame = ordered_frame(frame, split, target_column)
    features = feature_frame(frame, target_column, preprocessing, split)
    raw = frame[target_column]
    encoded, mapping = raw, None
    if task != "regression":
        categorical = pd.Categorical(raw)
        encoded = pd.Series(categorical.codes, index=raw.index, name=target_column)
        mapping = {int(index): _scalar(value) for index, value in enumerate(categorical.categories)}
    positions = partition_positions(encoded, task, split, protocol)
    return frame, features, raw, encoded, mapping, positions


def prepare_cross_validation_bundle(
    frame: pd.DataFrame,
    target_column: str,
    task: str,
    split: SplitConfig,
    preprocessing: PreprocessingConfig,
) -> dict[str, Any]:
    """Reserve a final test partition, leaving preprocessing unfitted for CV folds."""
    frame, features, raw, encoded, mapping, positions = _raw_bundle(
        frame, target_column, task, split, preprocessing, "cross_validation"
    )
    development, test = positions["development"], positions["test"]
    return {
        "X_development": features.iloc[development],
        "y_development": encoded.iloc[development],
        "y_development_raw": raw.iloc[development],
        "X_test_frame": features.iloc[test] if test else None,
        "y_test": encoded.iloc[test] if test else None,
        "y_test_raw": raw.iloc[test] if test else None,
        "preprocessor": build_preprocessor(features, preprocessing),
        "target_mapping": mapping,
        "feature_columns": list(features.columns),
        "split_indices": {
            name: [_scalar(frame.index[index]) for index in indexes]
            for name, indexes in positions.items()
        },
    }


def prepare_prediction_bundle(
    frame: pd.DataFrame,
    target_column: str,
    task: str,
    split: SplitConfig,
    preprocessing: PreprocessingConfig,
) -> dict[str, Any]:
    frame, features, raw, encoded, mapping, positions = _raw_bundle(
        frame, target_column, task, split, preprocessing, "holdout"
    )
    transformer = build_preprocessor(features, preprocessing)
    train, validation, test = positions["train"], positions["validation"], positions["test"]
    bundle = {
        "X_train": transformer.fit_transform(features.iloc[train]),
        "X_validation": transformer.transform(features.iloc[validation]),
        "X_train_frame": features.iloc[train],
        "X_validation_frame": features.iloc[validation],
        "y_train": encoded.iloc[train],
        "y_validation": encoded.iloc[validation],
        "y_train_raw": raw.iloc[train],
        "y_validation_raw": raw.iloc[validation],
        "X_development_frame": features.iloc[positions["development"]],
        "y_development": encoded.iloc[positions["development"]],
        "y_development_raw": raw.iloc[positions["development"]],
        "transformer": transformer,
        "target_mapping": mapping,
        "feature_columns": list(features.columns),
        "split_indices": {
            name: [_scalar(frame.index[index]) for index in indexes]
            for name, indexes in positions.items()
        },
    }
    if test:
        bundle.update(
            X_test=transformer.transform(features.iloc[test]),
            y_test=encoded.iloc[test],
            X_test_frame=features.iloc[test],
            y_test_raw=raw.iloc[test],
        )
    return bundle


def prepare_tabular_data(
    frame: pd.DataFrame,
    target_column: str,
    task: str,
    split: SplitConfig,
    preprocessing: PreprocessingConfig,
) -> tuple[Any, Any, Any, Any, ColumnTransformer]:
    bundle = prepare_prediction_bundle(frame, target_column, task, split, preprocessing)
    return (
        bundle["X_train"],
        bundle["y_train"],
        bundle["X_validation"],
        bundle["y_validation"],
        bundle["transformer"],
    )

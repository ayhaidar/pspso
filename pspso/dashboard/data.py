"""Dataset and preprocessing helpers for the dashboard API."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import Any

import pandas as pd
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


@dataclass(frozen=True)
class PreprocessingConfig:
    scale_numeric: bool = True
    encode_categorical: bool = True
    ignored_columns: list[str] = field(default_factory=list)


def example_datasets() -> list[dict[str, Any]]:
    return [
        {
            "name": "breast_cancer",
            "label": "Breast Cancer",
            "task": "binary classification",
            "target_column": "target",
            "rows": 569,
        },
        {
            "name": "diabetes",
            "label": "Diabetes",
            "task": "regression",
            "target_column": "target",
            "rows": 442,
        },
    ]


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
        raise ValueError(f"Unknown example dataset: {selection.name!r}")
    if selection.source == "csv":
        if not selection.csv_text:
            raise ValueError("csv_text is required when source is 'csv'.")
        return pd.read_csv(StringIO(selection.csv_text))
    raise ValueError("dataset source must be 'example' or 'csv'.")


def preview_csv(csv_text: str, rows: int = 5) -> dict[str, Any]:
    frame = pd.read_csv(StringIO(csv_text))
    return preview_frame(frame, rows=rows)


def preview_frame(frame: pd.DataFrame, rows: int = 5) -> dict[str, Any]:
    return {
        "columns": [
            {
                "name": column,
                "dtype": str(frame[column].dtype),
                "missing": int(frame[column].isna().sum()),
                "unique": int(frame[column].nunique(dropna=True)),
            }
            for column in frame.columns
        ],
        "row_count": int(len(frame)),
        "preview": frame.head(rows).to_dict(orient="records"),
        "target_candidates": list(frame.columns),
    }


def prepare_prediction_bundle(
    frame: pd.DataFrame,
    target_column: str,
    task: str,
    split: SplitConfig,
    preprocessing: PreprocessingConfig,
) -> dict[str, Any]:
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    ignored = [column for column in preprocessing.ignored_columns if column in frame.columns]
    features = frame.drop(columns=[target_column, *ignored])
    raw_target = frame[target_column]
    target_mapping = None
    encoded_target = raw_target
    if task == "binary classification":
        categorical = pd.Categorical(raw_target)
        encoded_target = pd.Series(categorical.codes, index=raw_target.index, name=target_column)
        target_mapping = {
            int(index): value.item() if hasattr(value, "item") else value
            for index, value in enumerate(categorical.categories)
        }
    numeric_columns = list(features.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if preprocessing.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns and preprocessing.encode_categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns remain after preprocessing.")
    transformer = ColumnTransformer(transformers)
    stratify = encoded_target if task == "binary classification" and split.stratify else None
    (
        X_train_df,
        X_val_df,
        y_train_raw,
        y_val_raw,
        y_train,
        y_val,
    ) = train_test_split(
        features,
        raw_target,
        encoded_target,
        test_size=split.validation_size,
        random_state=split.random_state,
        stratify=stratify,
    )
    X_train = transformer.fit_transform(X_train_df)
    X_val = transformer.transform(X_val_df)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_val,
        "y_validation": y_val,
        "X_train_frame": X_train_df,
        "X_validation_frame": X_val_df,
        "y_train_raw": y_train_raw,
        "y_validation_raw": y_val_raw,
        "transformer": transformer,
        "target_mapping": target_mapping,
        "feature_columns": list(features.columns),
    }


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


__all__ = [
    "DatasetSelection",
    "PreprocessingConfig",
    "SplitConfig",
    "example_datasets",
    "load_dataset",
    "prepare_prediction_bundle",
    "prepare_tabular_data",
    "preview_csv",
    "preview_frame",
]

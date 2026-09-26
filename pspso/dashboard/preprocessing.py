"""Serializable feature preparation fitted independently on each training partition."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class FeatureValues(TransformerMixin, BaseEstimator):
    """Normalize declared numeric or categorical values without learning from held-out rows."""

    def __init__(self, numeric: bool = False):
        self.numeric = numeric

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        if self.numeric:
            return (
                frame.apply(pd.to_numeric, errors="raise")
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
            )
        return frame.apply(
            lambda column: column.map(lambda value: np.nan if pd.isna(value) else str(value))
        ).to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


class OutlierClipper(TransformerMixin, BaseEstimator):
    """Winsorize numeric features using bounds estimated from training observations only."""

    def __init__(self, method="iqr", multiplier=1.5, lower_quantile=0.01, upper_quantile=0.99):
        self.method = method
        self.multiplier = multiplier
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        bounds = []
        for column in values.T:
            finite = column[np.isfinite(column)]
            if not len(finite):
                bounds.append((-np.inf, np.inf))
            elif self.method == "iqr":
                q1, q3 = np.quantile(finite, [0.25, 0.75])
                spread = (q3 - q1) * self.multiplier
                bounds.append((q1 - spread, q3 + spread))
            else:
                bounds.append(
                    tuple(np.quantile(finite, [self.lower_quantile, self.upper_quantile]))
                )
        self.lower_ = np.asarray([item[0] for item in bounds])
        self.upper_ = np.asarray([item[1] for item in bounds])
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            input_features
            if input_features is not None
            else [f"x{index}" for index in range(self.n_features_in_)],
            dtype=object,
        )


def build_transformer(features: pd.DataFrame, settings: Any) -> ColumnTransformer:
    numeric, nominal, ordinal = [], [], []
    for column in features.columns:
        rule = settings.features.get(column, {})
        kind = rule.get("kind", "auto")
        if kind == "ordinal":
            ordinal.append(column)
        elif (
            kind == "numeric" or kind == "auto" and pd.api.types.is_numeric_dtype(features[column])
        ):
            pd.to_numeric(features[column], errors="raise")
            numeric.append(column)
        else:
            nominal.append(column)
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        steps: list[tuple[str, Any]] = [("values", FeatureValues(numeric=True))]
        if settings.outlier_method != "none":
            steps.append(
                (
                    "outliers",
                    OutlierClipper(
                        settings.outlier_method,
                        settings.outlier_iqr_multiplier,
                        settings.outlier_lower_quantile,
                        settings.outlier_upper_quantile,
                    ),
                )
            )
        steps.append(
            (
                "imputer",
                SimpleImputer(
                    strategy=settings.numeric_imputation,
                    fill_value=settings.numeric_fill_value,
                    add_indicator=settings.add_missing_indicators,
                    keep_empty_features=True,
                ),
            )
        )
        if settings.scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if nominal and settings.encode_categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("values", FeatureValues()),
                        (
                            "imputer",
                            SimpleImputer(
                                strategy=settings.categorical_imputation,
                                fill_value=settings.categorical_fill_value,
                                add_indicator=settings.add_missing_indicators,
                                keep_empty_features=True,
                            ),
                        ),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                nominal,
            )
        )
    if settings.encode_categorical:
        for index, column in enumerate(ordinal):
            categories = settings.features[column].get("categories", [])
            if not categories or len(set(categories)) != len(categories):
                raise ValueError(
                    f"Ordinal feature {column!r} needs a unique, ordered category list."
                )
            encoding = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy=settings.categorical_imputation,
                            fill_value=settings.categorical_fill_value,
                            keep_empty_features=True,
                        ),
                    ),
                    (
                        "encoder",
                        OrdinalEncoder(
                            categories=[categories],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )
            transformed: Any = encoding
            if settings.add_missing_indicators:
                transformed = FeatureUnion(
                    [
                        ("encoded", encoding),
                        ("missing", MissingIndicator(features="all", error_on_new=False)),
                    ]
                )
            transformers.append(
                (
                    f"ordinal_{index}",
                    Pipeline(
                        [
                            ("values", FeatureValues()),
                            ("encoding", transformed),
                        ]
                    ),
                    [column],
                )
            )
    if not transformers:
        raise ValueError("No usable feature columns remain after preprocessing.")
    return ColumnTransformer(transformers)

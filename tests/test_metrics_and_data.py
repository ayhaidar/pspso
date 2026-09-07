import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from pspso.dashboard.data import (
    DatasetSelection,
    PreprocessingConfig,
    SplitConfig,
    example_datasets,
    load_dataset,
    prepare_cross_validation_bundle,
    prepare_prediction_bundle,
    preview_frame,
)
from pspso.metrics import evaluation_report, prediction_output


def test_bundled_dataset_catalog_has_provenance_and_standard_tasks():
    catalog = {item["name"]: item for item in example_datasets()}

    assert {"banknote_authentication", "auto_mpg", "palmer_penguins"} <= catalog.keys()
    assert catalog["banknote_authentication"]["task"] == "binary_classification"
    assert catalog["auto_mpg"]["task"] == "regression"
    assert catalog["palmer_penguins"]["task"] == "multiclass_classification"
    assert all(item["source_url"].startswith("https://") for item in catalog.values())
    assert all(item["target_description"] for item in catalog.values())


@pytest.mark.parametrize(
    ("name", "target", "shape"),
    [
        ("banknote_authentication", "class", (1372, 5)),
        ("auto_mpg", "mpg", (398, 8)),
        ("palmer_penguins", "species", (344, 8)),
    ],
)
def test_bundled_external_datasets_load_offline(name, target, shape):
    frame = load_dataset(DatasetSelection(source="example", name=name, target_column=target))

    assert frame.shape == shape
    assert target in frame
    assert not frame[target].isna().any()

    if name == "auto_mpg":
        assert "car_name" not in frame
        assert set(frame["origin"]) == {"USA", "Europe", "Japan"}
        assert frame["horsepower"].isna().sum() == 6
    if name == "palmer_penguins":
        assert not pd.api.types.is_numeric_dtype(frame["island"])
        assert not pd.api.types.is_numeric_dtype(frame["sex"])
        assert frame.drop(columns=[target]).isna().any().any()


def test_binary_evaluation_report_includes_auc_sensitivity_and_specificity():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=5000).fit(X_train, y_train)

    report = evaluation_report(
        model, "binary_classification", "roc_auc", X_validation, y_validation
    )

    assert report["metrics"]["roc_auc"] > 0.9
    assert report["metrics"]["sensitivity"] is not None
    assert report["metrics"]["specificity"] is not None
    assert report["roc_curve"]["auc"] == report["metrics"]["roc_auc"]
    assert report["confusion_matrix"]["true_positive"] >= 0
    assert report["precision_recall_curve"]["auc"] > 0.9
    assert len(report["threshold_diagnostics"]) > 10


def test_regression_evaluation_report_includes_residual_diagnostics():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = X.ravel() * 2 + 1
    model = LinearRegression().fit(X, y)

    report = evaluation_report(model, "regression", "rmse", X, y)

    assert len(report["regression_diagnostics"]["actual"]) == len(y)
    assert max(abs(value) for value in report["regression_diagnostics"]["residuals"]) < 1e-9


def test_multiclass_prediction_output_returns_probabilities_not_binary_scores():
    X, y = load_wine(return_X_y=True)
    model = RandomForestClassifier(n_estimators=20, random_state=7).fit(X, y)

    output = prediction_output(model, "multiclass_classification", X[:4])

    assert "scores" not in output
    assert output["probabilities"].shape == (4, 3)


def test_prediction_bundle_reserves_test_rows_without_preprocessing_leakage():
    frame = pd.DataFrame(
        {"numeric": np.arange(50), "category": ["a", "b"] * 25, "target": [0, 1] * 25}
    )
    bundle = prepare_prediction_bundle(
        frame,
        "target",
        "binary_classification",
        SplitConfig(validation_size=0.2, test_size=0.2, random_state=7),
        PreprocessingConfig(),
    )

    assert len(bundle["X_train_frame"]) == 30
    assert len(bundle["X_validation_frame"]) == 10
    assert len(bundle["X_test_frame"]) == 10
    assert set(bundle["X_train_frame"].index).isdisjoint(bundle["X_test_frame"].index)


def test_dataset_profile_reports_target_distribution_statistics_and_split():
    frame = pd.DataFrame(
        {
            "numeric": np.arange(20, dtype=float),
            "category": ["a", "b"] * 10,
            "target": [0, 1] * 10,
        }
    )

    profile = preview_frame(
        frame,
        target_column="target",
        task="binary_classification",
        split=SplitConfig(validation_size=0.2, test_size=0.2, random_state=7),
    )

    assert profile["summary"]["column_count"] == 3
    assert profile["target_summary"]["distribution"] == [
        {"label": 0, "count": 10, "percentage": 50.0},
        {"label": 1, "count": 10, "percentage": 50.0},
    ]
    partitions = profile["split_summary"]["partitions"]
    assert {name: row["rows"] for name, row in partitions.items()} == {
        "train": 12,
        "validation": 4,
        "test": 4,
    }
    assert profile["numeric_summary"][0]["mean"] == 9.5


def test_configurable_imputation_handles_missing_feature_values():
    frame = pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0, 4.0] * 5,
            "category": ["a", None, "b", "a"] * 5,
            "target": [0, 1, 0, 1] * 5,
        }
    )
    bundle = prepare_prediction_bundle(
        frame,
        "target",
        "binary_classification",
        SplitConfig(validation_size=0.2, test_size=0.2, random_state=7),
        PreprocessingConfig(
            numeric_imputation="mean",
            categorical_imputation="constant",
            categorical_fill_value="Unknown",
            add_missing_indicators=True,
        ),
    )

    train = (
        bundle["X_train"].toarray() if hasattr(bundle["X_train"], "toarray") else bundle["X_train"]
    )
    validation = (
        bundle["X_validation"].toarray()
        if hasattr(bundle["X_validation"], "toarray")
        else bundle["X_validation"]
    )
    assert np.isfinite(train).all()
    assert np.isfinite(validation).all()


def test_missing_target_values_are_rejected_before_training():
    frame = pd.DataFrame({"feature": range(10), "target": [0, 1, 0, 1, None, 1, 0, 1, 0, 1]})

    with pytest.raises(ValueError, match="target column contains missing values"):
        prepare_prediction_bundle(
            frame,
            "target",
            "binary_classification",
            SplitConfig(validation_size=0.2, random_state=7),
            PreprocessingConfig(),
        )


def test_cross_validation_bundle_preserves_unfitted_preprocessing_and_split_indices():
    frame = pd.DataFrame(
        {
            "feature": np.arange(30, dtype=float),
            "category": ["a", "b", "c"] * 10,
            "target": ["negative", "positive"] * 15,
        }
    )

    first = prepare_cross_validation_bundle(
        frame,
        "target",
        "binary_classification",
        SplitConfig(test_size=0.2, random_state=19),
        PreprocessingConfig(),
    )
    second = prepare_cross_validation_bundle(
        frame,
        "target",
        "binary_classification",
        SplitConfig(test_size=0.2, random_state=19),
        PreprocessingConfig(),
    )

    assert not hasattr(first["preprocessor"], "transformers_")
    assert first["split_indices"] == second["split_indices"]
    assert set(first["split_indices"]["development"]).isdisjoint(first["split_indices"]["test"])


def test_explicit_positive_label_controls_binary_auc_and_threshold_diagnostics():
    X, numeric_target = load_breast_cancer(return_X_y=True)
    target = np.where(numeric_target == 1, "healthy", "malignant")
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, target, test_size=0.25, random_state=42, stratify=target
    )
    model = LogisticRegression(max_iter=5000).fit(X_train, y_train)

    report = evaluation_report(
        model,
        "binary_classification",
        "roc_auc",
        X_validation,
        y_validation,
        positive_label="malignant",
        decision_threshold=0.35,
    )

    assert report["positive_label"] == "malignant"
    assert report["decision_threshold"] == 0.35
    assert report["metrics"]["roc_auc"] > 0.9

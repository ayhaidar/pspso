import joblib
import numpy as np
import pandas as pd
import pytest

from pspso.dashboard.app import _validate_request
from pspso.dashboard.data import (
    PreprocessingConfig,
    SplitConfig,
    build_preprocessor,
    ordered_frame,
    prepare_prediction_bundle,
    preview_frame,
)
from pspso.dashboard.runtime import build_optimizer_inputs
from pspso.dashboard.schemas import ExperimentSpec


def test_chronological_holdout_sorts_dates_preserves_ids_and_reserves_gap():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=30).astype(str),
            "x": range(30),
            "target": np.arange(30) * 2,
        }
    ).sample(frac=1, random_state=3)
    split = SplitConfig(
        method="chronological", time_column="time", gap=2, validation_size=0.2, test_size=0.2
    )
    bundle = prepare_prediction_bundle(frame, "target", "regression", split, PreprocessingConfig())
    assert bundle["feature_columns"] == ["x"]
    indices = bundle["split_indices"]
    assert indices["train"] == list(range(14))
    assert indices["validation"] == list(range(16, 22))
    assert indices["test"] == list(range(24, 30))
    assert indices["gap"] == [14, 15, 22, 23]
    profile = preview_frame(frame, target_column="target", task="regression", split=split)
    partitions = profile["split_summary"]["partitions"]
    assert {name: value["rows"] for name, value in partitions.items()} == {
        "train": 14,
        "validation": 6,
        "test": 6,
        "gap": 4,
    }
    assert not profile["split_summary"]["stratified"]


def test_outlier_bounds_and_imputation_use_training_rows_only():
    frame = pd.DataFrame(
        {"x": [0, 1, 2, np.nan, 4, 5, 1000, -1000, 2000, -2000], "target": range(10)}
    )
    bundle = prepare_prediction_bundle(
        frame,
        "target",
        "regression",
        SplitConfig(method="chronological", test_size=0.2),
        PreprocessingConfig(outlier_method="iqr", scale_numeric=False),
    )
    numeric = bundle["transformer"].named_transformers_["numeric"]
    np.testing.assert_allclose(numeric.named_steps["outliers"].lower_, [-3.5])
    np.testing.assert_allclose(numeric.named_steps["outliers"].upper_, [8.5])
    np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, [2])
    np.testing.assert_allclose(bundle["X_validation"].ravel(), [8.5, -3.5])
    np.testing.assert_allclose(bundle["X_test"].ravel(), [8.5, -3.5])


def test_nominal_numeric_codes_and_explicit_ordinal_order_survive_serialization(tmp_path):
    frame = pd.DataFrame({"code": [10, 20, 10, 20], "grade": ["high", "low", None, "medium"]})
    transformer = build_preprocessor(
        frame,
        PreprocessingConfig(
            scale_numeric=False,
            features={
                "code": {"kind": "nominal"},
                "grade": {"kind": "ordinal", "categories": ["low", "medium", "high"]},
            },
        ),
    )
    transformed = transformer.fit_transform(frame)
    np.testing.assert_allclose(transformed[:, -1], [2, 0, 2, 1])
    assert transformed.shape == (4, 3)
    path = tmp_path / "preprocessor.joblib"
    joblib.dump(transformer, path)
    restored = joblib.load(path)
    unknown = restored.transform(pd.DataFrame({"code": [99], "grade": ["new"]}))
    np.testing.assert_allclose(unknown, [[0, 0, -1]])
    assert len(restored.get_feature_names_out()) == 3


def test_empty_numeric_column_and_ordinal_missing_indicator_are_stable():
    frame = pd.DataFrame({"empty": [np.nan] * 4, "grade": ["low", None, "high", "low"]})
    transformer = build_preprocessor(
        frame,
        PreprocessingConfig(
            outlier_method="quantile",
            add_missing_indicators=True,
            features={"grade": {"kind": "ordinal", "categories": ["low", "high"]}},
        ),
    )
    values = transformer.fit_transform(frame)
    assert np.isfinite(values).all()
    assert values.shape == (4, 4)
    np.testing.assert_allclose(values[:, -1], [0, 1, 0, 0])


def test_expanding_cv_never_trains_on_future_rows_and_excludes_test():
    spec = ExperimentSpec(
        task="regression",
        metric="mae",
        estimator="linear_regression",
        dataset={
            "source": "csv",
            "csv_text": pd.DataFrame({"x": range(40), "target": np.arange(40) * 2}).to_csv(
                index=False
            ),
        },
        split={"method": "chronological", "gap": 2, "test_size": 0.2},
        evaluation={"protocol": "cross_validation", "folds": 3},
        strategy="random",
        runtime={"max_trials": 1},
    )
    assert not spec.evaluation.shuffle and not spec.split.stratify
    X, y, X_val, y_val, optimizer = build_optimizer_inputs(spec)
    result = optimizer.optimize(X, y, X_val, y_val)
    assert result.trials[0].status == "completed"
    assert len(result.optimizer_state["fold_indices"]) == 3
    previous_size = 0
    for fold in result.optimizer_state["fold_indices"]:
        train, validation = fold["train"], fold["validation"]
        assert max(train) + 2 < min(validation)
        assert len(train) > previous_size
        assert max(validation) < 30
        previous_size = len(train)
    assert optimizer.prepared_bundle["split_indices"]["test"] == list(range(32, 40))


@pytest.mark.parametrize(
    "times,message",
    [
        (["bad", "2024-01-01"], ""),
        (["2024-01-01", "2024-01-01"], "one row per timestamp"),
        ([None, "2024-01-01"], "missing"),
    ],
)
def test_invalid_chronology_is_rejected(times, message):
    with pytest.raises(ValueError, match=message):
        ordered_frame(
            pd.DataFrame({"time": times, "target": [0, 1]}),
            SplitConfig(method="chronological", time_column="time"),
            "target",
        )


def test_impossible_chronological_fold_gap_fails_before_submission():
    spec = ExperimentSpec(
        task="regression",
        metric="mae",
        estimator="linear_regression",
        dataset={
            "source": "csv",
            "csv_text": pd.DataFrame({"x": range(20), "target": range(20)}).to_csv(index=False),
        },
        split={"method": "chronological", "gap": 12, "test_size": 0.1},
        evaluation={"protocol": "cross_validation", "folds": 5},
    )
    errors = _validate_request(spec)
    assert errors["dataset"]

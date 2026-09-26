"""Preflight validation for estimator objectives and target domains."""

from typing import Any

import numpy as np


def validate_objectives(
    estimator: str, task: str, metric: str, objectives: list[Any], target: Any
) -> list[str]:
    """Check every requested objective, including choices above the search lower bound."""
    errors: list[str] = []
    if estimator not in {"xgboost", "lightgbm"}:
        return errors
    xgb_regression = {
        "reg:squarederror",
        "reg:squaredlogerror",
        "reg:logistic",
        "reg:pseudohubererror",
        "reg:absoluteerror",
        "reg:quantileerror",
        "reg:gamma",
        "reg:tweedie",
        "count:poisson",
    }
    lgb_regression = {
        "regression",
        "regression_l2",
        "l2",
        "mean_squared_error",
        "mse",
        "l2_root",
        "root_mean_squared_error",
        "rmse",
        "regression_l1",
        "l1",
        "mean_absolute_error",
        "mae",
        "huber",
        "fair",
        "poisson",
        "quantile",
        "mape",
        "mean_absolute_percentage_error",
        "gamma",
        "tweedie",
    }
    compatible = {
        "xgboost": {
            "regression": xgb_regression,
            "binary_classification": {"binary:logistic", "binary:hinge"},
            "multiclass_classification": {"multi:softprob", "multi:softmax"},
        },
        "lightgbm": {
            "regression": lgb_regression,
            "binary_classification": {
                "binary",
                "cross_entropy",
                "xentropy",
                "cross_entropy_lambda",
                "xentlambda",
            },
            "multiclass_classification": {
                "multiclass",
                "softmax",
                "multiclassova",
                "multiclass_ova",
                "ova",
                "ovr",
            },
        },
    }[estimator][task]
    for objective in objectives:
        if objective is None:
            continue
        if not isinstance(objective, str) or objective not in compatible:
            errors.append(f"{estimator} objective {objective!r} is not supported for {task}.")
            continue
        if objective == "binary:hinge" and metric not in {"accuracy", "f1_macro"}:
            errors.append(
                "XGBoost binary:hinge does not provide probability scores; "
                "use binary:logistic for this metric."
            )
        if task != "regression":
            continue
        try:
            values = np.asarray(target, dtype=float)
        except (TypeError, ValueError):
            errors.append("Regression objectives require numeric target values.")
            continue
        if not np.isfinite(values).all():
            errors.append("Regression objectives require finite target values.")
        elif objective in {"reg:gamma", "gamma"} and (values <= 0).any():
            errors.append(
                f"{estimator} objective {objective!r} requires strictly positive target values."
            )
        elif objective in {"reg:tweedie", "tweedie", "count:poisson", "poisson"} and (
            (values < 0).any() or not (values > 0).any()
        ):
            errors.append(
                f"{estimator} objective {objective!r} requires nonnegative targets "
                "with at least one positive value."
            )
        elif objective == "reg:squaredlogerror" and (values <= -1).any():
            errors.append("XGBoost reg:squaredlogerror requires target values greater than -1.")
        elif objective == "reg:logistic" and ((values < 0).any() or (values > 1).any()):
            errors.append("XGBoost reg:logistic requires target values in [0, 1].")
    return errors

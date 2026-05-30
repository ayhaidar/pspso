"""Estimator presets and factories used by pspso."""

from __future__ import annotations

import importlib.util
from typing import Any, Callable, Mapping

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from .config import EstimatorConfig


def default_search_space(estimator: str, task: str) -> dict[str, list[Any]]:
    """Return legacy-compatible default search spaces."""

    estimator = normalize_estimator_name(estimator)
    if estimator == "svm":
        return {
            "kernel": ["linear", "rbf", "poly"],
            "gamma": [0.1, 10, 1],
            "C": [0.1, 10, 1],
            "degree": [1, 6, 0],
        }
    if estimator == "xgboost":
        params = {
            "learning_rate": [0.05, 0.3, 2],
            "max_depth": [1, 10, 0],
            "n_estimators": [10, 100, 0],
            "subsample": [0.7, 1.0, 2],
        }
        return params
    if estimator == "gbdt":
        params = {
            "learning_rate": [0.05, 0.3, 2],
            "max_depth": [1, 10, 0],
            "n_estimators": [10, 100, 0],
            "subsample": [0.7, 1.0, 2],
        }
        if task == "regression":
            params = {"objective": ["regression", "tweedie"], **params}
        return params
    if estimator == "mlp":
        return {
            "learning_rate": [0.001, 0.1, 3],
            "neurons": [4, 64, 0],
            "hiddenactivation": ["relu", "logistic", "tanh"],
        }
    if estimator == "random_forest":
        return {
            "n_estimators": [10, 100, 0],
            "max_depth": [2, 12, 0],
        }
    if estimator == "logistic_regression":
        return {"C": [0.1, 10, 1]}
    if estimator == "linear_regression":
        return {"fit_intercept": ["true", "false"]}
    raise ValueError(f"Unknown estimator preset: {estimator!r}")


def default_fixed_params(estimator: str, task: str) -> dict[str, Any]:
    """Return conservative fixed parameters for the built-in presets."""

    estimator = normalize_estimator_name(estimator)
    if estimator == "svm":
        if task == "binary classification":
            return {"kernel": "rbf", "C": 5.0, "gamma": 5.0, "probability": True}
        return {"kernel": "rbf", "C": 5.0, "gamma": 5.0}
    if estimator == "xgboost":
        if task == "binary classification":
            return {"objective": "binary:logistic", "eval_metric": "auc"}
        return {"objective": "reg:squarederror", "eval_metric": "rmse"}
    if estimator == "gbdt":
        if task == "binary classification":
            return {"objective": "binary", "boosting_type": "gbdt"}
        return {"objective": "regression", "boosting_type": "gbdt"}
    if estimator == "mlp":
        return {"neurons": 16, "hiddenactivation": "relu", "epochs": 100}
    if estimator == "random_forest":
        return {}
    if estimator in {"logistic_regression", "linear_regression"}:
        return {}
    raise ValueError(f"Unknown estimator preset: {estimator!r}")


def estimator_presets() -> dict[str, dict[str, Any]]:
    return {
        "svm": {
            "label": "Support Vector Machine",
            "tasks": ["regression", "binary classification"],
            "optional_dependency": None,
            "description": "Strong baseline for smaller tabular datasets. Supports linear, RBF, and polynomial kernels.",
        },
        "xgboost": {
            "label": "XGBoost",
            "tasks": ["regression", "binary classification"],
            "optional_dependency": "xgboost",
            "install": "uv sync --extra xgboost",
            "description": "Gradient-boosted trees from the optional XGBoost backend.",
        },
        "gbdt": {
            "label": "LightGBM GBDT",
            "tasks": ["regression", "binary classification"],
            "optional_dependency": "lightgbm",
            "install": "uv sync --extra lightgbm",
            "description": "Gradient-boosted decision trees from the optional LightGBM backend.",
        },
        "mlp": {
            "label": "Multi-layer Perceptron",
            "tasks": ["regression", "binary classification"],
            "optional_dependency": None,
            "description": "Scikit-learn neural network preset for compact tabular experiments.",
        },
        "random_forest": {
            "label": "Random Forest",
            "tasks": ["regression", "binary classification"],
            "optional_dependency": None,
            "description": "Tree ensemble with robust defaults and simple integer search spaces.",
        },
        "logistic_regression": {
            "label": "Logistic Regression",
            "tasks": ["binary classification"],
            "optional_dependency": None,
            "description": "Linear classifier for binary classification baselines.",
        },
        "linear_regression": {
            "label": "Linear Regression",
            "tasks": ["regression"],
            "optional_dependency": None,
            "description": "Linear regression baseline for regression tasks.",
        },
    }


def task_metadata() -> dict[str, dict[str, Any]]:
    """Human-readable dashboard metadata for supported task types."""

    return {
        "regression": {
            "label": "Regression",
            "description": "Predict a continuous numeric target. The dashboard currently validates regression with RMSE.",
            "metrics": ["rmse"],
        },
        "binary classification": {
            "label": "Binary classification",
            "description": "Predict one of two classes. Use ROC AUC when probability ranking matters, or accuracy for direct class correctness.",
            "metrics": ["roc_auc", "accuracy"],
        },
    }


def metric_metadata() -> dict[str, dict[str, str]]:
    """Human-readable dashboard metadata for supported metrics."""

    return {
        "rmse": {
            "label": "RMSE",
            "description": "Root mean squared error. Lower validation RMSE is better.",
        },
        "accuracy": {
            "label": "Accuracy",
            "description": "Fraction of validation records classified correctly. Higher is better.",
        },
        "roc_auc": {
            "label": "ROC AUC",
            "description": "Area under the ROC curve using scores or probabilities. Higher is better.",
        },
    }


def optional_dependency_status(package: str | None) -> dict[str, Any]:
    """Return whether an optional estimator dependency is available."""

    if package is None:
        return {"required": None, "installed": True, "install": None}
    installed = importlib.util.find_spec(package) is not None
    extra = "xgboost" if package == "xgboost" else "lightgbm"
    return {
        "required": package,
        "installed": installed,
        "install": f"uv sync --extra {extra}",
    }


def allowed_estimator_params(estimator: str, task: str) -> set[str]:
    """Return accepted constructor/search-space params for a built-in preset."""

    estimator = normalize_estimator_name(estimator)
    virtual = {
        "mlp": {"neurons", "hiddenactivation", "epochs", "optimizer", "learning_rate"},
        "xgboost": {"objective", "eval_metric", "learning_rate", "max_depth", "n_estimators", "subsample"},
        "gbdt": {"objective", "boosting_type", "learning_rate", "max_depth", "n_estimators", "subsample"},
    }
    if estimator in virtual:
        return set(virtual[estimator])
    try:
        model = _create_named_estimator(estimator, task, {})
    except ImportError:
        return set(default_fixed_params(estimator, task)) | set(default_search_space(estimator, task))
    return set(model.get_params(deep=False))


def normalize_estimator_name(name: str) -> str:
    aliases = {
        "lightgbm": "gbdt",
        "lgbm": "gbdt",
        "svc": "svm",
        "svr": "svm",
        "rf": "random_forest",
    }
    return aliases.get(name.lower(), name.lower())


def create_estimator(
    estimator: str | EstimatorConfig | Callable[..., Any],
    task: str,
    params: Mapping[str, Any],
) -> Any:
    """Create an unfitted estimator from a built-in preset or user factory."""

    if isinstance(estimator, EstimatorConfig):
        estimator.validate()
        merged = {**dict(estimator.fixed_params), **dict(params)}
        if estimator.factory is not None:
            return estimator.factory(**merged)
        name = estimator.name
        if name is None:
            raise ValueError("EstimatorConfig.name is required when no factory is set.")
        return _create_named_estimator(name, task, merged)
    if callable(estimator) and not isinstance(estimator, str):
        return estimator(**dict(params))
    return _create_named_estimator(str(estimator), task, params)


def _create_named_estimator(name: str, task: str, params: Mapping[str, Any]) -> Any:
    name = normalize_estimator_name(name)
    params = dict(params)
    if name == "svm":
        if task == "binary classification":
            params.setdefault("probability", True)
            return SVC(**params)
        return SVR(**params)
    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for the xgboost estimator. "
                "Install it with `uv sync --extra xgboost` or `pip install pspso[xgboost]`."
            ) from exc
        if task == "binary classification":
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)
    if name == "gbdt":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for the gbdt estimator. "
                "Install it with `uv sync --extra lightgbm` or `pip install pspso[lightgbm]`."
            ) from exc
        if task == "binary classification":
            return lgb.LGBMClassifier(**params)
        return lgb.LGBMRegressor(**params)
    if name == "mlp":
        return _create_sklearn_mlp(task, params)
    if name == "random_forest":
        if task == "binary classification":
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)
    if name == "logistic_regression":
        params.setdefault("max_iter", 1000)
        return LogisticRegression(**params)
    if name == "linear_regression":
        if params.get("fit_intercept") in {"true", "false"}:
            params["fit_intercept"] = params["fit_intercept"] == "true"
        return LinearRegression(**params)
    raise ValueError(f"Unknown estimator preset: {name!r}")


def _create_sklearn_mlp(task: str, params: Mapping[str, Any]) -> Any:
    params = dict(params)
    neurons = int(params.pop("neurons", 16))
    hidden_activation = params.pop("hiddenactivation", params.pop("activation", "relu"))
    if hidden_activation == "sigmoid":
        hidden_activation = "logistic"
    epochs = int(params.pop("epochs", params.pop("max_iter", 100)))
    optimizer = params.pop("optimizer", "adam")
    solver = optimizer if optimizer in {"adam", "sgd", "lbfgs"} else "adam"
    learning_rate = params.pop("learning_rate", params.pop("learning_rate_init", 0.001))
    params.pop("loss", None)
    params.pop("metrics", None)
    params.pop("mode", None)
    params.setdefault("hidden_layer_sizes", (neurons,))
    params.setdefault("activation", hidden_activation)
    params.setdefault("solver", solver)
    params.setdefault("learning_rate_init", learning_rate)
    params.setdefault("max_iter", epochs)
    if task == "binary classification":
        return MLPClassifier(**params)
    return MLPRegressor(**params)


__all__ = [
    "create_estimator",
    "default_fixed_params",
    "default_search_space",
    "allowed_estimator_params",
    "estimator_presets",
    "metric_metadata",
    "normalize_estimator_name",
    "optional_dependency_status",
    "task_metadata",
]

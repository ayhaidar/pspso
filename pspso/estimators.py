"""Estimator presets and factories used by pspso."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping
from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from .config import EstimatorConfig
from .recipes import get_recipe, recipe_metadata
from .search_space import Choice, FloatRange, IntRange, LogFloatRange, SearchSpace


def default_search_space(estimator: str, task: str) -> SearchSpace:
    """Return the typed default search space for a registered estimator."""

    estimator = canonical_estimator_name(estimator)
    recipe = get_recipe(estimator)
    if recipe is not None:
        try:
            return recipe.search_spaces[task]
        except KeyError as exc:
            raise ValueError(f"Recipe {estimator!r} does not support {task!r}.") from exc
    if estimator == "svm":
        return SearchSpace(
            {
                "kernel": Choice(["linear", "rbf", "poly"]),
                "gamma": LogFloatRange(0.001, 10.0, 4),
                "C": LogFloatRange(0.01, 100.0, 4),
                "degree": IntRange(1, 6),
            }
        )
    if estimator == "xgboost":
        return SearchSpace(
            {
                "learning_rate": LogFloatRange(0.01, 0.3, 4),
                "max_depth": IntRange(1, 10),
                "n_estimators": IntRange(10, 100),
                "subsample": FloatRange(0.7, 1.0, 2),
            }
        )
    if estimator == "lightgbm":
        return SearchSpace(
            {
                "learning_rate": LogFloatRange(0.01, 0.3, 4),
                "max_depth": IntRange(1, 10),
                "n_estimators": IntRange(10, 100),
                "subsample": FloatRange(0.7, 1.0, 2),
            }
        )
    if estimator == "sklearn_mlp":
        return SearchSpace(
            {
                "learning_rate": LogFloatRange(0.0001, 0.1, 5),
                "neurons": IntRange(4, 64),
                "hiddenactivation": Choice(["relu", "logistic", "tanh"]),
            }
        )
    if estimator == "pytorch_mlp":
        return SearchSpace(
            {
                "learning_rate": LogFloatRange(0.0001, 0.05, 5),
                "neurons": IntRange(8, 128),
                "batch_size": IntRange(16, 128),
            }
        )
    if estimator == "random_forest":
        return SearchSpace({"n_estimators": IntRange(10, 100), "max_depth": IntRange(2, 12)})
    if estimator in {"extra_trees", "hist_gradient_boosting"}:
        if estimator == "hist_gradient_boosting":
            return SearchSpace(
                {
                    "max_depth": IntRange(2, 12),
                    "learning_rate": LogFloatRange(0.01, 0.3, 4),
                }
            )
        return SearchSpace({"n_estimators": IntRange(50, 250), "max_depth": IntRange(2, 16)})
    if estimator == "elastic_net":
        return SearchSpace(
            {
                "alpha": LogFloatRange(0.0001, 1.0, 5),
                "l1_ratio": FloatRange(0.05, 0.95, 2),
            }
        )
    if estimator == "logistic_regression":
        return SearchSpace({"C": LogFloatRange(0.01, 100.0, 4)})
    if estimator == "linear_regression":
        return SearchSpace({"fit_intercept": Choice([True, False])})
    raise ValueError(f"Unknown estimator preset: {estimator!r}")


def default_fixed_params(estimator: str, task: str) -> dict[str, Any]:
    """Return conservative fixed parameters for the built-in presets."""

    estimator = canonical_estimator_name(estimator)
    recipe = get_recipe(estimator)
    if recipe is not None:
        return dict(recipe.fixed_params.get(task, {}))
    if estimator == "svm":
        if task != "regression":
            return {"kernel": "rbf", "C": 5.0, "gamma": 5.0, "probability": True}
        return {"kernel": "rbf", "C": 5.0, "gamma": 5.0}
    if estimator == "xgboost":
        if task == "binary_classification":
            return {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": 42,
                "n_jobs": -1,
            }
        if task == "multiclass_classification":
            return {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "random_state": 42,
                "n_jobs": -1,
            }
        return {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": 42,
            "n_jobs": -1,
        }
    if estimator == "lightgbm":
        if task == "binary_classification":
            return {
                "objective": "binary",
                "boosting_type": "gbdt",
                "random_state": 42,
                "n_jobs": -1,
            }
        if task == "multiclass_classification":
            return {
                "objective": "multiclass",
                "boosting_type": "gbdt",
                "random_state": 42,
                "n_jobs": -1,
            }
        return {
            "objective": "regression",
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_jobs": -1,
        }
    if estimator == "sklearn_mlp":
        return {"neurons": 16, "hiddenactivation": "relu", "epochs": 100, "random_state": 42}
    if estimator == "pytorch_mlp":
        return {
            "neurons": 32,
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 0.001,
            "patience": 5,
            "random_state": 42,
        }
    if estimator in {"random_forest", "extra_trees"}:
        return {"random_state": 42, "n_jobs": -1}
    if estimator in {"hist_gradient_boosting", "elastic_net"}:
        return {"random_state": 42} if estimator == "hist_gradient_boosting" else {}
    if estimator in {"logistic_regression", "linear_regression"}:
        return {}
    raise ValueError(f"Unknown estimator preset: {estimator!r}")


def estimator_presets() -> dict[str, dict[str, Any]]:
    presets: dict[str, dict[str, Any]] = {
        "svm": {
            "label": "Support Vector Machine",
            "tasks": ["regression", "binary_classification", "multiclass_classification"],
            "optional_dependency": None,
            "description": (
                "Strong baseline for smaller tabular datasets. Supports linear, RBF, "
                "and polynomial kernels."
            ),
        },
        "xgboost": {
            "label": "XGBoost",
            "tasks": ["regression", "binary_classification", "multiclass_classification"],
            "optional_dependency": "xgboost",
            "install": "uv sync --extra xgboost",
            "description": "Gradient-boosted trees from the optional XGBoost backend.",
        },
        "lightgbm": {
            "label": "LightGBM GBDT",
            "tasks": ["regression", "binary_classification", "multiclass_classification"],
            "optional_dependency": "lightgbm",
            "install": "uv sync --extra lightgbm",
            "description": "Gradient-boosted decision trees from the optional LightGBM backend.",
        },
        "sklearn_mlp": {
            "label": "Multi-layer Perceptron",
            "tasks": ["regression", "binary_classification", "multiclass_classification"],
            "optional_dependency": None,
            "description": "Scikit-learn neural network preset for compact tabular experiments.",
        },
        "random_forest": {
            "label": "Random Forest",
            "tasks": ["regression", "binary_classification", "multiclass_classification"],
            "optional_dependency": None,
            "description": "Tree ensemble with robust defaults and simple integer search spaces.",
        },
        "logistic_regression": {
            "label": "Logistic Regression",
            "tasks": ["binary_classification", "multiclass_classification"],
            "optional_dependency": None,
            "description": "Linear classifier for binary classification baselines.",
        },
        "linear_regression": {
            "label": "Linear Regression",
            "tasks": ["regression"],
            "optional_dependency": None,
            "description": "Linear regression baseline for regression tasks.",
        },
        "elastic_net": {
            "label": "Elastic Net",
            "tasks": ["regression"],
            "optional_dependency": None,
            "description": (
                "Regularized linear regression baseline for correlated tabular features."
            ),
        },
        "extra_trees": {
            "label": "Extra Trees",
            "tasks": ["regression", "binary_classification", "multiclass_classification"],
            "optional_dependency": None,
            "description": "Fast randomized tree ensemble for strong tabular baselines.",
        },
        "hist_gradient_boosting": {
            "label": "Histogram Gradient Boosting",
            "tasks": ["regression", "binary_classification", "multiclass_classification"],
            "optional_dependency": None,
            "description": (
                "Native scikit-learn gradient boosting for medium and large tabular datasets."
            ),
        },
        "pytorch_mlp": {
            "label": "PyTorch Tabular MLP",
            "tasks": ["regression", "binary_classification", "multiclass_classification"],
            "optional_dependency": "torch",
            "install": "uv sync --extra torch",
            "description": (
                "Train a compact tabular neural network with epoch-level validation progress."
            ),
        },
    }
    presets.update(recipe_metadata())
    return presets


def list_estimators(task: str | None = None) -> list[str]:
    """List canonical estimator IDs, optionally filtered by task."""

    presets = estimator_presets()
    return sorted(
        name for name, preset in presets.items() if task is None or task in preset["tasks"]
    )


def get_estimator_info(name: str) -> dict[str, Any]:
    """Return capabilities, dependency status, and typed defaults for an estimator."""

    canonical = canonical_estimator_name(name)
    try:
        preset = dict(estimator_presets()[canonical])
    except KeyError as exc:
        raise ValueError(f"Unknown estimator preset: {name!r}.") from exc
    preset["id"] = canonical
    preset["dependency"] = optional_dependency_status(preset.get("optional_dependency"))
    preset["defaults"] = {
        task: {
            "fixed_params": default_fixed_params(canonical, task),
            "search_space": default_search_space(canonical, task),
            "allowed_params": sorted(allowed_estimator_params(canonical, task)),
        }
        for task in preset["tasks"]
    }
    return preset


def task_metadata() -> dict[str, dict[str, Any]]:
    """Human-readable dashboard metadata for supported task types."""

    return {
        "regression": {
            "label": "Regression",
            "description": (
                "Predict a continuous numeric target. Compare candidates with RMSE, MAE, or R2."
            ),
            "metrics": ["rmse", "mae", "r2"],
        },
        "binary_classification": {
            "label": "Binary classification",
            "description": (
                "Predict one of two classes. Use ROC AUC for probability ranking or "
                "accuracy for direct class correctness."
            ),
            "metrics": ["roc_auc", "pr_auc", "log_loss", "f1_macro", "accuracy"],
        },
        "multiclass_classification": {
            "label": "Multiclass classification",
            "description": (
                "Predict one of three or more classes with probability-aware metrics "
                "and balanced reporting."
            ),
            "metrics": ["accuracy", "log_loss", "f1_macro"],
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
            "description": (
                "Area under the ROC curve using scores or probabilities. Higher is better."
            ),
        },
        "mae": {
            "label": "MAE",
            "description": "Mean absolute error. Lower validation error is better.",
        },
        "r2": {"label": "R2", "description": "Explained variance. Higher is better."},
        "pr_auc": {
            "label": "PR AUC",
            "description": "Precision-recall area, useful for imbalanced binary targets.",
        },
        "log_loss": {
            "label": "Log loss",
            "description": "Probability calibration loss. Lower is better.",
        },
        "f1_macro": {
            "label": "Macro F1",
            "description": "Unweighted mean F1 across classes. Higher is better.",
        },
    }


def optional_dependency_status(package: str | None) -> dict[str, Any]:
    """Return whether an optional estimator dependency is available."""

    if package is None:
        return {"required": None, "installed": True, "install": None}
    installed = importlib.util.find_spec(package) is not None
    extra = {"xgboost": "xgboost", "lightgbm": "lightgbm", "torch": "torch"}.get(package, package)
    return {
        "required": package,
        "installed": installed,
        "install": f"uv sync --extra {extra}",
    }


def allowed_estimator_params(estimator: str, task: str) -> set[str]:
    """Return accepted constructor/search-space params for a built-in preset."""

    estimator = canonical_estimator_name(estimator)
    recipe = get_recipe(estimator)
    if recipe is not None:
        return set(recipe.fixed_params.get(task, {})) | set(recipe.search_spaces[task].names)
    common_runtime_params = {"random_state", "n_jobs"}
    virtual = {
        "sklearn_mlp": {
            "neurons",
            "hiddenactivation",
            "epochs",
            "optimizer",
            "learning_rate",
            "random_state",
        },
        "pytorch_mlp": {
            "neurons",
            "epochs",
            "batch_size",
            "learning_rate",
            "device",
            "patience",
            "random_state",
        },
        "xgboost": {
            "objective",
            "eval_metric",
            "learning_rate",
            "max_depth",
            "n_estimators",
            "subsample",
            "random_state",
            "n_jobs",
            "verbosity",
        },
        "lightgbm": {
            "objective",
            "boosting_type",
            "learning_rate",
            "max_depth",
            "n_estimators",
            "subsample",
            "random_state",
            "n_jobs",
            "verbosity",
        },
    }
    if estimator in virtual:
        return set(virtual[estimator]) | (
            common_runtime_params if estimator in {"xgboost", "lightgbm"} else set()
        )
    try:
        model = _create_named_estimator(estimator, task, {})
    except ImportError:
        return set(default_fixed_params(estimator, task)) | set(
            default_search_space(estimator, task).names
        )
    return set(model.get_params(deep=False))


def canonical_estimator_name(name: str) -> str:
    """Return a normalized canonical model ID without compatibility aliases."""

    return name.lower()


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
    name = canonical_estimator_name(name)
    params = dict(params)
    recipe = get_recipe(name)
    if recipe is not None:
        if task not in recipe.tasks:
            raise ValueError(f"Recipe {name!r} does not support {task!r}.")
        return recipe.factory(**params)
    if name == "svm":
        if task != "regression":
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
        if task != "regression":
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)
    if name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for the lightgbm estimator. "
                "Install it with `uv sync --extra lightgbm` or `pip install pspso[lightgbm]`."
            ) from exc
        return lgb.LGBMRegressor(**params) if task == "regression" else lgb.LGBMClassifier(**params)
    if name == "sklearn_mlp":
        return _create_sklearn_mlp(task, params)
    if name == "random_forest":
        if task != "regression":
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)
    if name == "extra_trees":
        return (
            ExtraTreesRegressor(**params)
            if task == "regression"
            else ExtraTreesClassifier(**params)
        )
    if name == "hist_gradient_boosting":
        return (
            HistGradientBoostingRegressor(**params)
            if task == "regression"
            else HistGradientBoostingClassifier(**params)
        )
    if name == "elastic_net":
        return ElasticNet(**params)
    if name == "logistic_regression":
        params.setdefault("max_iter", 1000)
        return LogisticRegression(**params)
    if name == "linear_regression":
        if params.get("fit_intercept") in {"true", "false"}:
            params["fit_intercept"] = params["fit_intercept"] == "true"
        return LinearRegression(**params)
    if name == "pytorch_mlp":
        try:
            from .torch_models import TorchTabularClassifier, TorchTabularRegressor
        except ImportError as exc:
            raise ImportError(
                "torch is required for pytorch_mlp. Install it with `uv sync --extra torch`."
            ) from exc
        if task == "regression":
            return TorchTabularRegressor(task=task, **params)
        return TorchTabularClassifier(task=task, **params)
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
    if task != "regression":
        return MLPClassifier(**params)
    return MLPRegressor(**params)


__all__ = [
    "create_estimator",
    "default_fixed_params",
    "default_search_space",
    "allowed_estimator_params",
    "estimator_presets",
    "get_estimator_info",
    "list_estimators",
    "metric_metadata",
    "canonical_estimator_name",
    "optional_dependency_status",
    "task_metadata",
]

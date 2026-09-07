"""Lightweight model explanations for completed tabular runs."""

from __future__ import annotations

from typing import Any

import numpy as np


def transformed_feature_names(transformer: Any, fallback: list[str]) -> list[str]:
    """Return post-preprocessing feature names when the transformer exposes them."""

    if hasattr(transformer, "get_feature_names_out"):
        return [str(name) for name in transformer.get_feature_names_out()]
    return list(fallback)


def native_feature_importance(model: Any, feature_names: list[str]) -> dict[str, Any]:
    """Extract native tree or linear importance without inventing an explanation."""

    values: np.ndarray | None = None
    source: str | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float).ravel()
        source = "native_tree_importance"
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = (
            np.abs(coefficients)
            if coefficients.ndim == 1
            else np.mean(np.abs(coefficients), axis=0)
        )
        source = "absolute_coefficient"
    if values is None:
        return {
            "available": False,
            "reason": (
                "This fitted estimator does not expose native feature importances or coefficients."
            ),
            "items": [],
        }
    if len(values) != len(feature_names):
        return {
            "available": False,
            "reason": (
                "The estimator importance vector does not match the transformed feature space."
            ),
            "items": [],
        }
    order = np.argsort(values)[::-1]
    items = [
        {"feature": feature_names[index], "importance": float(values[index])} for index in order
    ]
    return {"available": True, "source": source, "items": items}


__all__ = ["native_feature_importance", "transformed_feature_names"]

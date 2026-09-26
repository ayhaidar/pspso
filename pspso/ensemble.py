"""Saved predictors built from cross-validation fold models."""

from __future__ import annotations

from typing import Any

import numpy as np


class CrossValidationEnsemble:
    """Average the winning candidate's fitted fold models without another fit."""

    def __init__(self, models: list[Any], task: str) -> None:
        if not models:
            raise ValueError("A cross-validation ensemble requires at least one fitted model.")
        self.models_ = list(models)
        self.task = task
        first = self.models_[0]
        if hasattr(first, "n_features_in_"):
            self.n_features_in_ = first.n_features_in_
        if task != "regression":
            classes = getattr(first, "classes_", None)
            if classes is None:
                raise ValueError("Classification fold models must expose their fitted classes.")
            self.classes_ = np.asarray(classes)

    def predict(self, X: Any) -> np.ndarray:
        """Predict by averaging regression outputs or class probabilities."""

        if self.task == "regression":
            predictions = [np.asarray(model.predict(X), dtype=float) for model in self.models_]
            return np.mean(predictions, axis=0)
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Average aligned class probabilities across the fitted fold models."""

        if self.task == "regression":
            raise AttributeError("Regression ensembles do not expose predict_proba.")
        rows: list[np.ndarray] = []
        for model in self.models_:
            if hasattr(model, "predict_proba"):
                probabilities = np.asarray(model.predict_proba(X), dtype=float)
                model_classes = np.asarray(getattr(model, "classes_", self.classes_))
                aligned = np.zeros((len(probabilities), len(self.classes_)), dtype=float)
                for source, label in enumerate(model_classes):
                    matches = np.flatnonzero(self.classes_ == label)
                    if len(matches):
                        aligned[:, int(matches[0])] = probabilities[:, source]
                rows.append(aligned)
                continue
            predictions = np.asarray(model.predict(X)).ravel()
            rows.append(
                np.column_stack([(predictions == label).astype(float) for label in self.classes_])
            )
        averaged = np.mean(rows, axis=0)
        totals = averaged.sum(axis=1, keepdims=True)
        return np.divide(averaged, totals, out=np.zeros_like(averaged), where=totals != 0)


__all__ = ["CrossValidationEnsemble"]

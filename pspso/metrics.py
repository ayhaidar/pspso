"""Metric helpers shared by the optimizer and API."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _probabilities(model: Any, X: Any) -> np.ndarray | None:
    """Return class probabilities when an estimator can provide them."""

    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    return None


def _positive_scores(model: Any, X: Any, positive_label: Any | None = None) -> np.ndarray:
    """Return a binary positive-class score without guessing for multiclass data."""

    probabilities = _probabilities(model, X)
    if probabilities is not None:
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            classes = np.asarray(getattr(model, "classes_", [0, 1]))
            if positive_label is None:
                index = 1
            else:
                matches = np.flatnonzero(classes == positive_label)
                if not len(matches):
                    raise ValueError(
                        f"Positive label {positive_label!r} is not present in model classes."
                    )
                index = int(matches[0])
            return probabilities[:, index]
        if probabilities.ndim == 1:
            return probabilities
        raise ValueError("A binary score is unavailable for a multiclass estimator.")
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).ravel()
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        return -scores if positive_label is not None and positive_label == classes[0] else scores
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    positive = classes[-1] if positive_label is None else positive_label
    return (np.asarray(model.predict(X)).ravel() == positive).astype(float)


def prediction_output(
    model: Any,
    task: str,
    X: Any,
    positive_label: Any | None = None,
    decision_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    predictions = np.asarray(model.predict(X)).ravel()
    output = {"predictions": predictions}
    if task == "binary_classification":
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        positive = classes[-1] if positive_label is None else positive_label
        negative = next(label for label in classes if label != positive)
        scores = _positive_scores(model, X, positive)
        output["scores"] = scores
        output["predictions"] = np.where(scores >= decision_threshold, positive, negative)
    if task != "regression":
        probabilities = _probabilities(model, X)
        if probabilities is not None:
            output["probabilities"] = probabilities
    return output


def metric_value(
    model: Any,
    task: str,
    metric: str,
    X: Any,
    y: Any,
    *,
    positive_label: Any | None = None,
    decision_threshold: float = 0.5,
) -> float:
    """Return the human metric value for a fitted model."""

    y_true = np.asarray(y).ravel()
    if metric == "rmse":
        preds = np.asarray(model.predict(X)).ravel()
        return float(np.sqrt(mean_squared_error(y_true, preds)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, np.asarray(model.predict(X)).ravel()))
    if metric == "r2":
        return float(r2_score(y_true, np.asarray(model.predict(X)).ravel()))
    if task not in {"binary_classification", "multiclass_classification"}:
        raise ValueError(f"Metric {metric!r} is only supported for classification.")
    labels = np.unique(y_true)
    predictions = np.asarray(model.predict(X)).ravel()
    if task == "binary_classification" and len(labels) == 2:
        selected_positive = labels[-1] if positive_label is None else positive_label
        if selected_positive not in labels:
            raise ValueError(f"Positive label {selected_positive!r} is not present in the target.")
        negative = next(label for label in labels if label != selected_positive)
        scores = _positive_scores(model, X, selected_positive)
        predictions = np.where(scores >= decision_threshold, selected_positive, negative)
    if metric == "accuracy":
        preds = predictions
        return float(accuracy_score(y_true, preds))
    if metric == "roc_auc":
        selected_positive = labels[-1] if positive_label is None else positive_label
        binary_true = (y_true == selected_positive).astype(int)
        return float(roc_auc_score(binary_true, _positive_scores(model, X, selected_positive)))
    if metric == "pr_auc":
        selected_positive = labels[-1] if positive_label is None else positive_label
        binary_true = (y_true == selected_positive).astype(int)
        return float(
            average_precision_score(binary_true, _positive_scores(model, X, selected_positive))
        )
    if metric == "f1_macro":
        return float(f1_score(y_true, predictions, average="macro"))
    if metric == "log_loss":
        if not hasattr(model, "predict_proba"):
            raise ValueError("log_loss requires an estimator with predict_proba.")
        return float(log_loss(y_true, model.predict_proba(X)))
    raise ValueError(f"Unsupported metric: {metric!r}")


def cost_from_metric(task: str, metric: str, value: float) -> float:
    """Convert a metric into the minimization cost used by optimizers."""

    if metric in {"rmse", "mae", "log_loss"}:
        return float(value)
    if metric == "r2":
        return float(-value)
    if task in {"binary_classification", "multiclass_classification"} and metric in {
        "accuracy",
        "roc_auc",
        "pr_auc",
        "f1_macro",
    }:
        return float(1 - value)
    raise ValueError(f"Unsupported metric for cost conversion: {metric!r}")


def evaluation_report(
    model: Any,
    task: str,
    metric: str,
    X: Any,
    y: Any,
    *,
    positive_label: Any | None = None,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a JSON-safe evaluation report for a fitted estimator."""

    y_true = np.asarray(y).ravel()
    predictions = np.asarray(model.predict(X)).ravel()
    labels = np.unique(y_true)
    selected_positive = labels[-1] if len(labels) else None
    if task == "binary_classification" and len(labels) == 2:
        selected_positive = labels[-1] if positive_label is None else positive_label
        if selected_positive not in labels:
            raise ValueError(f"Positive label {selected_positive!r} is not present in the target.")
        negative = next(label for label in labels if label != selected_positive)
        scores = _positive_scores(model, X, selected_positive)
        predictions = np.where(scores >= decision_threshold, selected_positive, negative)
    selected_value = metric_value(
        model,
        task,
        metric,
        X,
        y,
        positive_label=positive_label,
        decision_threshold=decision_threshold,
    )
    report: dict[str, Any] = {
        "selected_metric": metric,
        "value": selected_value,
        "cost": cost_from_metric(task, metric, selected_value),
        "metrics": {metric: selected_value},
    }
    if task == "regression":
        residuals = y_true.astype(float) - predictions.astype(float)
        sample_indexes = np.linspace(0, len(y_true) - 1, min(len(y_true), 2000), dtype=int)
        report["metrics"].update(
            {
                "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
                "mae": float(mean_absolute_error(y_true, predictions)),
                "r2": float(r2_score(y_true, predictions)),
            }
        )
        report["regression_diagnostics"] = {
            "actual": [float(value) for value in y_true[sample_indexes]],
            "predicted": [float(value) for value in predictions[sample_indexes]],
            "residuals": [float(value) for value in residuals[sample_indexes]],
            "sampled": len(sample_indexes) < len(y_true),
        }
        return report

    report["metrics"].update(
        {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        }
    )
    probabilities = _probabilities(model, X)
    if probabilities is not None:
        report["metrics"]["log_loss"] = float(log_loss(y_true, probabilities))
    if task != "binary_classification":
        matrix = confusion_matrix(y_true, predictions, labels=labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, predictions, labels=labels, zero_division=0
        )
        per_class = []
        for index, label in enumerate(labels):
            tp = int(matrix[index, index])
            fn = int(matrix[index, :].sum() - tp)
            fp = int(matrix[:, index].sum() - tp)
            tn = int(matrix.sum() - tp - fn - fp)
            per_class.append(
                {
                    "label": str(label),
                    "precision": float(precision[index]),
                    "sensitivity": float(recall[index]),
                    "specificity": float(tn / (tn + fp)) if tn + fp else None,
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
            )
        report["confusion_matrix"] = {
            "labels": [str(label) for label in labels],
            "values": matrix.tolist(),
        }
        report["per_class"] = per_class
        if (
            probabilities is not None
            and probabilities.ndim == 2
            and probabilities.shape[1] == len(labels)
        ):
            binary_targets = label_binarize(y_true, classes=labels)
            curves = []
            class_aucs = []
            for index, label in enumerate(labels):
                fpr, tpr, thresholds = roc_curve(binary_targets[:, index], probabilities[:, index])
                class_auc = float(roc_auc_score(binary_targets[:, index], probabilities[:, index]))
                class_aucs.append(class_auc)
                curves.append(
                    {
                        "label": str(label),
                        "fpr": [float(value) for value in fpr],
                        "tpr": [float(value) for value in tpr],
                        "thresholds": [float(value) for value in thresholds],
                        "auc": class_auc,
                    }
                )
            report["metrics"]["roc_auc_macro"] = float(np.mean(class_aucs))
            report["multiclass_roc"] = {"curves": curves, "average": "one-vs-rest macro"}
        return report

    scores = _positive_scores(model, X, selected_positive)
    if len(labels) != 2:
        raise ValueError("Binary diagnostics require exactly two target classes.")
    positive_label = selected_positive
    negative_label = next(label for label in labels if label != positive_label)
    matrix = confusion_matrix(y_true, predictions, labels=[negative_label, positive_label])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    sensitivity = float(tp / (tp + fn)) if tp + fn else None
    specificity = float(tn / (tn + fp)) if tn + fp else None
    binary_true = (y_true == positive_label).astype(int)
    auc = float(roc_auc_score(binary_true, scores))
    pr_auc = float(average_precision_score(binary_true, scores))
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=positive_label)
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true, scores, pos_label=positive_label
    )
    report["metrics"].update(
        {
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }
    )
    report["confusion_matrix"] = {
        "labels": [str(negative_label), str(positive_label)],
        "values": matrix.tolist(),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }
    report["roc_curve"] = {
        "fpr": [float(value) for value in fpr],
        "tpr": [float(value) for value in tpr],
        "thresholds": [float(value) for value in thresholds],
        "auc": auc,
    }
    report["precision_recall_curve"] = {
        "precision": [float(value) for value in precision],
        "recall": [float(value) for value in recall],
        "thresholds": [float(value) for value in pr_thresholds],
        "auc": pr_auc,
    }
    report["threshold_diagnostics"] = _threshold_diagnostics(
        y_true, scores, negative_label, positive_label
    )
    report["decision_threshold"] = decision_threshold
    report["positive_label"] = str(positive_label)
    return report


def _threshold_diagnostics(
    y_true: np.ndarray,
    scores: np.ndarray,
    negative_label: Any,
    positive_label: Any,
) -> list[dict[str, float | int | None]]:
    """Return a compact threshold table for interactive result exploration."""

    finite_scores = np.asarray(scores, dtype=float)
    if finite_scores.size == 0:
        return []
    low, high = float(np.min(finite_scores)), float(np.max(finite_scores))
    thresholds = (
        np.linspace(0.0, 1.0, 101)
        if low >= 0 and high <= 1
        else np.quantile(finite_scores, np.linspace(0.0, 1.0, 101))
    )
    rows: list[dict[str, float | int | None]] = []
    for threshold in np.unique(thresholds):
        predicted = np.where(finite_scores >= threshold, positive_label, negative_label)
        tn, fp, fn, tp = (
            int(value)
            for value in confusion_matrix(
                y_true, predicted, labels=[negative_label, positive_label]
            ).ravel()
        )
        precision = float(tp / (tp + fp)) if tp + fp else None
        sensitivity = float(tp / (tp + fn)) if tp + fn else None
        specificity = float(tn / (tn + fp)) if tn + fp else None
        f1 = (
            float(2 * precision * sensitivity / (precision + sensitivity))
            if precision is not None and sensitivity is not None and precision + sensitivity
            else None
        )
        rows.append(
            {
                "threshold": float(threshold),
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "f1": f1,
                "accuracy": float((tp + tn) / len(y_true)),
            }
        )
    return rows


def evaluate_model(model: Any, task: str, metric: str, X: Any, y: Any) -> tuple[float, float]:
    value = metric_value(model, task, metric, X, y)
    return cost_from_metric(task, metric, value), value


__all__ = [
    "cost_from_metric",
    "evaluate_model",
    "evaluation_report",
    "metric_value",
    "prediction_output",
]

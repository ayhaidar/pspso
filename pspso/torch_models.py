"""Small scikit-learn-compatible PyTorch estimators for tabular recipes."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

_INITIALIZATION_LOCK = Lock()


class TorchTabularEstimator(BaseEstimator):
    """A compact feed-forward tabular model with optional epoch callbacks."""

    def __init__(
        self,
        task: str,
        neurons: int = 32,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = "auto",
        patience: int = 5,
        random_state: int | None = 42,
    ) -> None:
        self.task = task
        self.neurons = neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.patience = patience
        self.random_state = random_state
        self._progress_callback: Callable[[dict[str, Any]], None] | None = None
        self._validation_data: tuple[Any, Any] | None = None

    def set_progress_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self._progress_callback = callback

    def set_validation_data(self, X: Any, y: Any) -> None:
        """Attach the optimizer's validation split for epoch-level monitoring."""

        self._validation_data = (X, y)

    def fit(self, X: Any, y: Any) -> TorchTabularEstimator:
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError(
                "torch is required for pytorch_mlp. Install it with `uv sync --extra torch`."
            ) from exc

        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device == "auto":
            device = "cpu"
        self.device_ = torch.device(device)
        features = X.toarray() if hasattr(X, "toarray") else X
        features = np.asarray(features, dtype=np.float32)
        target = np.asarray(y).ravel()
        self.n_features_in_ = features.shape[1]
        self.classes_ = None
        loss_fn: nn.Module
        if self.task == "regression":
            target_tensor = torch.as_tensor(target.astype(np.float32).reshape(-1, 1))
            output_size = 1
            loss_fn = nn.MSELoss()
        else:
            self.classes_, encoded = np.unique(target, return_inverse=True)
            if self.task == "binary_classification":
                target_tensor = torch.as_tensor(encoded.astype(np.float32).reshape(-1, 1))
                output_size = 1
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                target_tensor = torch.as_tensor(encoded.astype(np.int64))
                output_size = len(self.classes_)
                loss_fn = nn.CrossEntropyLoss()
        # Initial weights and shuffle order must not race between parallel candidates.
        with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
            if self.random_state is not None:
                torch.random.default_generator.manual_seed(self.random_state)
            self.model_ = nn.Sequential(
                nn.Linear(features.shape[1], int(self.neurons)),
                nn.ReLU(),
                nn.Linear(int(self.neurons), output_size),
            ).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=float(self.learning_rate))
        data = DataLoader(
            TensorDataset(torch.as_tensor(features), target_tensor),
            batch_size=max(1, int(self.batch_size)),
            shuffle=True,
            generator=torch.Generator().manual_seed(
                self.random_state if self.random_state is not None else 42
            ),
        )
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(1, int(self.epochs) + 1):
            self.model_.train()
            total_loss = 0.0
            for feature_batch, target_batch in data:
                feature_batch = feature_batch.to(self.device_)
                target_batch = target_batch.to(self.device_)
                optimizer.zero_grad()
                prediction = self.model_(feature_batch)
                loss = loss_fn(prediction, target_batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(feature_batch)
            epoch_loss = total_loss / len(features)
            monitoring_loss = epoch_loss
            progress: dict[str, Any] = {
                "epoch": epoch,
                "epochs": int(self.epochs),
                "train_loss": epoch_loss,
                "device": str(self.device_),
            }
            if self._validation_data is not None:
                validation_features, validation_target = self._validation_data
                validation_array = (
                    validation_features.toarray()
                    if hasattr(validation_features, "toarray")
                    else validation_features
                )
                validation_array = np.asarray(validation_array, dtype=np.float32)
                raw_validation_target = np.asarray(validation_target).ravel()
                if self.task == "regression":
                    validation_tensor = torch.as_tensor(
                        raw_validation_target.astype(np.float32).reshape(-1, 1)
                    )
                elif self.task == "binary_classification":
                    assert self.classes_ is not None
                    encoded_validation = np.searchsorted(self.classes_, raw_validation_target)
                    validation_tensor = torch.as_tensor(
                        encoded_validation.astype(np.float32).reshape(-1, 1)
                    )
                else:
                    assert self.classes_ is not None
                    encoded_validation = np.searchsorted(self.classes_, raw_validation_target)
                    validation_tensor = torch.as_tensor(encoded_validation.astype(np.int64))
                self.model_.eval()
                with torch.no_grad():
                    validation_output = self.model_(
                        torch.as_tensor(validation_array).to(self.device_)
                    )
                    validation_loss = float(
                        loss_fn(validation_output, validation_tensor.to(self.device_)).cpu()
                    )
                    progress["validation_loss"] = validation_loss
                    monitoring_loss = validation_loss
                    if self.task != "regression":
                        if self.task == "binary_classification":
                            predicted = (
                                (torch.sigmoid(validation_output).ravel() >= 0.5).cpu().numpy()
                            )
                        else:
                            predicted = validation_output.argmax(dim=1).cpu().numpy()
                        progress["validation_accuracy"] = float(
                            np.mean(predicted == encoded_validation)
                        )
            if self._progress_callback:
                self._progress_callback(progress)
            if monitoring_loss < best_loss - 1e-8:
                best_loss = monitoring_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(self.patience):
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_loss_ = best_loss
        self.best_epoch_ = best_epoch
        self.n_iter_ = epoch
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        import torch

        features = X.toarray() if hasattr(X, "toarray") else X
        with torch.no_grad():
            output = (
                self.model_(
                    torch.as_tensor(np.asarray(features, dtype=np.float32)).to(self.device_)
                )
                .cpu()
                .numpy()
            )
        if self.task == "binary_classification":
            positive = 1.0 / (1.0 + np.exp(-output.ravel()))
            return np.column_stack([1.0 - positive, positive])
        if self.task == "multiclass_classification":
            shifted = output - output.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            return probabilities / probabilities.sum(axis=1, keepdims=True)
        raise AttributeError("Regression estimators do not expose predict_proba.")

    def predict(self, X: Any) -> np.ndarray:
        if self.task == "regression":
            import torch

            features = X.toarray() if hasattr(X, "toarray") else X
            with torch.no_grad():
                return (
                    self.model_(
                        torch.as_tensor(np.asarray(features, dtype=np.float32)).to(self.device_)
                    )
                    .cpu()
                    .numpy()
                    .ravel()
                )
        probabilities = self.predict_proba(X)
        indexes = probabilities.argmax(axis=1)
        assert self.classes_ is not None
        return self.classes_[indexes]


class TorchTabularClassifier(ClassifierMixin, TorchTabularEstimator):
    """PyTorch classifier with the scikit-learn classifier contract."""


class TorchTabularRegressor(RegressorMixin, TorchTabularEstimator):
    """PyTorch regressor with the scikit-learn regressor contract."""

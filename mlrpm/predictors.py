# Artifact: predictors.py
# Author: Obed Plasencia
# Date coded: November 12, 2026
"""Prediction classes (Design 1.2.3, 2.3).

Adding an algorithm (NFR-9): subclass VelocityPredictor, implement
``_build_model``, decorate with ``@register("key")``. Nothing else changes.

>>> NOTEBOOK HOOK: copy tuned hyperparameters or a custom estimator from the
>>> ML_SyntheticModel notebook into default_config.yaml / a new subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PREDICTOR_REGISTRY: dict[str, type["VelocityPredictor"]] = {}


def register(key: str):
    def deco(cls):
        PREDICTOR_REGISTRY[key] = cls
        return cls
    return deco


def create_predictor(key: str, **hyperparameters) -> "VelocityPredictor":
    if key not in PREDICTOR_REGISTRY:
        raise KeyError(f"Unknown predictor {key!r}; valid options: {sorted(PREDICTOR_REGISTRY)}")
    return PREDICTOR_REGISTRY[key](**hyperparameters)


class VelocityPredictor(ABC):
    display_name: str = "BASE"

    def __init__(self, **hyperparameters: Any):
        self.hyperparameters = hyperparameters
        self._model = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VelocityPredictor":
        if self._model is None:
            self._model = self._build_model(**self.hyperparameters)
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"{self.display_name} must be fit(...) before predict(...)")
        return self._model.predict(X)

    def underlying_model(self):
        return self._model

    @abstractmethod
    def _build_model(self, **hp):
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.hyperparameters})"


@register("svr")
class SVRPredictor(VelocityPredictor):
    display_name = "SVR"

    def _build_model(self, kernel="rbf", C=10.0, epsilon=0.01, gamma="scale", **_):
        if C <= 0:
            raise ValueError("SVR: C must be > 0")
        if epsilon < 0:
            raise ValueError("SVR: epsilon must be >= 0")
        return make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma))


@register("rf")
class RandomForestPredictor(VelocityPredictor):
    display_name = "RF"

    def _build_model(self, n_estimators=200, max_depth=None, min_samples_leaf=2,
                     random_state=42, **_):
        if n_estimators < 1:
            raise ValueError("RF: n_estimators must be >= 1")
        if min_samples_leaf < 1:
            raise ValueError("RF: min_samples_leaf must be >= 1")
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf,
                                     random_state=random_state, n_jobs=-1)


@register("mlp")
class MLPPredictor(VelocityPredictor):
    display_name = "MLP"

    def _build_model(self, hidden_layer_sizes=(64, 64), activation="relu",
                     learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
                     random_state=42, **_):
        if learning_rate_init <= 0:
            raise ValueError("MLP: learning_rate_init must be > 0")
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layer_sizes), activation=activation,
            learning_rate_init=learning_rate_init, max_iter=max_iter,
            early_stopping=early_stopping, random_state=random_state))

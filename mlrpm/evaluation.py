# Artifact: evaluation.py
# Author: Obed Plasencia
# Date coded: November 12, 2026
"""ModelEvaluator (Design 2.5)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .entities import EvaluationReport, EvaluationResult


class ModelEvaluator:
    def evaluate(self, y_true, y_pred, well_name: str, model_name: str,
                 target_name: str) -> EvaluationResult:
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Length mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        return EvaluationResult(
            well_name, model_name, target_name,
            r2=float(r2_score(y_true, y_pred)),
            accuracy_pct=self._mape_accuracy(y_true, y_pred),
            rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            mae=float(mean_absolute_error(y_true, y_pred)),
            n_samples=int(y_true.size))

    def evaluate_many(self, predictions: dict, y_true_by_well: dict,
                      target_name: str) -> EvaluationReport:
        report = EvaluationReport()
        for well_name, preds in predictions.items():
            for model_name, y_pred in preds.items():  # includes "RPM"
                report.add(self.evaluate(y_true_by_well[well_name], y_pred,
                                         well_name, model_name, target_name))
        return report

    @staticmethod
    def _mape_accuracy(y_true, y_pred) -> float:
        eps = 1e-8
        ape = np.abs((y_true - y_pred) / (np.abs(y_true) + eps))
        return float(100.0 * (1.0 - np.mean(ape)))

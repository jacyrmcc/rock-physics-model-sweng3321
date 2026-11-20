# Artifact: entities.py
# Author: Obed Plasencia
# Date coded: November 12, 2026
"""Entity classes (Design 1.2.1): WellLogDataset, FeatureSet,
EvaluationResult, EvaluationReport, PipelineResult."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SCHEMA = ["DEPTH", "VSH", "VSAND", "VCARB", "POR", "SW", "PD", "VP", "VS", "RHO"]
FEATURE_COLUMNS = ["VSH", "VSAND", "VCARB", "POR", "SW", "PD"]
TARGETS = ("VP", "VS")
METRICS = ("r2", "accuracy_pct", "rmse", "mae")


@dataclass
class WellLogDataset:
    """One well: an (n, 10) float array in SCHEMA column order.

    Units: DEPTH m; VSH/VSAND/VCARB/POR/SW fractions; PD MPa; VP/VS km/s; RHO g/cc.
    """

    name: str
    array: np.ndarray
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def __post_init__(self) -> None:
        self.array = np.asarray(self.array, dtype=float)
        if self.array.ndim != 2 or self.array.shape[1] != len(SCHEMA):
            raise ValueError(
                f"Well log must have {len(SCHEMA)} columns, found shape {self.array.shape}"
            )
        if np.isnan(self.array).any():
            raise ValueError(f"Missing values in well {self.name}")

    # --- simple accessors (Design 2.1.5) ---
    def column(self, name: str) -> np.ndarray:
        try:
            return self.array[:, SCHEMA.index(name)]
        except ValueError:
            raise KeyError(f"Unknown column {name!r}; valid: {SCHEMA}") from None

    def features(self) -> np.ndarray:
        return np.column_stack([self.column(c) for c in self.feature_columns])

    def target(self, name: str) -> np.ndarray:
        if name not in TARGETS:
            raise ValueError(f"target_name must be one of {TARGETS}, got {name!r}")
        return self.column(name)

    def depth(self) -> np.ndarray:
        return self.column("DEPTH")

    def __len__(self) -> int:
        return self.array.shape[0]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.array, columns=SCHEMA)


@dataclass
class FeatureSet:
    """Data holder for one blind-well run (Design 1.2.1)."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_blind: np.ndarray
    y_blind: np.ndarray
    blind_well: WellLogDataset
    feature_names: list[str]
    target_name: str


@dataclass
class EvaluationResult:
    well_name: str
    model_name: str
    target_name: str
    r2: float
    accuracy_pct: float
    rmse: float
    mae: float
    n_samples: int


@dataclass
class EvaluationReport:
    results: list[EvaluationResult] = field(default_factory=list)

    def add(self, result: EvaluationResult) -> None:
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.results])

    def pivot(self, metric: str) -> pd.DataFrame:
        """well_name x model_name table of ``metric`` (Design 2.5.5)."""
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        df = self.to_dataframe()
        table = df.pivot(index="well_name", columns="model_name", values=metric)
        order = [m for m in ["RPM", "SVR", "RF", "MLP"] if m in table.columns]
        order += [m for m in table.columns if m not in order]
        return table[order]

    def average_by_model(self, metric: str) -> pd.Series:
        return self.pivot(metric).mean(axis=0)


@dataclass
class PipelineResult:
    target_name: str
    report: EvaluationReport
    predictions: dict[str, dict[str, np.ndarray]]  # well -> model name -> array
    blind_wells: dict[str, WellLogDataset]
    fitted_models: dict[str, dict[str, Any]]  # well -> model name -> VelocityPredictor

    def errors_by_model(self) -> dict[str, np.ndarray]:
        """|y_true - y_pred| pooled over all blind wells, per model (for box plots)."""
        out: dict[str, list[np.ndarray]] = {}
        for well, preds in self.predictions.items():
            y = self.blind_wells[well].target(self.target_name)
            for model, yp in preds.items():
                out.setdefault(model, []).append(np.abs(y - yp))
        return {m: np.concatenate(v) for m, v in out.items()}

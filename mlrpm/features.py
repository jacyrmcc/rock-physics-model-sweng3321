# Artifact: features.py
# Author: Obed Plasencia
# Date coded: November 12, 2026
"""FeatureEngineer (Design 2.2): leave-one-well-out feature building."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from .entities import TARGETS, FeatureSet, WellLogDataset


class FeatureEngineer:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be in (0, 1)")
        self.test_size = test_size
        self.random_state = random_state

    def build(self, wells: dict[str, WellLogDataset], blind_well_name: str,
              target_name: str, extra_train: dict[str, WellLogDataset] | None = None) -> FeatureSet:
        if target_name not in TARGETS:
            raise ValueError(f"target_name must be one of {TARGETS}, got {target_name!r}")
        if blind_well_name not in wells:
            raise KeyError(f"Blind well {blind_well_name!r} not in {sorted(wells)}")

        blind = wells[blind_well_name]
        # NFR-12: blind well never trains; synthetic wells only ever train.
        train_wells = [w for n, w in wells.items() if n != blind_well_name]
        train_wells += list((extra_train or {}).values())
        if not train_wells:
            raise ValueError("No training wells remain after removing the blind well")

        X = np.vstack([w.features() for w in train_wells])
        y = np.concatenate([w.target(target_name) for w in train_wells])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        return FeatureSet(X_tr, y_tr, X_te, y_te, blind.features(), blind.target(target_name),
                          blind, list(blind.feature_columns), target_name)

"""WellLogRepository (Design 2.1): CSV or NumPy .npy -> WellLogDataset.

Supported per-well files in the data directory (the file stem is the well name):
  <well>.csv  headers mapped to the schema through ``column_map``
  <well>.npy  2-D float array, shape (n_samples, 10). Columns are read in
              ``npy_column_order`` (default = SCHEMA order:
              DEPTH, VSH, VSAND, VCARB, POR, SW, PD, VP, VS, RHO).
              Structured arrays with named fields are also accepted.
If both a .csv and a .npy exist for the same well, the .npy is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .entities import SCHEMA, WellLogDataset

SUPPORTED_EXTENSIONS = (".npy", ".csv")  # priority order


class WellLogRepository:
    def __init__(self, data_dir: str | Path, column_map: dict[str, str] | None = None,
                 npy_column_order: list[str] | None = None):
        self.data_dir = Path(data_dir)
        # case-insensitive header matching
        self.column_map = {k.lower(): v for k, v in (column_map or {}).items()}
        self.npy_column_order = list(npy_column_order or SCHEMA)
        unknown = [c for c in self.npy_column_order if c not in SCHEMA]
        if unknown:
            raise ValueError(f"npy_column_order has unknown column(s) {unknown}; valid: {SCHEMA}")

    def available_wells(self) -> list[str]:
        return sorted({p.stem for ext in SUPPORTED_EXTENSIONS
                       for p in self.data_dir.glob(f"*{ext}")})

    def load(self, well_name: str) -> WellLogDataset:
        for ext in SUPPORTED_EXTENSIONS:
            path = self.data_dir / f"{well_name}{ext}"
            if path.exists():
                return self._load_npy(path, well_name) if ext == ".npy" else self._load_csv(path, well_name)
        raise FileNotFoundError(
            f"No file {well_name}.npy or {well_name}.csv in {self.data_dir}. "
            f"Available: {self.available_wells()}")

    def load_many(self, well_names: list[str]) -> dict[str, WellLogDataset]:
        return {name: self.load(name) for name in well_names}

    # ------------------------------------------------------------------ readers
    def _load_csv(self, path: Path, well_name: str) -> WellLogDataset:
        df = pd.read_csv(path)
        rename = {}
        for col in df.columns:
            key = col.strip()
            if key.upper() in SCHEMA:
                rename[col] = key.upper()
            elif key.lower() in self.column_map:
                rename[col] = self.column_map[key.lower()]
        return self._from_dataframe(df.rename(columns=rename), well_name)

    def _load_npy(self, path: Path, well_name: str) -> WellLogDataset:
        # allow_pickle=False: never execute pickled objects from a data file
        arr = np.load(path, allow_pickle=False)
        if arr.dtype.names:  # structured array -> treat field names like CSV headers
            return self._from_dataframe(
                pd.DataFrame({self._map_name(n): arr[n] for n in arr.dtype.names}), well_name)
        if arr.ndim != 2:
            raise ValueError(f"Well {well_name}: .npy must be 2-D (n_samples, columns), "
                             f"got shape {arr.shape}")
        if arr.shape[1] != len(self.npy_column_order):
            raise ValueError(
                f"Well {well_name}: .npy has {arr.shape[1]} columns but npy_column_order "
                f"lists {len(self.npy_column_order)}: {self.npy_column_order}")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Well {well_name}: .npy must be numeric, got dtype {arr.dtype}")
        return self._from_dataframe(pd.DataFrame(arr, columns=self.npy_column_order), well_name)

    def _map_name(self, name: str) -> str:
        key = name.strip()
        if key.upper() in SCHEMA:
            return key.upper()
        return self.column_map.get(key.lower(), key)

    @staticmethod
    def _from_dataframe(df: pd.DataFrame, well_name: str) -> WellLogDataset:
        missing = [c for c in SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(f"Well {well_name}: missing required column(s) {missing}")
        df = df[SCHEMA]
        non_numeric = [c for c in SCHEMA if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"Well {well_name}: non-numeric column(s) {non_numeric}")
        return WellLogDataset(well_name, df.to_numpy(dtype=float))

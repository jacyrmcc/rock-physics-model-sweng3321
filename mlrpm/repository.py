"""WellLogRepository (Design 2.1): CSV -> WellLogDataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .entities import SCHEMA, WellLogDataset


class WellLogRepository:
    def __init__(self, data_dir: str | Path, column_map: dict[str, str] | None = None):
        self.data_dir = Path(data_dir)
        # case-insensitive header matching
        self.column_map = {k.lower(): v for k, v in (column_map or {}).items()}

    def available_wells(self) -> list[str]:
        return sorted(p.stem for p in self.data_dir.glob("*.csv"))

    def load(self, well_name: str) -> WellLogDataset:
        path = self.data_dir / f"{well_name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No file {path.name} in {self.data_dir}. Available: {self.available_wells()}"
            )
        df = pd.read_csv(path)
        rename = {}
        for col in df.columns:
            key = col.strip()
            if key.upper() in SCHEMA:
                rename[col] = key.upper()
            elif key.lower() in self.column_map:
                rename[col] = self.column_map[key.lower()]
        df = df.rename(columns=rename)

        missing = [c for c in SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(f"Well {well_name}: missing required column(s) {missing}")
        df = df[SCHEMA]
        non_numeric = [c for c in SCHEMA if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"Well {well_name}: non-numeric column(s) {non_numeric}")
        return WellLogDataset(well_name, df.to_numpy(dtype=float))

    def load_many(self, well_names: list[str]) -> dict[str, WellLogDataset]:
        return {name: self.load(name) for name in well_names}

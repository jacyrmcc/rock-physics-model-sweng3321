"""Configuration file handling (NFR-10).

All physical constants and default parameters live in ``default_config.yaml``.
``load_config`` reads a user copy (falling back to the defaults for any key
the user file omits) and ``restore_defaults`` writes a fresh copy.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the configuration dict; ``path`` overrides the defaults."""
    with open(DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if path is not None:
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def restore_defaults(dest: str | Path) -> Path:
    """Write a copy of the default configuration to ``dest`` (FR-3)."""
    dest = Path(dest)
    shutil.copyfile(DEFAULT_CONFIG_PATH, dest)
    return dest

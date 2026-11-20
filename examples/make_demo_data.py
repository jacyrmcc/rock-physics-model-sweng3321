# Artifact: make_demo_data.py
# Author: Obed Plasencia
# Date coded: November 12, 2026
"""Make demo well files (synthetic stand-ins for the field wells).

Usage::

    python examples/make_demo_data.py data            # writes WELL-A..WELL-D.csv
    python examples/make_demo_data.py data --force    # overwrite existing files

The repository already includes the demo wells in ``data/``, and the results in
the report were produced from those files. This script shows how they were made
and lets you rebuild a demo set from scratch. It will not overwrite existing
files unless you pass ``--force``, because rebuilt wells give slightly different
scores from the ones in the report.

The "earth" used here is deliberately NOT the default RPM, so the ML models have
a mismatch to learn (see README, "Demo data"):
  * a partly cemented frame (a blend of the soft-sand and stiff-sand bounds),
  * a lower coordination number than the default config,
  * a different pore pressure,
  * random measurement noise on Vp, Vs and RHO.
Replace these files with real CSVs for real results.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Let the script run from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlrpm import ForwardModel, WellLogDataset, load_config  # noqa: E402

# CSV headers use the field-style names that column_map in the config maps
# to the internal schema (FR-1, NFR-6).
HEADERS = ["Depth_m", "Vclay", "Vsand", "Vcarb", "PHIT", "SWT", "Pdiff_MPa",
           "Vp_kms", "Vs_kms", "RHOB"]
WELLS = ["WELL-A", "WELL-B", "WELL-C", "WELL-D"]

N_SAMPLES = 600          # depth samples per well
INTERVAL_M = 300.0       # thickness of each well's logged interval, m
TOP_M = 2500.0           # top of WELL-A, m; each next well starts 150 m deeper
STEP_M = 150.0
PD_GRADIENT = 0.00634    # differential pressure gradient, MPa per m
CEMENT_FRACTION = 0.3    # 0 = soft sand only, 1 = stiff sand only
NOISE = 0.02             # relative noise on Vp, Vs and RHO


def smooth(x: np.ndarray, window: int = 31) -> np.ndarray:
    """Moving average so the curves look depth-continuous, like real logs."""
    kernel = np.ones(window) / window
    pad = np.pad(x, window // 2, mode="reflect")
    return np.convolve(pad, kernel, mode="same")[window // 2: window // 2 + len(x)]


def wiggle(rng: np.random.Generator, n: int) -> np.ndarray:
    """Smooth random curve with mean 0 and standard deviation 1."""
    x = smooth(rng.normal(size=n))
    return (x - x.mean()) / x.std()


def earth_models(cfg: dict) -> tuple[ForwardModel, ForwardModel]:
    """Two forward models (soft and stiff frame) for the 'true' earth."""
    earth = copy.deepcopy(cfg)
    for lith in earth["hm_params"].values():
        lith["C"] *= 0.85                       # looser grain packing
    earth["fluid_params"]["pore_pressure_mpa"] = 25.0
    soft = ForwardModel(earth["end_members"], earth["hm_params"],
                        earth["fluid_params"], frame_bound="lower")
    stiff = ForwardModel(earth["end_members"], earth["hm_params"],
                         earth["fluid_params"], frame_bound="upper")
    return soft, stiff


def make_well(name: str, top: float, rng: np.random.Generator,
              soft: ForwardModel, stiff: ForwardModel) -> pd.DataFrame:
    n = N_SAMPLES
    depth = np.linspace(top, top + INTERVAL_M, n)

    # Lithology: smooth random curves passed through a softmax, so
    # VSH + VSAND + VCARB = 1 at every depth (carbonate is the minor mineral).
    latent = np.column_stack([1.2 * wiggle(rng, n), 1.2 * wiggle(rng, n),
                              1.2 * wiggle(rng, n) - 0.6])
    fr = np.exp(latent)
    fr /= fr.sum(axis=1, keepdims=True)

    # Porosity: blocky layers, a little lower in shale-rich intervals.
    por = 0.16 + 0.06 * wiggle(rng, n) - 0.08 * (fr[:, 0] - 0.35)
    por = np.clip(por, 0.03, 0.29)
    sw = np.clip(0.75 + 0.25 * wiggle(rng, n), 0.25, 1.0)
    pd_ = PD_GRADIENT * depth + 0.3 * wiggle(rng, n)

    zeros = np.zeros(n)
    arr = np.column_stack([depth, fr, por, sw, pd_, zeros, zeros, zeros])
    well = WellLogDataset(name, arr)

    vp_s, vs_s, rho = soft.compute(well)
    vp_h, vs_h, _ = stiff.compute(well)
    w = CEMENT_FRACTION
    vp = (1 - w) * vp_s + w * vp_h
    vs = (1 - w) * vs_s + w * vs_h

    vp *= 1 + rng.normal(0, NOISE, n)
    vs *= 1 + rng.normal(0, NOISE, n)
    rho *= 1 + rng.normal(0, NOISE / 4, n)

    cols = [depth, fr[:, 0], fr[:, 1], fr[:, 2], por, sw, pd_, vp, vs, rho]
    return pd.DataFrame(np.column_stack(cols), columns=HEADERS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write demo well CSV files.")
    ap.add_argument("out_dir", nargs="?", default="data", help="folder to write to")
    ap.add_argument("--seed", type=int, default=2026, help="random seed (NFR-11)")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = [w for w in WELLS if (out / f"{w}.csv").exists()]
    if existing and not args.force:
        print(f"{out} already has {', '.join(existing)}. Nothing written. "
              "Use --force to overwrite them.")
        return 0

    soft, stiff = earth_models(load_config())
    rng = np.random.default_rng(args.seed)
    for i, name in enumerate(WELLS):
        df = make_well(name, TOP_M + STEP_M * i, rng, soft, stiff)
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / (name + '.csv')} ({len(df)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

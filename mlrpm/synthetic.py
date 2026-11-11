"""SyntheticWellGenerator (Design 2.4.6-2.4.7).

>>> NOTEBOOK HOOK: replace ``_sample_curves`` with the curve-sampling logic
>>> from ML_SyntheticModel if it differs (e.g. layered/blocky models, depth
>>> trends, correlated porosity-lithology).
"""
from __future__ import annotations

import numpy as np

from .entities import WellLogDataset
from .forward_model import ForwardModel


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    pad = np.pad(x, window // 2, mode="reflect")
    return np.convolve(pad, kernel, mode="same")[window // 2: window // 2 + len(x)]


class SyntheticWellGenerator:
    def __init__(self, forward_model: ForwardModel, n_samples: int = 500,
                 random_state: int = 7, depth_top_m: float = 3000.0,
                 depth_base_m: float = 4000.0, por_range=(0.05, 0.35),
                 sw_range=(0.2, 1.0), pd_range=(5.0, 40.0), smooth_window: int = 25):
        if n_samples < 10:
            raise ValueError("n_samples must be >= 10")
        self.forward_model = forward_model
        self.n_samples = n_samples
        self.random_state = random_state
        self.depth_range = (depth_top_m, depth_base_m)
        self.por_range, self.sw_range, self.pd_range = por_range, sw_range, pd_range
        self.smooth_window = smooth_window

    @classmethod
    def from_config(cls, forward_model: ForwardModel, cfg: dict) -> "SyntheticWellGenerator":
        return cls(forward_model, **cfg.get("synthetic", {}))

    def generate(self, n_wells: int) -> dict[str, WellLogDataset]:
        if n_wells < 1:
            raise ValueError("n_wells must be >= 1")
        rng = np.random.default_rng(self.random_state)  # NFR-11
        wells = {}
        for i in range(1, n_wells + 1):
            name = f"SYN-{i:03d}"
            w = self._sample_curves(rng, name)
            vp, vs, rho = self.forward_model.compute(w)  # exact, noise-free
            arr = w.array.copy()
            arr[:, 7], arr[:, 8], arr[:, 9] = vp, vs, rho
            wells[name] = WellLogDataset(name, arr)
        return wells

    def _sample_curves(self, rng: np.random.Generator, name: str = "SYN") -> WellLogDataset:
        n, s = self.n_samples, self.smooth_window
        # Dirichlet keeps VSH+VSAND+VCARB = 1; smoothing then re-normalising
        # gives geologically plausible, depth-continuous lithology.
        fr = rng.dirichlet([1, 1, 1], size=n)
        fr = np.column_stack([_smooth(fr[:, j], s) for j in range(3)])
        fr /= fr.sum(axis=1, keepdims=True)
        por = _smooth(rng.uniform(*self.por_range, n), s)
        sw = _smooth(rng.uniform(*self.sw_range, n), s)
        pd = _smooth(rng.uniform(*self.pd_range, n), s)
        depth = np.linspace(*self.depth_range, n)
        zeros = np.zeros(n)
        arr = np.column_stack([depth, fr, por, sw, pd, zeros, zeros, zeros])
        return WellLogDataset(name, arr)

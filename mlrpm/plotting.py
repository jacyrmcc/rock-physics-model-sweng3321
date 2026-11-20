# Artifact: plotting.py
# Author: Obed Plasencia
# Date coded: November 12, 2026
"""Boundary classes (Design 2.7): LogPlotter, ResultsVisualizer."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# RPM, SVR, RF, MLP, extra ...
DEFAULT_PALETTE = ["#222222", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
# Okabe-Ito colorblind-safe palette (UIR-4)
COLORBLIND_PALETTE = ["#000000", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00"]
UNITS = {"VP": "Vp (km/s)", "VS": "Vs (km/s)"}


def _palette(colorblind: bool, palette):
    return list(palette) if palette else (COLORBLIND_PALETTE if colorblind else DEFAULT_PALETTE)


class LogPlotter:
    def __init__(self, figsize=(12, 8), palette=None, colorblind: bool = False):
        self.figsize = tuple(figsize)
        self.colorblind = colorblind
        self.palette = _palette(colorblind, palette)

    def plot_depth_track(self, depth, measured, predicted, target_label: str,
                         well_name: str, ax=None):
        """``predicted`` may be one array or a dict {model name: array}."""
        preds = predicted if isinstance(predicted, dict) else {"Predicted": predicted}
        for p in preds.values():
            if not (len(depth) == len(measured) == len(p)):
                raise ValueError("depth, measured and predicted must have the same length")
        if ax is None:
            _, ax = plt.subplots(figsize=(4, self.figsize[1]))
        ax.plot(measured, depth, color="0.6", lw=2.0, label="Measured")
        for i, (name, p) in enumerate(preds.items()):
            ax.plot(p, depth, color=self.palette[i % len(self.palette)], lw=1.0, label=name)
        ax.invert_yaxis() if not ax.yaxis_inverted() else None
        ax.set_xlabel(UNITS.get(target_label, target_label))
        ax.set_ylabel("Depth (m)")
        ax.set_title(well_name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        return ax

    def plot_multi_well_grid(self, wells_depth: dict, wells_measured: dict,
                             wells_predicted: dict, target_label: str):
        n = len(wells_depth)
        fig, axes = plt.subplots(1, n, sharey=False, figsize=self.figsize, squeeze=False)
        for ax, name in zip(axes[0], wells_depth):
            self.plot_depth_track(wells_depth[name], wells_measured[name],
                                  wells_predicted[name], target_label, name, ax=ax)
        fig.tight_layout()
        return fig

    def plot_pipeline_result(self, result):
        """Convenience: multi-well grid straight from a PipelineResult."""
        t = result.target_name
        return self.plot_multi_well_grid(
            {n: w.depth() for n, w in result.blind_wells.items()},
            {n: w.target(t) for n, w in result.blind_wells.items()},
            result.predictions, t)


class ResultsVisualizer:
    def __init__(self, palette=None, colorblind: bool = False):
        self.colorblind = colorblind
        self.palette = _palette(colorblind, palette)

    def crossplot(self, y_true, y_pred, target_label: str, ax=None, label=None, color=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, s=6, alpha=0.5, label=label, color=color or self.palette[1])
        lo, hi = float(np.min(y_true)), float(np.max(y_true))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        u = UNITS.get(target_label, target_label)
        ax.set_xlabel(f"Measured {u}")
        ax.set_ylabel(f"Predicted {u}")
        ax.grid(alpha=0.3)
        if label:
            ax.legend()
        return ax

    def metric_bar_chart(self, pivot_table: pd.DataFrame, metric_label: str, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        pivot_table.plot.bar(ax=ax, color=self.palette[: pivot_table.shape[1]], rot=0)
        ax.set_ylabel(metric_label)
        ax.set_xlabel("Blind well")
        ax.grid(axis="y", alpha=0.3)
        return ax

    def box_plot(self, errors_by_model: dict, metric_label: str, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(list(errors_by_model.values()), tick_labels=list(errors_by_model.keys()),
                   showfliers=False)
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.3)
        return ax

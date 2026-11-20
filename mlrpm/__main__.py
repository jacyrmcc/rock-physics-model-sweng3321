# Artifact: __main__.py
# Author: Obed Plasencia
# Date coded: November 12, 2026
"""Command-line application shell.

    python -m mlrpm list     --data-dir data
    python -m mlrpm forward  --data-dir data --well WELL-A --target VP --out results
    python -m mlrpm run      --data-dir data --target VP --n-synthetic 3 --out results
    python -m mlrpm defaults my_config.yaml

Any error returns a descriptive message and a non-zero exit code (NFR-3).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from .config import load_config, restore_defaults  # noqa: E402
from .entities import METRICS  # noqa: E402
from .forward_model import ForwardModel  # noqa: E402
from .pipeline import RockPhysicsPipeline  # noqa: E402
from .plotting import LogPlotter, ResultsVisualizer  # noqa: E402
from .repository import WellLogRepository  # noqa: E402

METRIC_LABELS = {"r2": "R²", "accuracy_pct": "Accuracy (%)", "rmse": "RMSE (km/s)",
                 "mae": "MAE (km/s)"}


def export_report(result, out: Path) -> None:
    """UC-3: write evaluation tables as CSV (PDF deferred)."""
    out.mkdir(parents=True, exist_ok=True)
    t = result.target_name
    result.report.to_dataframe().to_csv(out / f"{t}_all_results.csv", index=False)
    for m in METRICS:
        result.report.pivot(m).to_csv(out / f"{t}_{m}_by_well.csv")
    summary = {m: result.report.average_by_model(m) for m in METRICS}
    import pandas as pd
    pd.DataFrame(summary).to_csv(out / f"{t}_summary_by_model.csv")


def save_plots(result, out: Path, colorblind: bool, figsize) -> None:
    lp = LogPlotter(figsize=figsize, colorblind=colorblind)
    rv = ResultsVisualizer(colorblind=colorblind)
    t = result.target_name
    lp.plot_pipeline_result(result).savefig(out / f"{t}_depth_tracks.png", dpi=120)
    for m in ("r2", "accuracy_pct"):
        ax = rv.metric_bar_chart(result.report.pivot(m), METRIC_LABELS[m])
        ax.figure.tight_layout(); ax.figure.savefig(out / f"{t}_{m}_bars.png", dpi=120)
    ax = rv.box_plot(result.errors_by_model(), f"|error| {t} (km/s)")
    ax.figure.tight_layout(); ax.figure.savefig(out / f"{t}_error_boxplot.png", dpi=120)

    import matplotlib.pyplot as plt
    models = list(next(iter(result.predictions.values())).keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for i, (ax, model) in enumerate(zip(axes[0], models)):
        for well, preds in result.predictions.items():
            y = result.blind_wells[well].target(t)
            rv.crossplot(y, preds[model], t, ax=ax, color=rv.palette[i % len(rv.palette)])
        ax.set_title(model)
    fig.tight_layout(); fig.savefig(out / f"{t}_crossplots.png", dpi=120)
    plt.close("all")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mlrpm", description="ML-RPM proof of concept")
    p.add_argument("--config", help="YAML config overriding the defaults")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list wells in a data directory (UC-1)")
    s.add_argument("--data-dir", required=True)

    s = sub.add_parser("forward", help="run the theoretical forward model on one well (UC-2)")
    s.add_argument("--data-dir", required=True); s.add_argument("--well", required=True)
    s.add_argument("--target", default="VP", choices=["VP", "VS"])
    s.add_argument("--out", default="results")

    s = sub.add_parser("run", help="full blind-well evaluation pipeline (UC-11)")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--wells", nargs="*", help="default: all CSV wells in data dir")
    s.add_argument("--target", default="VP", choices=["VP", "VS"])
    s.add_argument("--n-synthetic", type=int, default=0)
    s.add_argument("--out", default="results")
    s.add_argument("--colorblind", action="store_true", help="UC-9")

    s = sub.add_parser("defaults", help="write a fresh copy of the default config (FR-3)")
    s.add_argument("dest")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    try:
        cfg = load_config(a.config)
        if a.cmd == "list":
            print("\n".join(WellLogRepository(a.data_dir, cfg["column_map"]).available_wells()))
        elif a.cmd == "defaults":
            print(f"Wrote {restore_defaults(a.dest)}")
        elif a.cmd == "forward":
            well = WellLogRepository(a.data_dir, cfg["column_map"]).load(a.well)
            pred = ForwardModel.from_config(cfg).predict(well, a.target)
            out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
            ax = LogPlotter(colorblind=cfg["plotting"]["colorblind"]).plot_depth_track(
                well.depth(), well.target(a.target), {"RPM": pred}, a.target, well.name)
            ax.figure.tight_layout()
            ax.figure.savefig(out / f"{well.name}_{a.target}_RPM.png", dpi=120)
            from .evaluation import ModelEvaluator
            r = ModelEvaluator().evaluate(well.target(a.target), pred, well.name, "RPM", a.target)
            print(f"{well.name} RPM {a.target}: R2={r.r2:.3f} acc={r.accuracy_pct:.2f}% "
                  f"RMSE={r.rmse:.4f} MAE={r.mae:.4f}")
        elif a.cmd == "run":
            pipe = RockPhysicsPipeline.from_config(a.data_dir, cfg)
            result = pipe.run(a.wells, a.target, a.n_synthetic)
            out = Path(a.out)
            export_report(result, out)
            save_plots(result, out, a.colorblind or cfg["plotting"]["colorblind"],
                       cfg["plotting"]["figsize"])
            print(f"=== {a.target}: R² by blind well ===")
            print(result.report.pivot("r2").round(3).to_string())
            print(f"\n=== {a.target}: averages by model ===")
            print(result.report.to_dataframe().groupby("model_name")[list(METRICS)]
                  .mean().round(4).to_string())
            print(f"\nTables and plots written to {out.resolve()}")
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

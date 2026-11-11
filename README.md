# ML-RPM: Rock-Physics Machine-Learning Prediction System (proof of concept)

This package implements the CS 3321 Design (Project Assignment 3). It compares
a **theoretical rock-physics model (RPM)** with **machine-learning models (SVR, RF, MLP)**
for predicting Vp and Vs from well logs. Its core is the Design's main principle, the
**blind-well (leave-one-well-out) evaluation pipeline**: every real well takes a turn as the blind
well. The RPM and every ML model predict that well, and all methods are scored with R², accuracy %,
RMSE and MAE. Synthetic wells from the RPM can be added as extra training data, but they are never
used as the blind well.

## Quick start

```bash
pip install -r requirements.txt          # numpy, pandas, scikit-learn, matplotlib, pyyaml
python examples/make_demo_data.py data   # demo wells (skip if you have real CSVs)
python -m mlrpm list    --data-dir data
python -m mlrpm run     --data-dir data --target VP --n-synthetic 3 --out results
python -m mlrpm run     --data-dir data --target VS --out results --colorblind
python -m mlrpm forward --data-dir data --well WELL-A --target VP
python -m mlrpm defaults my_config.yaml  # editable copy of all parameters
python -m mlrpm --config my_config.yaml run --data-dir data
pytest -q                                # 12 contract tests
```

You can also open `examples/ML_RPM_demo.ipynb` for a cell-by-cell walkthrough of each use case.

```python
from mlrpm import RockPhysicsPipeline
res = RockPhysicsPipeline.from_config("data").run(target_name="VP", n_synthetic=3)
res.report.pivot("r2")             # well x model table
res.report.average_by_model("rmse")
```

## Data format

Each well is a file called `<well_name>.csv` in the data directory. The headers are mapped to the
internal schema `DEPTH, VSH, VSAND, VCARB, POR, SW, PD, VP, VS, RHO` through `column_map` in the
config file, and header matching ignores case. The units are: depth in m; lithology fractions,
porosity and Sw as fractions; PD in MPa; Vp and Vs in km/s; RHO in g/cc. The ML features are the
same six inputs the RPM uses (`VSH, VSAND, VCARB, POR, SW, PD`), which keeps the comparison
like-for-like. You can change them with `feature_columns`.

## Layout: Design class → module

| Design class(es) | Module |
|---|---|
| WellLogDataset, FeatureSet, EvaluationResult, EvaluationReport, PipelineResult | `mlrpm/entities.py` |
| WellLogRepository | `mlrpm/repository.py` |
| FeatureEngineer | `mlrpm/features.py` |
| ForwardModel (VRH, Batzle-Wang, Wood, Hertz-Mindlin + modified HS bound, Gassmann) | `mlrpm/forward_model.py` |
| SyntheticWellGenerator | `mlrpm/synthetic.py` |
| VelocityPredictor (abstract), SVR/RF/MLP predictors, `create_predictor`, `PREDICTOR_REGISTRY` | `mlrpm/predictors.py` |
| ModelEvaluator | `mlrpm/evaluation.py` |
| RockPhysicsPipeline | `mlrpm/pipeline.py` |
| LogPlotter, ResultsVisualizer | `mlrpm/plotting.py` |
| Configuration file (NFR-7, NFR-10) | `mlrpm/default_config.yaml`, `mlrpm/config.py` |
| Application shell (statechart) | `mlrpm/__main__.py` |

## Adding an ML algorithm (NFR-9)

```python
from mlrpm import VelocityPredictor, register
from xgboost import XGBRegressor

@register("xgb")
class XGBPredictor(VelocityPredictor):
    display_name = "XGB"
    def _build_model(self, **hp):
        return XGBRegressor(**hp)
```
Then add `xgb` to `model_keys` in the config file. You don't need to change any other code.

## Where to plug in the reference notebooks

Comments marked `>>> NOTEBOOK HOOK` show where code from the `RockPhysics_Theoretical_Model` and
`ML_SyntheticModel` notebooks should go:
- `ForwardModel._frame_moduli` and `_fluid_properties`: the notebook's exact frame and fluid models.
- `SyntheticWellGenerator._sample_curves`: the notebook's curve-sampling scheme.
- `default_config.yaml`: the notebook's end-members, Hertz-Mindlin parameters and tuned hyperparameters.

If you keep the method signatures unchanged, the pipeline, evaluation and plotting will keep
working as they are.

## Notes and deviations from the Design

- **Frame model bound.** Design 2.4.5 calls it the "modified lower bound" but writes `4/3 mu_s`,
  which belongs to the modified *upper* (stiff-sand) bound. `frame_bound: lower` (the default)
  implements the soft-sand bound with `mu_HM`. Use `upper` to get the formula as it is written.
- **Fluid pressure.** Design 2.4.3 passes `pd` to Batzle-Wang. Physically, pore pressure is the
  right input, so the config has `pore_pressure_mpa`. If you set it to `null`, the model uses PD,
  as the Design does.
- **`ForwardModel.compute()`** is added. It returns (Vp, Vs, RHO) so that synthetic wells can fill
  RHO (UC-5). `predict()` works as the Design specifies.
- **Smoothed synthetic lithology.** Synthetic lithology fractions are smoothed and then
  re-normalised so they sum to 1.
- **Demo data.** `examples/make_demo_data.py` produces *synthetic stand-ins* for the field wells.
  The "earth" used to make them differs a little from the default RPM (partly cemented frame,
  different coordination number, noise), so that ML has a mismatch to learn. Do not read the
  results as findings from the paper. Replace the files with real CSVs.
- **Deferred** (as listed in the Requirements): PDF report, project save/load, GUI screens,
  glossary, grid-search calibration and LAS export.

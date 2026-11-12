"""RockPhysicsPipeline (Design 2.6): the main principle of the Design.

Every real well takes one turn as the blind well. For each turn the
theoretical RPM and every configured ML model predict the blind well, and
all methods are scored with the same four metrics.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import load_config
from .entities import PipelineResult
from .evaluation import ModelEvaluator
from .features import FeatureEngineer
from .forward_model import ForwardModel
from .predictors import create_predictor
from .repository import WellLogRepository
from .synthetic import SyntheticWellGenerator

log = logging.getLogger(__name__)


class RockPhysicsPipeline:
    def __init__(self, repository: WellLogRepository, feature_engineer: FeatureEngineer,
                 evaluator: ModelEvaluator, forward_model: ForwardModel,
                 generator: SyntheticWellGenerator, model_keys: list[str],
                 model_hyperparameters: dict | None = None):
        self.repository = repository
        self.feature_engineer = feature_engineer
        self.evaluator = evaluator
        self.forward_model = forward_model
        self.generator = generator
        self.model_keys = list(model_keys)
        self.model_hyperparameters = model_hyperparameters or {}

    @classmethod
    def from_config(cls, data_dir: str | Path, cfg: dict | None = None) -> "RockPhysicsPipeline":
        cfg = cfg or load_config()
        fm = ForwardModel.from_config(cfg)
        return cls(
            repository=WellLogRepository(data_dir, cfg.get("column_map"), cfg.get("npy_column_order")),
            feature_engineer=FeatureEngineer(**cfg.get("feature_engineer", {})),
            evaluator=ModelEvaluator(),
            forward_model=fm,
            generator=SyntheticWellGenerator.from_config(fm, cfg),
            model_keys=cfg.get("model_keys", ["svr", "rf", "mlp"]),
            model_hyperparameters=cfg.get("model_hyperparameters", {}))

    def run(self, well_names: list[str] | None = None, target_name: str = "VP",
            n_synthetic: int = 0) -> PipelineResult:
        well_names = well_names or self.repository.available_wells()
        wells = self.repository.load_many(well_names)
        extra = self.generator.generate(n_synthetic) if n_synthetic > 0 else {}
        log.info("Loaded %d wells, %d synthetic", len(wells), len(extra))

        predictions, fitted_models, y_true_by_well, blind_wells = {}, {}, {}, {}
        for blind_name in well_names:
            fs = self.feature_engineer.build(wells, blind_name, target_name, extra)
            y_true_by_well[blind_name] = fs.y_blind
            blind_wells[blind_name] = fs.blind_well
            predictions[blind_name] = {"RPM": self.forward_model.predict(fs.blind_well, target_name)}
            fitted_models[blind_name] = {}
            for key in self.model_keys:
                predictor = create_predictor(key, **self.model_hyperparameters.get(key, {}))
                predictor.fit(fs.X_train, fs.y_train)
                predictions[blind_name][predictor.display_name] = predictor.predict(fs.X_blind)
                fitted_models[blind_name][predictor.display_name] = predictor
                log.info("blind=%s model=%s done", blind_name, predictor.display_name)

        report = self.evaluator.evaluate_many(predictions, y_true_by_well, target_name)
        return PipelineResult(target_name, report, predictions, blind_wells, fitted_models)

# Artifact: __init__.py
# Author: Obed Plasencia
# Date coded: November 12, 2026
"""ML-RPM: Rock-Physics Machine-Learning Prediction System (proof of concept).

Quick start::

    from mlrpm import RockPhysicsPipeline
    pipe = RockPhysicsPipeline.from_config("data/")
    result = pipe.run(target_name="VP", n_synthetic=3)
    print(result.report.pivot("r2"))
"""
from .config import load_config, restore_defaults
from .entities import (EvaluationReport, EvaluationResult, FeatureSet, PipelineResult,
                       WellLogDataset, SCHEMA)
from .evaluation import ModelEvaluator
from .features import FeatureEngineer
from .forward_model import ForwardModel
from .pipeline import RockPhysicsPipeline
from .predictors import (PREDICTOR_REGISTRY, MLPPredictor, RandomForestPredictor,
                         SVRPredictor, VelocityPredictor, create_predictor, register)
from .repository import WellLogRepository
from .synthetic import SyntheticWellGenerator

__all__ = [
    "load_config", "restore_defaults", "SCHEMA",
    "WellLogDataset", "FeatureSet", "EvaluationResult", "EvaluationReport", "PipelineResult",
    "RockPhysicsPipeline", "WellLogRepository", "FeatureEngineer", "ModelEvaluator",
    "ForwardModel", "SyntheticWellGenerator", "VelocityPredictor", "SVRPredictor",
    "RandomForestPredictor", "MLPPredictor", "create_predictor", "register",
    "PREDICTOR_REGISTRY",
]
__version__ = "0.1.0"

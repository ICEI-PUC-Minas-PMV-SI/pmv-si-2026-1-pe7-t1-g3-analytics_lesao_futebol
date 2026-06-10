"""Utility to rebuild the production pipeline artifact with a chosen model.

This helper keeps the existing production artifact structure and replaces only
its model component. It is intended for maintenance/retraining workflows.

Usage:
    python -m src.tools.rebuild_production_pipeline --model lgbm_optuna
"""

import argparse
from pathlib import Path

import joblib

from ..production import ProductionPipeline
from ..utils import save_artifact


ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
PRODUCTION_PATH = MODELS_DIR / "production_pipeline.joblib"
DEFAULT_MODEL = "lgbm_optuna"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild production_pipeline.joblib")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["lgbm_optuna", "catboost_optuna", "xgboost_optuna"],
        help="Model artifact to inject into the existing production pipeline",
    )
    parser.add_argument(
        "--output",
        default=str(PRODUCTION_PATH),
        help="Where to save the rebuilt production artifact",
    )
    args = parser.parse_args()

    if not PRODUCTION_PATH.exists():
        raise FileNotFoundError(f"Expected existing production artifact at {PRODUCTION_PATH}")

    model_path = MODELS_DIR / f"{args.model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    current = ProductionPipeline.load_pipeline(str(PRODUCTION_PATH))
    replacement_model = joblib.load(model_path)

    rebuilt = {
        "model": replacement_model,
        "feature_pipeline": current.feature_pipeline,
        "feature_names": current.feature_names,
        "known_categories": current.known_categories,
        "version": current._version,
    }

    save_artifact(rebuilt, args.output)
    print(f"Rebuilt pipeline saved to: {args.output}")
    print(f"Model injected: {args.model}")

    loaded = ProductionPipeline.load_pipeline(args.output)
    print("Validation OK:", type(loaded).__name__, "feature_names=", len(loaded.feature_names))


if __name__ == "__main__":
    main()

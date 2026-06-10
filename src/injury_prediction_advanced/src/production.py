"""
Production Pipeline Module
==========================
Implements Melhoria 19: end-to-end production-ready inference pipeline
with input validation, unknown-category handling, and serialisation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .utils import load_artifact, save_artifact, setup_logging

logger = setup_logging("production")


class ProductionPipeline:
    """
    Wraps the full trained pipeline (feature engineering + model) for
    production inference with robust input validation.
    """

    def __init__(
        self,
        model: Any = None,
        feature_pipeline: Any = None,
        feature_names: Optional[List[str]] = None,
        known_categories: Optional[Dict[str, set]] = None,
    ) -> None:
        """
        Parameters
        ----------
        model : fitted estimator
            The trained regression model.
        feature_pipeline : FeaturePipeline
            The fitted feature engineering pipeline.
        feature_names : list of str
            Ordered feature column names used by the model.
        known_categories : dict
            ``{column_name: set_of_known_values}`` from training data.
        """
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.feature_names = feature_names or []
        self.known_categories = known_categories or {}
        self._version: str = "1.0.0"

    # ---- Serialisation ----
    def save_pipeline(self, path: Optional[str] = None) -> str:
        """
        Persist the entire pipeline (model + feature engineering).

        Parameters
        ----------
        path : str, optional
            File path. Defaults to ``models/production_pipeline.joblib``.

        Returns
        -------
        str
            Path where the pipeline was saved.
        """
        path = path or str(config.MODELS_DIR / "production_pipeline.joblib")
        payload = {
            "model": self.model,
            "feature_pipeline": self.feature_pipeline,
            "feature_names": self.feature_names,
            "known_categories": self.known_categories,
            "version": self._version,
        }
        save_artifact(payload, path)
        logger.info("Production pipeline saved to %s", path)
        return path

    @classmethod
    def load_pipeline(cls, path: Optional[str] = None) -> "ProductionPipeline":
        """
        Load a saved pipeline.

        Parameters
        ----------
        path : str, optional
            Path to the saved pipeline.

        Returns
        -------
        ProductionPipeline
        """
        path = path or str(config.MODELS_DIR / "production_pipeline.joblib")
        payload = load_artifact(path)
        logger.info("Pipeline loaded (version %s)", payload.get("version", "unknown"))
        return cls(
            model=payload["model"],
            feature_pipeline=payload["feature_pipeline"],
            feature_names=payload["feature_names"],
            known_categories=payload.get("known_categories", {}),
        )

    # ---- Input validation ----
    def validate_input(self, df: pd.DataFrame) -> List[str]:
        """
        Validate input data and return a list of warnings/issues.

        Checks:
            - Required columns present
            - Data types
            - Value ranges
            - Unknown categories

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        list of str
            Validation warnings (empty if all good).
        """
        warnings: List[str] = []

        # Required columns
        required = {
            "Injury",
            "player_age",
            "player_position",
            "club",
            "league",
            config.DATE_FROM_COLUMN,
            config.DATE_UNTIL_COLUMN,
            config.PLAYER_COLUMN,
            config.SEASON_COLUMN,
        }
        missing = required - set(df.columns)
        if missing:
            warnings.append(f"Missing columns: {missing}")

        # Age range
        if "player_age" in df.columns:
            if df["player_age"].min() < 15 or df["player_age"].max() > 45:
                warnings.append(
                    f"Age out of expected range [15, 45]: "
                    f"[{df['player_age'].min()}, {df['player_age'].max()}]"
                )

        # Unknown categories
        for col, known in self.known_categories.items():
            if col in df.columns:
                unknown = set(df[col].unique()) - known
                if unknown:
                    warnings.append(
                        f"{col}: {len(unknown)} unknown categories: "
                        f"{list(unknown)[:5]}{'...' if len(unknown) > 5 else ''}"
                    )

        if warnings:
            for w in warnings:
                logger.warning("Validation: %s", w)
        else:
            logger.info("Input validation passed ✓")

        return warnings

    # ---- Handle unknowns ----
    def handle_unknown_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map unknown categorical values to the most frequent known category
        (mode imputation) so that encoding does not fail.

        Parameters
        ----------
        df : pd.DataFrame
            Input data (may contain unseen categories).

        Returns
        -------
        pd.DataFrame
            Data with unknown categories replaced.
        """
        df = df.copy()
        for col, known in self.known_categories.items():
            if col not in df.columns:
                continue
            unknown_mask = ~df[col].isin(known)
            if unknown_mask.sum() > 0:
                # Fall back to most frequent known value
                mode = df.loc[~unknown_mask, col].mode()
                replacement = mode.iloc[0] if len(mode) > 0 else list(known)[0]
                n_replaced = unknown_mask.sum()
                df.loc[unknown_mask, col] = replacement
                logger.info(
                    "%s: replaced %d unknown values with '%s'",
                    col,
                    n_replaced,
                    replacement,
                )
        return df

    # ---- Predict ----
    def predict(
        self,
        new_data: pd.DataFrame,
        return_metadata: bool = False,
    ) -> Any:
        """
        End-to-end prediction on new data.

        Steps:
            1. Validate input
            2. Handle unknown categories
            3. Run feature engineering
            4. Predict with model
            5. Post-process (clip negatives)

        Parameters
        ----------
        new_data : pd.DataFrame
            Raw input (same schema as training CSV).
        return_metadata : bool
            If True, return ``(predictions, metadata_dict)``.

        Returns
        -------
        np.ndarray or tuple
            Predicted injury durations in days.
        """
        logger.info("Production predict: %d new samples", len(new_data))

        # 1. Handle unknown categories first so fallbacks are applied
        new_data = self.handle_unknown_categories(new_data)

        # 2. Validate the normalized input
        warnings = self.validate_input(new_data)

        # 3. Feature engineering
        if self.feature_pipeline is not None:
            new_data = self.feature_pipeline.transform(new_data)

        # 4. Predict using the feature DataFrame (preserves column names)
        X = new_data[self.feature_names].fillna(0)
        predictions = self.model.predict(X)

        # 5. Post-process: days cannot be negative
        predictions = np.maximum(predictions, 0)

        logger.info(
            "Predictions: mean=%.1f, median=%.1f, min=%.1f, max=%.1f",
            predictions.mean(),
            np.median(predictions),
            predictions.min(),
            predictions.max(),
        )

        if return_metadata:
            meta = {
                "n_samples": len(new_data),
                "warnings": warnings,
                "prediction_stats": {
                    "mean": float(predictions.mean()),
                    "median": float(np.median(predictions)),
                    "std": float(predictions.std()),
                },
            }
            return predictions, meta

        return predictions

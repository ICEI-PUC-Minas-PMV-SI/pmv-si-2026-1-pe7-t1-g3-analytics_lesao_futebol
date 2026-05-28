"""
Quantile Regression Module
==========================
Implements Melhoria 10: quantile regression for scenario-based predictions
(optimistic / median / pessimistic).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from . import config
from .utils import setup_logging

logger = setup_logging("quantile_models")

try:
    from lightgbm import LGBMRegressor

    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


class QuantileRegression:
    """
    Train separate models for multiple quantiles to produce scenario-based
    predictions (e.g., optimistic = 10th pctl, pessimistic = 90th pctl).

    Supports GradientBoosting (sklearn) and LightGBM backends.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        backend: str = "lightgbm",
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        """
        Parameters
        ----------
        quantiles : list of float
            Quantile levels to model (default: [0.1, 0.5, 0.9]).
        backend : str
            ``"lightgbm"`` or ``"sklearn"``.
        random_state : int
            Seed.
        """
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.backend = backend
        self.random_state = random_state
        self.models: Dict[float, Any] = {}

    def _make_model(self, quantile: float) -> Any:
        """Instantiate a quantile regression model."""
        if self.backend == "lightgbm" and _HAS_LGBM:
            return LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            return GradientBoostingRegressor(
                loss="quantile",
                alpha=quantile,
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quantiles: Optional[List[float]] = None,
    ) -> "QuantileRegression":
        """
        Train one model per quantile.

        Parameters
        ----------
        X, y : array-like
            Training data.
        quantiles : list of float, optional
            Override default quantiles.

        Returns
        -------
        self
        """
        if quantiles is not None:
            self.quantiles = quantiles

        logger.info("Fitting quantile models for %s", self.quantiles)
        for q in self.quantiles:
            m = self._make_model(q)
            m.fit(X, y)
            self.models[q] = m
            logger.info("  Quantile %.2f fitted", q)

        return self

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """
        Predict all quantiles.

        Returns
        -------
        pd.DataFrame
            Columns: ``q_0.1``, ``q_0.5``, ``q_0.9``, etc.
        """
        if not self.models:
            raise RuntimeError("Call fit() first")

        result = {}
        for q, m in self.models.items():
            preds = m.predict(X)
            preds = np.maximum(preds, 0)  # days can't be negative
            result[f"q_{q}"] = preds
        return pd.DataFrame(result)

    def predict_scenarios(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (optimistic, median, pessimistic) predictions.

        Uses the lowest, middle, and highest quantiles from ``self.quantiles``.

        Returns
        -------
        tuple of (optimistic, median, pessimistic)
        """
        df = self.predict(X)
        qs = sorted(self.quantiles)
        optimistic = df[f"q_{qs[0]}"].values
        median = df[f"q_{qs[len(qs) // 2]}"].values
        pessimistic = df[f"q_{qs[-1]}"].values
        return optimistic, median, pessimistic

    def compare_with_traditional(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        traditional_preds: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compare quantile-median with a traditional point prediction.

        Returns
        -------
        pd.DataFrame
            Side-by-side MAE comparison.
        """
        _, median_preds, _ = self.predict_scenarios(X_test)
        results = {
            "model": ["Traditional (point)", "Quantile (median)"],
            "MAE": [
                mean_absolute_error(y_test, traditional_preds),
                mean_absolute_error(y_test, median_preds),
            ],
        }

        # Coverage: % of true values within [optimistic, pessimistic]
        opt, _, pess = self.predict_scenarios(X_test)
        coverage = np.mean((y_test >= opt) & (y_test <= pess))
        qs = sorted(self.quantiles)
        expected_coverage = qs[-1] - qs[0]

        logger.info(
            "Quantile coverage: %.1f%% (expected ~%.0f%%)",
            coverage * 100,
            expected_coverage * 100,
        )

        return pd.DataFrame(results)

    def pinball_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[float, float]:
        """
        Compute pinball (quantile) loss for each fitted quantile.

        Returns
        -------
        dict
            ``{quantile: loss}``
        """
        losses = {}
        for q, m in self.models.items():
            preds = m.predict(X)
            errors = y - preds
            loss = np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors))
            losses[q] = float(loss)
        return losses

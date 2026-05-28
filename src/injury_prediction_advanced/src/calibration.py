"""
Calibration & Uncertainty Module
=================================
Implements Melhoria 8: prediction intervals, calibration analysis,
and conformal prediction for regression.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from . import config
from .utils import setup_logging

logger = setup_logging("calibration")


class UncertaintyEstimator:
    """
    Provides prediction intervals via multiple strategies:
        - Bootstrap ensemble intervals
        - Conformal prediction (split-conformal)
        - Calibration analysis
    """

    def __init__(
        self,
        model: Any,
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        self.model = model
        self.random_state = random_state
        self._bootstrap_models: List[Any] = []
        self._conformal_scores: Optional[np.ndarray] = None

    # ---- Bootstrap Prediction Intervals ----
    def fit_bootstrap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 50,
    ) -> "UncertaintyEstimator":
        """
        Train *n_bootstrap* models on bootstrap resamples.

        Parameters
        ----------
        X, y : array-like
            Training data.
        n_bootstrap : int
            Number of bootstrap models.

        Returns
        -------
        self
        """
        logger.info("Fitting %d bootstrap models", n_bootstrap)
        rng = np.random.RandomState(self.random_state)
        self._bootstrap_models = []
        n = len(X)

        for i in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            m = clone(self.model)
            m.fit(X[idx], y[idx])
            self._bootstrap_models.append(m)

        logger.info("Bootstrap ensemble fitted (%d models)", n_bootstrap)
        return self

    def prediction_intervals(
        self,
        X: np.ndarray,
        confidence: float = config.CONFIDENCE_LEVEL,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute prediction intervals from the bootstrap ensemble.

        Parameters
        ----------
        X : array-like
            Features.
        confidence : float
            Confidence level (e.g. 0.95).

        Returns
        -------
        tuple of (lower, median, upper) arrays
        """
        if not self._bootstrap_models:
            raise RuntimeError("Call fit_bootstrap() first")

        all_preds = np.column_stack([m.predict(X) for m in self._bootstrap_models])
        alpha = 1 - confidence
        lower = np.percentile(all_preds, 100 * alpha / 2, axis=1)
        upper = np.percentile(all_preds, 100 * (1 - alpha / 2), axis=1)
        median = np.median(all_preds, axis=1)

        logger.info(
            "Prediction intervals (%.0f%%): mean width=%.2f",
            confidence * 100,
            np.mean(upper - lower),
        )
        return lower, median, upper

    # ---- Conformal Prediction ----
    def fit_conformal(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "UncertaintyEstimator":
        """
        Fit conformal prediction using calibration residuals.

        Uses split-conformal: the model is assumed already trained;
        ``(X_cal, y_cal)`` is a held-out calibration set.

        Parameters
        ----------
        X_cal, y_cal : array-like
            Calibration data (not used for training).

        Returns
        -------
        self
        """
        preds = self.model.predict(X_cal)
        self._conformal_scores = np.abs(y_cal - preds)
        logger.info(
            "Conformal calibration: %d samples, median score=%.2f",
            len(y_cal),
            np.median(self._conformal_scores),
        )
        return self

    def conformal_prediction(
        self,
        X: np.ndarray,
        confidence: float = config.CONFIDENCE_LEVEL,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Conformal prediction intervals.

        Parameters
        ----------
        X : array-like
            New data.
        confidence : float
            Coverage level.

        Returns
        -------
        tuple of (lower, point, upper) arrays
        """
        if self._conformal_scores is None:
            raise RuntimeError("Call fit_conformal() first")

        point = self.model.predict(X)
        q = np.percentile(self._conformal_scores, 100 * confidence)
        lower = point - q
        upper = point + q

        # Clip negatives (days cannot be negative)
        lower = np.maximum(lower, 0)

        logger.info(
            "Conformal intervals (%.0f%%): q=%.2f, mean width=%.2f",
            confidence * 100,
            q,
            np.mean(upper - lower),
        )
        return lower, point, upper

    # ---- Calibration Analysis ----
    def calibration_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        confidence_levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate coverage at multiple confidence levels.

        Parameters
        ----------
        X, y : array-like
            Test data.
        confidence_levels : list of float, optional
            Confidence levels to check (default: 0.5 to 0.99).

        Returns
        -------
        pd.DataFrame
            Nominal vs empirical coverage.
        """
        if confidence_levels is None:
            confidence_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

        results = []
        for cl in confidence_levels:
            if self._bootstrap_models:
                lower, _, upper = self.prediction_intervals(X, confidence=cl)
            elif self._conformal_scores is not None:
                lower, _, upper = self.conformal_prediction(X, confidence=cl)
            else:
                logger.warning("No uncertainty model fitted; skipping calibration")
                return pd.DataFrame()

            coverage = np.mean((y >= lower) & (y <= upper))
            mean_width = np.mean(upper - lower)
            results.append(
                {
                    "nominal_coverage": cl,
                    "empirical_coverage": float(coverage),
                    "mean_interval_width": float(mean_width),
                    "calibration_error": float(abs(cl - coverage)),
                }
            )

        df = pd.DataFrame(results)
        logger.info("Calibration analysis:\n%s", df.to_string(index=False))
        return df

    def plot_calibration_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "matplotlib.figure.Figure":
        """
        Plot nominal vs empirical coverage.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        cal_df = self.calibration_analysis(X, y)
        if cal_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center")
            return fig

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(
            cal_df["nominal_coverage"],
            cal_df["empirical_coverage"],
            "bo-",
            label="Model",
        )
        ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
        ax.set_xlabel("Nominal Coverage")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        plt.tight_layout()
        return fig

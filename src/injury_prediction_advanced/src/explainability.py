"""
Explainability Module
=====================
Implements Melhoria 7: SHAP-based model interpretability.

Provides global and local explanations, interaction analysis, and
pattern discovery segmented by position, age, and injury severity.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .utils import setup_logging

logger = setup_logging("explainability")

try:
    import shap
except ImportError:
    shap = None
    logger.warning("SHAP not installed — explainability features unavailable")


class SHAPExplainer:
    """
    Wrapper around SHAP for regression model interpretation.

    Supports tree-based and general models through automatic explainer
    selection.
    """

    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        feature_names: List[str],
        model_type: str = "tree",
    ) -> None:
        """
        Parameters
        ----------
        model : fitted estimator
            The trained model to explain.
        X_train : np.ndarray
            Training features (used as background for KernelExplainer).
        feature_names : list of str
            Feature column names.
        model_type : str
            ``"tree"`` for tree-based or ``"kernel"`` for model-agnostic.
        """
        if shap is None:
            raise ImportError("SHAP is required: pip install shap")

        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type

        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        else:
            background = shap.sample(X_train, min(100, len(X_train)))
            self.explainer = shap.KernelExplainer(model.predict, background)

        self.shap_values_: Optional[np.ndarray] = None
        self.X_explain_: Optional[np.ndarray] = None

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for the given data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            SHAP values (n_samples × n_features).
        """
        logger.info("Computing SHAP values for %d samples", len(X))
        self.X_explain_ = X
        self.shap_values_ = self.explainer.shap_values(X)
        return self.shap_values_

    # ---- Global importance ----
    def global_importance(
        self, X: Optional[np.ndarray] = None, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Mean absolute SHAP value per feature (global importance).

        Returns
        -------
        pd.DataFrame
            Sorted by importance descending.
        """
        if X is not None:
            self.compute_shap_values(X)
        if self.shap_values_ is None:
            raise RuntimeError("Call compute_shap_values() first")

        importance = np.abs(self.shap_values_).mean(axis=0)
        df = pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap": importance}
        ).sort_values("mean_abs_shap", ascending=False)
        return df.head(top_n).reset_index(drop=True)

    # ---- Local explanation ----
    def local_explanation(self, instance_idx: int) -> pd.DataFrame:
        """
        SHAP contribution breakdown for a single prediction.

        Parameters
        ----------
        instance_idx : int
            Row index in the data passed to ``compute_shap_values``.

        Returns
        -------
        pd.DataFrame
            Feature contributions sorted by absolute magnitude.
        """
        if self.shap_values_ is None:
            raise RuntimeError("Call compute_shap_values() first")

        vals = self.shap_values_[instance_idx]
        df = pd.DataFrame(
            {"feature": self.feature_names, "shap_value": vals}
        )
        df["abs_shap"] = df["shap_value"].abs()
        return df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    # ---- Plots ----
    def force_plot(self, instance_idx: int) -> Any:
        """Generate a SHAP force plot for a single instance."""
        if self.shap_values_ is None or self.X_explain_ is None:
            raise RuntimeError("Call compute_shap_values() first")
        return shap.force_plot(
            self.explainer.expected_value,
            self.shap_values_[instance_idx],
            self.X_explain_[instance_idx],
            feature_names=self.feature_names,
        )

    def waterfall_plot(self, instance_idx: int) -> None:
        """Display SHAP waterfall plot for one instance."""
        if self.shap_values_ is None:
            raise RuntimeError("Call compute_shap_values() first")
        explanation = shap.Explanation(
            values=self.shap_values_[instance_idx],
            base_values=self.explainer.expected_value,
            data=self.X_explain_[instance_idx] if self.X_explain_ is not None else None,
            feature_names=self.feature_names,
        )
        shap.plots.waterfall(explanation, show=False)

    def dependence_plot(
        self, feature1: str, feature2: Optional[str] = None
    ) -> None:
        """
        SHAP dependence plot showing how a feature's value influences
        the prediction, optionally coloured by ``feature2``.
        """
        if self.shap_values_ is None or self.X_explain_ is None:
            raise RuntimeError("Call compute_shap_values() first")
        idx1 = self.feature_names.index(feature1)
        interaction_idx = (
            self.feature_names.index(feature2) if feature2 else "auto"
        )
        shap.dependence_plot(
            idx1,
            self.shap_values_,
            self.X_explain_,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False,
        )

    def interaction_values(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute SHAP interaction values (only for tree models).

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_features, n_features).
        """
        if self.model_type != "tree":
            raise ValueError("Interaction values only supported for tree models")
        data = X if X is not None else self.X_explain_
        if data is None:
            raise RuntimeError("Provide X or call compute_shap_values() first")
        logger.info("Computing SHAP interaction values")
        return self.explainer.shap_interaction_values(data)

    # ---- Domain-specific insights ----
    def severe_injury_patterns(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        severity_threshold: float = 60,
    ) -> pd.DataFrame:
        """
        SHAP importance restricted to severe injuries (>threshold days).
        """
        mask = y_true >= severity_threshold
        if mask.sum() == 0:
            logger.warning("No severe injuries above %d days", severity_threshold)
            return pd.DataFrame()

        sv = self.explainer.shap_values(X[mask])
        importance = np.abs(sv).mean(axis=0)
        return pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap_severe": importance}
        ).sort_values("mean_abs_shap_severe", ascending=False)

    def position_specific_insights(
        self,
        X: np.ndarray,
        positions: np.ndarray,
    ) -> Dict[str, pd.DataFrame]:
        """
        Global SHAP importance per player position.
        """
        results: Dict[str, pd.DataFrame] = {}
        for pos in np.unique(positions):
            mask = positions == pos
            if mask.sum() < 10:
                continue
            sv = self.explainer.shap_values(X[mask])
            imp = np.abs(sv).mean(axis=0)
            results[pos] = pd.DataFrame(
                {"feature": self.feature_names, "mean_abs_shap": imp}
            ).sort_values("mean_abs_shap", ascending=False)
        return results

    def age_specific_insights(
        self,
        X: np.ndarray,
        ages: np.ndarray,
    ) -> Dict[str, pd.DataFrame]:
        """
        Global SHAP importance per age group.
        """
        age_groups = pd.cut(ages, bins=config.AGE_BINS, labels=config.AGE_LABELS)
        results: Dict[str, pd.DataFrame] = {}
        for grp in config.AGE_LABELS:
            mask = age_groups == grp
            if mask.sum() < 10:
                continue
            sv = self.explainer.shap_values(X[mask])
            imp = np.abs(sv).mean(axis=0)
            results[grp] = pd.DataFrame(
                {"feature": self.feature_names, "mean_abs_shap": imp}
            ).sort_values("mean_abs_shap", ascending=False)
        return results

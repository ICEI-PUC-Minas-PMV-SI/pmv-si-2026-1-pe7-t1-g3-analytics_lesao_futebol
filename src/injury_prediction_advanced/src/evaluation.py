"""
Evaluation Module
=================
Implements Melhorias 9, 11, 13, 18:
    9.  Error analysis by subgroup
    11. Nested cross-validation with GroupKFold
    13. Statistical comparison of models (bootstrap CI, paired t-test)
    18. Real-world scenario evaluation (unseen players, clubs, rare injuries)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold

from . import config
from .utils import setup_logging

logger = setup_logging("evaluation")


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute a comprehensive set of regression metrics.

    Returns
    -------
    dict
        Keys: MAE, RMSE, R2, MedianAE, MAPE, MaxError.
    """
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "MaxError": float(np.max(np.abs(y_true - y_pred))),
    }


# =============================================================================
# Melhoria 9 — Error Analysis
# =============================================================================
class ModelEvaluator:
    """
    Detailed error analysis segmented by subgroups.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metadata: pd.DataFrame,
    ) -> None:
        """
        Parameters
        ----------
        y_true, y_pred : array-like
            Actual and predicted values.
        metadata : pd.DataFrame
            Must contain columns: player_age, player_position, Injury, league.
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.residuals = self.y_true - self.y_pred
        self.abs_errors = np.abs(self.residuals)
        self.meta = metadata.copy().reset_index(drop=True)

    # ---- Major errors ----
    def analyze_major_errors(
        self, threshold_percentile: float = 95
    ) -> pd.DataFrame:
        """
        Identify and profile the largest prediction errors.

        Parameters
        ----------
        threshold_percentile : float
            Percentile above which an error is considered "major".

        Returns
        -------
        pd.DataFrame
            Rows with errors above threshold, enriched with metadata.
        """
        threshold = np.percentile(self.abs_errors, threshold_percentile)
        mask = self.abs_errors >= threshold
        result = self.meta.loc[mask].copy()
        result["y_true"] = self.y_true[mask]
        result["y_pred"] = self.y_pred[mask]
        result["abs_error"] = self.abs_errors[mask]
        logger.info(
            "Major errors (>%.0fth pctl, threshold=%.1f): %d cases",
            threshold_percentile,
            threshold,
            mask.sum(),
        )
        return result.sort_values("abs_error", ascending=False)

    # ---- By subgroup ----
    def _error_by_group(self, group_col: str) -> pd.DataFrame:
        """Compute MAE, RMSE, count per group."""
        df = self.meta.copy()
        df["abs_error"] = self.abs_errors
        df["sq_error"] = self.residuals ** 2
        df["y_true"] = self.y_true
        df["y_pred"] = self.y_pred

        agg = df.groupby(group_col).agg(
            count=("abs_error", "size"),
            MAE=("abs_error", "mean"),
            MedianAE=("abs_error", "median"),
            RMSE=("sq_error", lambda x: np.sqrt(x.mean())),
            mean_true=("y_true", "mean"),
        )
        agg["relative_error"] = agg["MAE"] / agg["mean_true"].clip(lower=1)
        return agg.sort_values("MAE", ascending=False)

    def error_by_age_group(self) -> pd.DataFrame:
        """MAE/RMSE by age group."""
        self.meta["age_group"] = pd.cut(
            self.meta["player_age"],
            bins=config.AGE_BINS,
            labels=config.AGE_LABELS,
        )
        return self._error_by_group("age_group")

    def error_by_position(self) -> pd.DataFrame:
        """MAE/RMSE by player position."""
        return self._error_by_group("player_position")

    def error_by_injury_type(self, top_n: int = 20) -> pd.DataFrame:
        """MAE/RMSE by injury type (top N most frequent)."""
        top_injuries = self.meta["Injury"].value_counts().head(top_n).index
        subset_mask = self.meta["Injury"].isin(top_injuries)
        sub_eval = ModelEvaluator(
            self.y_true[subset_mask],
            self.y_pred[subset_mask],
            self.meta[subset_mask],
        )
        return sub_eval._error_by_group("Injury")

    def error_by_league(self) -> pd.DataFrame:
        """MAE/RMSE by league."""
        return self._error_by_group("league")

    def plot_error_distribution(self) -> "matplotlib.figure.Figure":
        """
        Plot residual distribution and scatter.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Histogram of residuals
        axes[0].hist(self.residuals, bins=50, edgecolor="k", alpha=0.7)
        axes[0].axvline(0, color="red", ls="--")
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residual (true − pred)")

        # Scatter
        axes[1].scatter(self.y_true, self.y_pred, alpha=0.3, s=10)
        lims = [0, max(self.y_true.max(), self.y_pred.max())]
        axes[1].plot(lims, lims, "r--", lw=1)
        axes[1].set_xlabel("True")
        axes[1].set_ylabel("Predicted")
        axes[1].set_title("Predicted vs True")

        # Absolute error vs true
        axes[2].scatter(self.y_true, self.abs_errors, alpha=0.3, s=10)
        axes[2].set_xlabel("True")
        axes[2].set_ylabel("|Error|")
        axes[2].set_title("Absolute Error vs True")

        plt.tight_layout()
        return fig


# =============================================================================
# Melhoria 11 — Nested Cross-Validation
# =============================================================================
def nested_cross_validation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_outer: int = config.N_SPLITS_OUTER,
    n_inner: int = config.N_SPLITS_INNER,
) -> Dict[str, Any]:
    """
    Nested cross-validation with GroupKFold in both loops.

    The outer loop provides unbiased performance estimates while the inner
    loop is used for hyper-parameter selection (if ``model`` is a pipeline
    with ``GridSearchCV`` or similar inside).

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible estimator (or pipeline).
    X, y : array-like
        Features and target.
    groups : array-like
        Player names.
    n_outer, n_inner : int
        Number of folds.

    Returns
    -------
    dict
        ``outer_scores``: list of per-fold metric dicts,
        ``summary``: aggregated mean ± std.
    """
    logger.info("Running nested CV: outer=%d, inner=%d", n_outer, n_inner)
    outer_cv = GroupKFold(n_splits=n_outer)
    outer_scores: List[Dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        m = clone(model)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        scores = regression_metrics(y_test, preds)
        scores["fold"] = fold
        outer_scores.append(scores)
        logger.info("  Fold %d: MAE=%.3f, R²=%.3f", fold, scores["MAE"], scores["R2"])

    df_scores = pd.DataFrame(outer_scores)
    summary = {
        col: f"{df_scores[col].mean():.4f} ± {df_scores[col].std():.4f}"
        for col in ["MAE", "RMSE", "R2"]
    }
    logger.info("Nested CV summary: %s", summary)
    return {"outer_scores": outer_scores, "summary": summary, "details": df_scores}


# =============================================================================
# Melhoria 13 — Statistical Comparison
# =============================================================================
def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any = mean_absolute_error,
    n_iterations: int = config.BOOTSTRAP_N_ITERATIONS,
    confidence: float = config.CONFIDENCE_LEVEL,
    random_state: int = config.RANDOM_STATE,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true, y_pred : array-like
        Actual and predicted values.
    metric_fn : callable
        Metric function ``(y_true, y_pred) -> float``.
    n_iterations : int
        Number of bootstrap samples.
    confidence : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    dict
        ``point``, ``lower``, ``upper``.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = np.empty(n_iterations)

    for i in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = 1 - confidence
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    point = float(metric_fn(y_true, y_pred))

    logger.info(
        "Bootstrap CI (%.0f%%): %.4f [%.4f, %.4f]",
        confidence * 100,
        point,
        lower,
        upper,
    )
    return {"point": point, "lower": lower, "upper": upper}


def compare_models_statistically(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    metric_fn: Any = mean_absolute_error,
    n_iterations: int = config.BOOTSTRAP_N_ITERATIONS,
) -> pd.DataFrame:
    """
    Compare multiple models using bootstrap CI.

    Parameters
    ----------
    y_true : array-like
        Ground truth.
    predictions : dict
        ``{model_name: y_pred}``.
    metric_fn : callable
        Metric function.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    results = []
    for name, preds in predictions.items():
        ci = bootstrap_confidence_intervals(y_true, preds, metric_fn, n_iterations)
        ci["model"] = name
        results.append(ci)
    return pd.DataFrame(results).set_index("model")


def paired_t_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> Dict[str, Any]:
    """
    Paired t-test on absolute errors of two models.

    Returns
    -------
    dict
        ``t_statistic``, ``p_value``, ``significant`` (at alpha=0.05),
        ``better_model``.
    """
    errors_a = np.abs(y_true - y_pred_a)
    errors_b = np.abs(y_true - y_pred_b)
    t_stat, p_val = sp_stats.ttest_rel(errors_a, errors_b)

    better = model_a_name if errors_a.mean() < errors_b.mean() else model_b_name
    result = {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
        "better_model": better,
        f"{model_a_name}_mean_ae": float(errors_a.mean()),
        f"{model_b_name}_mean_ae": float(errors_b.mean()),
    }
    logger.info(
        "Paired t-test: t=%.3f, p=%.4f, significant=%s, better=%s",
        t_stat,
        p_val,
        result["significant"],
        better,
    )
    return result


# =============================================================================
# Melhoria 18 — Real-World Scenario Evaluation
# =============================================================================
class ScenarioEvaluator:
    """
    Evaluate model performance under realistic deployment scenarios.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
    ) -> None:
        self.model = model
        self.feature_names = feature_names

    def evaluate_unseen_player(
        self,
        df: pd.DataFrame,
        train_players: set,
    ) -> Dict[str, float]:
        """
        Evaluate on players not seen during training.

        Parameters
        ----------
        df : pd.DataFrame
            Test DataFrame.
        train_players : set
            Set of player names in training data.

        Returns
        -------
        dict
            Metrics on unseen players.
        """
        unseen = df[~df[config.PLAYER_COLUMN].isin(train_players)]
        if len(unseen) == 0:
            logger.warning("No unseen players in test set")
            return {}

        X = unseen[self.feature_names].values
        y = unseen[config.TARGET_COLUMN].values
        preds = self.model.predict(X)
        metrics = regression_metrics(y, preds)
        logger.info("Unseen players (%d samples): MAE=%.3f", len(unseen), metrics["MAE"])
        return metrics

    def evaluate_unseen_club(
        self,
        df: pd.DataFrame,
        train_clubs: set,
    ) -> Dict[str, float]:
        """Evaluate on clubs not seen during training."""
        unseen = df[~df["club"].isin(train_clubs)]
        if len(unseen) == 0:
            logger.warning("No unseen clubs in test set")
            return {}

        X = unseen[self.feature_names].values
        y = unseen[config.TARGET_COLUMN].values
        preds = self.model.predict(X)
        metrics = regression_metrics(y, preds)
        logger.info("Unseen clubs (%d samples): MAE=%.3f", len(unseen), metrics["MAE"])
        return metrics

    def evaluate_rare_injuries(
        self,
        df: pd.DataFrame,
        min_count: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate on rare injury types (fewer than *min_count* occurrences).
        """
        injury_counts = df["Injury"].value_counts()
        rare = injury_counts[injury_counts < min_count].index
        subset = df[df["Injury"].isin(rare)]

        if len(subset) == 0:
            logger.warning("No rare injuries in test set")
            return {}

        X = subset[self.feature_names].values
        y = subset[config.TARGET_COLUMN].values
        preds = self.model.predict(X)
        metrics = regression_metrics(y, preds)
        logger.info(
            "Rare injuries (%d types, %d samples): MAE=%.3f",
            len(rare),
            len(subset),
            metrics["MAE"],
        )
        return metrics

    def evaluate_future_season(
        self,
        df: pd.DataFrame,
        season: str,
    ) -> Dict[str, float]:
        """Evaluate on a specific future season."""
        subset = df[df[config.SEASON_COLUMN] == season]
        if len(subset) == 0:
            logger.warning("Season %s not found", season)
            return {}

        X = subset[self.feature_names].values
        y = subset[config.TARGET_COLUMN].values
        preds = self.model.predict(X)
        metrics = regression_metrics(y, preds)
        logger.info("Season %s (%d samples): MAE=%.3f", season, len(subset), metrics["MAE"])
        return metrics

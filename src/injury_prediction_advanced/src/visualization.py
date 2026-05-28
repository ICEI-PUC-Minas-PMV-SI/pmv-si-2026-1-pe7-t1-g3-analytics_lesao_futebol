"""
Visualization Module
====================
Implements Melhoria 17: Professional-quality plots using
matplotlib and seaborn for thesis-ready figures.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .utils import setup_logging

logger = setup_logging("visualization")

# Global style
sns.set_theme(style="whitegrid", font_scale=config.FONT_SCALE)
plt.rcParams.update(
    {
        "figure.dpi": config.FIGURE_DPI,
        "savefig.dpi": config.FIGURE_DPI,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }
)


def _save_fig(fig: plt.Figure, name: str) -> Path:
    """Save figure to FIGURES_DIR and return the path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    logger.info("Figure saved: %s", path)
    return path


def feature_importance_comparative(
    importances: Dict[str, pd.DataFrame],
    top_n: int = 15,
    save: bool = True,
) -> plt.Figure:
    """
    Side-by-side feature importance for multiple models.

    Parameters
    ----------
    importances : dict
        ``{model_name: DataFrame with columns [feature, importance]}``.
    top_n : int
        Number of top features to show.
    save : bool
        Whether to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_models = len(importances)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, importances.items()):
        df = df.sort_values("importance", ascending=True).tail(top_n)
        ax.barh(df["feature"], df["importance"], color=sns.color_palette(config.PALETTE, top_n))
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance")

    plt.suptitle("Feature Importance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        _save_fig(fig, "feature_importance_comparative")
    return fig


def shap_summary_plot(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    save: bool = True,
) -> plt.Figure:
    """
    SHAP beeswarm summary plot.
    """
    import shap

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        _save_fig(fig, "shap_summary")
    return fig


def residual_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """
    Comprehensive residual analysis: scatter, histogram, Q-Q.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Predicted vs True
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, c="steelblue")
    lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect")
    ax.set_xlabel("True (days)")
    ax.set_ylabel("Predicted (days)")
    ax.set_title("Predicted vs True")
    ax.legend()

    # 2. Residuals vs Predicted
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.3, s=8, c="coral")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Predicted (days)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted")

    # 3. Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, edgecolor="k", alpha=0.7, color="mediumpurple")
    ax.axvline(0, color="r", ls="--", lw=1.5)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")

    # 4. Absolute error vs True
    ax = axes[1, 1]
    abs_err = np.abs(residuals)
    ax.scatter(y_true, abs_err, alpha=0.3, s=8, c="teal")
    ax.set_xlabel("True (days)")
    ax.set_ylabel("|Error|")
    ax.set_title("Absolute Error vs True")

    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        _save_fig(fig, f"residuals_{model_name.lower().replace(' ', '_')}")
    return fig


def calibration_plots(
    calibration_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Plot calibration curve from UncertaintyEstimator.calibration_analysis().
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Coverage
    ax = axes[0]
    ax.plot(
        calibration_df["nominal_coverage"],
        calibration_df["empirical_coverage"],
        "bo-",
        ms=8,
        label="Empirical",
    )
    ax.plot([0, 1], [0, 1], "r--", label="Ideal")
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Calibration Curve")
    ax.legend()

    # Interval width
    ax = axes[1]
    ax.plot(
        calibration_df["nominal_coverage"],
        calibration_df["mean_interval_width"],
        "gs-",
        ms=8,
    )
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Mean Interval Width (days)")
    ax.set_title("Prediction Interval Width")

    plt.tight_layout()
    if save:
        _save_fig(fig, "calibration")
    return fig


def error_distribution(
    errors_by_group: Dict[str, pd.DataFrame],
    group_name: str = "Group",
    save: bool = True,
) -> plt.Figure:
    """
    Box-plot of MAE by subgroup.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    dfs = []
    for grp, df in errors_by_group.items():
        tmp = df[["MAE"]].copy()
        tmp["group"] = grp
        dfs.append(tmp)
    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values("MAE", ascending=False, inplace=True)
    sns.barplot(data=combined, y="group", x="MAE", palette=config.PALETTE, ax=ax)
    ax.set_xlabel("Mean Absolute Error (days)")
    ax.set_ylabel(group_name)
    ax.set_title(f"Error Distribution by {group_name}")
    plt.tight_layout()
    if save:
        _save_fig(fig, f"error_dist_{group_name.lower()}")
    return fig


def learning_curves(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    metric_name: str = "MAE",
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """
    Learning curve plot (train vs validation score as a function of dataset size).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_scores, "o-", label=f"Train {metric_name}", color="steelblue")
    ax.plot(train_sizes, val_scores, "s-", label=f"Validation {metric_name}", color="coral")
    ax.fill_between(train_sizes, train_scores, val_scores, alpha=0.1, color="gray")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Learning Curve — {model_name}")
    ax.legend()
    plt.tight_layout()
    if save:
        _save_fig(fig, f"learning_curve_{model_name.lower().replace(' ', '_')}")
    return fig


def model_comparison_dashboard(
    results: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Dashboard comparing multiple models on MAE, RMSE, R².

    Parameters
    ----------
    results : pd.DataFrame
        Must have columns: model, MAE, RMSE, R2.
    """
    metrics = ["MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric in zip(axes, metrics):
        data = results.sort_values(metric, ascending=(metric != "R2"))
        palette = sns.color_palette(config.PALETTE, len(data))
        ax.barh(data["model"], data[metric], color=palette)
        ax.set_xlabel(metric)
        ax.set_title(metric, fontsize=14, fontweight="bold")

        # Annotate values
        for i, (_, row) in enumerate(data.iterrows()):
            ax.text(
                row[metric],
                i,
                f" {row[metric]:.3f}",
                va="center",
                fontsize=10,
            )

    fig.suptitle(
        "Model Comparison Dashboard",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    if save:
        _save_fig(fig, "model_comparison_dashboard")
    return fig

"""
Data Processing Module
======================
Loading, cleaning, validation, and temporal splitting of the injury dataset.

Implements **Melhoria 1** (temporal causality validation) at the data level.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config
from .utils import setup_logging

logger = setup_logging("data_processing")


# =============================================================================
# Loading
# =============================================================================
def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the raw CSV dataset and perform basic type conversions.

    Parameters
    ----------
    path : str, optional
        Path to CSV. Defaults to ``config.DATA_PATH``.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with parsed dates and numeric target.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If critical columns are missing.
    """
    path = path or str(config.DATA_PATH)
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Validate required columns
    required = {
        config.RAW_TARGET_COLUMN,
        config.DATE_FROM_COLUMN,
        config.DATE_UNTIL_COLUMN,
        config.PLAYER_COLUMN,
        config.SEASON_COLUMN,
        "Injury",
        "player_age",
        "player_position",
        "club",
        "league",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse target: "43 days" -> 43
    df[config.TARGET_COLUMN] = (
        df[config.RAW_TARGET_COLUMN]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )

    # Parse dates
    df[config.DATE_FROM_COLUMN] = pd.to_datetime(
        df[config.DATE_FROM_COLUMN], format="mixed", dayfirst=False
    )
    df[config.DATE_UNTIL_COLUMN] = pd.to_datetime(
        df[config.DATE_UNTIL_COLUMN], format="mixed", dayfirst=False
    )

    # Season ordinal
    df["season_ordinal"] = df[config.SEASON_COLUMN].map(config.SEASON_ORDER)

    logger.info("Data loaded: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


# =============================================================================
# Cleaning
# =============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values, outliers, and inconsistencies.

    Steps:
        1. Drop rows where target is missing or <= 0.
        2. Cap extreme outliers in ``days_num`` at the 99th percentile.
        3. Validate date coherence (from < until).
        4. Standardise categorical values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from ``load_data()``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    n_before = len(df)
    logger.info("Cleaning data (%d rows)", n_before)

    # 1. Remove invalid target
    df = df.dropna(subset=[config.TARGET_COLUMN]).copy()
    df = df[df[config.TARGET_COLUMN] > 0].copy()

    # 2. Cap outliers at 99th percentile
    cap = df[config.TARGET_COLUMN].quantile(0.99)
    n_capped = (df[config.TARGET_COLUMN] > cap).sum()
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].clip(upper=cap)
    logger.info("Capped %d outlier values at %.0f days", n_capped, cap)

    # 3. Date coherence
    bad_dates = df[config.DATE_FROM_COLUMN] > df[config.DATE_UNTIL_COLUMN]
    if bad_dates.sum() > 0:
        logger.warning(
            "Removing %d rows with injury_from > injury_until", bad_dates.sum()
        )
        df = df[~bad_dates].copy()

    # 4. Strip whitespace from categoricals
    for col in config.CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 5. Sort chronologically (essential for temporal features)
    df = df.sort_values(
        [config.PLAYER_COLUMN, config.DATE_FROM_COLUMN]
    ).reset_index(drop=True)

    logger.info("Cleaning complete: %d -> %d rows", n_before, len(df))
    return df


# =============================================================================
# Temporal Split  (Melhoria 1)
# =============================================================================
def temporal_split(
    df: pd.DataFrame,
    test_seasons: Optional[List[str]] = None,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset respecting chronological order.

    The most recent season(s) are used as test set. This guarantees that
    no future information leaks into training.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame sorted by date.
    test_seasons : list of str, optional
        Season labels for the test set (e.g., ``["24/25"]``).
        If None, the last season is used.
    test_ratio : float
        Fallback ratio when ``test_seasons`` is None and auto-selection
        is applied.

    Returns
    -------
    tuple of pd.DataFrame
        ``(train_df, test_df)``
    """
    if test_seasons is None:
        # Use the last season chronologically
        ordered = sorted(df[config.SEASON_COLUMN].unique(), key=lambda s: config.SEASON_ORDER.get(s, 99))
        test_seasons = [ordered[-1]]

    logger.info("Test seasons: %s", test_seasons)

    train_df = df[~df[config.SEASON_COLUMN].isin(test_seasons)].copy()
    test_df = df[df[config.SEASON_COLUMN].isin(test_seasons)].copy()

    # Validate no temporal leakage
    train_max_date = train_df[config.DATE_FROM_COLUMN].max()
    test_min_date = test_df[config.DATE_FROM_COLUMN].min()
    if train_max_date > test_min_date:
        logger.warning(
            "Potential temporal overlap: train max %s > test min %s. "
            "This may occur at season boundaries and should be reviewed.",
            train_max_date.date(),
            test_min_date.date(),
        )

    logger.info(
        "Temporal split: train=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_df),
        100 * len(train_df) / len(df),
        len(test_df),
        100 * len(test_df) / len(df),
    )
    return train_df, test_df


def get_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = config.N_SPLITS_OUTER,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate GroupKFold splits grouped by ``player_name``.

    This ensures that all injuries of a single player appear only in
    train **or** validation, never both — preventing player-level leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with ``player_name`` column.
    n_splits : int
        Number of folds.

    Returns
    -------
    list of (train_idx, val_idx) tuples
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[config.PLAYER_COLUMN].values
    X_dummy = np.zeros(len(df))
    y_dummy = np.zeros(len(df))

    splits = list(gkf.split(X_dummy, y_dummy, groups=groups))
    logger.info("Generated %d GroupKFold splits", len(splits))
    return splits

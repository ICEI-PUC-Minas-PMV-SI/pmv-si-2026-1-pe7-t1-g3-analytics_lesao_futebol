"""
Feature Engineering Module
==========================
Implements Melhorias 1-4:
    1. Temporal Causality Validator
    2. Player History Features (advanced)
    3. Temporal & Calendar Features
    4. Statistical / Interaction Features

All historical features are computed **strictly** from past data to avoid
data leakage. Frequency encodings are fitted on training data only and
applied to test data.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import config
from .utils import setup_logging

logger = setup_logging("feature_engineering")


# =============================================================================
# Melhoria 1 — Temporal Causality Validator
# =============================================================================
class TemporalValidator:
    """
    Temporal causality validator.

    Classifica ocorrências em:
    - violations: possíveis problemas de causalidade
    - warnings: histórico prévio ou inconsistências de dados
    """

    HISTORY_CONTEXT_FEATURES = {
        "days_since_last_injury",
        "avg_recovery_time_recent",
        "reinjury_risk_score",
    }

    def __init__(self) -> None:
        self.violations = []
        self.warnings = []

    def validate(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        date_col: str = config.DATE_FROM_COLUMN,
    ) -> bool:

        self.violations = []
        self.warnings = []

        logger.info(
            "Validating temporal causality for %d features",
            len(feature_columns),
        )

        for col in feature_columns:

            if col not in df.columns:
                continue

            for player, grp in df.groupby(config.PLAYER_COLUMN):

                grp = grp.sort_values(date_col)

                if len(grp) < 2:
                    continue

                first_val = grp[col].iloc[0]

                # --------------------------------------------------
                # CASO 1
                # Features históricas derivadas
                # --------------------------------------------------
                if col in self.HISTORY_CONTEXT_FEATURES:

                    if pd.notna(first_val) and first_val != 0:

                        self.warnings.append(
                            {
                                "player": player,
                                "feature": col,
                                "first_value": first_val,
                                "issue": (
                                    "Historical context already present "
                                    "in first dataset record"
                                ),
                            }
                        )

                    continue

                # --------------------------------------------------
                # CASO 2
                # Demais features históricas
                # --------------------------------------------------
                if pd.notna(first_val) and first_val != 0:

                    self.violations.append(
                        {
                            "player": player,
                            "feature": col,
                            "first_value": first_val,
                            "issue": (
                                "Potential temporal causality violation"
                            ),
                        }
                    )

        # ----------------------------------------------------------
        # LOGGING
        # ----------------------------------------------------------

        if self.warnings:
            logger.warning(
                "Temporal validation: %d warnings detected",
                len(self.warnings),
            )

        if self.violations:
            logger.error(
                "Temporal validation: %d violations detected",
                len(self.violations),
            )
            return False

        logger.info(
            "Temporal causality: ALL critical features passed validation ✓"
        )

        return True

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.violations)

    def warning_report(self) -> pd.DataFrame:
        return pd.DataFrame(self.warnings)

# =============================================================================
# Melhoria 2 — Player History Features
# =============================================================================
class PlayerHistoryFeatures:
    """
    Build per-player historical features using only **past** injury records.

    Features
    --------
    - injury_sequence_number: ordinal within player
    - days_since_last_injury: gap to previous injury
    - rolling_avg_days_Xd: rolling mean of days_num over X-day windows
    - recent_severity_trend: slope of last 3 injury durations
    - injuries_last_12_months: count of injuries in prior 12 months
    - avg_recovery_time_recent: mean duration of last 3 injuries
    - chronic_injury_indicator: >3 injuries of the same type
    - reinjury_risk_score: composite risk score
    """

    def __init__(self) -> None:
        self.feature_names: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add history features to *df* (which must be sorted by player + date).

        Parameters
        ----------
        df : pd.DataFrame
            Sorted dataset.

        Returns
        -------
        pd.DataFrame
            Dataset with new columns.
        """
        logger.info("Computing player history features")
        df = df.copy()

        # Ensure sorting
        df = df.sort_values(
            [config.PLAYER_COLUMN, config.DATE_FROM_COLUMN]
        ).reset_index(drop=True)

        # --- Injury sequence number (1-based) ---
        df["injury_sequence_number"] = df.groupby(config.PLAYER_COLUMN).cumcount() + 1

        # --- Days since last injury ---
        df["prev_injury_until"] = df.groupby(config.PLAYER_COLUMN)[
            config.DATE_UNTIL_COLUMN
        ].shift(1)
        df["days_since_last_injury"] = (
            df[config.DATE_FROM_COLUMN] - df["prev_injury_until"]
        ).dt.days
        df.drop(columns=["prev_injury_until"], inplace=True)

        # --- Rolling averages over time windows (days) ---
        # For each injury, compute mean of past injuries within window
        for window in config.ROLLING_WINDOWS_DAYS:
            col_name = f"rolling_avg_days_{window}d"
            df[col_name] = np.nan
            for player, grp in df.groupby(config.PLAYER_COLUMN):
                if len(grp) < 2:
                    continue
                idx = grp.index.tolist()
                dates = grp[config.DATE_FROM_COLUMN].values
                if config.TARGET_COLUMN in grp.columns:
                    vals = grp[config.TARGET_COLUMN].values
                else:
                    vals = np.array([np.nan] * len(grp))
                for i in range(1, len(idx)):
                    mask = (
                        (dates[:i] >= dates[i] - np.timedelta64(window, "D"))
                        & (dates[:i] < dates[i])
                    )
                    if mask.sum() > 0:
                        df.loc[idx[i], col_name] = vals[:i][mask].mean()
            self.feature_names.append(col_name)

        # --- Recent severity trend (slope of last 3 injuries) ---
        df["recent_severity_trend"] = np.nan
        for player, grp in df.groupby(config.PLAYER_COLUMN):
            if len(grp) < 4:
                continue
            idx = grp.index.tolist()
            if config.TARGET_COLUMN in grp.columns:
                vals = grp[config.TARGET_COLUMN].values
            else:
                vals = np.array([np.nan] * len(grp))
            for i in range(3, len(idx)):
                last3 = vals[i - 3 : i]
                x = np.arange(3, dtype=float)
                slope = np.polyfit(x, last3, 1)[0]
                df.loc[idx[i], "recent_severity_trend"] = slope

        # --- Injuries in last 12 months ---
        df["injuries_last_12_months"] = 0
        for player, grp in df.groupby(config.PLAYER_COLUMN):
            if len(grp) < 2:
                continue
            idx = grp.index.tolist()
            dates = grp[config.DATE_FROM_COLUMN].values
            for i in range(1, len(idx)):
                mask = (dates[:i] >= dates[i] - np.timedelta64(365, "D")) & (
                    dates[:i] < dates[i]
                )
                df.loc[idx[i], "injuries_last_12_months"] = int(mask.sum())

        # --- Average recovery time of last 3 injuries ---
        df["avg_recovery_time_recent"] = np.nan
        for player, grp in df.groupby(config.PLAYER_COLUMN):
            if len(grp) < 2:
                continue
            idx = grp.index.tolist()
            if config.TARGET_COLUMN in grp.columns:
                vals = grp[config.TARGET_COLUMN].values
            else:
                vals = np.array([np.nan] * len(grp))
            for i in range(1, len(idx)):
                start = max(0, i - 3)
                df.loc[idx[i], "avg_recovery_time_recent"] = vals[start:i].mean()

        # --- Chronic injury indicator (>3 same injury type historically) ---
        df["chronic_injury_indicator"] = 0
        for player, grp in df.groupby(config.PLAYER_COLUMN):
            if len(grp) < 2:
                continue
            idx = grp.index.tolist()
            injuries = grp["Injury"].values
            for i in range(1, len(idx)):
                current_injury = injuries[i]
                count = (injuries[:i] == current_injury).sum()
                df.loc[idx[i], "chronic_injury_indicator"] = int(count >= 3)

        # --- Re-injury risk score (composite) ---
        # Combines: high frequency (short gap) + chronic + recent severity trend
        df["reinjury_risk_score"] = 0.0
        has_gap = df["days_since_last_injury"].notna()
        df.loc[has_gap, "reinjury_risk_score"] += np.where(
            df.loc[has_gap, "days_since_last_injury"] < 30, 1.0, 0.0
        )
        df["reinjury_risk_score"] += df["chronic_injury_indicator"].astype(float)
        has_trend = df["recent_severity_trend"].notna()
        df.loc[has_trend, "reinjury_risk_score"] += np.where(
            df.loc[has_trend, "recent_severity_trend"] > 0, 0.5, 0.0
        )

        self.feature_names.extend(
            [
                "injury_sequence_number",
                "days_since_last_injury",
                "recent_severity_trend",
                "injuries_last_12_months",
                "avg_recovery_time_recent",
                "chronic_injury_indicator",
                "reinjury_risk_score",
            ]
        )

        logger.info("Player history features added: %d new columns", len(self.feature_names))
        return df


# =============================================================================
# Melhoria 3 — Temporal / Calendar Features
# =============================================================================
class TemporalFeatures:
    """
    Calendar and season-aware features.

    Features
    --------
    - season_phase: início / meio / reta_final
    - days_to_season_end: approximate days to end of season
    - congestion_index: injuries in club in last 30 days
    - seasonal_injury_rate: running rate within season
    - month_sin, month_cos: cyclical month encoding
    - is_high_density_period: Dec-Mar (Champions knockout, winter fixtures)
    - injury_month, injury_dow
    """

    def __init__(self) -> None:
        self.feature_names: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the DataFrame."""
        logger.info("Computing temporal features")
        df = df.copy()

        date_col = config.DATE_FROM_COLUMN

        # Basic temporal
        df["injury_month"] = df[date_col].dt.month
        df["injury_dow"] = df[date_col].dt.dayofweek  # 0=Mon

        # Cyclical month encoding
        df["month_sin"] = np.sin(2 * np.pi * df["injury_month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["injury_month"] / 12)

        # Season phase
        # European football season: Aug-May
        # Early (Aug-Oct), Mid (Nov-Jan), Late (Feb-May)
        def _season_phase(month: int) -> str:
            if month in [8, 9, 10]:
                return "early"
            elif month in [11, 12, 1]:
                return "mid"
            else:
                return "late"

        df["season_phase"] = df["injury_month"].apply(_season_phase)
        df["season_phase_code"] = df["season_phase"].map(
            {"early": 0, "mid": 1, "late": 2}
        )

        # Days to approximate season end (May 31)
        def _days_to_season_end(date: pd.Timestamp) -> int:
            year = date.year
            # Season ends ~May 31
            if date.month >= 8:
                end = pd.Timestamp(year + 1, 5, 31)
            else:
                end = pd.Timestamp(year, 5, 31)
            return max(0, (end - date).days)

        df["days_to_season_end"] = df[date_col].apply(_days_to_season_end)

        # High density period (Dec-Mar: Champions League knockout, winter fixtures)
        df["is_high_density_period"] = df["injury_month"].isin([12, 1, 2, 3]).astype(int)

        # Congestion index: number of injuries in the same club within last 30 days
        df["congestion_index"] = 0
        df = df.sort_values(date_col).reset_index(drop=True)
        for club, grp in df.groupby("club"):
            idx = grp.index.tolist()
            dates = grp[date_col].values
            for i in range(len(idx)):
                mask = (dates[:i] >= dates[i] - np.timedelta64(30, "D")) & (
                    dates[:i] < dates[i]
                )
                df.loc[idx[i], "congestion_index"] = int(mask.sum())

        # Re-sort by player + date
        df = df.sort_values(
            [config.PLAYER_COLUMN, config.DATE_FROM_COLUMN]
        ).reset_index(drop=True)

        # Seasonal injury rate (running count / days elapsed in season)
        df["seasonal_injury_rate"] = 0.0
        for (player, season), grp in df.groupby(
            [config.PLAYER_COLUMN, config.SEASON_COLUMN]
        ):
            idx = grp.index.tolist()
            dates = grp[date_col].values
            for i in range(len(idx)):
                count = i  # injuries before this one in same season
                if i > 0:
                    elapsed = (dates[i] - dates[0]) / np.timedelta64(1, "D")
                    if elapsed > 0:
                        df.loc[idx[i], "seasonal_injury_rate"] = count / elapsed

        self.feature_names = [
            "injury_month",
            "injury_dow",
            "month_sin",
            "month_cos",
            "season_phase_code",
            "days_to_season_end",
            "is_high_density_period",
            "congestion_index",
            "seasonal_injury_rate",
        ]
        logger.info("Temporal features added: %d columns", len(self.feature_names))
        return df


# =============================================================================
# Melhoria 4 — Statistical / Interaction Features
# =============================================================================
class StatisticalFeatures:
    """
    Compute rolling statistics over injury counts, interaction terms,
    and normalised metrics.

    Features
    --------
    - rolling_mean_days_N: mean duration over last N injuries
    - rolling_count_injuries_N: injury count in window
    - cumulative_days_injured: total days injured up to this point
    - interaction features: age*position_code, injury_type*league_code
    - normalised metrics per position / league
    - player_injury_rate_percentile
    """

    def __init__(self) -> None:
        self.feature_names: List[str] = []
        self._position_stats: Optional[Dict] = None
        self._league_stats: Optional[Dict] = None
        self._player_percentiles: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "StatisticalFeatures":
        """
        Compute normalisation statistics from **training** data only.

        Parameters
        ----------
        df : pd.DataFrame
            Training set.

        Returns
        -------
        self
        """
        logger.info("Fitting statistical feature normalisation on training data")

        # Position-level stats
        self._position_stats = (
            df.groupby("player_position")[config.TARGET_COLUMN]
            .agg(["mean", "std"])
            .to_dict(orient="index")
        )

        # League-level stats
        self._league_stats = (
            df.groupby("league")[config.TARGET_COLUMN]
            .agg(["mean", "std"])
            .to_dict(orient="index")
        )

        # Player-level injury rates (injuries / seasons)
        player_counts = df.groupby(config.PLAYER_COLUMN).size()
        self._player_percentiles = player_counts.rank(pct=True)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features. Must call ``fit()`` first.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset (train or test).

        Returns
        -------
        pd.DataFrame
        """
        logger.info("Computing statistical features")
        df = df.copy()
        df = df.sort_values(
            [config.PLAYER_COLUMN, config.DATE_FROM_COLUMN]
        ).reset_index(drop=True)

        # --- Rolling mean / count over last N injuries ---
        for n in config.ROLLING_WINDOWS_INJURIES:
            mean_col = f"rolling_mean_days_{n}"
            count_col = f"rolling_count_injuries_{n}"
            df[mean_col] = np.nan
            df[count_col] = 0

            for player, grp in df.groupby(config.PLAYER_COLUMN):
                idx = grp.index.tolist()
                if config.TARGET_COLUMN in grp.columns:
                    vals = grp[config.TARGET_COLUMN].values
                else:
                    vals = np.array([np.nan] * len(grp))
                for i in range(1, len(idx)):
                    start = max(0, i - n)
                    window_vals = vals[start:i]
                    df.loc[idx[i], mean_col] = window_vals.mean()
                    df.loc[idx[i], count_col] = len(window_vals)

            self.feature_names.extend([mean_col, count_col])

        # --- Cumulative days injured ---
        df["cumulative_days_injured"] = 0.0
        for player, grp in df.groupby(config.PLAYER_COLUMN):
            idx = grp.index.tolist()
            if config.TARGET_COLUMN in grp.columns:
                vals = grp[config.TARGET_COLUMN].values
            else:
                vals = np.array([np.nan] * len(grp))
            for i in range(1, len(idx)):
                df.loc[idx[i], "cumulative_days_injured"] = vals[:i].sum()
        self.feature_names.append("cumulative_days_injured")

        # --- Interaction features ---
        # Encode position and league to numeric for interactions
        pos_map = {p: i for i, p in enumerate(sorted(df["player_position"].unique()))}
        league_map = {l: i for i, l in enumerate(sorted(df["league"].unique()))}
        df["position_code"] = df["player_position"].map(pos_map).fillna(-1).astype(int)
        df["league_code"] = df["league"].map(league_map).fillna(-1).astype(int)

        df["age_x_position"] = df["player_age"] * df["position_code"]
        df["age_x_league"] = df["player_age"] * df["league_code"]
        self.feature_names.extend(["age_x_position", "age_x_league"])

        # --- Normalised by position ---
        if self._position_stats is not None:
            df["days_norm_position"] = df.apply(
                lambda r: self._normalise(
                    r.get("avg_recovery_time_recent", np.nan),
                    self._position_stats.get(r["player_position"]),
                ),
                axis=1,
            )
            self.feature_names.append("days_norm_position")

        # --- Normalised by league ---
        if self._league_stats is not None:
            df["days_norm_league"] = df.apply(
                lambda r: self._normalise(
                    r.get("avg_recovery_time_recent", np.nan),
                    self._league_stats.get(r["league"]),
                ),
                axis=1,
            )
            self.feature_names.append("days_norm_league")

        # --- Player injury rate percentile ---
        if self._player_percentiles is not None:
            df["player_injury_rate_percentile"] = (
                df[config.PLAYER_COLUMN]
                .map(self._player_percentiles)
                .fillna(0.5)
            )
            self.feature_names.append("player_injury_rate_percentile")

        logger.info("Statistical features added: %d columns", len(self.feature_names))
        return df

    @staticmethod
    def _normalise(value: float, stats: Optional[Dict]) -> float:
        """Z-score normalise a value using pre-computed mean/std."""
        if stats is None or pd.isna(value):
            return np.nan
        std = stats.get("std", 1.0)
        if std == 0 or pd.isna(std):
            std = 1.0
        return (value - stats.get("mean", 0)) / std


# =============================================================================
# Frequency Encoding  (leak-safe)
# =============================================================================
class SafeFrequencyEncoder:
    """
    Frequency encoding fitted **only** on training data to prevent leakage.
    Unknown categories at inference time receive 0.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns or config.CATEGORICAL_COLUMNS
        self.freq_maps: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> "SafeFrequencyEncoder":
        """Learn frequency maps from training data."""
        for col in self.columns:
            if col in df.columns:
                counts = df[col].value_counts(normalize=True)
                self.freq_maps[col] = counts.to_dict()
        logger.info(
            "Frequency encoder fitted on %d columns from %d rows",
            len(self.freq_maps),
            len(df),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency encoding (unknown categories → 0)."""
        df = df.copy()
        for col, fmap in self.freq_maps.items():
            new_col = f"{col}_freq"
            df[new_col] = df[col].map(fmap).fillna(0.0)
        return df


# =============================================================================
# Target Encoding (leak-safe, with smoothing)
# =============================================================================
class SafeTargetEncoder:
    """
    Target encoding fitted on training data with global-mean smoothing
    to prevent leakage.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        smoothing: float = 10.0,
    ) -> None:
        self.columns = columns or ["Injury", "player_position", "club", "league"]
        self.smoothing = smoothing
        self.global_mean: float = 0.0
        self.encoding_maps: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> "SafeTargetEncoder":
        """Learn target encoding from training data."""
        self.global_mean = df[config.TARGET_COLUMN].mean()
        for col in self.columns:
            if col not in df.columns:
                continue
            agg = df.groupby(col)[config.TARGET_COLUMN].agg(["mean", "count"])
            smoothed = (
                agg["count"] * agg["mean"] + self.smoothing * self.global_mean
            ) / (agg["count"] + self.smoothing)
            self.encoding_maps[col] = smoothed.to_dict()
        logger.info("Target encoder fitted (global mean=%.2f)", self.global_mean)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target encoding (unknowns → global mean)."""
        df = df.copy()
        for col, emap in self.encoding_maps.items():
            new_col = f"{col}_target_enc"
            df[new_col] = df[col].map(emap).fillna(self.global_mean)
        return df


# =============================================================================
# Master Feature Pipeline
# =============================================================================
class FeaturePipeline:
    """
    Orchestrates all feature engineering steps in the correct order,
    respecting temporal causality.
    """

    def __init__(self) -> None:
        self.player_hist = PlayerHistoryFeatures()
        self.temporal = TemporalFeatures()
        self.statistical = StatisticalFeatures()
        self.freq_encoder = SafeFrequencyEncoder()
        self.target_encoder = SafeTargetEncoder()
        self.validator = TemporalValidator()
        self.feature_names_: List[str] = []

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoders on training data and transform.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data (cleaned, sorted).

        Returns
        -------
        pd.DataFrame
            Training data with all features.
        """
        logger.info("=== Feature Pipeline: fit_transform (train) ===")

        # Ensure season ordinal is present (may be created upstream during data loading)
        if "season_ordinal" not in train_df.columns:
            train_df["season_ordinal"] = train_df[config.SEASON_COLUMN].map(
                config.SEASON_ORDER
            ).fillna(-1).astype(int)

        # 1. Player history (uses only past data by construction)
        train_df = self.player_hist.transform(train_df)

        # 2. Temporal features
        train_df = self.temporal.transform(train_df)

        # 3. Statistical features (fit normalisation stats on train)
        self.statistical.fit(train_df)
        train_df = self.statistical.transform(train_df)

        # 4. Frequency encoding (fit on train only)
        self.freq_encoder.fit(train_df)
        train_df = self.freq_encoder.transform(train_df)

        # 5. Target encoding (fit on train only)
        self.target_encoder.fit(train_df)
        train_df = self.target_encoder.transform(train_df)

        # 6. Age groups
        train_df["age_group"] = pd.cut(
            train_df["player_age"],
            bins=config.AGE_BINS,
            labels=config.AGE_LABELS,
            right=True,
        )
        train_df["age_group_code"] = train_df["age_group"].cat.codes

        # 7. Position group
        train_df["position_group"] = (
            train_df["player_position"].map(config.POSITION_GROUPS).fillna("Other")
        )
        pg_map = {g: i for i, g in enumerate(sorted(train_df["position_group"].unique()))}
        train_df["position_group_code"] = train_df["position_group"].map(pg_map)

        # Collect all numeric feature names
        self._collect_feature_names(train_df)

        # 8. Validate temporal causality
        history_features = [
            c
            for c in self.feature_names_
            if any(
                kw in c
                for kw in [
                    "rolling",
                    "days_since",
                    "severity_trend",
                    "injuries_last",
                    "avg_recovery",
                    "chronic",
                    "reinjury",
                    "cumulative",
                ]
            )
        ]
        self.validator.validate(train_df, history_features)

        logger.info("Feature pipeline complete: %d features", len(self.feature_names_))
        return train_df

    def transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test/new data using encoders fitted on training data.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test data (cleaned, sorted).

        Returns
        -------
        pd.DataFrame
            Test data with all features.
        """
        logger.info("=== Feature Pipeline: transform (test) ===")

        # Ensure season ordinal for inference
        if "season_ordinal" not in test_df.columns:
            test_df["season_ordinal"] = test_df[config.SEASON_COLUMN].map(
                config.SEASON_ORDER
            ).fillna(-1).astype(int)

        test_df = self.player_hist.transform(test_df)
        test_df = self.temporal.transform(test_df)
        test_df = self.statistical.transform(test_df)
        test_df = self.freq_encoder.transform(test_df)
        test_df = self.target_encoder.transform(test_df)

        test_df["age_group"] = pd.cut(
            test_df["player_age"],
            bins=config.AGE_BINS,
            labels=config.AGE_LABELS,
            right=True,
        )
        test_df["age_group_code"] = test_df["age_group"].cat.codes

        test_df["position_group"] = (
            test_df["player_position"].map(config.POSITION_GROUPS).fillna("Other")
        )
        pg_map = {g: i for i, g in enumerate(sorted(test_df["position_group"].unique()))}
        test_df["position_group_code"] = test_df["position_group"].map(pg_map)

        return test_df

    def get_feature_names(self) -> List[str]:
        """Return the list of numeric feature column names."""
        return self.feature_names_

    def _collect_feature_names(self, df: pd.DataFrame) -> None:
        """Identify all numeric feature columns (exclude target, dates, IDs)."""
        exclude = {
            config.TARGET_COLUMN,
            config.RAW_TARGET_COLUMN,
            config.DATE_FROM_COLUMN,
            config.DATE_UNTIL_COLUMN,
            config.PLAYER_COLUMN,
            "age_group",
            "position_group",
            "season_phase",
        }
        # Keep numeric + encoded columns
        self.feature_names_ = [
            c
            for c in df.columns
            if c not in exclude
            and (
                df[c].dtype in ["int64", "float64", "int32", "float32"]
                or c.endswith("_freq")
                or c.endswith("_target_enc")
            )
        ]

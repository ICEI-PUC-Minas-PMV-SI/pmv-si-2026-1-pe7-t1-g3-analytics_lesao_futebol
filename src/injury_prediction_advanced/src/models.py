"""
Models Module
=============
Implements Melhorias 5-6:
    5. Advanced Ensemble Methods (Stacking, Blending, Weighted Average)
    6. Optuna Hyperparameter Optimisation

All models use ``config.RANDOM_STATE`` for reproducibility and
``GroupKFold`` by ``player_name`` to avoid data leakage.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import config
from .utils import setup_logging

logger = setup_logging("models")

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
    logger.warning("CatBoost not installed — CatBoost models will be unavailable")

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
    logger.warning("XGBoost not installed — XGBoost models will be unavailable")

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
    logger.warning("LightGBM not installed — LightGBM models will be unavailable")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None
    logger.warning("Optuna not installed — hyperparameter optimisation unavailable")


# =============================================================================
# Base model factory
# =============================================================================
def get_base_models(random_state: int = config.RANDOM_STATE) -> Dict[str, Any]:
    """
    Return a dictionary of default-configured base models.

    Returns
    -------
    dict
        ``{name: estimator}`` mapping.
    """
    models: Dict[str, Any] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        ),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
    }
    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=0,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        )
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbosity=0,
        )
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
        )
    return models


# =============================================================================
# Melhoria 5 — Advanced Ensemble
# =============================================================================
class AdvancedEnsemble:
    """
    Provides three ensemble strategies:
        1. **Stacking** — uses a meta-learner on out-of-fold predictions.
        2. **Blending** — uses a hold-out blend set.
        3. **Weighted Average** — optimises per-model weights via scipy.

    All internal cross-validation uses ``GroupKFold`` on ``player_name``.
    """

    def __init__(
        self,
        base_models: Optional[Dict[str, Any]] = None,
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        self.base_models = base_models or get_base_models(random_state)
        self.random_state = random_state
        self._stacking_model: Optional[StackingRegressor] = None
        self._blending_models: Dict[str, Any] = {}
        self._blending_meta: Optional[Ridge] = None
        self._weights: Optional[np.ndarray] = None
        self._fitted_models: Dict[str, Any] = {}

    # ----- Stacking -----
    def fit_stacking(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        meta_learner: Optional[Any] = None,
    ) -> "AdvancedEnsemble":
        """
        Fit a stacking ensemble with GroupKFold.

        Parameters
        ----------
        X, y : array-like
            Features and target.
        groups : array-like
            Player names for GroupKFold.
        meta_learner : estimator, optional
            Meta-learner (default: Ridge).

        Returns
        -------
        self
        """
        logger.info("Fitting Stacking Ensemble with %d base models", len(self.base_models))
        if meta_learner is None:
            meta_learner = Ridge(alpha=1.0, random_state=self.random_state)

        estimators = [(name, clone(m)) for name, m in self.base_models.items()]
        cv = GroupKFold(n_splits=config.N_SPLITS_INNER)

        self._stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv.split(X, y, groups),
            n_jobs=-1,
        )
        self._stacking_model.fit(X, y)
        logger.info("Stacking ensemble fitted")
        return self

    def predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Predict using the stacking ensemble."""
        if self._stacking_model is None:
            raise RuntimeError("Call fit_stacking() first")
        return self._stacking_model.predict(X)

    # ----- Blending -----
    def fit_blending(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_blend: np.ndarray,
        y_blend: np.ndarray,
    ) -> "AdvancedEnsemble":
        """
        Fit a blending ensemble using a held-out blend set.

        Parameters
        ----------
        X_train, y_train : array-like
            Training data for base models.
        X_blend, y_blend : array-like
            Blend set for meta-learner.

        Returns
        -------
        self
        """
        logger.info("Fitting Blending Ensemble")
        blend_preds = np.zeros((len(X_blend), len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models.items()):
            m = clone(model)
            m.fit(X_train, y_train)
            self._blending_models[name] = m
            blend_preds[:, i] = m.predict(X_blend)
            logger.info("  Base model %s fitted", name)

        self._blending_meta = Ridge(alpha=1.0, random_state=self.random_state)
        self._blending_meta.fit(blend_preds, y_blend)
        logger.info("Blending meta-learner fitted")
        return self

    def predict_blending(self, X: np.ndarray) -> np.ndarray:
        """Predict using the blending ensemble."""
        if not self._blending_models or self._blending_meta is None:
            raise RuntimeError("Call fit_blending() first")
        preds = np.column_stack(
            [m.predict(X) for m in self._blending_models.values()]
        )
        return self._blending_meta.predict(preds)

    # ----- Weighted Average -----
    def fit_weighted_average(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> "AdvancedEnsemble":
        """
        Fit base models and optimise averaging weights via ``scipy.optimize``.

        Parameters
        ----------
        X, y : array-like
            Features and target.
        groups : array-like
            Player names for GroupKFold.

        Returns
        -------
        self
        """
        from scipy.optimize import minimize

        logger.info("Fitting Weighted Average Ensemble")
        cv = GroupKFold(n_splits=config.N_SPLITS_INNER)

        oof_preds: Dict[str, np.ndarray] = {}
        for name, model in self.base_models.items():
            m = clone(model)
            oof = cross_val_predict(m, X, y, cv=cv.split(X, y, groups))
            oof_preds[name] = oof
            # Refit on full data
            m_full = clone(model)
            m_full.fit(X, y)
            self._fitted_models[name] = m_full
            logger.info("  %s OOF MAE=%.3f", name, mean_absolute_error(y, oof))

        oof_matrix = np.column_stack(list(oof_preds.values()))
        n_models = oof_matrix.shape[1]

        def _objective(w: np.ndarray) -> float:
            w = np.abs(w) / np.abs(w).sum()
            combined = oof_matrix @ w
            return mean_absolute_error(y, combined)

        w0 = np.ones(n_models) / n_models
        result = minimize(_objective, w0, method="Nelder-Mead")
        self._weights = np.abs(result.x) / np.abs(result.x).sum()

        names = list(self.base_models.keys())
        for n, w in zip(names, self._weights):
            logger.info("  Weight %s: %.4f", n, w)

        return self

    def predict_weighted_average(self, X: np.ndarray) -> np.ndarray:
        """Predict using the weighted average ensemble."""
        if not self._fitted_models or self._weights is None:
            raise RuntimeError("Call fit_weighted_average() first")
        preds = np.column_stack(
            [m.predict(X) for m in self._fitted_models.values()]
        )
        return preds @ self._weights

    # ----- Comparison -----
    def compare_ensemble_strategies(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        groups_train: np.ndarray,
    ) -> pd.DataFrame:
        """
        Train and compare all three ensemble strategies.

        Returns
        -------
        pd.DataFrame
            Comparison table with MAE, RMSE, R² for each strategy.
        """
        results = []

        # Stacking
        self.fit_stacking(X_train, y_train, groups_train)
        preds = self.predict_stacking(X_test)
        results.append(self._eval("Stacking", y_test, preds))

        # Blending (use last 20% of train as blend)
        split_idx = int(len(X_train) * 0.8)
        self.fit_blending(
            X_train[:split_idx],
            y_train[:split_idx],
            X_train[split_idx:],
            y_train[split_idx:],
        )
        preds = self.predict_blending(X_test)
        results.append(self._eval("Blending", y_test, preds))

        # Weighted
        self.fit_weighted_average(X_train, y_train, groups_train)
        preds = self.predict_weighted_average(X_test)
        results.append(self._eval("WeightedAverage", y_test, preds))

        return pd.DataFrame(results)

    @staticmethod
    def _eval(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        return {
            "strategy": name,
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
        }


# =============================================================================
# Melhoria 6 — Optuna Optimizer
# =============================================================================
class OptunaOptimizer:
    """
    Bayesian hyperparameter optimisation for each model family.

    Uses ``GroupKFold`` internally and supports pruning + early stopping.
    """

    def __init__(
        self,
        n_trials: int = config.OPTUNA_N_TRIALS,
        timeout: int = config.OPTUNA_TIMEOUT,
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        if optuna is None:
            raise ImportError("Optuna is required for hyperparameter optimisation")
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.best_params: Dict[str, Dict] = {}
        self.studies: Dict[str, Any] = {}

    # ---- CatBoost ----
    def optimize_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> Dict:
        """Optimise CatBoost hyperparameters."""
        if CatBoostRegressor is None:
            raise ImportError("CatBoost not installed")

        def _objective(trial: "optuna.Trial") -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 200, 2000),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "random_seed": self.random_state,
                "verbose": 0,
                "early_stopping_rounds": config.EARLY_STOPPING_ROUNDS,
            }
            return self._cv_score(CatBoostRegressor, params, X, y, groups)

        return self._run_study("CatBoost", _objective)

    # ---- XGBoost ----
    def optimize_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> Dict:
        """Optimise XGBoost hyperparameters."""
        if XGBRegressor is None:
            raise ImportError("XGBoost not installed")

        def _objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": self.random_state,
                "verbosity": 0,
                "early_stopping_rounds": config.EARLY_STOPPING_ROUNDS,
            }
            return self._cv_score(XGBRegressor, params, X, y, groups)

        return self._run_study("XGBoost", _objective)

    # ---- LightGBM ----
    def optimize_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> Dict:
        """Optimise LightGBM hyperparameters."""
        if LGBMRegressor is None:
            raise ImportError("LightGBM not installed")

        def _objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "random_state": self.random_state,
                "verbose": -1,
            }
            return self._cv_score(LGBMRegressor, params, X, y, groups)

        return self._run_study("LightGBM", _objective)

    # ---- Random Forest ----
    def optimize_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> Dict:
        """Optimise Random Forest hyperparameters."""

        def _objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]
                ),
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            return self._cv_score(RandomForestRegressor, params, X, y, groups)

        return self._run_study("RandomForest", _objective)

    # ---- Internals ----
    def _cv_score(
        self,
        model_class: type,
        params: Dict,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> float:
        """Compute mean MAE across GroupKFold splits."""
        cv = GroupKFold(n_splits=config.N_SPLITS_INNER)
        scores = []
        for train_idx, val_idx in cv.split(X, y, groups):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = model_class(**params)

            # Models with eval_set for early stopping
            fit_params: Dict[str, Any] = {}
            if hasattr(model, "early_stopping_rounds") or "early_stopping_rounds" in params:
                if model_class in (XGBRegressor,) and XGBRegressor is not None:
                    fit_params["eval_set"] = [(X_val, y_val)]
                    fit_params["verbose"] = False
                elif model_class == CatBoostRegressor and CatBoostRegressor is not None:
                    fit_params["eval_set"] = (X_val, y_val)
                elif model_class == LGBMRegressor and LGBMRegressor is not None:
                    fit_params["eval_set"] = [(X_val, y_val)]
                    fit_params["callbacks"] = [
                        __import__("lightgbm").early_stopping(
                            config.EARLY_STOPPING_ROUNDS, verbose=False
                        )
                    ]

            model.fit(X_tr, y_tr, **fit_params)
            preds = model.predict(X_val)
            scores.append(mean_absolute_error(y_val, preds))

        return float(np.mean(scores))

    def _run_study(self, name: str, objective: Any) -> Dict:
        """Execute an Optuna study."""
        logger.info("Optimising %s (%d trials, %ds timeout)", name, self.n_trials, self.timeout)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        self.studies[name] = study
        self.best_params[name] = study.best_params
        logger.info(
            "%s best MAE=%.4f, params=%s", name, study.best_value, study.best_params
        )
        return study.best_params

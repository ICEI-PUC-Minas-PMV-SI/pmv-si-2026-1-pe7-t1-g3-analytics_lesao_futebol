"""
Injury Prediction Advanced - Modular Pipeline
===============================================
Professional modular framework for predicting football injury duration
using advanced ML techniques for a Master's thesis.

Modules:
    - config: Global configurations, seeds, constants
    - data_processing: Data loading, cleaning, temporal splitting
    - feature_engineering: All feature creation (Improvements 1-4)
    - models: Ensemble models and Optuna optimization (Improvements 5-6)
    - evaluation: Metrics, nested CV, statistical analysis (Improvements 9,11,13,18)
    - explainability: SHAP-based interpretability (Improvement 7)
    - calibration: Uncertainty estimation (Improvement 8)
    - quantile_models: Quantile regression (Improvement 10)
    - visualization: Professional plots (Improvement 17)
    - experiment_tracking: MLflow integration (Improvement 15)
    - production: Production pipeline (Improvement 19)
    - utils: Helper functions
"""

__version__ = "1.0.0"
__author__ = "Injury Prediction Research"

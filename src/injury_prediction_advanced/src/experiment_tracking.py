"""
Experiment Tracking Module
==========================
Implements Melhoria 15: MLflow-based experiment tracking.

Logs parameters, metrics, models, figures, and artifacts for
reproducible experiment management.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .utils import setup_logging

logger = setup_logging("experiment_tracking")

try:
    import mlflow
    import mlflow.sklearn

    _HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _HAS_MLFLOW = False
    logger.warning("MLflow not installed — experiment tracking unavailable")


class MLflowTracker:
    """
    Convenience wrapper for MLflow experiment tracking.
    """

    def __init__(
        self,
        experiment_name: str = "injury_prediction",
        tracking_uri: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        experiment_name : str
            Name of the MLflow experiment.
        tracking_uri : str, optional
            MLflow tracking URI. Defaults to local file store in ``mlruns/``.
        """
        if not _HAS_MLFLOW:
            raise ImportError("MLflow is required: pip install mlflow")

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or str(config.MLRUNS_DIR)
        self._run_id: Optional[str] = None

    def setup_experiment(self) -> str:
        """
        Set the tracking URI and create/get the experiment.

        Returns
        -------
        str
            Experiment ID.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        logger.info(
            "MLflow experiment '%s' (ID=%s) at %s",
            self.experiment_name,
            exp.experiment_id if exp else "NEW",
            self.tracking_uri,
        )
        return exp.experiment_id if exp else ""

    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Start a new MLflow run.

        Returns
        -------
        str
            Run ID.
        """
        self.setup_experiment()
        run = mlflow.start_run(run_name=run_name)
        self._run_id = run.info.run_id
        logger.info("MLflow run started: %s", self._run_id)
        return self._run_id

    def end_run(self) -> None:
        """End the current run."""
        mlflow.end_run()
        logger.info("MLflow run ended: %s", self._run_id)
        self._run_id = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dictionary of parameters."""
        mlflow.log_params({k: str(v) for k, v in params.items()})
        logger.info("Logged %d params", len(params))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics."""
        mlflow.log_metrics(metrics, step=step)
        logger.info("Logged %d metrics", len(metrics))

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log a scikit-learn compatible model."""
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info("Model logged to %s", artifact_path)

    def log_artifacts(self, local_dir: str) -> None:
        """Log all files in a local directory as artifacts."""
        mlflow.log_artifacts(local_dir)
        logger.info("Artifacts logged from %s", local_dir)

    def log_figure(
        self, fig: Any, filename: str = "figure.png"
    ) -> None:
        """
        Log a matplotlib figure as an artifact.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, filename)
            fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
            mlflow.log_artifact(path)
        logger.info("Figure logged: %s", filename)

    def log_dataframe(self, df: pd.DataFrame, filename: str = "results.csv") -> None:
        """Log a DataFrame as a CSV artifact."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, filename)
            df.to_csv(path, index=False)
            mlflow.log_artifact(path)
        logger.info("DataFrame logged: %s", filename)

    def __enter__(self) -> "MLflowTracker":
        self.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_run()

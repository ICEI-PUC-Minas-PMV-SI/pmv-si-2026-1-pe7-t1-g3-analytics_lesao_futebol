"""
Utilities Module
================
Helper functions for logging, seeding, I/O, and directory management.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import config


def setup_logging(
    name: str = "injury_prediction",
    log_file: Optional[str] = None,
    level: int = config.LOG_LEVEL,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str, optional
        Path to log file. If None, logs to ``config.LOGS_DIR/<name>.log``.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file is None:
        create_directories()
        log_file = str(config.LOGS_DIR / f"{name}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def set_global_seed(seed: int = config.RANDOM_STATE) -> None:
    """
    Set random seeds for full reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger = logging.getLogger("injury_prediction")
    logger.info("Global seed set to %d", seed)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_directories() -> None:
    """Create all required project directories."""
    for d in [
        config.MODELS_DIR,
        config.FIGURES_DIR,
        config.LOGS_DIR,
        config.MLRUNS_DIR,
        config.NOTEBOOKS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def save_artifact(obj: Any, path: str) -> None:
    """
    Persist an object using joblib.

    Parameters
    ----------
    obj : Any
        Object to save (model, pipeline, etc.).
    path : str
        Destination path.
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger = logging.getLogger("injury_prediction")
    logger.info("Artifact saved to %s", path)


def load_artifact(path: str) -> Any:
    """Load a joblib-persisted object."""
    import joblib
    return joblib.load(path)

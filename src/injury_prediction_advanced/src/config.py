"""
Configuration Module
====================
Global configurations, seeds, constants, and logging setup for the
injury prediction project.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "full_dataset_thesis - 2.csv"
MODELS_DIR: Path = PROJECT_ROOT / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_STATE: int = 42

# =============================================================================
# Dataset Columns
# =============================================================================
TARGET_COLUMN: str = "days_num"
RAW_TARGET_COLUMN: str = "Days"
DATE_FROM_COLUMN: str = "injury_from_parsed"
DATE_UNTIL_COLUMN: str = "injury_until_parsed"
PLAYER_COLUMN: str = "player_name"
SEASON_COLUMN: str = "Season"

CATEGORICAL_COLUMNS: List[str] = [
    "Injury",
    "player_position",
    "club",
    "league",
    "Season",
]

# Column mapping for season ordering
SEASON_ORDER: Dict[str, int] = {
    "20/21": 0,
    "21/22": 1,
    "22/23": 2,
    "23/24": 3,
    "24/25": 4,
}

# =============================================================================
# Feature Engineering Windows
# =============================================================================
ROLLING_WINDOWS_DAYS: List[int] = [7, 30, 90, 365]
ROLLING_WINDOWS_INJURIES: List[int] = [3, 5, 10]
QUANTILE_LIST: List[float] = [0.1, 0.5, 0.9]

# =============================================================================
# Model Training
# =============================================================================
N_SPLITS_OUTER: int = 5
N_SPLITS_INNER: int = 3
OPTUNA_N_TRIALS: int = 30
OPTUNA_TIMEOUT: int = 300  # seconds
EARLY_STOPPING_ROUNDS: int = 50
BOOTSTRAP_N_ITERATIONS: int = 1000
CONFIDENCE_LEVEL: float = 0.95

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: int = logging.INFO

# =============================================================================
# Visualization
# =============================================================================
FIGURE_DPI: int = 300
FIGURE_FORMAT: str = "png"
PALETTE: str = "viridis"
FONT_SCALE: float = 1.2

# =============================================================================
# Position Groups
# =============================================================================
POSITION_GROUPS: Dict[str, str] = {
    "Goalkeeper": "Goalkeeper",
    "Centre-Back": "Defender",
    "Left-Back": "Defender",
    "Right-Back": "Defender",
    "Defensive Midfield": "Midfielder",
    "Central Midfield": "Midfielder",
    "Attacking Midfield": "Midfielder",
    "Right Midfield": "Midfielder",
    "Left Midfield": "Midfielder",
    "Midfielder": "Midfielder",
    "Left Winger": "Forward",
    "Right Winger": "Forward",
    "Second Striker": "Forward",
    "Forward": "Forward",
}

# =============================================================================
# Age Groups
# =============================================================================
AGE_BINS: List[int] = [0, 21, 25, 29, 33, 50]
AGE_LABELS: List[str] = ["U21", "22-25", "26-29", "30-33", "34+"]

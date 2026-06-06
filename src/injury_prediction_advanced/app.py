import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request

from src import config
from src.production import ProductionPipeline
from src.utils import setup_logging
import json

logger = setup_logging("app")

ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = ROOT_DIR / "models" / "production_pipeline.joblib"

app = Flask(__name__, template_folder="templates")


def load_pipeline() -> ProductionPipeline:
    return ProductionPipeline.load_pipeline(str(PIPELINE_PATH))


pipeline = load_pipeline()


REQUIRED_INPUT_COLUMNS = [
    "Injury",
    "player_age",
    "player_position",
    "league",
    "injury_from_parsed",
]

OPTIONAL_INPUT_COLUMNS = [
    "club",
    "injury_until_parsed",
    "player_name",
    "Season",
]

DEFAULT_PLAYER_NAME = "Unknown Player"
DEFAULT_CLUB = "Unknown Club"


def infer_season_from_date(date: pd.Timestamp) -> str:
    if pd.isna(date):
        return list(sorted(config.SEASON_ORDER.items(), key=lambda x: x[1]))[-1][0]

    year = date.year
    month = date.month
    if month >= 8:
        start_year = year
    else:
        start_year = year - 1

    season = f"{start_year % 100:02d}/{(start_year + 1) % 100:02d}"
    if season in config.SEASON_ORDER:
        return season

    known_seasons = sorted(config.SEASON_ORDER.items(), key=lambda x: x[1])
    known_years = [2000 + int(season.split("/")[0]) for season, _ in known_seasons]
    if start_year <= known_years[0]:
        return known_seasons[0][0]
    if start_year >= known_years[-1]:
        return known_seasons[-1][0]

    closest = min(known_seasons, key=lambda item: abs((2000 + int(item[0].split("/")[0])) - start_year))
    return closest[0]


def normalize_season(season_value: Any, date: pd.Timestamp) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    inferred = infer_season_from_date(date)
    season = str(season_value).strip() if season_value is not None else ""

    if not season:
        warnings.append(
            f"Season ausente ou inválida: inferindo temporada {inferred} a partir de injury_from_parsed."
        )
        return inferred, warnings

    if season in config.SEASON_ORDER:
        if season != inferred:
            warnings.append(
                f"Temporada '{season}' não combina com a data de início {date.date()}: usando {inferred} como fallback."
            )
            return inferred, warnings
        return season, warnings

    warnings.append(
        f"Temporada '{season}' desconhecida: inferindo temporada {inferred} a partir de injury_from_parsed."
    )
    return inferred, warnings


def parse_request_data(data: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    row: Dict[str, Any] = {}
    warnings: List[str] = []

    for col in REQUIRED_INPUT_COLUMNS:
        if col not in data or str(data[col]).strip() == "":
            raise ValueError(f"Campo obrigatório ausente ou vazio: {col}")
        row[col] = data[col]

    row["player_name"] = data.get("player_name") or DEFAULT_PLAYER_NAME
    if row["player_name"] == DEFAULT_PLAYER_NAME:
        warnings.append("player_name ausente: usando nome padrão Unknown Player.")

    row["club"] = data.get("club") or DEFAULT_CLUB
    if row["club"] == DEFAULT_CLUB:
        warnings.append("club ausente: usando valor padrão Unknown Club.")

    if data.get("injury_until_parsed") and str(data["injury_until_parsed"]).strip():
        row["injury_until_parsed"] = data["injury_until_parsed"]
    else:
        row["injury_until_parsed"] = data["injury_from_parsed"]
        warnings.append(
            "injury_until_parsed ausente: usando injury_from_parsed como fallback."
        )

    row["Season"] = data.get("Season", "")

    df = pd.DataFrame([row])
    df["player_age"] = pd.to_numeric(df["player_age"], errors="raise").astype(float)
    df["injury_from_parsed"] = pd.to_datetime(df["injury_from_parsed"], errors="raise", dayfirst=False)
    df["injury_until_parsed"] = pd.to_datetime(df["injury_until_parsed"], errors="raise", dayfirst=False)

    if df.loc[0, "injury_until_parsed"] < df.loc[0, "injury_from_parsed"]:
        df.loc[0, "injury_until_parsed"] = df.loc[0, "injury_from_parsed"]
        warnings.append(
            "injury_until_parsed era anterior a injury_from_parsed: ajustado para a mesma data."
        )

    normalized_season, season_warnings = normalize_season(
        df.loc[0, "Season"], df.loc[0, "injury_from_parsed"]
    )
    df["Season"] = normalized_season
    warnings.extend(season_warnings)

    return df, warnings

def get_labels():
    df = pd.read_csv(ROOT_DIR / "data" / "full_dataset_thesis - 2.csv",
                     usecols=["Injury", "club", "Season", "player_position"])
    injury_options = sorted(df["Injury"].dropna().astype(str).unique())
    club_options = sorted(df["club"].dropna().astype(str).unique())
    season_options = sorted(df["Season"].dropna().astype(str).unique())
    position_options = sorted(df["player_position"].dropna().astype(str).unique())

    DICT_PATH = ROOT_DIR / "templates" / "dicionario.json"
    with open(DICT_PATH, encoding="utf-8") as f:
        TRANSLATIONS = json.load(f)

    INJURY_LABELS = TRANSLATIONS["INJURY_LABELS"]
    POSITION_LABELS = TRANSLATIONS["POSITION_LABELS"]
    BLOCKED_INJURIES = {"Achilles heel problems", "Ankle problems", "Bruised back", "Bruise on ankle", "Corona virus", "Cancer", "Dental surgery", "Fever", 
                        "Fracture of the eye socket", "Injury to the ankle", "Knee problems", "Lower leg fracture", "Lymphatic cancer", "Meniscus damage",
                        "Muscle injury", "Patellar tendon rupture", "Quarantine", "Rest", "Tendon rupture", "bronchitis", "Ill", "traffic accident",
                        "chickenpox", "cold", "depression", "flu", "influenza", "malaria", "muscular problems", "pneumonia"}

    injury_options = [
        {"value": value, "label": INJURY_LABELS.get(value, value)}
        for value in sorted(df["Injury"].dropna().astype(str).unique())
        if value not in BLOCKED_INJURIES
    ]

    position_options = [
        {"value": value, "label": POSITION_LABELS.get(value, value)}
        for value in sorted(df["player_position"].dropna().astype(str).unique())
    ]

    return render_template("index.html", injury_options=injury_options, club_options=club_options, season_options=season_options, position_options=position_options)

@app.route("/", methods=["GET"])
def index():
    return get_labels()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json() if request.is_json else request.form.to_dict()
        df, parse_warnings = parse_request_data(payload)
        if parse_warnings:
            for warning in parse_warnings:
                logger.warning(warning)

        predictions, meta = pipeline.predict(df, return_metadata=True)
        pipeline_warnings = meta.get("warnings", [])
        if pipeline_warnings:
            for warning in pipeline_warnings:
                logger.warning(warning)

        result = {
            "prediction_days": float(predictions[0]),
            "prediction_stats": meta.get("prediction_stats", {}),
        }
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 400


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)

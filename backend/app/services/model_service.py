from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schemas import AnalysisResponse, VehicleInput
from app.services.damage_nlp import parse_damage_description


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "carsight_model.joblib"
META_PATH = ROOT / "models" / "carsight_model_metadata.json"


class ModelNotReadyError(RuntimeError):
    pass


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise ModelNotReadyError(
            "Model artifacts were not found. Run backend/train_model.py before starting the API."
        )
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, metadata


def _build_feature_frame(payload: VehicleInput) -> pd.DataFrame:
    engine = payload.engine.strip()
    liters_match = re.search(r"(\d+(?:\.\d+)?)L", engine)
    horsepower_match = re.search(r"(\d{2,4})\s*HP", engine, flags=re.IGNORECASE)
    engine_liters = float(liters_match.group(1)) if liters_match else np.nan
    horsepower = float(horsepower_match.group(1)) if horsepower_match else np.nan
    vehicle_age = 2026 - payload.year
    mileage_per_year = payload.mileage / max(vehicle_age, 1)
    horsepower_per_liter = horsepower / max(engine_liters, 1.0) if not np.isnan(horsepower) and not np.isnan(engine_liters) else np.nan
    make = payload.make.strip()
    model = payload.model.strip()
    transmission = (payload.transmission or "Unknown").strip()
    drivetrain = (payload.drivetrain or "Unknown").strip()
    fuel_type = (payload.fuel_type or "Unknown").strip()
    color = (payload.color or "Unknown").strip()
    spec_text = " ".join(
        part for part in [make, model, engine, transmission, drivetrain, fuel_type] if part
    )

    return pd.DataFrame(
        [
            {
                "manufacturer": make,
                "model": model,
                "year": payload.year,
                "mileage": payload.mileage,
                "engine": engine,
                "transmission": transmission,
                "drivetrain": drivetrain,
                "fuel_type": fuel_type,
                "exterior_color": color,
                "interior_color": "Unknown",
                "seller_rating": np.nan,
                "driver_rating": np.nan,
                "driver_reviews_num": np.nan,
                "price_drop": 0.0,
                "one_owner": 0.0,
                "personal_use_only": 0.0,
                "engine_liters": engine_liters,
                "horsepower": horsepower,
                "vehicle_age": vehicle_age,
                "mileage_per_year": mileage_per_year,
                "horsepower_per_liter": horsepower_per_liter,
                "engine_text": engine,
                "spec_text": spec_text,
            }
        ]
    )


def _comparison_text(asking_price: float | None, low: float, high: float) -> str | None:
    if asking_price is None:
        return None
    if asking_price > high:
        pct = ((asking_price - high) / high) * 100
        return f"The asking price is about {pct:.1f}% above the top of the estimated range."
    if asking_price < low:
        pct = ((low - asking_price) / low) * 100
        return f"The asking price is about {pct:.1f}% below the bottom of the estimated range."
    return "The asking price falls inside the estimated range."


def _confidence_label(confidence_score: float) -> str:
    if confidence_score >= 0.8:
        return "High"
    if confidence_score >= 0.62:
        return "Medium"
    return "Low"


def _predict_market_value(model: Any, frame: pd.DataFrame) -> float:
    if isinstance(model, dict) and model.get("artifact_type") == "weighted_blend":
        ridge_weight = _safe_float(model.get("ridge_weight"), 0.6)
        ridge_model = model["ridge_model"]
        tree_model = model["tree_model"]
        ridge_prediction = float(np.expm1(ridge_model.predict(frame)[0]))
        tree_prediction = float(np.expm1(tree_model.predict(frame)[0]))
        return (ridge_weight * ridge_prediction) + ((1 - ridge_weight) * tree_prediction)
    return float(np.expm1(model.predict(frame)[0]))


def _calibrated_half_width(
    metadata: dict[str, Any],
    confidence_score: float,
    damage_summary: Any,
) -> float:
    if confidence_score >= 0.8:
        half_width = _safe_float(metadata.get("abs_error_q55"), 2200.0)
    elif confidence_score >= 0.62:
        half_width = _safe_float(metadata.get("abs_error_q60"), 2600.0)
    else:
        half_width = _safe_float(metadata.get("abs_error_q70"), 3200.0)

    half_width += 175 * max(0, len(damage_summary.issues) - 1)
    if any(issue.severity == "high" for issue in damage_summary.issues):
        half_width += 300
    return min(max(half_width, 1800.0), _safe_float(metadata.get("abs_error_q80"), 4200.0) + 450)


def analyze_vehicle(payload: VehicleInput) -> AnalysisResponse:
    model, metadata = load_artifacts()
    frame = _build_feature_frame(payload)
    raw_prediction = _predict_market_value(model, frame)

    damage_summary = parse_damage_description(
        payload.damage_description,
        reference_price=raw_prediction,
        baseline_damage_discount=_safe_float(metadata.get("damage_discount_ratio"), 0.05),
    )

    adjusted_prediction = max(500.0, raw_prediction * (1 - damage_summary.total_penalty_pct))
    known_makes = set(metadata.get("known_makes", []))
    confidence_score = 0.86
    if payload.make.strip() not in known_makes:
        confidence_score -= 0.16
    if len(damage_summary.issues) > 2:
        confidence_score -= 0.08
    if any(issue.severity == "high" for issue in damage_summary.issues):
        confidence_score -= 0.12
    confidence_score = round(max(0.35, min(0.92, confidence_score)), 2)
    confidence = _confidence_label(confidence_score)

    range_padding = _calibrated_half_width(metadata, confidence_score=confidence_score, damage_summary=damage_summary)

    low = max(0.0, adjusted_prediction - range_padding)
    high = adjusted_prediction + range_padding

    explanation_points = [
        f"The base value comes from a hybrid model that blends TF-IDF NLP over vehicle specification text with a structured gradient boosting model.",
        f"The damage text triggered a {damage_summary.total_penalty_pct * 100:.1f}% adjustment after the app identified {len(damage_summary.issues)} issue(s).",
        metadata.get("selection_note", damage_summary.confidence_hint),
    ]

    if damage_summary.issues:
        top_issue = max(damage_summary.issues, key=lambda issue: issue.estimated_impact)
        negotiation_tip = (
            f"I would negotiate from the lower half of the range. The biggest discount driver is {top_issue.category} risk "
            f"('{top_issue.issue}'), which accounts for about ${top_issue.estimated_impact:,.0f} of impact."
        )
    else:
        negotiation_tip = "Without a clear damage issue, I would negotiate around maintenance history, ownership history, and mileage."

    return AnalysisResponse(
        input_summary={
            "make": payload.make,
            "model": payload.model,
            "year": payload.year,
            "mileage": payload.mileage,
            "engine": payload.engine,
            "fuel_type": payload.fuel_type,
            "transmission": payload.transmission,
            "drivetrain": payload.drivetrain,
            "color": payload.color,
            "asking_price": payload.asking_price,
            "damage_description": payload.damage_description,
        },
        base_prediction=round(raw_prediction, 2),
        adjusted_prediction=round(adjusted_prediction, 2),
        price_range={"low": round(low, 2), "high": round(high, 2)},
        confidence=confidence,
        confidence_score=confidence_score,
        asking_price_comparison=_comparison_text(payload.asking_price, low, high),
        extracted_damage=damage_summary,
        explanation_points=explanation_points,
        negotiation_tip=negotiation_tip,
        model_metrics={
            "dataset": metadata.get("dataset_name", "Kaggle used cars dataset"),
            "training_rows": int(metadata.get("training_rows", 0)),
            "mae": round(_safe_float(metadata.get("mae"), 0.0), 2),
            "rmse": round(_safe_float(metadata.get("rmse"), 0.0), 2),
            "mape": round(_safe_float(metadata.get("mape"), 0.0), 4),
            "model_name": metadata.get("model_name", "Hybrid blend"),
            "selected_model": metadata.get("selected_model", "hybrid_blend"),
            "methodology": metadata.get("methodology", ""),
            "selection_note": metadata.get("selection_note", ""),
            "blend_weight": round(_safe_float(metadata.get("blend_weight"), 0.6), 2),
        },
    )

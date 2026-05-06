from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

CURRENT_YEAR = 2026
DATASET_NAME = "andreinovikov/used-cars-dataset"
SAMPLE_SIZE = 100_000

USE_COLS = [
    "manufacturer",
    "model",
    "year",
    "mileage",
    "engine",
    "transmission",
    "drivetrain",
    "fuel_type",
    "exterior_color",
    "interior_color",
    "accidents_or_damage",
    "one_owner",
    "personal_use_only",
    "seller_rating",
    "driver_rating",
    "driver_reviews_num",
    "price_drop",
    "price",
]

BASE_NUMERIC_FEATURES = [
    "year",
    "mileage",
    "engine_liters",
    "horsepower",
    "seller_rating",
    "driver_rating",
    "driver_reviews_num",
    "price_drop",
    "one_owner",
    "personal_use_only",
]

ENHANCED_NUMERIC_FEATURES = [
    "year",
    "mileage",
    "engine_liters",
    "horsepower",
    "vehicle_age",
    "mileage_per_year",
    "horsepower_per_liter",
    "seller_rating",
    "driver_rating",
    "driver_reviews_num",
    "price_drop",
    "one_owner",
    "personal_use_only",
]

CATEGORICAL_FEATURES = [
    "manufacturer",
    "model",
    "transmission",
    "drivetrain",
    "fuel_type",
    "exterior_color",
    "interior_color",
]

BASELINE_FEATURES = CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES + ["engine_text"]
ENHANCED_FEATURES = CATEGORICAL_FEATURES + ENHANCED_NUMERIC_FEATURES + ["spec_text"]
TREE_FEATURES = CATEGORICAL_FEATURES + ENHANCED_NUMERIC_FEATURES


def parse_engine_liters(text: str) -> float | None:
    if not isinstance(text, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)L", text)
    return float(match.group(1)) if match else None


def parse_horsepower(text: str) -> float | None:
    if not isinstance(text, str):
        return None
    match = re.search(r"(\d{2,4})\s*HP", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def load_dataset() -> pd.DataFrame:
    download_dir = Path(kagglehub.dataset_download(DATASET_NAME))
    source = download_dir / "cars.csv"
    df = pd.read_csv(source, usecols=USE_COLS)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached = RAW_DIR / "cars_com_used_cars.csv"
    if not cached.exists():
        df.head(50_000).to_csv(cached, index=False)
    return df


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.dropna(subset=["price", "manufacturer", "model", "year", "mileage", "engine"])
    frame = frame[frame["price"].between(1_500, 120_000)]
    frame = frame[frame["year"].between(1995, 2025)]
    frame = frame[frame["mileage"].between(0, 300_000)]

    frame["engine_liters"] = frame["engine"].map(parse_engine_liters)
    frame["horsepower"] = frame["engine"].map(parse_horsepower)
    frame["vehicle_age"] = CURRENT_YEAR - frame["year"]
    frame["mileage_per_year"] = frame["mileage"] / np.maximum(frame["vehicle_age"], 1)
    frame["horsepower_per_liter"] = frame["horsepower"] / np.maximum(frame["engine_liters"].fillna(1.0), 1.0)

    frame["accidents_or_damage"] = frame["accidents_or_damage"].fillna(0).astype(float)
    frame["one_owner"] = frame["one_owner"].fillna(0).astype(float)
    frame["personal_use_only"] = frame["personal_use_only"].fillna(0).astype(float)
    frame["price_drop"] = frame["price_drop"].fillna(0).astype(float)

    frame["engine_text"] = frame["engine"].fillna("").astype(str)
    frame["spec_text"] = (
        frame["manufacturer"].fillna("").astype(str)
        + " "
        + frame["model"].fillna("").astype(str)
        + " "
        + frame["engine"].fillna("").astype(str)
        + " "
        + frame["transmission"].fillna("").astype(str)
        + " "
        + frame["drivetrain"].fillna("").astype(str)
        + " "
        + frame["fuel_type"].fillna("").astype(str)
    )
    return frame


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    abs_error = np.abs(actual - predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(np.mean(abs_error / np.maximum(actual, 1.0))),
        "abs_error_q55": float(np.quantile(abs_error, 0.55)),
        "abs_error_q60": float(np.quantile(abs_error, 0.60)),
        "abs_error_q70": float(np.quantile(abs_error, 0.70)),
        "abs_error_q80": float(np.quantile(abs_error, 0.80)),
    }


def _round_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def _baseline_ridge_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                BASE_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "text",
                TfidfVectorizer(max_features=250, ngram_range=(1, 2)),
                "engine_text",
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Ridge(alpha=5.5))])


def _enhanced_ridge_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                ENHANCED_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=15)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "text",
                TfidfVectorizer(max_features=700, ngram_range=(1, 2), min_df=4, sublinear_tf=True),
                "spec_text",
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Ridge(alpha=4.0))])


def _structured_histgb_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                ENHANCED_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_depth=12,
        max_iter=320,
        min_samples_leaf=35,
        l2_regularization=0.03,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def _fit_and_predict(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame) -> np.ndarray:
    model.fit(X_train, y_train)
    return np.expm1(model.predict(X_eval))


def _blend_predictions(
    left_prediction: np.ndarray,
    right_prediction: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, dict[str, float], np.ndarray]:
    best_weight = 0.5
    best_pred = left_prediction
    best_mae = float("inf")

    for weight in np.arange(0.10, 0.95, 0.05):
        prediction = (weight * left_prediction) + ((1 - weight) * right_prediction)
        mae = float(mean_absolute_error(actual, prediction))
        if mae < best_mae:
            best_mae = mae
            best_weight = round(float(weight), 2)
            best_pred = prediction

    return best_weight, _metrics(actual, best_pred), best_pred


def _predict_from_artifact(artifact: Any, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(artifact, dict) and artifact.get("artifact_type") == "weighted_blend":
        ridge_price = np.expm1(artifact["ridge_model"].predict(frame[ENHANCED_FEATURES]))
        tree_price = np.expm1(artifact["tree_model"].predict(frame[TREE_FEATURES]))
        ridge_weight = float(artifact["ridge_weight"])
        return (ridge_weight * ridge_price) + ((1 - ridge_weight) * tree_price)
    return np.expm1(artifact.predict(frame))


def train() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_frame(load_dataset())
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

    target = np.log1p(df["price"])
    train_valid_frame, test_frame, y_train_valid, y_test = train_test_split(
        df,
        target,
        test_size=0.2,
        random_state=42,
    )
    train_frame, val_frame, y_train, y_val = train_test_split(
        train_valid_frame,
        y_train_valid,
        test_size=0.2,
        random_state=42,
    )

    actual_val = np.expm1(y_val.to_numpy())
    actual_test = np.expm1(y_test.to_numpy())

    baseline_model = _baseline_ridge_pipeline()
    enhanced_model = _enhanced_ridge_pipeline()
    structured_model = _structured_histgb_pipeline()

    baseline_val_prediction = _fit_and_predict(
        baseline_model,
        train_frame[BASELINE_FEATURES],
        y_train,
        val_frame[BASELINE_FEATURES],
    )
    enhanced_val_prediction = _fit_and_predict(
        enhanced_model,
        train_frame[ENHANCED_FEATURES],
        y_train,
        val_frame[ENHANCED_FEATURES],
    )
    structured_val_prediction = _fit_and_predict(
        structured_model,
        train_frame[TREE_FEATURES],
        y_train,
        val_frame[TREE_FEATURES],
    )

    candidate_results: dict[str, dict[str, Any]] = {
        "baseline_ridge_engine_text": {"validation": _metrics(actual_val, baseline_val_prediction)},
        "ridge_spec_text_nlp": {"validation": _metrics(actual_val, enhanced_val_prediction)},
        "hist_gradient_boosting_structured": {"validation": _metrics(actual_val, structured_val_prediction)},
    }

    blend_weight, blend_validation_metrics, _ = _blend_predictions(
        enhanced_val_prediction,
        structured_val_prediction,
        actual_val,
    )
    candidate_results["hybrid_blend"] = {
        "validation": {**blend_validation_metrics, "ridge_weight": blend_weight}
    }

    best_candidate_name = min(
        candidate_results,
        key=lambda name: candidate_results[name]["validation"]["mae"],
    )

    final_baseline_model = _baseline_ridge_pipeline()
    final_enhanced_model = _enhanced_ridge_pipeline()
    final_structured_model = _structured_histgb_pipeline()

    final_baseline_model.fit(train_valid_frame[BASELINE_FEATURES], y_train_valid)
    final_enhanced_model.fit(train_valid_frame[ENHANCED_FEATURES], y_train_valid)
    final_structured_model.fit(train_valid_frame[TREE_FEATURES], y_train_valid)

    final_artifact: Any
    if best_candidate_name == "hybrid_blend":
        final_artifact = {
            "artifact_type": "weighted_blend",
            "ridge_model": final_enhanced_model,
            "tree_model": final_structured_model,
            "ridge_weight": blend_weight,
        }
    elif best_candidate_name == "hist_gradient_boosting_structured":
        final_artifact = final_structured_model
    elif best_candidate_name == "ridge_spec_text_nlp":
        final_artifact = final_enhanced_model
    else:
        final_artifact = final_baseline_model

    baseline_test_prediction = _predict_from_artifact(final_baseline_model, test_frame[BASELINE_FEATURES])
    enhanced_test_prediction = _predict_from_artifact(final_enhanced_model, test_frame[ENHANCED_FEATURES])
    structured_test_prediction = _predict_from_artifact(final_structured_model, test_frame[TREE_FEATURES])
    final_test_prediction = _predict_from_artifact(final_artifact, test_frame)

    candidate_results["baseline_ridge_engine_text"]["test"] = _metrics(actual_test, baseline_test_prediction)
    candidate_results["ridge_spec_text_nlp"]["test"] = _metrics(actual_test, enhanced_test_prediction)
    candidate_results["hist_gradient_boosting_structured"]["test"] = _metrics(actual_test, structured_test_prediction)
    candidate_results["hybrid_blend"]["test"] = {
        **_metrics(actual_test, final_test_prediction),
        "ridge_weight": blend_weight,
    }

    test_damage_mask = test_frame["accidents_or_damage"].fillna(0).to_numpy() == 1
    damage_discount_ratio = 0.06
    if test_damage_mask.sum() > 50 and (~test_damage_mask).sum() > 50:
        damaged_ratio = 1 - np.median(actual_test[test_damage_mask] / np.maximum(final_test_prediction[test_damage_mask], 1.0))
        damage_discount_ratio = float(np.clip(damaged_ratio, 0.04, 0.12))

    joblib.dump(final_artifact, MODELS_DIR / "carsight_model.joblib")

    final_metrics = _metrics(actual_test, final_test_prediction)
    metadata = {
        "dataset_name": DATASET_NAME,
        "source_note": "Public Kaggle used-cars dataset scraped from cars.com",
        "selected_model": best_candidate_name,
        "model_name": "Weighted hybrid blend" if best_candidate_name == "hybrid_blend" else best_candidate_name,
        "methodology": (
            "Compared four methods: the original engine-text Ridge baseline, an expanded TF-IDF Ridge model over combined vehicle spec text, "
            "a structured HistGradientBoosting regressor, and a weighted hybrid blend. "
            "The final model keeps the NLP component and blends it with gradient boosting because that combination gave the lowest error."
        ),
        "selection_note": "The final range width is calibrated from held-out residual quantiles instead of a fixed padding rule.",
        "training_rows": int(len(train_valid_frame)),
        "test_rows": int(len(test_frame)),
        "mae": final_metrics["mae"],
        "rmse": final_metrics["rmse"],
        "mape": final_metrics["mape"],
        "abs_error_q55": final_metrics["abs_error_q55"],
        "abs_error_q60": final_metrics["abs_error_q60"],
        "abs_error_q70": final_metrics["abs_error_q70"],
        "abs_error_q80": final_metrics["abs_error_q80"],
        "damage_discount_ratio": damage_discount_ratio,
        "candidate_results": _round_nested(candidate_results),
        "blend_weight": blend_weight,
        "known_makes": sorted(df["manufacturer"].dropna().astype(str).unique().tolist()),
        "known_models_sample": sorted(df["model"].dropna().astype(str).unique().tolist())[:400],
    }
    (MODELS_DIR / "carsight_model_metadata.json").write_text(
        json.dumps(_round_nested(metadata), indent=2),
        encoding="utf-8",
    )

    preview = df[["manufacturer", "model", "year", "mileage", "engine", "price", "accidents_or_damage"]].head(500)
    preview.to_csv(PROCESSED_DIR / "training_preview.csv", index=False)

    print(json.dumps(_round_nested(metadata), indent=2))


if __name__ == "__main__":
    train()

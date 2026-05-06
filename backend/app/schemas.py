from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VehicleInput(BaseModel):
    make: str = Field(..., min_length=1, max_length=80)
    model: str = Field(..., min_length=1, max_length=120)
    year: int = Field(..., ge=1980, le=2030)
    mileage: int = Field(..., ge=0, le=500_000)
    engine: str = Field(..., min_length=1, max_length=200)
    fuel_type: str | None = Field(default="Gasoline", max_length=80)
    transmission: str | None = Field(default="Automatic", max_length=80)
    drivetrain: str | None = Field(default="Front-wheel Drive", max_length=80)
    color: str | None = Field(default="Unknown", max_length=80)
    damage_description: str = Field(default="", max_length=600)
    asking_price: float | None = Field(default=None, ge=0)


class DamageIssue(BaseModel):
    issue: str
    category: str
    severity: str
    penalty_pct: float
    estimated_impact: float


class DamageSummary(BaseModel):
    cleaned_text: str
    severity_score: float
    total_penalty_pct: float
    confidence_hint: str
    issues: list[DamageIssue]
    category_counts: dict[str, int]


class AnalysisResponse(BaseModel):
    input_summary: dict[str, Any]
    base_prediction: float
    adjusted_prediction: float
    price_range: dict[str, float]
    confidence: str
    confidence_score: float
    asking_price_comparison: str | None
    extracted_damage: DamageSummary
    explanation_points: list[str]
    negotiation_tip: str
    model_metrics: dict[str, float | int | str]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=400)
    analysis: AnalysisResponse


class ChatResponse(BaseModel):
    answer: str
    bullets: list[str]

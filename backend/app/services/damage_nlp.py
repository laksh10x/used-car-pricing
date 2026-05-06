from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import DamageIssue, DamageSummary


SEVERITY_MAP = {
    "low": {"score": 1.0, "multiplier": 1.0},
    "medium": {"score": 2.0, "multiplier": 1.5},
    "high": {"score": 3.0, "multiplier": 2.1},
}

SEVERITY_HINTS = {
    "high": [
        "engine knocking",
        "engine knock",
        "knocks",
        "transmission slipping",
        "frame damage",
        "flood damage",
        "salvage",
        "airbag",
        "won't start",
        "major accident",
        "check engine",
        "check engine light",
        "misfire",
    ],
    "medium": [
        "cracked",
        "dent",
        "damaged",
        "needs repair",
        "oil leak",
        "rough idle",
        "bumper",
        "paint peeling",
        "warning light",
        "alignment",
    ],
    "low": [
        "scratch",
        "scratches",
        "minor",
        "small",
        "cosmetic",
        "wear",
        "stain",
        "scuff",
        "chip",
        "service soon",
    ],
}

CATEGORY_KEYWORDS = {
    "body": {
        "keywords": ["scratch", "scratches", "dent", "bumper", "paint", "fender", "door", "hood", "rust", "cracked mirror"],
        "penalties": {"low": 0.015, "medium": 0.04, "high": 0.08},
    },
    "engine": {
        "keywords": ["engine", "knock", "misfire", "oil leak", "overheat", "won't start", "stall", "timing", "smoke"],
        "penalties": {"low": 0.04, "medium": 0.08, "high": 0.14},
    },
    "transmission": {
        "keywords": ["transmission", "slipping", "gear", "clutch", "shifts hard", "shift"],
        "penalties": {"low": 0.035, "medium": 0.07, "high": 0.12},
    },
    "interior": {
        "keywords": ["seat", "interior", "dashboard", "odor", "smell", "tear", "stain", "upholstery"],
        "penalties": {"low": 0.01, "medium": 0.025, "high": 0.05},
    },
    "electrical": {
        "keywords": ["battery", "electrical", "sensor", "warning light", "airbag", "camera", "radio", "window"],
        "penalties": {"low": 0.025, "medium": 0.05, "high": 0.09},
    },
    "suspension": {
        "keywords": ["suspension", "alignment", "axle", "frame", "steering", "brake", "rotor", "wheel bearing"],
        "penalties": {"low": 0.03, "medium": 0.06, "high": 0.11},
    },
    "tires": {
        "keywords": ["tire", "tires", "tread", "flat", "wheel"],
        "penalties": {"low": 0.01, "medium": 0.02, "high": 0.035},
    },
}


@dataclass
class ParsedIssue:
    issue: str
    category: str
    severity: str
    penalty_pct: float


def clean_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s\-/]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def detect_severity(text: str) -> str:
    if not text:
        return "low"
    for severity in ("high", "medium", "low"):
        for pattern in SEVERITY_HINTS[severity]:
            if pattern in text:
                return severity
    return "medium"


def parse_damage_description(text: str, reference_price: float, baseline_damage_discount: float) -> DamageSummary:
    cleaned = clean_text(text)
    if not cleaned:
        return DamageSummary(
            cleaned_text="",
            severity_score=0.0,
            total_penalty_pct=0.0,
            confidence_hint="No damage text provided, so the valuation leans mostly on the base market model.",
            issues=[],
            category_counts={},
        )

    issues: list[ParsedIssue] = []
    category_counts: dict[str, int] = {}

    for category, config in CATEGORY_KEYWORDS.items():
        matches = [kw for kw in config["keywords"] if kw in cleaned]
        if not matches:
            continue
        severity = detect_severity(cleaned)
        penalty = config["penalties"][severity]
        category_counts[category] = len(matches)
        issues.append(
            ParsedIssue(
                issue=", ".join(sorted(set(matches)))[:80],
                category=category,
                severity=severity,
                penalty_pct=penalty,
            )
        )

    if not issues:
        fallback_penalty = round(max(0.01, baseline_damage_discount * 0.5), 4)
        issues.append(
            ParsedIssue(
                issue="general condition concern",
                category="general",
                severity=detect_severity(cleaned),
                penalty_pct=fallback_penalty,
            )
        )
        category_counts["general"] = 1

    total_penalty = sum(issue.penalty_pct for issue in issues)
    high_count = sum(1 for issue in issues if issue.severity == "high")
    if high_count:
        total_penalty = max(total_penalty, baseline_damage_discount * 1.4)
    elif issues:
        total_penalty = max(total_penalty, baseline_damage_discount * 0.8)
    total_penalty = min(round(total_penalty, 4), 0.35)

    severity_score = round(
        sum(SEVERITY_MAP[issue.severity]["score"] for issue in issues) / max(len(issues), 1),
        2,
    )

    confidence_hint = "The damage language is specific enough to estimate a direction of impact."
    if any(issue.severity == "high" for issue in issues):
        confidence_hint = "The text points to a major risk item, so the app uses a wider range and a stronger discount."
    elif len(issues) > 2:
        confidence_hint = "Multiple issues were detected, so the adjustment is spread across several categories."

    damage_issues = [
        DamageIssue(
            issue=issue.issue,
            category=issue.category,
            severity=issue.severity,
            penalty_pct=round(issue.penalty_pct, 4),
            estimated_impact=round(reference_price * issue.penalty_pct, 2),
        )
        for issue in issues
    ]

    return DamageSummary(
        cleaned_text=cleaned,
        severity_score=severity_score,
        total_penalty_pct=total_penalty,
        confidence_hint=confidence_hint,
        issues=damage_issues,
        category_counts=category_counts,
    )

from app.schemas import AnalysisResponse, DamageIssue, DamageSummary
from app.services.chat_service import answer_question


def test_chat_risk_question_mentions_main_issue():
    analysis = AnalysisResponse(
        input_summary={"asking_price": 14900},
        base_prediction=21000,
        adjusted_prediction=18400,
        price_range={"low": 16000, "high": 20800},
        confidence="Medium",
        confidence_score=0.74,
        asking_price_comparison="The asking price is below the estimated range.",
        extracted_damage=DamageSummary(
            cleaned_text="engine knocks and check engine light is on",
            severity_score=3.0,
            total_penalty_pct=0.18,
            confidence_hint="The text points to a major risk item, so the app uses a wider range and a stronger discount.",
            issues=[
                DamageIssue(
                    issue="engine, knocks",
                    category="engine",
                    severity="high",
                    penalty_pct=0.14,
                    estimated_impact=2940.0,
                )
            ],
            category_counts={"engine": 2},
        ),
        explanation_points=[
            "The base value comes from market patterns in a Kaggle used-car pricing dataset.",
            "The damage text triggered an 18.0% adjustment.",
            "The text points to a major risk item.",
        ],
        negotiation_tip="Negotiate from the lower half of the range.",
        model_metrics={"dataset": "demo", "training_rows": 96000, "mae": 3322.25, "rmse": 5655.38, "mape": 0.1142},
    )

    response = answer_question("What is the main risk here?", analysis)

    assert "engine" in response.answer.lower()
    assert "inspection" in response.answer.lower()

from app.services.damage_nlp import parse_damage_description


def test_engine_issue_gets_high_penalty():
    summary = parse_damage_description(
        "Engine knocking badly and check engine light is on.",
        reference_price=12000,
        baseline_damage_discount=0.06,
    )
    assert summary.issues
    assert any(issue.category == "engine" for issue in summary.issues)
    assert summary.total_penalty_pct >= 0.08


def test_blank_text_returns_zero_penalty():
    summary = parse_damage_description(
        "",
        reference_price=12000,
        baseline_damage_discount=0.06,
    )
    assert summary.total_penalty_pct == 0
    assert summary.issues == []

from __future__ import annotations

from app.schemas import AnalysisResponse, ChatResponse


def answer_question(question: str, analysis: AnalysisResponse) -> ChatResponse:
    q = question.lower().strip()
    low = analysis.price_range["low"]
    high = analysis.price_range["high"]
    adjusted = analysis.adjusted_prediction
    asking = analysis.input_summary.get("asking_price")
    damage = analysis.extracted_damage
    main_issue = max(damage.issues, key=lambda issue: issue.estimated_impact) if damage.issues else None

    bullets = [
        f"Estimated fair range: ${low:,.0f} to ${high:,.0f}.",
        f"Confidence: {analysis.confidence}.",
    ]

    if asking:
        bullets.append(analysis.asking_price_comparison or "")

    if "overprice" in q or "overpriced" in q or "fair" in q:
        if asking is None:
            answer = f"My fair range is about ${low:,.0f} to ${high:,.0f}. Without an asking price, I would use that band as the benchmark."
        elif asking > high:
            answer = f"Based on the current inputs, it looks overpriced. The asking price is above my upper estimate of ${high:,.0f}, so I would negotiate down."
        elif asking < low:
            answer = f"It does not look overpriced. The asking price is actually below my estimated range, which could mean it is a strong value or that the listing needs closer inspection."
        else:
            answer = f"It looks reasonably close to fair. The asking price sits inside the estimated range of ${low:,.0f} to ${high:,.0f}."
    elif "negot" in q or "offer" in q:
        answer = analysis.negotiation_tip
        bullets.append(f"Adjusted midpoint: ${adjusted:,.0f}.")
    elif "why" in q or "explain" in q or "reason" in q:
        if main_issue:
            answer = (
                f"The biggest adjustment came from {main_issue.category} risk. I flagged '{main_issue.issue}' as {main_issue.severity} severity, "
                f"which lowered the value by about ${main_issue.estimated_impact:,.0f}."
            )
        else:
            answer = "Most of the estimate comes from the car's year, mileage, make, model, and engine profile because no clear damage issue was detected."
    elif "risk" in q or "safe" in q or "worry" in q:
        if main_issue:
            answer = (
                f"The main risk is {main_issue.category} condition tied to '{main_issue.issue}'. "
                f"I treated that as {main_issue.severity} severity, so I would want service history or an inspection before trusting the deal."
            )
        else:
            answer = "The current description suggests manageable risk, but I would still verify maintenance history and look closely at the problem areas."
    else:
        answer = (
            f"I estimate this vehicle around ${low:,.0f} to ${high:,.0f}. "
            f"The main adjustment comes from the damage description and the base market profile learned from used-car listings."
        )

    bullets.extend(analysis.explanation_points[:2])
    bullets = [bullet for bullet in bullets if bullet]
    return ChatResponse(answer=answer, bullets=bullets)

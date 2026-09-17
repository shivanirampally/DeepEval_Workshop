def make_final_decision(guardrail_decision: str, evaluation_verdict: str) -> str:
    """Apply simple safety-first decision precedence."""
    if guardrail_decision == "BLOCK":
        return "BLOCK"

    if guardrail_decision == "REVIEW":
        return "MANUAL_REVIEW"

    if evaluation_verdict != "PASS":
        return "MANUAL_REVIEW"

    return "PASS"

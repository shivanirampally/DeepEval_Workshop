import pandas as pd
from config import QUALITY_THRESHOLD, REVIEW_THRESHOLD

METRIC_ACTIONS = {
    "hallucination": "Strengthen source-only instructions and require explicit acknowledgement when information is missing.",
    "faithfulness": "Tell the model to stay strictly aligned with the supplied source and avoid unsupported details.",
    "correctness": "Tell the model to prefer source-supported facts and avoid assumptions or unsupported inference.",
    "completeness": "Tell the model to cover supported information and clearly identify missing information.",
    "answer_relevancy": "Keep the response focused directly on the question and remove unnecessary discussion.",
    "bias": "Use neutral, evidence-based wording and avoid unsupported judgments.",
}


def verdict(score):
    if score is None:
        return "ERROR"
    if score >= QUALITY_THRESHOLD:
        return "PASS"
    if score >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "FAIL"


def analyze(details):
    if details.empty:
        return pd.DataFrame(), []

    if "status" in details.columns:
        failed_mask = (details["passed"] != True) | (details["status"] != "COMPLETED")
    else:
        failed_mask = details["passed"] != True
    failures = details[failed_mask].copy()
    if failures.empty:
        failures = pd.DataFrame(columns=[
            "test_id", "metric", "score", "verdict", "reason", "recommended_action"
        ])
    else:
        failures["verdict"] = failures["score"].apply(verdict)
        failures["recommended_action"] = failures["metric"].map(METRIC_ACTIONS).fillna(
            "Review the failed metric and determine whether a prompt change is appropriate."
        )
        failures = failures[
            ["test_id", "metric", "score", "verdict", "reason", "recommended_action"]
        ]

    recommendations = []
    for metric in failures["metric"].dropna().unique():
        recommendations.append({
            "Metric": metric,
            "Failure Count": int((failures["metric"] == metric).sum()),
            "Recommended Action": METRIC_ACTIONS.get(
                metric, "Review whether the failure is prompt-related."
            ),
        })
    return failures, recommendations

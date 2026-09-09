import pandas as pd

METRIC_ACTIONS = {
    "hallucination": "Strengthen source-only instructions and explicitly prohibit invented details.",
    "faithfulness": "Require factual claims to remain supported by the supplied source.",
    "correctness": "Tell the model to prefer source-supported facts and avoid assumptions.",
    "completeness": "Tell the model to cover supported information and identify missing information.",
    "answer_relevancy": "Keep the response focused directly on the question.",
    "bias": "Use neutral, evidence-based wording and avoid unsupported judgments.",
}

def verdict(score):
    if score is None:
        return "ERROR"
    if score >= 0.90:
        return "PASS"
    if score >= 0.70:
        return "REVIEW"
    return "FAIL"

def analyze(details):
    failures = details[details["passed"] != True].copy()
    failure_rows = []

    for _, row in failures.iterrows():
        failure_rows.append({
            "Test Case": row["test_id"],
            "Metric": row["metric"],
            "Score (%)": "" if pd.isna(row["score"]) else round(float(row["score"]) * 100, 1),
            "Verdict": verdict(row["score"]),
            "Reason": row["reason"],
            "Recommended Action": METRIC_ACTIONS.get(
                row["metric"],
                "Review the prompt guidance for this metric.",
            ),
        })

    recommendations = []
    for metric in failures["metric"].dropna().unique():
        count = len(failures[failures["metric"] == metric])
        recommendations.append({
            "Metric": metric,
            "Failures / Reviews": count,
            "Recommended Action": METRIC_ACTIONS.get(
                metric,
                "Review the prompt guidance for this metric.",
            ),
        })

    return failure_rows, recommendations

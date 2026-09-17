import pandas as pd
from config import QUALITY_THRESHOLD, REVIEW_THRESHOLD
from governance_observability.excel_reporting import parse_excel_bool

METRIC_ACTIONS = {
    "hallucination": "Strengthen source-only instructions and require explicit acknowledgement when information is missing.",
    "faithfulness": "Tell the model to stay strictly aligned with the supplied source and avoid unsupported details.",
    "correctness": "Tell the model to prefer source-supported facts and avoid assumptions or unsupported inference.",
    "completeness": "Tell the model to cover supported information and clearly identify missing information.",
    "answer_relevancy": "Keep the response focused directly on the question and remove unnecessary discussion.",
    "bias": "Use neutral, evidence-based wording and avoid unsupported judgments.",
}


def verdict(score):
    # pd.isna catches both a real None and the float NaN that None becomes
    # after any Excel round-trip (every --resume path that re-reads a saved
    # evaluation sheet) -- `score is None` alone missed the NaN case and let
    # errored metrics silently fall through to a plain "FAIL" verdict.
    if pd.isna(score):
        return "ERROR"
    if score >= QUALITY_THRESHOLD:
        return "PASS"
    if score >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "FAIL"


def analyze(details):
    if details.empty:
        return pd.DataFrame(), []

    # A judge call that errored out (network blip, malformed response, etc.)
    # is a technical failure, not a verdict on the prompt -- optimizing a
    # prompt against a metric that never actually ran would be tuning against
    # noise. Keep those rows separate (surfaced via failures.attrs) so the
    # caller can block synthetic data generation / optimization until the
    # technical issue is resolved, instead of quietly folding it into the
    # ordinary prompt-quality failure list below.
    # Same normalization excel_reporting.py's pass-rate calculation uses --
    # a "passed" cell that comes back as the literal text "TRUE"/"FALSE"
    # after an Excel round-trip must be read the same way in both places,
    # or a --resume could classify a genuinely-passed row as a failure here
    # while the comparison sheet correctly counts it as passed.
    passed_bool = details["passed"].map(parse_excel_bool)
    if "status" in details.columns:
        technical_mask = details["status"].astype(str).str.upper().ne("COMPLETED")
        quality_failed_mask = passed_bool.ne(True) & ~technical_mask
        technical = details[technical_mask].copy()
    else:
        technical = pd.DataFrame()
        quality_failed_mask = passed_bool.ne(True)
    failures = details[quality_failed_mask].copy()
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
    failures.attrs["technical_errors"] = technical[
        [c for c in ["test_id", "metric", "status", "error"] if c in technical.columns]
    ].to_dict("records")
    return failures, recommendations

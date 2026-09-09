import pandas as pd

from analysis.failure_analysis import analyze, verdict

def test_verdict():
    assert verdict(0.95) == "PASS"
    assert verdict(0.80) == "REVIEW"
    assert verdict(0.60) == "FAIL"
    assert verdict(None) == "ERROR"

def test_failure_analysis_has_action():
    details = pd.DataFrame([{
        "test_id": "TC01",
        "metric": "hallucination",
        "score": 0.60,
        "passed": False,
        "reason": "Unsupported claim.",
    }])
    failures, recommendations = analyze(details)
    assert failures[0]["Recommended Action"]
    assert recommendations[0]["Metric"] == "hallucination"

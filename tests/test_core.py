import pandas as pd
from analysis.failure_analysis import analyze, verdict

def test_verdict_boundaries():
    assert verdict(0.95) == "PASS"
    assert verdict(0.80) == "REVIEW"
    assert verdict(0.60) == "FAIL"
    assert verdict(None) == "ERROR"

def test_failure_analysis_creates_action():
    details = pd.DataFrame([{
        "test_id": "T001", "metric": "correctness", "score": 0.60,
        "passed": False, "status": "COMPLETED",
        "reason": "Unsupported inference", "error": "",
    }])
    failures, recommendations = analyze(details)
    assert len(failures) == 1
    assert failures.iloc[0]["recommended_action"]
    assert recommendations[0]["Metric"] == "correctness"

from evaluation.scoring import (
    is_testcase_passed,
    testcase_status as get_testcase_status,
    weighted_score,
)


METRICS = (
    "hallucination",
    "faithfulness",
    "correctness",
    "completeness",
    "answer_relevancy",
    "bias",
)


def test_testcase_requires_every_metric():
    result = {
        name: {"status": "COMPLETED", "passed": True, "score": 0.9}
        for name in METRICS
    }
    assert is_testcase_passed(result)
    assert get_testcase_status(result) == "PASS"


def test_weighted_score_is_between_zero_and_one():
    result = {
        name: {"status": "COMPLETED", "score": 0.9}
        for name in METRICS
    }
    assert weighted_score(result) == 0.9


def test_weighted_score_is_none_when_any_metric_is_missing():
    # A rate-limited/errored metric must invalidate the score rather than
    # silently re-normalizing over only the metrics that completed - a
    # partial evaluation should never report a fabricated "perfect" score.
    result = {
        name: {"status": "COMPLETED", "score": 0.9}
        for name in METRICS
    }
    result["bias"] = {"status": "COMPLETED", "score": None}

    assert weighted_score(result) is None
    assert get_testcase_status(result) == "TECHNICAL ERROR"


def test_weighted_score_is_none_when_metric_status_is_error():
    result = {
        name: {"status": "COMPLETED", "score": 0.9}
        for name in METRICS
    }
    result["faithfulness"] = {"status": "ERROR", "score": 0.99}

    assert weighted_score(result) is None
    assert get_testcase_status(result) == "TECHNICAL ERROR"


def test_quality_fail_is_distinguished_from_technical_error():
    result = {
        name: {"status": "COMPLETED", "score": 0.9}
        for name in METRICS
    }
    result["hallucination"] = {
        "status": "COMPLETED",
        "score": 0.50,
        "passed": False,
        "verdict": "FAIL",
    }

    assert get_testcase_status(result) == "QUALITY FAIL"
    assert not is_testcase_passed(result)

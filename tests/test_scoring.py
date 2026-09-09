from deepeval_framework.scoring import is_testcase_passed, weighted_score


def test_testcase_requires_every_metric():
    result = {
        name: {"status": "COMPLETED", "passed": True, "score": 0.9}
        for name in (
            "hallucination",
            "faithfulness",
            "correctness",
            "completeness",
            "answer_relevancy",
            "bias",
        )
    }
    assert is_testcase_passed(result)


def test_weighted_score_is_between_zero_and_one():
    result = {
        name: {"score": 0.9}
        for name in (
            "hallucination",
            "faithfulness",
            "correctness",
            "completeness",
            "answer_relevancy",
            "bias",
        )
    }
    assert weighted_score(result) == 0.9

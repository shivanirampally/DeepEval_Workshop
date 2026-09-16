from observability.analysis import recommend


def _row(generator, score, passed, total, technical=0):
    return {
        "generator": generator,
        "overall_score": score,
        "passed_testcases": passed,
        "total_testcases": total,
        "quality_gate": "PASS" if passed == total else "FAIL",
        "technical_error_count": technical,
        "metric_scores": {},
    }


def test_small_sample_does_not_claim_a_winner():
    message = recommend([
        _row("llama3:instruct", 0.91, 3, 3),
        _row("gpt-oss:20b", 0.90, 3, 3),
    ])
    assert "3-test sample" in message
    assert "No clear winner" in message


def test_technical_error_blocks_final_comparison():
    message = recommend([
        _row("gpt-oss:20b", None, 2, 3, technical=1),
        _row("llama3:instruct", 0.92, 3, 3),
    ])
    assert "technical evaluation errors" in message


def test_large_sample_with_clear_gap_names_a_winner():
    message = recommend([
        _row("llama3:instruct", 0.95, 10, 10),
        _row("qwen2.5-coder:14b", 0.70, 8, 10),
    ], minimum_testcases=10)
    assert "Preferred generator: llama3:instruct" in message

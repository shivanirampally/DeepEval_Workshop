from config import (
    METRIC_THRESHOLDS,
    METRIC_WEIGHTS,
    QUALITY_THRESHOLD,
    WARNING_THRESHOLD,
)


CRITICAL_METRICS = {
    "hallucination",
    "faithfulness",
    "correctness",
}


def weighted_score(metric_results):
    values = []
    weights = []

    for name, weight in METRIC_WEIGHTS.items():
        result = metric_results.get(name, {})
        value = result.get("score")

        # A missing/errored metric invalidates the weighted score, even if
        # a stale score happens to be present alongside a non-COMPLETED
        # status. Silently re-normalizing over only the metrics that
        # happened to complete would let a partially-failed evaluation
        # (e.g. a rate-limited judge) report a misleadingly high - even a
        # fabricated "perfect" - score instead of surfacing that the
        # evaluation was incomplete.
        if value is None or result.get("status") not in (None, "COMPLETED"):
            return None

        values.append(float(value) * weight)
        weights.append(weight)

    if not weights:
        return None

    return round(
        sum(values) / sum(weights),
        4,
    )


def metric_verdict(metric_name, score):
    if score is None:
        return "ERROR"

    thresholds = METRIC_THRESHOLDS[metric_name]

    if score >= thresholds["pass"]:
        return "PASS"

    if score >= thresholds["review"]:
        return "REVIEW"

    return "FAIL"


def testcase_verdict(metric_results):
    if not metric_results:
        return "FAIL"

    for name, result in metric_results.items():
        if result.get("status") != "COMPLETED":
            return "FAIL"

        score = result.get("score")

        if score is None:
            return "FAIL"

        if (
            name in CRITICAL_METRICS
            and score < METRIC_THRESHOLDS[name]["review"]
        ):
            return "FAIL"

    score = weighted_score(metric_results)

    if score is None:
        return "FAIL"

    if score >= QUALITY_THRESHOLD:
        return "PASS"

    if score >= WARNING_THRESHOLD:
        return "REVIEW"

    return "FAIL"


def testcase_status(metric_results):
    """Distinguish a technical evaluation failure from a genuine quality result.

    testcase_verdict() collapses both into "FAIL", which reads the same in
    a report whether the judge scored the response poorly or the judge call
    itself errored (timeout, rate limit, connection failure). Those need
    different follow-up (fix the generator's response vs. fix the judge/
    infrastructure), so this returns one of PASS, QUALITY REVIEW,
    QUALITY FAIL, or TECHNICAL ERROR instead.
    """
    if not metric_results:
        return "TECHNICAL ERROR"

    for result in metric_results.values():
        if result.get("status") != "COMPLETED" or result.get("score") is None:
            return "TECHNICAL ERROR"

    verdict = testcase_verdict(metric_results)
    if verdict == "PASS":
        return "PASS"
    if verdict == "REVIEW":
        return "QUALITY REVIEW"
    return "QUALITY FAIL"


def is_testcase_passed(metric_results):
    return testcase_verdict(metric_results) == "PASS"

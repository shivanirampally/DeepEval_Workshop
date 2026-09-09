from config import METRIC_THRESHOLDS, METRIC_WEIGHTS


CRITICAL_METRICS = {
    "hallucination",
    "faithfulness",
    "correctness",
}


def weighted_score(metric_results):
    values = []
    weights = []

    for name, weight in METRIC_WEIGHTS.items():
        value = metric_results.get(name, {}).get("score")

        if value is not None:
            values.append(value * weight)
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

    if score >= 0.80:
        return "PASS"

    if score >= 0.60:
        return "REVIEW"

    return "FAIL"


def is_testcase_passed(metric_results):
    return testcase_verdict(metric_results) == "PASS"

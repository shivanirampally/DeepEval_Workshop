from config import METRIC_WEIGHTS


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
    return round(sum(values) / sum(weights), 4)


def is_testcase_passed(metric_results):
    return bool(metric_results) and all(
        item.get("status") == "COMPLETED" and item.get("passed") is True
        for item in metric_results.values()
    )


def judge_agreement(judge_results):
    scores = [
        result.get("score")
        for result in judge_results
        if result.get("score") is not None
    ]
    if len(scores) < 2:
        return None
    return round(1 - abs(scores[0] - scores[1]), 4)

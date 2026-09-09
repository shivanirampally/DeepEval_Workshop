from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from deepeval_framework.metrics import build_test_case, create_metrics
from config import JUDGE_CONCURRENCY, METRIC_NAMES, RETRIES


def _run_metric(metric_name, metric, test_case):
    last_error = ""
    for attempt in range(RETRIES + 1):
        try:
            started = time.perf_counter()
            metric.measure(test_case)
            score = float(metric.score)
            return {
                "score": round(score, 4),
                "passed": bool(metric.is_successful()),
                "reason": metric.reason or "",
                "status": "COMPLETED",
                "duration_seconds": round(time.perf_counter() - started, 2),
                "error": "",
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < RETRIES:
                continue

    return {
        "score": None,
        "passed": False,
        "reason": "",
        "status": "ERROR",
        "duration_seconds": "",
        "error": last_error,
    }


def evaluate_one(response_row, source_row, judge_name, base_url):
    if response_row["status"] != "COMPLETED":
        return {
            metric: {
                "score": None,
                "passed": False,
                "reason": "",
                "status": "SKIPPED",
                "duration_seconds": "",
                "error": response_row["error"],
            }
            for metric in METRIC_NAMES
        }

    test_case = build_test_case(source_row, response_row["response"])
    metrics = create_metrics(judge_name, base_url)
    results = {}

    workers = min(JUDGE_CONCURRENCY, len(METRIC_NAMES))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_metric, name, metrics[name], test_case): name
            for name in METRIC_NAMES
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return {name: results[name] for name in METRIC_NAMES}

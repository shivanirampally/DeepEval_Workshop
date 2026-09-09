from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from deepeval_framework.metrics import (
    build_test_case,
    create_metrics,
)
from deepeval_framework.scoring import metric_verdict

from config import (
    JUDGE_CONCURRENCY,
    METRIC_NAMES,
    RETRIES,
)


def _run_metric(
    metric_name,
    metric,
    test_case,
):
    last_error = ""

    for attempt in range(RETRIES + 1):
        try:
            started = time.perf_counter()

            metric.measure(test_case)

            score = float(metric.score)
            verdict = metric_verdict(
                metric_name,
                score,
            )

            return {
                "score": round(score, 4),
                "passed": verdict == "PASS",
                "verdict": verdict,
                "reason": metric.reason or "",
                "status": "COMPLETED",
                "duration_seconds": round(
                    time.perf_counter() - started,
                    2,
                ),
                "error": "",
            }

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < RETRIES:
                continue

    return {
        "score": None,
        "passed": False,
        "verdict": "ERROR",
        "reason": "",
        "status": "ERROR",
        "duration_seconds": "",
        "error": last_error,
    }


def evaluate_one(
    response_row,
    source_row,
    judge_name,
    base_url,
):
    if response_row["status"] != "COMPLETED":
        return {
            metric: {
                "score": None,
                "passed": False,
                "verdict": "ERROR",
                "reason": "",
                "status": "SKIPPED",
                "duration_seconds": "",
                "error": response_row["error"],
            }
            for metric in METRIC_NAMES
        }

    try:
        test_case = build_test_case(
            source_row,
            response_row["response"],
        )

        metrics = create_metrics(
            judge_name,
            base_url,
        )

    except Exception as exc:
        error = (
            f"{type(exc).__name__}: {exc}"
        )

        return {
            metric: {
                "score": None,
                "passed": False,
                "verdict": "ERROR",
                "reason": "",
                "status": "ERROR",
                "duration_seconds": "",
                "error": error,
            }
            for metric in METRIC_NAMES
        }

    results = {}

    workers = min(
        JUDGE_CONCURRENCY,
        len(METRIC_NAMES),
    )

    with ThreadPoolExecutor(
        max_workers=workers
    ) as pool:
        futures = {
            pool.submit(
                _run_metric,
                name,
                metrics[name],
                test_case,
            ): name
            for name in METRIC_NAMES
        }

        for future in as_completed(futures):
            name = futures[future]

            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {
                    "score": None,
                    "passed": False,
                    "verdict": "ERROR",
                    "reason": "",
                    "status": "ERROR",
                    "duration_seconds": "",
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }

    return {
        name: results[name]
        for name in METRIC_NAMES
    }
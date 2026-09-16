from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
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


class TruthsCache:
    """Thread-safe cache for FaithfulnessMetric's extracted "truths".

    Truths are derived only from the shared source/retrieval context, not
    from the generator's response, so they are identical for every
    generator evaluated against the same test case. Caching them by
    test_id avoids re-running the same "extract truths" LLM call once per
    generator. setdefault semantics ensure that if two generators race to
    populate the same key, only the first result is kept and both callers
    observe the same cached value.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._store = {}

    def get(self, key):
        with self._lock:
            return self._store.get(key)

    def set(self, key, value):
        with self._lock:
            self._store.setdefault(key, value)
            return self._store[key]


def _apply_cached_truths(metric, cached_truths):
    metric._generate_truths = lambda *args, **kwargs: cached_truths


def _instrument_judge_calls(metric, context_label):
    """Print each internal judge LLM call as it completes.

    DeepEval metrics make several sequential calls to the judge model per
    measure() (e.g. Faithfulness: extract truths, extract claims, generate
    verdicts, generate reason). Every one of those calls goes through
    model.generate(prompt, schema=...), and the schema class DeepEval
    passes (Truths, Claims, Verdicts, FaithfulnessScoreReason, ...) names
    the step, so wrapping generate() gives an accurate, real-time view of
    backend activity instead of a single opaque wait per metric. This adds
    a print() and two perf_counter() reads around a call that already
    takes seconds - no measurable latency.
    """
    original_generate = metric.model.generate

    def logged_generate(*args, **kwargs):
        schema = kwargs.get("schema")
        step = schema.__name__ if schema is not None else "response"
        started = time.perf_counter()
        result = original_generate(*args, **kwargs)
        elapsed = time.perf_counter() - started
        print(f"      {context_label} {step}: {elapsed:.1f}s")
        return result

    metric.model.generate = logged_generate


def _run_metric(
    metric_name,
    metric,
    test_case,
    truths_cache=None,
    cache_key=None,
    context_label=None,
):
    last_error = ""
    use_truths_cache = (
        metric_name == "faithfulness"
        and truths_cache is not None
        and cache_key is not None
    )
    cached_truths = truths_cache.get(cache_key) if use_truths_cache else None
    if cached_truths is not None:
        _apply_cached_truths(metric, cached_truths)

    if context_label is not None:
        _instrument_judge_calls(
            metric, f"{context_label}[{metric_name}]"
        )

    for attempt in range(RETRIES + 1):
        try:
            started = time.perf_counter()

            metric.measure(test_case)

            if use_truths_cache and cached_truths is None:
                truths_cache.set(cache_key, metric.truths)

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
    truths_cache=None,
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

    cache_key = str(response_row.get("test_id", ""))
    context_label = (
        f"[{response_row.get('generator', '')}][{cache_key}]"
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
                truths_cache,
                cache_key,
                context_label,
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
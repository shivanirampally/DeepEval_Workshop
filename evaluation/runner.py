"""Per-testcase metric execution against the judge.

Evaluation layer: runs the configured DeepEval metrics for one generator
response, caches Faithfulness's source-only truths extraction across
generators for the same test case, and captures per-internal-call timing
for observability/reporting.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from evaluation.metrics import build_test_case, create_metrics
from evaluation.scorer import metric_verdict
from observability import console

from settings.config import (
    JUDGE_CONCURRENCY,
    METRIC_NAMES,
    RETRIES,
)


class TruthsCache:
    """Thread-safe, race-safe cache for FaithfulnessMetric's "truths".

    Truths are derived only from the shared source/retrieval context, not
    from the generator's response, so they are identical for every
    generator evaluated against the same test case. get_or_compute()
    ensures exactly one real extraction call happens per test_id even when
    multiple generators' Faithfulness metrics reach a cache miss at the
    same time (concurrent evaluation_concurrency > 1): the first caller
    computes it; concurrent callers for the same key block until that
    finishes, then all reuse the same value. A naive get()-then-set()
    (check, then compute, then store) has a race window where several
    callers can all see a miss and all pay for a real extraction call.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._store = {}
        self._inflight = set()

    def get(self, key):
        with self._condition:
            return self._store.get(key)

    def get_or_compute(self, key, producer):
        """Return the cached value, or compute it exactly once for this key."""
        with self._condition:
            while key in self._inflight and key not in self._store:
                self._condition.wait()

            if key in self._store:
                return self._store[key], False

            self._inflight.add(key)

        try:
            value = producer()
            with self._condition:
                self._store[key] = value
                self._inflight.discard(key)
                self._condition.notify_all()
            return value, True
        except Exception:
            with self._condition:
                self._inflight.discard(key)
                self._condition.notify_all()
            raise


def _apply_cached_truths(metric, cached_truths):
    metric._generate_truths = lambda *args, **kwargs: cached_truths


def _instrument_judge_calls(metric, context_label, call_timings):
    """Capture and print every internal judge LLM call as it completes.

    DeepEval metrics make several sequential calls to the judge model per
    measure() (e.g. Faithfulness: extract truths, extract claims, generate
    verdicts, generate reason). Every one of those calls goes through
    model.generate(prompt, schema=...), and the schema class DeepEval
    passes (Truths, Claims, Verdicts, FaithfulnessScoreReason, ...) names
    the step, so wrapping generate() gives an accurate, real-time view of
    backend activity - not a guessed label - and a durable per-call timing
    record for the report. This adds a print() and two perf_counter()
    reads around a call that already takes seconds - no measurable latency
    (verified live: run time was unchanged before/after adding this).
    """
    original_generate = metric.model.generate

    def logged_generate(*args, **kwargs):
        schema = kwargs.get("schema")
        step = getattr(schema, "__name__", "response")
        started = time.perf_counter()
        try:
            return original_generate(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            call_timings.append({
                "step": step,
                "duration_seconds": round(elapsed, 2),
            })
            console.judge_call(context_label, step, elapsed)

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
    call_timings = []
    if context_label is not None:
        _instrument_judge_calls(metric, context_label, call_timings)

    use_truths_cache = (
        metric_name == "faithfulness"
        and truths_cache is not None
        and cache_key is not None
    )

    for attempt in range(RETRIES + 1):
        try:
            started = time.perf_counter()

            if use_truths_cache:
                cached_truths = truths_cache.get(cache_key)
                if cached_truths is not None:
                    _apply_cached_truths(metric, cached_truths)

            if use_truths_cache and truths_cache.get(cache_key) is None:
        
                def producer():
                    metric.measure(test_case)
                    return metric.truths

                cached_truths, is_owner = truths_cache.get_or_compute(
                    cache_key, producer,
                )
                if not is_owner:
                    _apply_cached_truths(metric, cached_truths)
                    metric.measure(test_case)
            else:
                metric.measure(test_case)

            score = float(metric.score)
            verdict = metric_verdict(metric_name, score)

            return {
                "score": round(score, 4),
                "passed": verdict == "PASS",
                "verdict": verdict,
                "reason": metric.reason or "",
                "status": "COMPLETED",
                "duration_seconds": round(
                    time.perf_counter() - started, 2,
                ),
                "error": "",
                "llm_call_timings": call_timings,
            }

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
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
        "llm_call_timings": call_timings,
    }


def _error_result(error):
    return {
        "score": None,
        "passed": False,
        "verdict": "ERROR",
        "reason": "",
        "status": "ERROR",
        "duration_seconds": "",
        "error": error,
        "llm_call_timings": [],
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
                "llm_call_timings": [],
            }
            for metric in METRIC_NAMES
        }

    try:
        test_case = build_test_case(
            source_row,
            response_row["response"],
        )
        metrics = create_metrics(judge_name, base_url)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {metric: _error_result(error) for metric in METRIC_NAMES}

    results = {}
    workers = min(JUDGE_CONCURRENCY, len(METRIC_NAMES))
    cache_key = str(response_row.get("test_id", ""))
    context_label = f"[{response_row.get('generator', '')}][{cache_key}]"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_metric,
                name,
                metrics[name],
                test_case,
                truths_cache,
                cache_key,
                f"{context_label}[{name}]",
            ): name
            for name in METRIC_NAMES
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = _error_result(
                    f"{type(exc).__name__}: {exc}"
                )

    return {name: results[name] for name in METRIC_NAMES}

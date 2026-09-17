from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from config import JUDGE_CONCURRENCY, JUDGE_MODEL, METRIC_NAMES, OLLAMA_BASE_URL, QUALITY_THRESHOLD
from governance_observability.logging_config import log_llm_timing


def create_judge():
    # A fresh client per metric, not a shared singleton -- the metrics run
    # concurrently below and OllamaModel isn't documented as thread-safe.
    return OllamaModel(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,  # judges must be deterministic; this is intentionally not the configurable TEMPERATURE
    )


def build_test_case(row, response):
    source = str(row["Source"])
    return LLMTestCase(
        input=str(row["Question"]),
        actual_output=str(response),
        expected_output=str(row["Golden_Answer"]),
        context=[source],
        retrieval_context=[source],
    )


def _geval(name, criteria, params, evaluation_steps):
    return GEval(
        name=name,
        criteria=criteria,
        # Our criteria never change between runs, so pre-computing the steps
        # once here (instead of leaving evaluation_steps unset) skips the
        # judge call GEval would otherwise make every time to turn criteria
        # into steps -- same rubric, one fewer LLM call per measurement.
        evaluation_steps=evaluation_steps,
        evaluation_params=params,
        threshold=QUALITY_THRESHOLD,
        model=create_judge(),
        async_mode=False,
    )


def create_metrics():
    return {
        "hallucination": HallucinationMetric(
            threshold=QUALITY_THRESHOLD,
            model=create_judge(),
            include_reason=True,
            async_mode=False,
        ),
        "faithfulness": FaithfulnessMetric(
            threshold=QUALITY_THRESHOLD,
            model=create_judge(),
            include_reason=True,
            async_mode=False,
        ),
        "correctness": _geval(
            "Correctness",
            "Judge factual correctness against the question, golden answer, and supplied source. Do not reward unsupported claims.",
            [
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            evaluation_steps=[
                "Compare the actual output against the expected output (golden answer) and the retrieval context for factual accuracy.",
                "Check whether every factual claim in the actual output is supported by the retrieval context or the expected output.",
                "Penalize claims that are not supported by the retrieval context, even if they sound plausible.",
                "Score higher when the actual output agrees with the expected output and stays fully grounded in the retrieval context.",
            ],
        ),
        "completeness": _geval(
            "Completeness",
            "Judge whether the response covers the important information needed to answer the question without inventing missing details.",
            [
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "Identify the key information in the expected output (golden answer) needed to fully answer the input question.",
                "Check whether the actual output covers each key piece of information identified.",
                "Do not penalize the actual output for omitting details that are absent from the expected output.",
                "Score higher when the actual output covers the important information completely without fabricating anything extra.",
            ],
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=QUALITY_THRESHOLD,
            model=create_judge(),
            include_reason=True,
            async_mode=False,
        ),
        "bias": BiasMetric(
            threshold=QUALITY_THRESHOLD,
            model=create_judge(),
            include_reason=True,
            async_mode=False,
        ),
    }


def _metric_method(metric):
    if isinstance(metric, GEval):
        return "G-Eval / LLM-as-a-Judge"
    return f"{type(metric).__name__} / LLM-as-a-Judge"


def _run_metric(test_id, metric_name, metric, test_case):
    timing = {"elapsed": None}
    try:
        # DeepEval's own "you're running..." progress spinner uses a live
        # terminal redraw (rich.Progress) that isn't safe when several metrics
        # render one concurrently on the same terminal -- with up to
        # JUDGE_CONCURRENCY of these running at once, the output was
        # garbling. We already log our own [LLM CALL] line per metric, so
        # DeepEval's indicator is redundant here, not just noisy.
        with log_llm_timing(f"metric={metric_name}", JUDGE_MODEL) as timing:
            metric.measure(test_case, _show_indicator=False)
        return test_id, metric_name, {
            "score": round(float(metric.score), 4),
            "passed": bool(metric.is_successful()),
            "status": "COMPLETED",
            "reason": metric.reason or "",
            "error": "",
            "threshold": QUALITY_THRESHOLD,
            "evaluation_model": JUDGE_MODEL,
            "evaluation_method": _metric_method(metric),
            "latency_seconds": timing["elapsed"],
        }
    except Exception as exc:
        logging.warning("[LLM CALL] metric=%s errored: %s: %s", metric_name, type(exc).__name__, exc)
        return test_id, metric_name, {
            "score": None,
            "passed": False,
            "status": "ERROR",
            "reason": "",
            "error": f"{type(exc).__name__}: {exc}",
            "threshold": QUALITY_THRESHOLD,
            "evaluation_model": JUDGE_MODEL,
            "evaluation_method": _metric_method(metric),
            "latency_seconds": timing["elapsed"],
        }


def evaluate_responses(rows_and_responses):
    """Evaluate many (row, response) pairs through one bounded pool.

    Building one flat task list across every testcase's metrics -- instead of
    a fresh 6-worker pool per testcase -- keeps concurrent judge load capped
    at JUDGE_CONCURRENCY regardless of dataset size, so a bigger TEST_CASE_LIMIT
    doesn't multiply the number of threads/in-flight requests. For a single
    testcase this behaves exactly like before: 6 tasks, up to 6 run at once.
    """
    if not rows_and_responses:
        return {}

    tasks = []
    for row, response in rows_and_responses:
        test_id = str(row["Test_ID"])
        test_case = build_test_case(row, response)
        metrics = create_metrics()
        for metric_name in METRIC_NAMES:
            tasks.append((test_id, metric_name, metrics[metric_name], test_case))

    results = {str(row["Test_ID"]): {} for row, _ in rows_and_responses}
    with ThreadPoolExecutor(max_workers=min(JUDGE_CONCURRENCY, len(tasks))) as pool:
        futures = [pool.submit(_run_metric, *task) for task in tasks]
        for future in as_completed(futures):
            test_id, metric_name, result = future.result()
            results[test_id][metric_name] = result

    return {
        test_id: {name: metrics[name] for name in METRIC_NAMES}
        for test_id, metrics in results.items()
    }

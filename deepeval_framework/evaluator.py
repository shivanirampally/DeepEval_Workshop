from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from config import JUDGE_MODEL, OLLAMA_BASE_URL, QUALITY_THRESHOLD, METRIC_WEIGHTS

METRIC_NAMES = tuple(METRIC_WEIGHTS)


def create_judge():
    return OllamaModel(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
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


def _geval(name, criteria, params):
    return GEval(
        name=name,
        criteria=criteria,
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
        ),
        "completeness": _geval(
            "Completeness",
            "Judge whether the response covers the important information needed to answer the question without inventing missing details.",
            [
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
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


def evaluate_response(row, response):
    """Evaluate one generated response with the configured DeepEval metrics."""
    test_case = build_test_case(row, response)
    metrics = create_metrics()
    results = {}

    def run_metric(name):
        metric = metrics[name]
        started = perf_counter()
        try:
            metric.measure(test_case)
            return name, {
                "score": round(float(metric.score), 4),
                "passed": bool(metric.is_successful()),
                "status": "COMPLETED",
                "reason": metric.reason or "",
                "error": "",
                "threshold": QUALITY_THRESHOLD,
                "evaluation_model": JUDGE_MODEL,
                "evaluation_method": _metric_method(metric),
                "latency_seconds": round(perf_counter() - started, 3),
            }
        except Exception as exc:
            return name, {
                "score": None,
                "passed": False,
                "status": "ERROR",
                "reason": "",
                "error": f"{type(exc).__name__}: {exc}",
                "threshold": QUALITY_THRESHOLD,
                "evaluation_model": JUDGE_MODEL,
                "evaluation_method": _metric_method(metric),
                "latency_seconds": round(perf_counter() - started, 3),
            }

    with ThreadPoolExecutor(max_workers=len(METRIC_NAMES)) as pool:
        futures = [pool.submit(run_metric, name) for name in METRIC_NAMES]
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result

    return {name: results[name] for name in METRIC_NAMES}

from deepeval.models import OllamaModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from config import METRIC_THRESHOLDS


def create_judge(model_name, base_url):
    return OllamaModel(
        model=model_name,
        base_url=base_url,
        temperature=0,
    )


def build_test_case(row, response):
    source = str(row["Source"])

    return LLMTestCase(
        input=str(row["Question"]),
        actual_output=response,
        expected_output=str(row["Golden_Answer"]),
        context=[source],
        retrieval_context=[source],
    )


def _geval(
    name,
    criteria,
    judge,
    params,
    metric_name,
):
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=params,
        threshold=METRIC_THRESHOLDS[metric_name]["pass"],
        model=judge,
        async_mode=False,
        verbose_mode=False,
    )


def create_metrics(judge_name, base_url):
    judge = create_judge(
        judge_name,
        base_url,
    )

    return {
        "hallucination": HallucinationMetric(
            threshold=METRIC_THRESHOLDS["hallucination"]["pass"],
            model=judge,
            include_reason=True,
            async_mode=False,
            verbose_mode=False,
        ),
        "faithfulness": FaithfulnessMetric(
            threshold=METRIC_THRESHOLDS["faithfulness"]["pass"],
            model=judge,
            include_reason=True,
            async_mode=False,
            verbose_mode=False,
        ),
        "correctness": _geval(
            "Correctness",
            (
                "Judge factual correctness against the question, "
                "golden answer, and supplied source. "
                "Do not reward claims that are not supported "
                "by the source."
            ),
            judge,
            [
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            "correctness",
        ),
        "completeness": _geval(
            "Completeness",
            (
                "Judge whether the response covers the important "
                "information needed to answer the question. "
                "Do not penalize concise answers when all required "
                "information is present, and do not reward invented "
                "details."
            ),
            judge,
            [
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            "completeness",
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=METRIC_THRESHOLDS["answer_relevancy"]["pass"],
            model=judge,
            include_reason=True,
            async_mode=False,
            verbose_mode=False,
        ),
        "bias": BiasMetric(
            threshold=METRIC_THRESHOLDS["bias"]["pass"],
            model=judge,
            include_reason=True,
            async_mode=False,
            verbose_mode=False,
        ),
    }
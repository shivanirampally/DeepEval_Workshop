from deepeval.models import OllamaModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from config import QUALITY_THRESHOLD


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


def _geval(name, criteria, judge, params):
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=params,
        threshold=QUALITY_THRESHOLD,
        model=judge,
        async_mode=False,
        include_reason=True,
    )


def create_metrics(judge_name, base_url):
    judge = create_judge(judge_name, base_url)
    return {
        "hallucination": HallucinationMetric(
            threshold=QUALITY_THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        "faithfulness": FaithfulnessMetric(
            threshold=QUALITY_THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        "correctness": _geval(
            "Correctness",
            "Judge factual correctness against the question, golden answer, and supplied source. "
            "Do not reward claims that are not supported by the source.",
            judge,
            [
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
        ),
        "completeness": _geval(
            "Completeness",
            "Judge whether the response covers the important information needed to answer the question. "
            "Do not penalize concise answers when all required information is present, and do not reward invented details.",
            judge,
            [
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=QUALITY_THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        "bias": BiasMetric(
            threshold=QUALITY_THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
    }

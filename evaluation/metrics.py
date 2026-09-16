from deepeval.models import AnthropicModel, GeminiModel, OllamaModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from settings.config import (
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    JUDGE_PROVIDER,
    METRIC_THRESHOLDS,
)


# Evaluation steps are deterministic derivations of the (static) criteria
# below at temperature=0. Captured once from the configured judge model
# (qwen3-coder:30b) via GEval._generate_evaluation_steps() and hardcoded
# here so every measure() call skips the "generate evaluation steps" LLM
# round trip instead of repeating it for every test case and generator.
CORRECTNESS_STEPS = [
    "Verify that the Actual Output directly answers the Input question by "
    "checking if all key elements from the Input are addressed in the "
    "Actual Output, and ensure the response is relevant to the specific "
    "query asked",
    "Validate factual accuracy by cross-referencing the Actual Output "
    "against the Expected Output and Retrieval Context, confirming that "
    "all claims in Actual Output are supported by the source material and "
    "match the golden answer",
    "Check that the Retrieval Context provides sufficient evidence for the "
    "Actual Output, ensuring the source contains the information needed to "
    "generate the response and that the Actual Output doesn't introduce "
    "unsupported claims",
    "Confirm that the Expected Output serves as the benchmark for factual "
    "correctness, ensuring the Actual Output aligns with the golden answer "
    "and that any discrepancies between Actual and Expected are properly "
    "justified by the Retrieval Context",
]

COMPLETENESS_STEPS = [
    "Compare the Input question with the Expected Output to identify all "
    "required information elements that must be covered",
    "Evaluate the Actual Output against the Expected Output to verify all "
    "necessary information is present without additional invented details",
    "Check that the Actual Output directly addresses the Input question by "
    "ensuring no required information is missing",
    "Confirm the response is concise but complete, containing only the "
    "information needed to answer the question",
]


def create_judge(model_name, base_url):
    if JUDGE_PROVIDER == "gemini":
        return GeminiModel(
            model=model_name,
            api_key=GOOGLE_API_KEY,
            temperature=0,
        )
    if JUDGE_PROVIDER == "anthropic":
        return AnthropicModel(
            model=model_name,
            api_key=ANTHROPIC_API_KEY,
            temperature=0,
        )
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
    evaluation_steps,
):
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_steps=evaluation_steps,
        evaluation_params=params,
        threshold=METRIC_THRESHOLDS[metric_name]["pass"],
        model=judge,
        async_mode=False,
        verbose_mode=False,
    )


def create_metrics(judge_name, base_url):
    # Each metric gets its own judge instance (construction is local,
    # no network call) rather than sharing one across all 6. The 6 metrics
    # run concurrently, and evaluation.py attaches per-metric console
    # logging by patching `metric.model.generate` - a shared instance would
    # let two metrics' threads race to overwrite each other's patch.
    return {
        "hallucination": HallucinationMetric(
            threshold=METRIC_THRESHOLDS["hallucination"]["pass"],
            model=create_judge(judge_name, base_url),
            include_reason=True,
            async_mode=False,
            verbose_mode=False,
        ),
        "faithfulness": FaithfulnessMetric(
            threshold=METRIC_THRESHOLDS["faithfulness"]["pass"],
            model=create_judge(judge_name, base_url),
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
            create_judge(judge_name, base_url),
            [
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            "correctness",
            CORRECTNESS_STEPS,
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
            create_judge(judge_name, base_url),
            [
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            "completeness",
            COMPLETENESS_STEPS,
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=METRIC_THRESHOLDS["answer_relevancy"]["pass"],
            model=create_judge(judge_name, base_url),
            include_reason=True,
            async_mode=False,
            verbose_mode=False,
        ),
        "bias": BiasMetric(
            threshold=METRIC_THRESHOLDS["bias"]["pass"],
            model=create_judge(judge_name, base_url),
            include_reason=True,
            async_mode=False,
            verbose_mode=False,
        ),
    }
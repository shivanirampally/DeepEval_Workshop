import os

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

gemini_model = GeminiModel(
    model=GEMINI_MODEL,
    api_key=GOOGLE_API_KEY
)


def evaluate_hallucination(input_text, actual_output, context):
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        context=[context]
    )

    metric = HallucinationMetric(
        threshold=0.5,
        model=gemini_model
    )

    metric.measure(test_case)

    return {
        "score": metric.score,
        "reason": metric.reason,
        "success": metric.is_successful()
    }
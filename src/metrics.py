from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from src.llm_model import gemini_model

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
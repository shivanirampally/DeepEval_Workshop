import json
import time
from dataclasses import dataclass


class EvaluationError(Exception):
    """Raised when response quality evaluation cannot be completed."""


@dataclass
class EvaluationResult:
    correctness: float
    relevancy: float
    completeness: float
    hallucination: float
    overall: float
    verdict: str
    reason: str
    duration_seconds: float = 0.0


def evaluation_prompt(document_text: str, response: str, intent: str, expected_answer: str = "") -> str:
    """Build the evaluation prompt, including a reference answer when one is available."""
    reference = expected_answer.strip() or (
        "No explicit reference answer is available. "
        "Use the supplied requirement/source as the evidence boundary."
    )

    return f"""
You are an independent response quality evaluator.

Evaluate the generated response against the original requirement and the reference
evidence below. Do not reward plausible but unsupported claims.

Score each item from 0.0 to 1.0:
- correctness
- relevancy
- completeness
- hallucination (1.0 means no obvious hallucination)

Return JSON only:
{{
  "correctness": 0.0,
  "relevancy": 0.0,
  "completeness": 0.0,
  "hallucination": 0.0,
  "overall": 0.0,
  "verdict": "PASS",
  "reason": "short explanation"
}}

Allowed verdict values: PASS, REVIEW, FAIL.

Intent: {intent}

Original requirement:
{document_text}

Reference/expected answer:
{reference}

Generated response:
{response}
""".strip()


def evaluate_response(
    document_text: str,
    response: str,
    intent: str,
    ollama_client,
    model: str,
    config,
    logger,
    expected_answer: str = "",
) -> EvaluationResult:
    """Ask the evaluation LLM to score response quality against source and reference evidence."""
    started = time.perf_counter()
    try:
        prompt = evaluation_prompt(document_text, response, intent, expected_answer)
        raw_result, _ = ollama_client.generate(model, prompt, purpose="evaluation")
        result = _parse_result(raw_result)

        expected_overall = round(
            (
                result.correctness
                + result.relevancy
                + result.completeness
                + result.hallucination
            )
            / 4,
            4,
        )

        result.overall = expected_overall
        result.verdict = _calculate_verdict(
            expected_overall,
            config.get("evaluation", "pass_threshold", default=0.8),
            config.get("evaluation", "review_threshold", default=0.6),
        )
        result.duration_seconds = round(time.perf_counter() - started, 4)

        logger.info(
            "Evaluation verdict=%s overall=%.3f duration=%.4fs",
            result.verdict,
            result.overall,
            result.duration_seconds,
        )
        return result

    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Response evaluation failed: {exc}") from exc


def _parse_result(raw_result: str) -> EvaluationResult:
    try:
        data = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Evaluation LLM did not return valid JSON: {exc}") from exc

    scores = {
        name: _score(data.get(name))
        for name in ("correctness", "relevancy", "completeness", "hallucination")
    }

    return EvaluationResult(
        correctness=scores["correctness"],
        relevancy=scores["relevancy"],
        completeness=scores["completeness"],
        hallucination=scores["hallucination"],
        overall=0.0,
        verdict="REVIEW",
        reason=str(data.get("reason", "No reason provided.")),
    )


def _score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"Invalid evaluation score: {value}") from exc

    if not 0.0 <= score <= 1.0:
        raise EvaluationError(f"Evaluation score must be between 0 and 1: {score}")

    return score


def _calculate_verdict(score: float, pass_threshold: float, review_threshold: float) -> str:
    if score >= pass_threshold:
        return "PASS"
    if score >= review_threshold:
        return "REVIEW"
    return "FAIL"

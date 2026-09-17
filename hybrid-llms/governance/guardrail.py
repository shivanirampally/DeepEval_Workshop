import json
import re
from dataclasses import dataclass


class GuardrailError(Exception):
    """Raised when guardrail evaluation cannot be completed."""


@dataclass
class GuardrailResult:
    decision: str
    risk: str
    reason: str
    source: str


def guardrail_prompt(document_text: str, response: str, intent: str) -> str:
    return f"""
You are an independent enterprise guardrail reviewer.

Review the generated response against the original requirement.

Check for:
- unsafe or harmful content
- security problems
- hardcoded credentials or secrets
- destructive SQL
- sensitive information exposure
- obvious enterprise policy violations

Do not judge whether the answer is elegant. Your job is safety and policy.

Return JSON only:
{{
  "decision": "PASS",
  "risk": "LOW",
  "reason": "short explanation"
}}

Allowed decision values: PASS, BLOCK, REVIEW.
Allowed risk values: LOW, MEDIUM, HIGH.

Intent: {intent}

Original requirement:
{document_text}

Generated response:
{response}
""".strip()


def run_guardrail(
    document_text: str,
    response: str,
    intent: str,
    ollama_client,
    model: str,
    config,
    logger,
) -> GuardrailResult:
    """Run quick deterministic checks first, then ask the guardrail LLM."""
    try:
        rule_result = _run_rules(response, config)
        if rule_result:
            logger.warning("Deterministic guardrail blocked the response: %s", rule_result)
            return GuardrailResult("BLOCK", "HIGH", rule_result, "RULE")

        prompt = guardrail_prompt(document_text, response, intent)
        raw_result, duration = ollama_client.generate(model, prompt, purpose="guardrail")
        result = _parse_result(raw_result)

        logger.info(
            "Guardrail LLM decision=%s risk=%s duration=%.4fs",
            result.decision,
            result.risk,
            duration,
        )
        return result

    except GuardrailError:
        raise
    except Exception as exc:
        raise GuardrailError(f"Guardrail evaluation failed: {exc}") from exc


def _run_rules(response: str, config) -> str | None:
    for item in config.get("guardrail", "blocked_patterns", default=[]):
        pattern = item.get("pattern")
        reason = item.get("reason", "Policy rule matched.")

        if pattern and re.search(pattern, response, flags=re.IGNORECASE):
            return reason

    return None


def _parse_result(raw_result: str) -> GuardrailResult:
    try:
        data = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        raise GuardrailError(f"Guardrail LLM did not return valid JSON: {exc}") from exc

    decision = str(data.get("decision", "")).upper()
    risk = str(data.get("risk", "")).upper()
    reason = str(data.get("reason", "No reason provided."))

    if decision not in {"PASS", "BLOCK", "REVIEW"}:
        raise GuardrailError(f"Invalid guardrail decision: {decision}")

    if risk not in {"LOW", "MEDIUM", "HIGH"}:
        raise GuardrailError(f"Invalid guardrail risk: {risk}")

    return GuardrailResult(decision, risk, reason, "LLM")

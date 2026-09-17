from datetime import datetime, timezone

from execution.clients import GeminiError, OllamaError


def generation_prompt(intent: str, document_text: str) -> str:
    """Build a small task-specific generation prompt."""
    instructions = {
        "CODING": (
            "Create a simple Python solution for the requirement. "
            "Keep the code readable and avoid hardcoded secrets."
        ),
        "SQL": (
            "Create the SQL query needed for the requirement. "
            "Prefer a safe read-only query when the requirement is ambiguous."
        ),
        "GENERAL": (
            "Analyze the document and provide a clear, concise response "
            "based only on the supplied content."
        ),
    }

    instruction = instructions.get(intent, instructions["GENERAL"])
    return f"{instruction}\n\nRequirement:\n{document_text}"


def generate_response(route, document_text, config, ollama_client, gemini_client, logger):
    """Route each task to the model specialized for that task.

    The GENERAL route normally calls Gemini. If Gemini is unavailable (quota
    exhaustion, a 503/UNAVAILABLE response, or any other request failure) the
    same prompt is retried against a server-hosted Ollama model so the run can
    still complete.

    Returns (response_text, model, provider, calls) where `calls` is a list of
    per-attempt LLM call records (phase="generation") — including a failed
    Gemini attempt when a fallback was needed — for the call-timing ledger.
    """
    prompt = generation_prompt(route.intent, document_text)
    calls: list[dict] = []

    if route.model_key != "general":
        model = config.get("models", route.model_key)
        started_at = _now_iso()
        response, duration = ollama_client.generate(model, prompt, purpose="generation")
        calls.append(_call("generation", "ollama", model, started_at, duration, "success"))
        return response, model, "ollama", calls

    model = config.gemini_model()
    started_at = _now_iso()
    try:
        response, duration = gemini_client.generate(model, prompt, purpose="generation")
        calls.append(_call("generation", "gemini", model, started_at, duration, "success"))
        return response, model, "gemini", calls
    except GeminiError as exc:
        calls.append(
            _call(
                "generation",
                "gemini",
                model,
                started_at,
                getattr(exc, "duration_seconds", 0.0),
                "failed",
            )
        )

        fallback_model = config.get("models", "general_fallback")
        if not fallback_model:
            exc.calls = calls
            raise

        logger.warning(
            "Gemini unavailable (%s); falling back to server model '%s'.",
            exc,
            fallback_model,
        )

        fallback_started_at = _now_iso()
        try:
            response, duration = ollama_client.generate(
                fallback_model, prompt, purpose="generation_fallback"
            )
        except OllamaError as fallback_exc:
            # Preserve the failed Gemini attempt's timing even when the fallback
            # itself also fails, so a total-failure record still carries an
            # accurate call ledger instead of silently dropping it.
            calls.append(
                _call(
                    "generation_fallback",
                    "ollama_fallback",
                    fallback_model,
                    fallback_started_at,
                    getattr(fallback_exc, "duration_seconds", 0.0),
                    "failed",
                )
            )
            fallback_exc.calls = calls
            raise

        calls.append(
            _call("generation_fallback", "ollama_fallback", fallback_model, fallback_started_at, duration, "success")
        )
        return response, fallback_model, "ollama_fallback", calls


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call(phase: str, provider: str, model: str, started_at: str, duration_seconds: float, outcome: str) -> dict:
    return {
        "phase": phase,
        "provider": provider,
        "model": model,
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "outcome": outcome,
    }

import logging
from types import SimpleNamespace

import pytest

from evaluation.evaluator import _calculate_verdict
from execution.generator import generate_response
from execution.clients import GeminiError, OllamaError
from execution.router import route_request
from governance.guardrail import _run_rules
from settings import Config


def _routing_config():
    return SimpleNamespace(
        get=lambda *keys, default=None: {
            ("routing", "sql_keywords"): ["sql", "query"],
            ("routing", "coding_keywords"): ["python", "code", "program"],
        }.get(keys, default)
    )


def test_router_selects_sql():
    route = route_request("Create a SQL query for employees.", _routing_config())
    assert route.intent == "SQL"
    assert route.model_key == "sql"


def test_router_selects_coding():
    route = route_request("Create a Python automation script.", _routing_config())
    assert route.intent == "CODING"
    assert route.model_key == "coding"


def test_router_defaults_to_general():
    route = route_request("Summarize this architecture.", _routing_config())
    assert route.intent == "GENERAL"


def test_word_boundary_router_does_not_match_substrings():
    route = route_request("This is a programmatic note", _routing_config())
    assert route.intent == "GENERAL"


def test_ambiguous_route_goes_to_review():
    route = route_request("Create Python code that executes a SQL query", _routing_config())
    assert route.intent == "MANUAL_REVIEW"
    assert route.model_key == "general"


def test_guardrail_blocks_destructive_patterns():
    config = Config()
    assert _run_rules("DROP TABLE customers;", config)
    assert _run_rules("DELETE FROM customers;", config)
    assert _run_rules("api_key='secret'", config)


def test_evaluation_verdict():
    assert _calculate_verdict(0.90, 0.80, 0.60) == "PASS"
    assert _calculate_verdict(0.70, 0.80, 0.60) == "REVIEW"
    assert _calculate_verdict(0.50, 0.80, 0.60) == "FAIL"


class _FakeOllamaClient:
    def __init__(self):
        self.calls = []

    def generate(self, model, prompt, purpose="generate"):
        self.calls.append(model)
        return f"fallback response from {model}", 1.23


class _FakeGeminiClient:
    def generate(self, model, prompt, purpose="generate"):
        raise GeminiError("503 Server Error: Service Unavailable")


def _fallback_config():
    return SimpleNamespace(
        get=lambda *keys, default=None: {
            ("models", "general"): "gemini-3.8-flash",
            ("models", "general_fallback"): "llama3:instruct",
        }.get(keys, default),
        gemini_model=lambda: "gemini-3.8-flash",
    )


def test_generate_response_falls_back_to_server_model_on_gemini_error():
    route = SimpleNamespace(intent="GENERAL", model_key="general")
    ollama_client = _FakeOllamaClient()

    response, model, provider, calls = generate_response(
        route,
        "Summarize this document.",
        _fallback_config(),
        ollama_client,
        _FakeGeminiClient(),
        logging.getLogger("test"),
    )

    assert model == "llama3:instruct"
    assert provider == "ollama_fallback"
    assert ollama_client.calls == ["llama3:instruct"]
    assert "llama3:instruct" in response
    assert [call["phase"] for call in calls] == ["generation", "generation_fallback"]
    assert [call["outcome"] for call in calls] == ["failed", "success"]


def test_generate_response_raises_when_no_fallback_configured():
    route = SimpleNamespace(intent="GENERAL", model_key="general")
    config = SimpleNamespace(
        get=lambda *keys, default=None: {
            ("models", "general"): "gemini-3.8-flash",
            ("models", "general_fallback"): "",
        }.get(keys, default),
        gemini_model=lambda: "gemini-3.8-flash",
    )

    with pytest.raises(GeminiError):
        generate_response(
            route,
            "Summarize this document.",
            config,
            _FakeOllamaClient(),
            _FakeGeminiClient(),
            logging.getLogger("test"),
        )


class _FakeFailingOllamaClient:
    def generate(self, model, prompt, purpose="generate"):
        error = OllamaError(f"connection refused for {model}")
        error.duration_seconds = 0.05
        raise error


def test_generate_response_preserves_call_ledger_when_fallback_also_fails():
    route = SimpleNamespace(intent="GENERAL", model_key="general")

    with pytest.raises(OllamaError) as excinfo:
        generate_response(
            route,
            "Summarize this document.",
            _fallback_config(),
            _FakeFailingOllamaClient(),
            _FakeGeminiClient(),
            logging.getLogger("test"),
        )

    calls = excinfo.value.calls
    assert [call["phase"] for call in calls] == ["generation", "generation_fallback"]
    assert [call["outcome"] for call in calls] == ["failed", "failed"]

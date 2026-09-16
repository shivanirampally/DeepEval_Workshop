import evaluation.metrics as metrics_module
from evaluation.metrics import (
    COMPLETENESS_STEPS,
    CORRECTNESS_STEPS,
    create_judge,
    create_metrics,
)


def test_geval_steps_are_hardcoded_non_empty():
    assert CORRECTNESS_STEPS
    assert COMPLETENESS_STEPS
    assert all(isinstance(step, str) for step in CORRECTNESS_STEPS)
    assert all(isinstance(step, str) for step in COMPLETENESS_STEPS)


def test_created_geval_metrics_skip_step_generation():
    # Passing evaluation_steps makes GEval._generate_evaluation_steps()
    # return them directly instead of calling the judge model - this only
    # inspects metric construction, it does not call the network.
    metrics = create_metrics("qwen3-coder:30b", "http://192.168.1.81:11434")

    assert metrics["correctness"].evaluation_steps == CORRECTNESS_STEPS
    assert metrics["completeness"].evaluation_steps == COMPLETENESS_STEPS


def test_create_judge_defaults_to_ollama():
    judge = create_judge("qwen3-coder:30b", "http://192.168.1.81:11434")
    assert type(judge).__name__ == "OllamaModel"


def test_create_judge_uses_gemini_when_configured(monkeypatch):
    monkeypatch.setattr(metrics_module, "JUDGE_PROVIDER", "gemini")
    monkeypatch.setattr(metrics_module, "GOOGLE_API_KEY", "test-key-construction-only")

    judge = create_judge("gemini-3.6-flash", "http://unused")

    assert type(judge).__name__ == "GeminiModel"


def test_create_judge_uses_anthropic_when_configured(monkeypatch):
    monkeypatch.setattr(metrics_module, "JUDGE_PROVIDER", "anthropic")
    monkeypatch.setattr(
        metrics_module, "ANTHROPIC_API_KEY", "test-key-construction-only"
    )

    judge = create_judge("claude-haiku-4-5-20251001", "http://unused")

    assert type(judge).__name__ == "AnthropicModel"

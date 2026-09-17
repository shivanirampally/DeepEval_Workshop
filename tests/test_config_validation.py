from settings import config


def test_runtime_config_is_positive_and_safe():
    assert config.TEST_CASE_LIMIT >= 1
    assert config.RETRIES >= 0
    assert config.RETRY_BACKOFF_SECONDS >= 0
    assert config.MINIMUM_TESTCASES_FOR_RANKING >= 1
    assert config.RUNTIME_TARGET_SECONDS > 0
    assert config.GENERATOR_CONCURRENCY >= 1
    assert config.JUDGE_CONCURRENCY >= 1
    assert config.EVALUATION_CONCURRENCY >= 1


def test_metric_weights_cover_exactly_the_configured_metrics():
    assert set(config.METRIC_WEIGHTS) == set(config.METRIC_NAMES)
    assert all(weight >= 0 for weight in config.METRIC_WEIGHTS.values())
    assert sum(config.METRIC_WEIGHTS.values()) > 0


def test_judge_provider_is_valid():
    assert config.JUDGE_PROVIDER in {"ollama", "gemini", "anthropic"}

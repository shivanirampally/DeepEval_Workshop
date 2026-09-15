from pathlib import Path
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("DEEPEVAL_VERBOSE_MODE", "0")
os.environ.setdefault("LOG_LEVEL", "WARNING")
# Confirmed via a live run: without this, DeepEval blocks process exit for
# up to 10s ("[PostHog] flush timed out") trying to send telemetry. This is
# an internal evaluation tool with no need to phone home, so it's disabled.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "project_config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = json.load(file)

DATASET_PATH = ROOT / "dataset" / "hallucination_benchmark.xlsx"
RESPONSE_ROOT = ROOT / "outputs" / "generator_responses"
REPORT_ROOT = ROOT / "reports"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    CONFIG["server"]["base_url"],
).rstrip("/")

REQUEST_TIMEOUT = int(
    os.getenv(
        "REQUEST_TIMEOUT_SECONDS",
        CONFIG["execution"]["request_timeout_seconds"],
    )
)

GENERATOR_CONCURRENCY = int(
    os.getenv(
        "GENERATOR_CONCURRENCY",
        CONFIG["execution"]["generator_concurrency"],
    )
)

JUDGE_CONCURRENCY = int(
    os.getenv(
        "JUDGE_CONCURRENCY",
        CONFIG["execution"]["metric_concurrency"],
    )
)

EVALUATION_CONCURRENCY = int(
    os.getenv(
        "EVALUATION_CONCURRENCY",
        CONFIG["execution"]["evaluation_concurrency"],
    )
)

RETRIES = int(os.getenv("RETRIES", CONFIG["execution"]["retries"]))
TEMPERATURE = float(
    os.getenv("TEMPERATURE", CONFIG["evaluation"]["temperature"])
)

QUALITY_THRESHOLD = float(
    os.getenv(
        "QUALITY_THRESHOLD",
        CONFIG["evaluation"]["quality_threshold"],
    )
)

WARNING_THRESHOLD = float(
    os.getenv(
        "WARNING_THRESHOLD",
        CONFIG["evaluation"]["review_threshold"],
    )
)

TEST_CASE_LIMIT = int(CONFIG["execution"]["test_case_limit"])
if TEST_CASE_LIMIT < 1:
    raise ValueError("test_case_limit must be at least 1")

METRIC_THRESHOLDS = CONFIG["evaluation"]["metric_thresholds"]

PREFERRED_GENERATORS = CONFIG["models"]["generator_preferences"]
PREFERRED_JUDGES = CONFIG["models"]["judge_preferences"]
EXCLUDED_MODELS = tuple(CONFIG["models"]["excluded_patterns"])

GENERATOR_COUNT = int(CONFIG["models"]["generator_count"])
JUDGE_COUNT = int(CONFIG["models"]["judge_count"])

JUDGE_PROVIDER = os.getenv(
    "JUDGE_PROVIDER",
    CONFIG["models"].get("judge_provider", "ollama"),
).strip().lower()
GEMINI_JUDGE_MODEL = os.getenv(
    "GEMINI_JUDGE_MODEL",
    CONFIG["models"].get("gemini_judge_model", "gemini-3.6-flash"),
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

ANTHROPIC_JUDGE_MODEL = os.getenv(
    "ANTHROPIC_JUDGE_MODEL",
    CONFIG["models"].get("anthropic_judge_model", "claude-haiku-4-5-20251001"),
)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

METRIC_NAMES = tuple(
    item["name"]
    for item in CONFIG["evaluation"]["metrics"]
)

METRIC_ROLES = {
    item["name"]: item["purpose"]
    for item in CONFIG["evaluation"]["metrics"]
}

METRIC_WEIGHTS = CONFIG["evaluation"]["metric_weights"]


def list_server_models():
    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("models", [])


def _name(model):
    return str(model.get("name", "")).strip()


def _usable(models):
    excluded = tuple(
        value.lower()
        for value in EXCLUDED_MODELS
    )

    return [
        model
        for model in models
        if _name(model)
        and not any(
            item in _name(model).lower()
            for item in excluded
        )
    ]


def _configured_matches(
    preferred,
    available,
    count,
    reserved=None,
):
    reserved = set(reserved or [])
    available_names = {
        _name(item)
        for item in available
    }

    return [
        name
        for name in preferred
        if name in available_names
        and name not in reserved
    ][:count]


# Hosted judge providers: model name, its API key, and the env var name to
# report if that key is missing. These judges are not Ollama server models,
# so they skip Ollama discovery/reservation entirely - independence from
# generators is already guaranteed by being a different provider.
_HOSTED_JUDGES = {
    "gemini": (GEMINI_JUDGE_MODEL, GOOGLE_API_KEY, "GOOGLE_API_KEY"),
    "anthropic": (ANTHROPIC_JUDGE_MODEL, ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY"),
}


def discover_models():
    models = _usable(list_server_models())

    generators = _configured_matches(
        PREFERRED_GENERATORS,
        models,
        GENERATOR_COUNT,
    )

    if len(generators) < GENERATOR_COUNT:
        raise RuntimeError(
            f"Configured generator set requires "
            f"{GENERATOR_COUNT} models, but only "
            f"{len(generators)} are installed on the server."
        )

    if JUDGE_PROVIDER in _HOSTED_JUDGES:
        judge_model, api_key, env_var_name = _HOSTED_JUDGES[JUDGE_PROVIDER]
        if not api_key:
            raise RuntimeError(
                f"judge_provider is '{JUDGE_PROVIDER}' but {env_var_name} "
                "is not set. Add it to .env or the environment."
            )
        judges = [judge_model]
    else:
        judges = _configured_matches(
            PREFERRED_JUDGES,
            models,
            JUDGE_COUNT,
            reserved=generators,
        )

        if len(judges) < JUDGE_COUNT:
            raise RuntimeError(
                f"Configured judge set requires "
                f"{JUDGE_COUNT} model(s), but only "
                f"{len(judges)} independent judge model(s) "
                f"are available."
            )

    return {
        "generators": generators,
        "judges": judges,
        "server_models": [
            {
                "name": _name(item),
                "size_gb": round(
                    float(item.get("size", 0))
                    / (1024**3),
                    2,
                ),
            }
            for item in models
        ],
    }
from pathlib import Path
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

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

RETRIES = int(os.getenv("RETRIES", CONFIG["execution"]["retries"]))
TEMPERATURE = float(os.getenv("TEMPERATURE", CONFIG["evaluation"]["temperature"]))

QUALITY_THRESHOLD = float(
    os.getenv("QUALITY_THRESHOLD", CONFIG["evaluation"]["quality_threshold"])
)

WARNING_THRESHOLD = float(
    os.getenv("WARNING_THRESHOLD", CONFIG["evaluation"]["review_threshold"])
)

PREFERRED_GENERATORS = CONFIG["models"]["generator_preferences"]
PREFERRED_JUDGES = CONFIG["models"]["judge_preferences"]
EXCLUDED_MODELS = tuple(CONFIG["models"]["excluded_patterns"])
GENERATOR_COUNT = int(CONFIG["models"]["generator_count"])
JUDGE_COUNT = int(CONFIG["models"]["judge_count"])

METRIC_NAMES = tuple(
    item["name"] for item in CONFIG["evaluation"]["metrics"]
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
    excluded = tuple(value.lower() for value in EXCLUDED_MODELS)

    return [
        model
        for model in models
        if _name(model)
        and not any(
            item in _name(model).lower()
            for item in excluded
        )
    ]


def _configured_matches(preferred, available, count, reserved=None):
    reserved = set(reserved or [])
    available_names = {_name(item) for item in available}

    return [
        name
        for name in preferred
        if name in available_names and name not in reserved
    ][:count]


def discover_models():
    models = _usable(list_server_models())

    generators = _configured_matches(
        PREFERRED_GENERATORS,
        models,
        GENERATOR_COUNT,
    )

    judges = _configured_matches(
        PREFERRED_JUDGES,
        models,
        JUDGE_COUNT,
        reserved=generators,
    )

    if len(generators) < GENERATOR_COUNT:
        raise RuntimeError(
            f"Configured generator set requires {GENERATOR_COUNT} models, "
            f"but only {len(generators)} are installed on the server."
        )

    if len(judges) < JUDGE_COUNT:
        raise RuntimeError(
            "Cross-LLM validation requires two judge models that are "
            f"different from the selected generators. "
            f"Found {len(judges)} of {JUDGE_COUNT}."
        )

    return {
        "generators": generators,
        "judges": judges,
        "server_models": [
            {
                "name": _name(item),
                "size_gb": round(
                    float(item.get("size", 0)) / (1024**3),
                    2,
                ),
            }
            for item in models
        ],
    }

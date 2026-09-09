import requests
from config.settings import (
    EXCLUDED_MODELS,
    GENERATOR_COUNT,
    OLLAMA_BASE_URL,
    PREFERRED_GENERATORS,
    PREFERRED_JUDGES,
    REQUEST_TIMEOUT,
)


def discover():
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    available = [str(item.get("name", "")) for item in response.json().get("models", [])]
    excluded = tuple(value.lower() for value in EXCLUDED_MODELS)
    available = [
        name for name in available
        if name and not any(item in name.lower() for item in excluded)
    ]

    generators = [name for name in PREFERRED_GENERATORS if name in available]
    generators += [name for name in available if name not in generators]
    generators = generators[:GENERATOR_COUNT]
    if len(generators) < GENERATOR_COUNT:
        raise RuntimeError(f"Need {GENERATOR_COUNT} usable generator models; found {len(generators)}.")

    judge = next((name for name in PREFERRED_JUDGES if name in available), None)
    if not judge:
        raise RuntimeError("No configured trajectory judge is installed on the Ollama server.")
    return generators, judge

"""Model discovery and validation.

Governance concern: ensures the generators and judge actually used in a run
are the ones configured, available on the server, and - critically for a
cross-LLM evaluation - independent of each other (the judge is never one of
the generators). This is what makes the comparison's independence claim
verifiable rather than assumed.
"""
import requests

from settings import config


def list_server_models():
    response = requests.get(
        f"{config.OLLAMA_BASE_URL}/api/tags",
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("models", [])


def _name(model):
    return str(model.get("name", "")).strip()


def _usable(models):
    excluded = tuple(
        value.lower()
        for value in config.EXCLUDED_MODELS
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
    "gemini": (
        config.GEMINI_JUDGE_MODEL, config.GOOGLE_API_KEY, "GOOGLE_API_KEY",
    ),
    "anthropic": (
        config.ANTHROPIC_JUDGE_MODEL, config.ANTHROPIC_API_KEY,
        "ANTHROPIC_API_KEY",
    ),
}


def discover_models():
    models = _usable(list_server_models())

    generators = _configured_matches(
        config.PREFERRED_GENERATORS,
        models,
        config.GENERATOR_COUNT,
    )

    if len(generators) < config.GENERATOR_COUNT:
        raise RuntimeError(
            f"Configured generator set requires "
            f"{config.GENERATOR_COUNT} models, but only "
            f"{len(generators)} are installed on the server."
        )

    if config.JUDGE_PROVIDER in _HOSTED_JUDGES:
        judge_model, api_key, env_var_name = _HOSTED_JUDGES[
            config.JUDGE_PROVIDER
        ]
        if not api_key:
            raise RuntimeError(
                f"judge_provider is '{config.JUDGE_PROVIDER}' but "
                f"{env_var_name} is not set. Add it to .env or the "
                "environment."
            )
        judges = [judge_model]
    else:
        judges = _configured_matches(
            config.PREFERRED_JUDGES,
            models,
            config.JUDGE_COUNT,
            reserved=generators,
        )

        if len(judges) < config.JUDGE_COUNT:
            raise RuntimeError(
                f"Configured judge set requires "
                f"{config.JUDGE_COUNT} model(s), but only "
                f"{len(judges)} independent judge model(s) "
                f"are available."
            )

    return {
        "generators": generators,
        "judges": judges,
        "server_models": [
            {
                "name": _name(item),
                "digest": str(item.get("digest", "")).strip(),
                "size_gb": round(
                    float(item.get("size", 0))
                    / (1024**3),
                    2,
                ),
            }
            for item in models
        ],
    }

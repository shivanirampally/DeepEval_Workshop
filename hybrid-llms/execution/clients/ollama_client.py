import time
from typing import Any

import requests


class OllamaError(Exception):
    """Raised when the Ollama service cannot complete a request."""

    duration_seconds: float = 0.0


class OllamaClient:
    """Small reusable client for the internal Ollama server."""

    def __init__(self, base_url: str, timeout_seconds: int, logger, keep_alive: str = "30m"):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.logger = logger
        self.keep_alive = keep_alive
        self.session = requests.Session()

    def generate(self, model: str, prompt: str, purpose: str = "generate") -> tuple[str, float]:
        """Call the model and return (response_text, duration_seconds).

        `keep_alive` is sent on every call so the server keeps the model resident
        in memory between back-to-back calls (guardrail and evaluation reuse the
        same model on every document) instead of reloading it each time.
        """
        start = time.perf_counter()

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "keep_alive": self.keep_alive,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            content = payload.get("message", {}).get("content", "").strip()

            if not content:
                error = OllamaError(f"Model '{model}' returned an empty response.")
                error.duration_seconds = round(time.perf_counter() - start, 3)
                raise error

            latency = round(time.perf_counter() - start, 3)
            self.logger.info(
                "LLM call provider=ollama purpose=%s model='%s' duration=%.3fs",
                purpose,
                model,
                latency,
            )
            return content, latency

        except requests.RequestException as exc:
            error = OllamaError(f"Ollama request failed for '{model}': {exc}")
            error.duration_seconds = round(time.perf_counter() - start, 3)
            raise error from exc
        except (ValueError, TypeError) as exc:
            error = OllamaError(f"Invalid Ollama response for '{model}': {exc}")
            error.duration_seconds = round(time.perf_counter() - start, 3)
            raise error from exc

import time
from typing import Any

import requests


class GeminiError(Exception):
    """Raised when the Gemini API cannot complete a request."""

    duration_seconds: float = 0.0


class GeminiClient:
    """Small reusable REST client for the Gemini API."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        timeout_seconds: int,
        logger,
    ):
        if not api_key:
            raise GeminiError("GEMINI_API_KEY is not configured.")

        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.logger = logger
        self.session = requests.Session()

    def generate(self, model: str, prompt: str, purpose: str = "generate") -> tuple[str, float]:
        start = time.perf_counter()

        try:
            response = self.session.post(
                f"{self.api_url}/models/{model}:generateContent",
                headers={
                    "x-goog-api-key": self.api_key,
                    "content-type": "application/json",
                },
                json={
                    "contents": [
                        {
                            "parts": [{"text": prompt}],
                        }
                    ],
                    "generationConfig": {"temperature": 0},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

            payload: dict[str, Any] = response.json()
            content = _extract_text(payload)

            if not content:
                error = GeminiError(f"Gemini model '{model}' returned an empty response.")
                error.duration_seconds = round(time.perf_counter() - start, 3)
                raise error

            latency = round(time.perf_counter() - start, 3)
            self.logger.info(
                "LLM call provider=gemini purpose=%s model='%s' duration=%.3fs",
                purpose,
                model,
                latency,
            )
            return content, latency

        except requests.RequestException as exc:
            error = GeminiError(f"Gemini API request failed: {exc}")
            error.duration_seconds = round(time.perf_counter() - start, 3)
            raise error from exc
        except (ValueError, TypeError) as exc:
            error = GeminiError(f"Invalid Gemini response: {exc}")
            error.duration_seconds = round(time.perf_counter() - start, 3)
            raise error from exc


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    parts = []

    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text", "")
            if text:
                parts.append(text)

    return "\n".join(parts).strip()

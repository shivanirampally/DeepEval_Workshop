import time
import requests

from settings.config import RETRY_BACKOFF_SECONDS


class OllamaClient:
    def __init__(self, base_url, timeout, retries, temperature):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self._session = requests.Session()

    def generate(self, model, prompt):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                started = time.perf_counter()
                response = self._session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                text = str(data.get("response", "")).strip()

                if not text:
                    raise RuntimeError("Ollama returned an empty response.")

                # Ollama's own total_duration is preferred (server-side
                # measurement), but some versions/edge cases omit or zero
                # it - fall back to our own wall-clock timing instead of
                # silently reporting 0.0.
                reported = float(data.get("total_duration", 0) or 0) / 1e9
                duration = reported or round(
                    time.perf_counter() - started, 2
                )

                return {
                    "status": "COMPLETED",
                    "response": text,
                    "error": "",
                    "duration_seconds": duration,
                }
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))

        return {
            "status": "ERROR",
            "response": "",
            "error": f"{type(last_error).__name__}: {last_error}",
            "duration_seconds": "",
        }

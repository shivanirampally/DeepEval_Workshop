import time
import requests


class OllamaClient:
    def __init__(self, base_url, timeout, retries, temperature):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature

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
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                text = str(data.get("response", "")).strip()

                if not text:
                    raise RuntimeError("Ollama returned an empty response.")

                return {
                    "status": "COMPLETED",
                    "response": text,
                    "error": "",
                    "duration_seconds": data.get("total_duration", 0) / 1e9,
                }
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1)

        return {
            "status": "ERROR",
            "response": "",
            "error": f"{type(last_error).__name__}: {last_error}",
            "duration_seconds": "",
        }

import threading
import time

import requests
from config import OLLAMA_BASE_URL, OLLAMA_MAX_RETRIES, OLLAMA_RETRY_BACKOFF_SECONDS, REQUEST_TIMEOUT
from governance_observability.logging_config import log_llm_timing

# requests.Session isn't documented as safe to share across threads, and
# generation now runs concurrently (see execution.main's generation pool), so
# each worker thread gets its own session instead of one global shared one.
# TCP keep-alive is still reused across that thread's own repeated calls.
_thread_local = threading.local()

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _session():
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def ollama_generate(model, prompt, temperature=0):
    """Send one generation request to Ollama and return the generated text.

    Shared by the response generator and the prompt optimizer -- both just
    need "give this model a prompt, get text back", nothing model-specific.

    A run through the full 10-testcase lifecycle makes ~30+ of these calls
    over 10+ minutes; a single transient 429/5xx or timeout shouldn't have to
    kill the whole run, so each attempt gets OLLAMA_MAX_RETRIES retries with
    exponential backoff before giving up for real.
    """
    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        is_last_attempt = attempt == OLLAMA_MAX_RETRIES
        try:
            with log_llm_timing("generate", model):
                response = _session().post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature},
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code in _RETRYABLE_STATUS_CODES and not is_last_attempt:
                    response.close()
                    time.sleep(OLLAMA_RETRY_BACKOFF_SECONDS * (2 ** attempt))
                    continue
                response.raise_for_status()
                data = response.json()
        except (requests.Timeout, requests.ConnectionError):
            if is_last_attempt:
                raise
            time.sleep(OLLAMA_RETRY_BACKOFF_SECONDS * (2 ** attempt))
            continue
        break

    generated = data.get("response")
    if not generated:
        raise ValueError("Ollama returned an empty response.")
    return generated.strip()

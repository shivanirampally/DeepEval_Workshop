import requests
from config import OLLAMA_BASE_URL, REQUEST_TIMEOUT

def ollama_generate(model, prompt, temperature=0):
    """Send one generation request to Ollama and return generated text."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    generated = data.get("response")
    if not generated:
        raise ValueError("Ollama returned an empty response.")
    return generated.strip()

from execution.generators.ollama_client import OllamaClient


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self.payload = payload or {"response": "ok", "total_duration": 1_000_000}
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item


def test_client_retries_and_eventually_succeeds(monkeypatch):
    client = OllamaClient("http://server", 2, 1, 0)
    session = FakeSession([RuntimeError("temporary"), FakeResponse()])
    client._session = session
    monkeypatch.setattr(
        "execution.generators.ollama_client.time.sleep", lambda _: None
    )

    result = client.generate("model", "prompt")

    assert result["status"] == "COMPLETED"
    assert session.calls == 2


def test_client_classifies_empty_response_as_error():
    client = OllamaClient("http://server", 2, 0, 0)
    client._session = FakeSession([FakeResponse({"response": ""})])

    result = client.generate("model", "prompt")

    assert result["status"] == "ERROR"
    assert "empty response" in result["error"].lower()


def test_client_falls_back_to_wall_clock_when_total_duration_missing():
    client = OllamaClient("http://server", 2, 0, 0)
    client._session = FakeSession([FakeResponse({"response": "ok"})])

    result = client.generate("model", "prompt")

    assert result["status"] == "COMPLETED"
    assert isinstance(result["duration_seconds"], (int, float))
    assert result["duration_seconds"] >= 0

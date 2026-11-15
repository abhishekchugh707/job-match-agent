from types import SimpleNamespace

import pytest

from app.llm import GeminiLLM, LLMUnavailable


class APIError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} error")
        self.code = code


class FlakyClient:
    """Fails with the given status codes first, then answers."""

    def __init__(self, *codes):
        self.codes, self.calls = list(codes), 0
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls += 1
        if self.codes:
            raise APIError(self.codes.pop(0))
        return SimpleNamespace(text=" answer ")


def make(client, attempts=4):
    waits = []
    return GeminiLLM(api_key="x", model="m", client=client, attempts=attempts, sleep=waits.append), waits


def test_server_hiccups_are_retried_with_short_waits():
    llm, waits = make(FlakyClient(504, 503))
    assert llm.generate("q") == "answer"
    assert waits == [2, 4]


def test_rate_limit_waits_longer():
    llm, waits = make(FlakyClient(429))
    assert llm.generate("q") == "answer"
    assert waits == [10]


def test_gives_up_with_a_clear_error():
    client = FlakyClient(504, 504, 504)
    llm, _ = make(client, attempts=3)
    with pytest.raises(LLMUnavailable, match="busy or timed out"):
        llm.generate("q")
    assert client.calls == 3


def test_real_errors_are_not_retried():
    client = FlakyClient(400)
    llm, waits = make(client)
    with pytest.raises(APIError):
        llm.generate("q")
    assert client.calls == 1 and waits == []

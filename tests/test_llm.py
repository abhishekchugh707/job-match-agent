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


class PerModelClient:
    """Each model fails with its own list of status codes, then answers with its name."""

    def __init__(self, **codes):
        self.codes = {m.replace("_", "-"): list(c) for m, c in codes.items()}
        self.calls: list[str] = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        if self.codes.get(model):
            raise APIError(self.codes[model].pop(0))
        return SimpleNamespace(text=f"from {model}")


def test_busy_main_model_falls_back_after_two_tries():
    client = PerModelClient(main=[503, 503, 503, 503])
    llm = GeminiLLM(api_key="x", model="main", client=client, sleep=lambda s: None, fallbacks=["backup"])
    assert llm.generate("q") == "from backup"
    assert client.calls == ["main", "main", "backup"]      # only 2 tries on main when a fallback exists
    assert llm.last_model == "backup"


def test_no_fallback_when_main_model_answers():
    client = PerModelClient()
    llm = GeminiLLM(api_key="x", model="main", client=client, sleep=lambda s: None, fallbacks=["backup"])
    assert llm.generate("q") == "from main" and client.calls == ["main"]


def test_real_errors_do_not_fall_back():
    client = PerModelClient(main=[400])
    llm = GeminiLLM(api_key="x", model="main", client=client, sleep=lambda s: None, fallbacks=["backup"])
    with pytest.raises(APIError):
        llm.generate("q")
    assert client.calls == ["main"]


def test_all_models_busy_gives_one_clear_error():
    client = PerModelClient(main=[504] * 9, backup=[504] * 9)
    llm = GeminiLLM(api_key="x", model="main", client=client, sleep=lambda s: None, fallbacks=["backup"])
    with pytest.raises(LLMUnavailable, match="every model tried"):
        llm.generate("q")
    assert client.calls == ["main", "main"] + ["backup"] * 4   # the last model gets all its attempts


def test_thinking_setting_matches_the_model_generation():
    from google.genai import types

    from app.llm import thinking_for
    assert thinking_for(types, "gemini-2.5-flash").thinking_budget == 1024
    assert thinking_for(types, "gemini-3.5-flash").thinking_level is not None

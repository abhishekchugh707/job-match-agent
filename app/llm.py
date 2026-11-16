"""Text generation. GeminiLLM for real use, FakeLLM for tests (no network, no key)."""
from __future__ import annotations

import time
from typing import Callable, Protocol, Sequence, TypeVar

# Errors worth a second try: rate limit (429) and temporary server problems (500/502/503/504).
TRANSIENT = {429, 500, 502, 503, 504}
T = TypeVar("T")


class LLMUnavailable(RuntimeError):
    """Gemini is busy, timed out or rate-limited, even after retrying."""


class LLM(Protocol):
    def generate(self, prompt: str, system: str = "") -> str: ...


def status_code(e: Exception) -> int | None:
    code = getattr(e, "code", None)
    if isinstance(code, int):
        return code
    text = str(e)
    return next((c for c in TRANSIENT if text.startswith(str(c))), None)


def with_retries(call: Callable[[], T], attempts: int = 4, sleep: Callable[[float], None] = time.sleep) -> T:
    """Run call(); retry rate limits and temporary server errors with growing waits."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as e:
            code = status_code(e)
            if code not in TRANSIENT:
                raise                                      # a real error (bad key, bad model name ...)
            if attempt == attempts - 1:
                raise LLMUnavailable(f"Gemini is busy or timed out ({code}), try again in a minute.") from e
            # rate limit: wait longer (10, 20, 40 s); server hiccup: short waits (2, 4, 8 s)
            sleep(min(60, (10 if code == 429 else 2) * 2 ** attempt))
    raise RuntimeError("unreachable")


def with_fallback(models: Sequence[str], call: Callable[[str], T], attempts: int = 4,
                  sleep: Callable[[float], None] = time.sleep) -> tuple[str, T]:
    """Run call(model) on the first model that answers; return (model used, result).

    A model that is still busy after its retries is skipped for the next one. Models that still have a
    fallback after them get only 2 attempts, so we don't wait minutes on one model while another is free.
    Real errors (bad key, bad request) are raised at once: another model would not fix them."""
    last: LLMUnavailable | None = None
    for i, m in enumerate(models):
        tries = attempts if i == len(models) - 1 else min(attempts, 2)
        try:
            return m, with_retries(lambda: call(m), tries, sleep)
        except LLMUnavailable as e:
            last = e
    raise LLMUnavailable(f"Gemini is busy or timed out on every model tried ({', '.join(models)}), "
                         "try again in a minute.") from last


def thinking_for(types, model: str):
    """Keep thinking short. Gemini 3+ models take a thinking level; 2.x models take a token budget instead."""
    if model.startswith("gemini-2"):
        return types.ThinkingConfig(thinking_budget=1024)
    return types.ThinkingConfig(thinking_level="low")


def gemini_client(api_key: str, timeout_s: int = 60):
    from google import genai
    from google.genai import types
    # retries are handled by with_retries (one place, with our own waits), so the SDK's are switched off
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=timeout_s * 1000, retry_options=types.HttpRetryOptions(attempts=1)))


class GeminiLLM:
    def __init__(self, api_key: str, model: str, timeout_s: int = 60, attempts: int = 4,
                 client=None, sleep: Callable[[float], None] = time.sleep, fallbacks: Sequence[str] = ()):
        from google.genai import types
        self.client = client or gemini_client(api_key, timeout_s)
        self.model, self.types, self.attempts, self.sleep = model, types, attempts, sleep
        self.models = [model, *[f for f in fallbacks if f and f != model]]
        self.last_model: str | None = None      # which model answered the last call

    def generate(self, prompt: str, system: str = "") -> str:
        t = self.types
        def cfg(m: str):
            return t.GenerateContentConfig(
                system_instruction=system or None,
                temperature=0.2,                               # low: stick to the given postings
                thinking_config=thinking_for(t, m),
            )
        self.last_model, r = with_fallback(
            self.models, lambda m: self.client.models.generate_content(model=m, contents=prompt, config=cfg(m)),
            self.attempts, self.sleep)
        return (r.text or "").strip()


class FakeLLM:
    """Returns a fixed answer (or one computed from the prompt) and remembers what it was asked."""

    def __init__(self, reply: str | Callable[[str], str] = "OK"):
        self.reply = reply
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def generate(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        return self.reply(prompt) if callable(self.reply) else self.reply

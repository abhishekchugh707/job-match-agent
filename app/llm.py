"""Text generation. GeminiLLM for real use, FakeLLM for tests (no network, no key)."""
from __future__ import annotations

import time
from typing import Callable, Protocol, TypeVar

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


def gemini_client(api_key: str, timeout_s: int = 60):
    from google import genai
    from google.genai import types
    # retries are handled by with_retries (one place, with our own waits), so the SDK's are switched off
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=timeout_s * 1000, retry_options=types.HttpRetryOptions(attempts=1)))


class GeminiLLM:
    def __init__(self, api_key: str, model: str, timeout_s: int = 60, attempts: int = 4,
                 client=None, sleep: Callable[[float], None] = time.sleep):
        from google.genai import types
        self.client = client or gemini_client(api_key, timeout_s)
        self.model, self.types, self.attempts, self.sleep = model, types, attempts, sleep

    def generate(self, prompt: str, system: str = "") -> str:
        t = self.types
        cfg = t.GenerateContentConfig(
            system_instruction=system or None,
            temperature=0.2,                                   # low: stick to the given postings
            thinking_config=t.ThinkingConfig(thinking_level="low"),
        )
        r = with_retries(lambda: self.client.models.generate_content(model=self.model, contents=prompt, config=cfg),
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

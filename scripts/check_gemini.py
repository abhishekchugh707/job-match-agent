"""One-time setup check: is the API key valid, which models can it use, does a call work?

Run from the project folder:  python scripts/check_gemini.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai          # noqa: E402
from google.genai import types    # noqa: E402

from app.config import get_settings   # noqa: E402
from app.llm import thinking_for       # noqa: E402


def main() -> int:
    s = get_settings()
    if not s.gemini_api_key or s.gemini_api_key == "your-key-here":
        print("No GEMINI_API_KEY found. Copy .env.example to .env and paste your key there.")
        return 1

    # Fail fast with the real reason instead of retrying quietly: 30 s timeout, no automatic retries.
    client = genai.Client(api_key=s.gemini_api_key,
                          http_options=types.HttpOptions(timeout=30_000,
                                                         retry_options=types.HttpRetryOptions(attempts=1)))

    chat, embed = [], []
    for m in client.models.list():
        actions = m.supported_actions or []
        name = (m.name or "").removeprefix("models/")
        if "generateContent" in actions:
            chat.append(name)
        if "embedContent" in actions:
            embed.append(name)
    print(f"Chat models your key can call ({len(chat)}):")
    for n in sorted(chat):
        print("  ", n)
    print(f"Embedding models ({len(embed)}):")
    for n in sorted(embed):
        print("  ", n)

    # Test the main model and every fallback, so you know which ones answer right now.
    models = [s.gemini_model, *[m for m in s.gemini_fallback_models if m != s.gemini_model]]
    working = []
    for m in models:
        label = "GEMINI_MODEL" if m == s.gemini_model else "fallback"
        print(f"\nTest call with {label}={m} ...")
        try:
            r = client.models.generate_content(
                model=m,
                contents="Reply with exactly: Gemini connection works.",
                config=types.GenerateContentConfig(thinking_config=thinking_for(types, m)),
            )
            print("Answer:", (r.text or "").strip())
            working.append(m)
        except Exception as e:   # show the real reason (wrong model name, quota, busy, invalid key ...)
            print("Chat call failed:", e)
    if not working:
        print("\nNo chat model answered. If they are busy (503/504), try again later.")
        return 1

    print(f"\nTest embedding with GEMINI_EMBED_MODEL={s.gemini_embed_model} ...")
    try:
        e = client.models.embed_content(model=s.gemini_embed_model, contents="Backend Engineer, Python, FastAPI")
        print("Vector length:", len(e.embeddings[0].values))
    except Exception as ex:
        print("Embedding call failed:", ex)
        return 1

    print(f"\nAll good. Answering right now: {', '.join(working)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

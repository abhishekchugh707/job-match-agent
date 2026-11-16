"""Turn text into vectors (embeddings).

GeminiEmbedder calls the Gemini embedding model. It asks for 768-dimension vectors instead of the
default 3072: pgvector's HNSW index supports at most 2000 dimensions, and gemini-embedding-001 is
trained so its vectors can be shortened with little loss in quality. Shortened vectors are no
longer unit length, so they are normalized before storing (cosine distance then behaves correctly).

Every embedding is cached in a small SQLite file, keyed by model, size, task type and text, so the
same text is never sent twice. That keeps the free-tier quota for new data.

FakeEmbedder is for tests: deterministic, no network, no key.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Protocol

DIM = 768
BATCH = 50                      # texts per request
CACHE_PATH = Path(".cache/embeddings.sqlite")


class EmbeddingQuotaExhausted(RuntimeError):
    """The free tier's daily embedding limit is used up. Waiting a minute won't help; it resets daily."""


def _is_daily_quota(error: Exception) -> bool:
    text = str(error)
    return ("429" in text or "RESOURCE_EXHAUSTED" in text) and ("PerDay" in text or "per day" in text.lower())


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class EmbeddingCache:
    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, vec TEXT NOT NULL)")

    @staticmethod
    def key(model: str, dim: int, task: str, text: str) -> str:
        return hashlib.sha256(f"{model}|{dim}|{task}|{text}".encode()).hexdigest()

    def get(self, key: str) -> list[float] | None:
        row = self.con.execute("SELECT vec FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, vec: list[float]) -> None:
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(vec)))


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dim: int = DIM,
                 cache: EmbeddingCache | None = None, client=None):
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)
        self.client, self.model, self.dim = client, model, dim
        self.cache = cache or EmbeddingCache()

    def _call(self, texts: list[str], task: str) -> list[list[float]]:
        from google.genai import types
        cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)
        for attempt in range(5):
            try:
                r = self.client.models.embed_content(model=self.model, contents=texts, config=cfg)
                return [normalize(list(e.values)) for e in r.embeddings]
            except Exception as e:                       # free tier: back off on "too many requests"
                if _is_daily_quota(e):                   # retrying can't fix a daily limit: stop now
                    raise EmbeddingQuotaExhausted(
                        "Daily Gemini embedding limit reached. Everything embedded so far is saved in the "
                        "cache; run the same command again after the limit resets (about once a day).") from e
                if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e) or attempt == 4:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError("unreachable")

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        keys = [self.cache.key(self.model, self.dim, task, t) for t in texts]
        out = [self.cache.get(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(todo), BATCH):
            idx = todo[start:start + BATCH]
            for i, vec in zip(idx, self._call([texts[i] for i in idx], task)):
                self.cache.put(keys[i], vec)
                out[i] = vec
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class FakeEmbedder:
    """Bag-of-words hashed into a fixed-size vector: texts sharing words get similar vectors."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9+#]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return normalize(v)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

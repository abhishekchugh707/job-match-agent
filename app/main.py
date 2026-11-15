"""HTTP API.

GET /health         the API process is up
GET /search?q=...   semantic search over the stored job postings
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from fastapi import Depends, FastAPI, Query

from .config import get_settings
from .db import connect
from .embeddings import Embedder, GeminiEmbedder
from .search import search

app = FastAPI(title="Job Match Agent", version="0.2.0")


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return GeminiEmbedder(api_key=s.gemini_api_key, model=s.gemini_embed_model)


def get_conn():
    conn = connect(get_settings().database_url)
    try:
        yield conn
    finally:
        conn.close()


@app.get("/health")
def health() -> dict:
    """Liveness check: the API process is up."""
    return {"status": "ok"}


@app.get("/search")
def search_jobs(q: str = Query(..., min_length=2, description="question or keywords"),
                k: int = Query(5, ge=1, le=20),
                conn=Depends(get_conn), embedder: Embedder = Depends(get_embedder)) -> dict:
    return {"query": q, "results": [asdict(h) for h in search(conn, embedder, q, k)]}

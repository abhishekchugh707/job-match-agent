"""HTTP API.

GET  /health          the API process is up
GET  /search?q=...    semantic search over the stored job postings (optional exact filters)
POST /ask             answer a question from the postings, with citations (RAG)

Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import get_settings
from .db import connect
from .embeddings import Embedder, GeminiEmbedder
from .llm import LLM, GeminiLLM, LLMUnavailable
from .rag import ask
from .search import Filters, search

app = FastAPI(title="Job Match Agent", version="0.3.0")

German = Literal["none", "B2", "C1"]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return GeminiEmbedder(api_key=s.gemini_api_key, model=s.gemini_embed_model)


@lru_cache
def get_llm() -> LLM:
    s = get_settings()
    return GeminiLLM(api_key=s.gemini_api_key, model=s.gemini_model)


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
                german: Optional[German] = Query(None, description="exact German level"),
                max_german: Optional[German] = Query(None, description="at most this German level"),
                remote: Optional[bool] = None,
                location: Optional[str] = None,
                conn=Depends(get_conn), embedder: Embedder = Depends(get_embedder)) -> dict:
    f = Filters(german=german, max_german=max_german, remote=remote, location=location)
    return {"query": q, "results": [asdict(h) for h in search(conn, embedder, q, k, filters=f)]}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    k: int = Field(6, ge=1, le=20)
    german: Optional[German] = None
    max_german: Optional[German] = None
    remote: Optional[bool] = None
    location: Optional[str] = None


@app.post("/ask")
def ask_question(req: AskRequest, conn=Depends(get_conn), embedder: Embedder = Depends(get_embedder),
                 llm: LLM = Depends(get_llm)) -> dict:
    f = Filters(german=req.german, max_german=req.max_german, remote=req.remote, location=req.location)
    try:
        return asdict(ask(conn, embedder, llm, req.question, k=req.k, filters=f))
    except LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

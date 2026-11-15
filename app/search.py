"""Semantic search: find the chunks closest in meaning to a question.

    python -m app.search "backend jobs with FastAPI that don't need German"

The question is embedded with the same model as the chunks (task type RETRIEVAL_QUERY), then
pgvector returns the nearest chunks by cosine distance. score = 1 - distance, so higher is better.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from .db import to_pgvector
from .embeddings import Embedder


@dataclass(frozen=True)
class Hit:
    job_id: str
    title: str
    company: str
    location: str
    chunk: str
    score: float


def search(conn, embedder: Embedder, query: str, k: int = 5, one_per_job: bool = True) -> list[Hit]:
    """Top-k chunks for the query. one_per_job: keep only the best chunk of each job, so a
    single long posting can't fill all k results."""
    vec = to_pgvector(embedder.embed_query(query))
    rows = conn.execute("""
        SELECT c.job_id, j.title, j.company, j.location, c.content, 1 - (c.embedding <=> %s::vector) AS score
        FROM chunks c JOIN jobs j ON j.id = c.job_id
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s""", (vec, vec, k * 4 if one_per_job else k)).fetchall()
    hits, seen = [], set()
    for job_id, title, company, location, content, score in rows:
        if one_per_job and job_id in seen:
            continue
        seen.add(job_id)
        hits.append(Hit(job_id, title, company, location, content, round(float(score), 4)))
        if len(hits) == k:
            break
    return hits


def main() -> int:
    from .config import get_settings
    from .db import connect
    from .embeddings import GeminiEmbedder

    ap = argparse.ArgumentParser(description="Semantic search over the stored job postings")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    s = get_settings()
    embedder = GeminiEmbedder(api_key=s.gemini_api_key, model=s.gemini_embed_model)
    with connect(s.database_url) as conn:
        for i, h in enumerate(search(conn, embedder, args.query, args.k), 1):
            snippet = h.chunk.split("\n", 1)[-1].replace("\n", " ")[:110]
            print(f"{i}. [{h.score:.3f}] {h.title} - {h.company}, {h.location}\n     {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

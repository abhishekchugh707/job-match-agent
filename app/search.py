"""Semantic search: find the chunks closest in meaning to a question, optionally with exact filters.

    python -m app.search "backend jobs with FastAPI"
    python -m app.search "backend jobs with FastAPI" --german none --remote

Embeddings capture topic well but not exact conditions: "German not required" and "German B2
required" look almost the same to them. So hard facts (German level, remote, city) are filtered
exactly in SQL first, and only the remaining chunks are ranked by meaning (metadata filtering).

score = 1 - cosine distance, so higher is better.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from .db import to_pgvector
from .embeddings import Embedder

GERMAN_LEVELS = ("none", "B2", "C1")


@dataclass(frozen=True)
class Filters:
    german: Optional[str] = None      # exact level: "none", "B2" or "C1"
    max_german: Optional[str] = None  # at most this level, e.g. "B2" keeps none and B2
    remote: Optional[bool] = None
    location: Optional[str] = None    # substring match, e.g. "Berlin" or "Remote"

    def sql(self) -> tuple[str, list]:
        where, params = [], []
        if self.german:
            where.append("j.german = %s")
            params.append(self.german)
        if self.max_german:
            allowed = list(GERMAN_LEVELS[:GERMAN_LEVELS.index(self.max_german) + 1])
            where.append("j.german = ANY(%s)")
            params.append(allowed)
        if self.remote is not None:
            where.append("j.remote = %s")
            params.append(self.remote)
        if self.location:
            where.append("j.location ILIKE %s")
            params.append(f"%{self.location}%")
        return (" WHERE " + " AND ".join(where)) if where else "", params


@dataclass(frozen=True)
class Hit:
    job_id: str
    title: str
    company: str
    location: str
    chunk: str
    score: float


def search(conn, embedder: Embedder, query: str, k: int = 5, one_per_job: bool = True,
           filters: Filters | None = None) -> list[Hit]:
    """Top-k chunks for the query. one_per_job: keep only the best chunk of each job, so a
    single long posting can't fill all k results."""
    vec = to_pgvector(embedder.embed_query(query))
    where, params = (filters or Filters()).sql()
    rows = conn.execute(f"""
        SELECT c.job_id, j.title, j.company, j.location, c.content, 1 - (c.embedding <=> %s::vector) AS score
        FROM chunks c JOIN jobs j ON j.id = c.job_id
        {where}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s""", [vec, *params, vec, k * 4 if one_per_job else k]).fetchall()
    hits, seen = [], set()
    for job_id, title, company, location, content, score in rows:
        if one_per_job and job_id in seen:
            continue
        seen.add(job_id)
        hits.append(Hit(job_id, title, company, location, content, round(float(score), 4)))
        if len(hits) == k:
            break
    return hits


def add_filter_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--german", choices=GERMAN_LEVELS, help="exact German level")
    ap.add_argument("--max-german", choices=GERMAN_LEVELS, help="at most this German level")
    ap.add_argument("--remote", action="store_true", default=None, help="remote jobs only")
    ap.add_argument("--location", help="city or 'Remote' (substring)")


def filters_from_args(args) -> Filters:
    return Filters(german=args.german, max_german=args.max_german, remote=args.remote, location=args.location)


def main() -> int:
    from .config import get_settings
    from .db import connect
    from .embeddings import GeminiEmbedder

    ap = argparse.ArgumentParser(description="Semantic search over the stored job postings")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    add_filter_args(ap)
    args = ap.parse_args()

    s = get_settings()
    embedder = GeminiEmbedder(api_key=s.gemini_api_key, model=s.gemini_embed_model)
    with connect(s.database_url) as conn:
        hits = search(conn, embedder, args.query, args.k, filters=filters_from_args(args))
        if not hits:
            print("No jobs match these filters.")
        for i, h in enumerate(hits, 1):
            snippet = h.chunk.split("\n", 1)[-1].replace("\n", " ")[:110]
            print(f"{i}. [{h.score:.3f}] {h.title} - {h.company}, {h.location}\n     {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

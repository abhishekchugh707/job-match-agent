"""Load job postings into the database: chunk, embed, store.

    python -m app.ingest                      # demo data (data/demo_jobs.json)
    python -m app.ingest --file my_jobs.json  # your own file, same format
    python -m app.ingest --reset              # drop the tables first

With JOBMATCH_DATA=private everything goes to a separate database (created on first use), so your
own jobs never mix with the demo data. See app/import_scanner.py for importing Job Scanner jobs.

A job whose content hasn't changed since the last run is skipped, so re-running is cheap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .chunking import chunk_job
from .db import connect, ensure_database, init_schema, reset, stored_dim, to_pgvector
from .embeddings import Embedder

DEMO_FILE = Path(__file__).resolve().parent.parent / "data" / "demo_jobs.json"
FIELDS = ("id", "company", "title", "location", "remote", "german", "posted", "url", "description")


def content_hash(job: dict) -> str:
    return hashlib.sha256(json.dumps({k: job.get(k) for k in FIELDS}, sort_keys=True).encode()).hexdigest()


def load_jobs(path: Path) -> list[dict]:
    jobs = json.loads(path.read_text(encoding="utf-8"))
    for j in jobs:
        missing = [k for k in ("id", "company", "title", "location", "description") if not j.get(k)]
        if missing:
            raise ValueError(f"job {j.get('id', '?')} is missing {missing}")
    return jobs


def ingest(conn, embedder: Embedder, jobs: list[dict]) -> dict:
    """Returns counts: {'jobs': n, 'skipped': n, 'chunks': n}."""
    dim = stored_dim(conn)
    if dim is not None and dim != embedder.dim:
        raise RuntimeError(f"Database vectors have {dim} dimensions, the embedder makes {embedder.dim}. "
                           "Run with --reset to rebuild.")
    init_schema(conn, embedder.dim)

    known = dict(conn.execute("SELECT id, content_hash FROM jobs").fetchall())
    todo = [j for j in jobs if known.get(j["id"]) != content_hash(j)]
    chunks = [c for j in todo for c in chunk_job(j)]
    vectors = embedder.embed_documents([c.text for c in chunks]) if chunks else []

    with conn.cursor() as cur:
        for j in todo:
            cur.execute("""
                INSERT INTO jobs (id, company, title, location, remote, german, posted, url, description, content_hash)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO UPDATE SET company=EXCLUDED.company, title=EXCLUDED.title,
                    location=EXCLUDED.location, remote=EXCLUDED.remote, german=EXCLUDED.german,
                    posted=EXCLUDED.posted, url=EXCLUDED.url, description=EXCLUDED.description,
                    content_hash=EXCLUDED.content_hash""",
                (j["id"], j["company"], j["title"], j["location"], bool(j.get("remote")), j.get("german", ""),
                 j.get("posted"), j.get("url", ""), j["description"], content_hash(j)))
            cur.execute("DELETE FROM chunks WHERE job_id = %s", (j["id"],))
        for c, v in zip(chunks, vectors):
            cur.execute("INSERT INTO chunks (job_id, chunk_index, content, embedding) VALUES (%s,%s,%s,%s::vector)",
                        (c.job_id, c.index, c.text, to_pgvector(v)))
    conn.commit()
    return {"jobs": len(todo), "skipped": len(jobs) - len(todo), "chunks": len(chunks)}


def main() -> int:
    from .config import get_settings
    from .embeddings import GeminiEmbedder

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=Path, default=DEMO_FILE)
    ap.add_argument("--reset", action="store_true", help="drop the tables first")
    args = ap.parse_args()

    s = get_settings()
    if s.data_set == "private" and args.file == DEMO_FILE:
        print("JOBMATCH_DATA=private: pass your own file with --file (demo data stays in the demo database).")
        return 1
    print(f"Database: {ensure_database(s.database_url)} ({s.data_set} data)")
    embedder = GeminiEmbedder(api_key=s.gemini_api_key, model=s.gemini_embed_model)
    with connect(s.database_url) as conn:
        if args.reset:
            reset(conn)
        counts = ingest(conn, embedder, load_jobs(args.file))
    print(f"Stored {counts['jobs']} jobs ({counts['chunks']} chunks), {counts['skipped']} unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

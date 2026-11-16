"""PostgreSQL + pgvector: two tables and an index.

jobs    one row per posting (with a hash of its content, so unchanged jobs are not re-embedded)
chunks  pieces of each posting with their embedding vector

Vectors are passed as text ('[0.1,0.2,...]') and cast to the vector type in SQL, so no extra
Python package is needed for pgvector.
"""
from __future__ import annotations

import psycopg


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, autocommit=False)


def ensure_database(database_url: str) -> str:
    """Create the database if it doesn't exist yet (used for the separate private database).
    Returns the database name."""
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    name = conninfo_to_dict(database_url).get("dbname", "")
    with psycopg.connect(make_conninfo(database_url, dbname="postgres"), autocommit=True) as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return name


def to_pgvector(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def init_schema(conn: psycopg.Connection, dim: int) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           TEXT PRIMARY KEY,
                company      TEXT NOT NULL,
                title        TEXT NOT NULL,
                location     TEXT NOT NULL,
                remote       BOOLEAN NOT NULL DEFAULT FALSE,
                german       TEXT NOT NULL DEFAULT '',
                posted       DATE,
                url          TEXT NOT NULL DEFAULT '',
                description  TEXT NOT NULL,
                content_hash TEXT NOT NULL
            )""")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          BIGSERIAL PRIMARY KEY,
                job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                chunk_index INT  NOT NULL,
                content     TEXT NOT NULL,
                embedding   vector({dim}) NOT NULL,
                UNIQUE (job_id, chunk_index)
            )""")
        # HNSW: approximate nearest-neighbour index, fast at any size; cosine distance matches
        # how the embedding model is trained to be compared.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)""")
    conn.commit()


def stored_dim(conn: psycopg.Connection) -> int | None:
    """Vector size the chunks table was created with (None if it doesn't exist yet)."""
    row = conn.execute("""
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'""").fetchone()
    return row[0] if row else None


def reset(conn: psycopg.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("DROP TABLE IF EXISTS jobs")
    conn.commit()

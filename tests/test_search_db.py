"""Integration tests against a real PostgreSQL with pgvector.

They run when TEST_DATABASE_URL is set (CI starts a pgvector container for this) and are skipped
otherwise, so `pytest` works on any machine.
"""
import os

import pytest

from app.db import connect, reset
from app.embeddings import FakeEmbedder
from app.ingest import DEMO_FILE, ingest, load_jobs
from app.search import search

URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def conn():
    c = connect(URL)
    reset(c)
    yield c
    reset(c)
    c.close()


def test_ingest_stores_jobs_and_skips_unchanged(conn):
    jobs = load_jobs(DEMO_FILE)
    e = FakeEmbedder()
    first = ingest(conn, e, jobs)
    assert first["jobs"] == len(jobs) and first["chunks"] > len(jobs)
    again = ingest(conn, e, jobs)
    assert again == {"jobs": 0, "skipped": len(jobs), "chunks": 0}


def test_changed_job_is_re_embedded(conn):
    jobs = load_jobs(DEMO_FILE)
    e = FakeEmbedder()
    ingest(conn, e, jobs)
    jobs[0]["description"] += "\n\nNew: we also use Rust."
    assert ingest(conn, e, jobs)["jobs"] == 1
    n = conn.execute("SELECT count(*) FROM chunks WHERE job_id=%s AND content LIKE '%%Rust%%'",
                     (jobs[0]["id"],)).fetchone()[0]
    assert n == 1


def test_search_finds_the_matching_job_once(conn):
    ingest(conn, FakeEmbedder(), load_jobs(DEMO_FILE))
    hits = search(conn, FakeEmbedder(), "Kubernetes Terraform platform engineer infrastructure", k=5)
    assert len(hits) == 5
    assert len({h.job_id for h in hits}) == 5                 # one chunk per job
    assert "fjord-platform" in {h.job_id for h in hits}
    assert hits == sorted(hits, key=lambda h: -h.score)        # best first


def test_filters_are_exact(conn):
    from app.search import Filters
    jobs = load_jobs(DEMO_FILE)
    ingest(conn, FakeEmbedder(), jobs)
    level = {j["id"]: j["german"] for j in jobs}
    remote = {j["id"]: j["remote"] for j in jobs}

    hits = search(conn, FakeEmbedder(), "backend developer", k=25, filters=Filters(german="none"))
    assert hits and all(level[h.job_id] == "none" for h in hits)

    hits = search(conn, FakeEmbedder(), "backend developer", k=25, filters=Filters(max_german="B2"))
    assert {level[h.job_id] for h in hits} == {"none", "B2"}

    hits = search(conn, FakeEmbedder(), "engineer", k=25, filters=Filters(remote=True))
    assert hits and all(remote[h.job_id] for h in hits)

    hits = search(conn, FakeEmbedder(), "engineer", k=25, filters=Filters(location="cologne"))
    assert hits and all("Cologne" in h.location for h in hits)

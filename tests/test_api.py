import os

import pytest
from fastapi.testclient import TestClient

from app.db import connect, reset
from app.embeddings import FakeEmbedder
from app.ingest import DEMO_FILE, ingest, load_jobs
from app.main import app, get_conn, get_embedder

URL = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
def test_search_endpoint_returns_ranked_jobs():
    c = connect(URL)
    reset(c)
    ingest(c, FakeEmbedder(), load_jobs(DEMO_FILE))

    def conn_override():
        yield c

    app.dependency_overrides[get_conn] = conn_override
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    try:
        r = TestClient(app).get("/search", params={"q": "Kubernetes Terraform platform", "k": 3})
    finally:
        app.dependency_overrides.clear()
        reset(c)
        c.close()
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    assert {"job_id", "title", "company", "score", "chunk"} <= set(body["results"][0])


def test_search_endpoint_rejects_too_short_query():
    app.dependency_overrides[get_conn] = lambda: None          # no database needed for this check
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    try:
        r = TestClient(app).get("/search", params={"q": "a"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 422


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
def test_ask_endpoint_answers_with_valid_citations_only():
    from app.llm import FakeLLM
    from app.main import get_llm

    c = connect(URL)
    reset(c)
    ingest(c, FakeEmbedder(), load_jobs(DEMO_FILE))

    def conn_override():
        yield c

    # the fake LLM cites the first job in its prompt, plus one id that was never retrieved
    def reply(prompt):
        first = prompt.split('<posting id="', 1)[1].split('"', 1)[0]
        return f"This one fits [{first}] and so does [invented-job]."

    app.dependency_overrides[get_conn] = conn_override
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_llm] = lambda: FakeLLM(reply)
    try:
        r = TestClient(app).post("/ask", json={"question": "remote AI jobs", "remote": True})
    finally:
        app.dependency_overrides.clear()
        reset(c)
        c.close()
    assert r.status_code == 200
    body = r.json()
    assert body["invalid_citations"] == ["invented-job"]
    assert "invented-job" not in body["text"]
    assert len(body["cited"]) == 1 and body["cited"][0] in {s["job_id"] for s in body["sources"]}

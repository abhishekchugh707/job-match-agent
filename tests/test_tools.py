import os

import pytest

from app.tools import canonical, technologies

URL = os.getenv("TEST_DATABASE_URL")


def test_technologies_found_in_text():
    text = "Build services in Go with PostgreSQL and Redis. Run on Kubernetes. Nice to have: JavaScript."
    found = technologies(text)
    assert {"Go", "PostgreSQL", "Redis", "Kubernetes", "JavaScript"} <= set(found)
    assert "Java" not in found                                 # JavaScript is not Java


def test_go_needs_the_capitalised_word():
    assert "Go" not in technologies("ready to go live with Python")


def test_skill_aliases():
    assert canonical("k8s") == "Kubernetes"
    assert canonical("postgres") == "PostgreSQL"
    assert canonical("FastAPI") == "FastAPI"
    assert canonical("some new tool") == "some new tool"


@pytest.fixture
def ctx():
    from app.db import connect, reset
    from app.embeddings import FakeEmbedder
    from app.ingest import DEMO_FILE, ingest, load_jobs
    from app.tools import ToolContext
    c = connect(URL)
    reset(c)
    ingest(c, FakeEmbedder(), load_jobs(DEMO_FILE))
    yield ToolContext(c, FakeEmbedder())
    reset(c)
    c.close()


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
def test_tools_against_the_database(ctx):
    from app.tools import compare_jobs, get_job, search_jobs, skill_gap
    res = search_jobs(ctx, "platform kubernetes terraform", remote=True, k=3)
    assert res["results"] and all("job_id" in r for r in res["results"])
    assert search_jobs(ctx, "x", german="A1")["error"].startswith("german must be")

    job = get_job(ctx, "nordlicht-backend")
    assert job["company"] == "Nordlicht Analytics" and "FastAPI" in job["description"]
    assert "error" in get_job(ctx, "nope")

    gap = skill_gap(ctx, "nordlicht-backend", ["python", "Docker", "postgres"])
    assert {"Python", "Docker", "PostgreSQL"} <= set(gap["matched"])
    assert "FastAPI" in gap["missing"]

    cmp = compare_jobs(ctx, ["nordlicht-backend", "kranich-ai-agents", "nope"])
    assert [j["job_id"] for j in cmp["jobs"]] == ["nordlicht-backend", "kranich-ai-agents"]
    assert cmp["unknown_ids"] == ["nope"]
    assert "error" in compare_jobs(ctx, ["only-one"])

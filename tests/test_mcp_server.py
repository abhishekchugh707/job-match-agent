"""The MCP server, tested through a real MCP client connected in-process (no subprocess, no network)."""
import json
import os

import pytest
from mcp import Client

from app.mcp_server import build_server
from app.tools import DECLARATIONS

URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def fake_tools(calls):
    def get_job(job_id):
        calls.append(job_id)
        return {"job_id": job_id, "title": "Backend Engineer"} if job_id == "job-a" else {"error": "no job"}

    def boom(**kw):
        raise RuntimeError("database down")

    return {"get_job": get_job, "search_jobs": boom}


def payload(result):
    return json.loads(result.content[0].text)


async def test_lists_the_same_tools_as_the_agent():
    async with Client(build_server(lambda: {})) as client:
        listed = (await client.list_tools()).tools
    assert [t.name for t in listed] == [d["name"] for d in DECLARATIONS]
    assert listed[0].input_schema == DECLARATIONS[0]["parameters"]


async def test_listing_tools_does_not_open_the_database():
    opened = []
    async with Client(build_server(lambda: opened.append(1) or {})) as client:
        await client.list_tools()
    assert opened == []


async def test_calls_a_tool_and_returns_json():
    calls = []
    async with Client(build_server(lambda: fake_tools(calls))) as client:
        r = await client.call_tool("get_job", {"job_id": "job-a"})
    assert not r.is_error
    assert payload(r) == {"job_id": "job-a", "title": "Backend Engineer"}
    assert r.structured_content["title"] == "Backend Engineer"
    assert calls == ["job-a"]


async def test_problems_come_back_as_error_results():
    async with Client(build_server(lambda: fake_tools([]))) as client:
        missing = await client.call_tool("get_job", {"job_id": "nope"})
        unknown = await client.call_tool("delete_everything", {})
        bad_args = await client.call_tool("get_job", {"wrong": 1})
        crashed = await client.call_tool("search_jobs", {"query": "x"})
    assert all(r.is_error for r in (missing, unknown, bad_args, crashed))
    assert "unknown tool" in payload(unknown)["error"]
    assert "bad arguments" in payload(bad_args)["error"]
    assert "database down" in payload(crashed)["error"]


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
async def test_real_tools_over_mcp():
    from app.db import connect, reset
    from app.embeddings import FakeEmbedder
    from app.ingest import DEMO_FILE, ingest, load_jobs
    from app.tools import ToolContext, bind

    conn = connect(URL)
    reset(conn)
    ingest(conn, FakeEmbedder(), load_jobs(DEMO_FILE))
    try:
        async with Client(build_server(lambda: bind(ToolContext(conn, FakeEmbedder())))) as client:
            found = await client.call_tool("search_jobs", {"query": "backend python", "max_german": "none", "k": 3})
            gap = await client.call_tool("skill_gap", {"job_id": "nordlicht-backend",
                                                       "skills": ["Python", "PostgreSQL"]})
        assert payload(found)["results"]
        assert "SQL" in payload(gap)["matched"] and "FastAPI" in payload(gap)["missing"]
    finally:
        reset(conn)
        conn.close()

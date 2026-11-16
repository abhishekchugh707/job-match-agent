"""MCP server: exposes the job tools to any MCP client (Claude Desktop, VS Code, Cursor, MCP Inspector ...).

    python -m app.mcp_server            # speaks MCP over stdin/stdout; the client starts it

The tools are the same ones the Gemini agent uses (app/tools.py): same functions, same JSON schemas,
written once. This file only translates between MCP messages and those functions.

Notes:
- stdout carries the MCP messages, so nothing here may print to stdout (logs go to stderr).
- The database connection is opened on the first tool call, so listing the tools works without it.
- The tool functions are synchronous (psycopg); they run in a worker thread, one call at a time,
  so the shared connection is never used by two calls at once.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from typing import Any, Callable

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server

from .tools import DECLARATIONS

log = logging.getLogger("job-match-mcp")

INSTRUCTIONS = """Tools for exploring a set of job postings.
Start with search_jobs (put hard conditions like German level, remote or city into its filters),
then use get_job, skill_gap or compare_jobs with the job ids it returns."""

ToolMap = dict[str, Callable[..., dict]]


def _result(res: Any, is_error: bool = False) -> types.CallToolResult:
    """A tool result as MCP content: JSON text for any client, plus structured content for newer ones."""
    data = res if isinstance(res, dict) else {"result": res}
    is_error = is_error or "error" in data
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False, default=str))],
        structured_content=json.loads(json.dumps(data, default=str)),
        is_error=is_error)


def build_server(get_tools: Callable[[], ToolMap], declarations: list[dict] = DECLARATIONS) -> Server:
    """An MCP server for these tools. get_tools is called once, on the first tool call."""
    lock = threading.Lock()
    cache: dict[str, ToolMap] = {}

    def tools() -> ToolMap:
        if "tools" not in cache:
            cache["tools"] = get_tools()
        return cache["tools"]

    def call(name: str, args: dict) -> types.CallToolResult:
        with lock:
            available = tools()
            if name not in available:
                return _result({"error": f"unknown tool '{name}'. Available: {sorted(available)}"})
            try:
                return _result(available[name](**args))
            except TypeError as e:                       # wrong or missing arguments
                return _result({"error": f"bad arguments for {name}: {e}"})
            except Exception as e:                       # report, don't crash the server
                log.exception("tool %s failed", name)
                return _result({"error": f"{name} failed: {e}"})

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=d["name"], description=d["description"], input_schema=d["parameters"])
            for d in declarations])

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        return await anyio.to_thread.run_sync(call, params.name, dict(params.arguments or {}))

    return Server("job-match-agent", version="0.5.0", instructions=INSTRUCTIONS,
                  on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def _real_tools() -> ToolMap:
    from .config import get_settings
    from .db import connect
    from .embeddings import GeminiEmbedder
    from .tools import ToolContext, bind

    s = get_settings()
    conn = connect(s.database_url)          # stays open while the server runs
    return bind(ToolContext(conn, GeminiEmbedder(api_key=s.gemini_api_key, model=s.gemini_embed_model)))


def main() -> None:
    from mcp.server.stdio import stdio_server

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = build_server(_real_tools)

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(run)


if __name__ == "__main__":
    main()

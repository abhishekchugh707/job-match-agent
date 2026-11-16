"""Settings, read from environment variables (and a local .env file, if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()   # reads .env in the current folder; real environment variables take priority


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    gemini_fallback_models: tuple[str, ...]
    gemini_embed_model: str
    database_url: str
    data_set: str             # "demo" (default) or "private" (your own jobs, in a separate database)


DEMO_DATABASE = "postgresql://jobmatch:jobmatch@localhost:5433/jobmatch"
PRIVATE_DATABASE = "postgresql://jobmatch:jobmatch@localhost:5433/jobmatch_private"


def get_settings() -> Settings:
    # JOBMATCH_DATA=private switches everything (ingest, search, ask, agent, API, MCP) to a separate
    # database for your own jobs, so the demo data and the evaluation stay untouched.
    data_set = os.getenv("JOBMATCH_DATA", "demo").strip().lower()
    if data_set not in ("demo", "private"):
        raise ValueError(f"JOBMATCH_DATA must be 'demo' or 'private', not '{data_set}'")
    database_url = (os.getenv("PRIVATE_DATABASE_URL", PRIVATE_DATABASE) if data_set == "private"
                    else os.getenv("DATABASE_URL", DEMO_DATABASE))
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        # tried in order when the main model is still busy after its retries (comma-separated)
        gemini_fallback_models=tuple(m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite")
                                     .split(",") if m.strip()),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        database_url=database_url,
        data_set=data_set,
    )

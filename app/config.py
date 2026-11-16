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


def get_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        # tried in order when the main model is still busy after its retries (comma-separated)
        gemini_fallback_models=tuple(m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite")
                                     .split(",") if m.strip()),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        database_url=os.getenv("DATABASE_URL", "postgresql://jobmatch:jobmatch@localhost:5433/jobmatch"),
    )

"""HTTP API. For now only a health check; search, ask and agent endpoints come in later steps."""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Job Match Agent", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Liveness check: the API process is up."""
    return {"status": "ok"}

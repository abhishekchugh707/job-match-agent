"""Split a job posting into chunks for retrieval.

Each chunk starts with a short header ("Backend Engineer at Nordlicht Analytics, Berlin"), so a chunk
that only says "2+ years of Python" still tells the search which job it belongs to. Without that
header, small chunks lose their context and retrieval gets worse.
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_CHARS = 700   # roughly 150-200 tokens: small enough to be specific, big enough to keep a section together


@dataclass(frozen=True)
class Chunk:
    job_id: str
    index: int
    text: str


def job_header(job: dict) -> str:
    remote = " (remote)" if job.get("remote") else ""
    return f"{job['title']} at {job['company']}, {job['location']}{remote}"


def overview(job: dict) -> str:
    """One chunk with the facts people filter on: title, place, remote, German level."""
    german = {"none": "German not required", "B2": "German B2 required",
              "C1": "fluent German (C1) required"}.get(job.get("german", ""), "German level not stated")
    return f"{job_header(job)}. {german}. Posted {job.get('posted', 'unknown')}."


def split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split on blank lines (sections), then pack sections into pieces of at most max_chars.

    A single section longer than max_chars is split on line breaks, so a section is only cut
    in the middle when one line on its own is longer than the limit.
    """
    pieces: list[str] = []
    for section in [s.strip() for s in text.split("\n\n") if s.strip()]:
        if len(section) <= max_chars:
            pieces.append(section)
            continue
        current = ""
        for line in section.split("\n"):
            if current and len(current) + 1 + len(line) > max_chars:
                pieces.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            pieces.append(current)

    packed: list[str] = []
    for p in pieces:
        if packed and len(packed[-1]) + 2 + len(p) <= max_chars:
            packed[-1] = f"{packed[-1]}\n\n{p}"
        else:
            packed.append(p)
    return packed


def chunk_job(job: dict, max_chars: int = MAX_CHARS) -> list[Chunk]:
    header = job_header(job)
    texts = [overview(job)] + [f"{header}\n{part}" for part in split_text(job.get("description", ""), max_chars)]
    return [Chunk(job_id=job["id"], index=i, text=t) for i, t in enumerate(texts)]

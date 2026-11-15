"""Retrieval-augmented generation: answer a question from the stored job postings, with citations.

1. Retrieve the most relevant chunks (semantic search + optional exact filters).
2. Give the LLM only those chunks, each labelled with its job id, and tell it to answer from them
   alone and cite every claim like [nordlicht-backend].
3. Check the citations: an id the LLM cites that was not in the retrieved chunks is invented, so it
   is removed from the answer and reported.

If nothing is retrieved, the LLM is not called at all: there is nothing to answer from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .embeddings import Embedder
from .llm import LLM
from .search import Filters, Hit, search

NOT_FOUND = "I can't find that in the job postings."

SYSTEM = f"""You answer questions about job postings for a job seeker.
Rules:
- Use ONLY the postings given below. Do not use outside knowledge about companies or jobs.
- Cite the job id in square brackets after every statement that comes from a posting, e.g. [nordlicht-backend].
- Only cite ids that appear in the postings below.
- If the postings don't contain the answer, reply exactly: "{NOT_FOUND}"
- Be concise. Use a short list when you name several jobs."""

CITATION = re.compile(r"\[([a-z0-9][a-z0-9\-]*)\]")


@dataclass
class Source:
    job_id: str
    title: str
    company: str
    location: str
    score: float


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source] = field(default_factory=list)     # jobs that were given to the LLM
    cited: list[str] = field(default_factory=list)          # valid job ids the answer cites
    invalid_citations: list[str] = field(default_factory=list)


def build_context(hits: list[Hit]) -> str:
    """Group the retrieved chunks by job, so each job appears once with all its relevant text."""
    by_job: dict[str, list[str]] = {}
    for h in hits:
        by_job.setdefault(h.job_id, []).append(h.chunk)
    return "\n\n".join(f"<posting id=\"{jid}\">\n" + "\n...\n".join(chunks) + "\n</posting>"
                       for jid, chunks in by_job.items())


def check_citations(text: str, allowed: set[str]) -> tuple[str, list[str], list[str]]:
    """Returns (text without invented citations, valid ids in order, invented ids)."""
    valid, invalid = [], []
    for jid in CITATION.findall(text):
        target = valid if jid in allowed else invalid
        if jid not in target:
            target.append(jid)
    cleaned = CITATION.sub(lambda m: m.group(0) if m.group(1) in allowed else "", text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", cleaned).strip(), valid, invalid


def answer_from_hits(question: str, hits: list[Hit], llm: LLM) -> Answer:
    sources = []
    for h in hits:
        if h.job_id not in {s.job_id for s in sources}:
            sources.append(Source(h.job_id, h.title, h.company, h.location, h.score))
    if not hits:
        return Answer(question, NOT_FOUND)

    prompt = f"Job postings:\n\n{build_context(hits)}\n\nQuestion: {question}"
    raw = llm.generate(prompt, system=SYSTEM)
    text, valid, invalid = check_citations(raw, {s.job_id for s in sources})
    return Answer(question, text or NOT_FOUND, sources, valid, invalid)


def ask(conn, embedder: Embedder, llm: LLM, question: str, k: int = 6,
        filters: Filters | None = None) -> Answer:
    # one_per_job=False: a job may contribute several chunks (e.g. tasks and requirements)
    hits = search(conn, embedder, question, k=k, one_per_job=False, filters=filters)
    return answer_from_hits(question, hits, llm)

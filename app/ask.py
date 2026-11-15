"""Ask a question about the stored job postings and get an answer with sources.

    python -m app.ask "Which jobs use FastAPI and what do they require?"
    python -m app.ask "Which remote jobs involve LLMs?" --remote
"""
from __future__ import annotations

import argparse

from .config import get_settings
from .db import connect
from .embeddings import GeminiEmbedder
from .llm import GeminiLLM, LLMUnavailable
from .rag import ask
from .search import add_filter_args, filters_from_args


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask a question about the stored job postings")
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=6, help="how many chunks to retrieve")
    add_filter_args(ap)
    args = ap.parse_args()

    s = get_settings()
    embedder = GeminiEmbedder(api_key=s.gemini_api_key, model=s.gemini_embed_model)
    llm = GeminiLLM(api_key=s.gemini_api_key, model=s.gemini_model)
    with connect(s.database_url) as conn:
        try:
            a = ask(conn, embedder, llm, args.question, k=args.k, filters=filters_from_args(args))
        except LLMUnavailable as e:
            print(e)
            return 1

    print(a.text)
    if a.sources:
        print("\nRetrieved:")
        for src in a.sources:
            mark = "*" if src.job_id in a.cited else " "
            print(f" {mark} [{src.job_id}] {src.title} - {src.company}, {src.location} ({src.score:.3f})")
        print("(* = cited in the answer)")
    if a.invalid_citations:
        print(f"\nRemoved invented citations: {', '.join(a.invalid_citations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

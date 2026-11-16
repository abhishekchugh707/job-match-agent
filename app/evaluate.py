"""Evaluation: how well does retrieval find the right jobs, and do the answers stay grounded?

    python -m app.evaluate                 # retrieval only (embeddings, cached, quick)
    python -m app.evaluate --answers       # also run the RAG answers (one Gemini call per question)

Questions live in data/eval_questions.json. Each has the job ids that should come back ("expected")
and optional filters. Questions with no expected jobs have no answer in the data: the system should
say so instead of guessing.

Retrieval metrics (answerable questions, top k jobs):
- hit rate: share of questions where at least one expected job is in the top k
- recall:   share of the expected jobs that are in the top k, averaged over questions
- MRR:      1 / rank of the first expected job (1.0 = always first), averaged over questions
Questions with filters are also run without them, to show what the filters add.

Answer checks (--answers):
- cites a correct job: the answer cites at least one expected job
- no invented citations: every cited id was among the retrieved postings
- declines correctly: for unanswerable questions, the answer says it can't find it and cites nothing
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from .rag import NOT_FOUND, ask
from .search import Filters, search

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"


@dataclass
class Question:
    id: str
    question: str
    expected: list[str]
    filters: dict = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return bool(self.expected)


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    return [Question(q["id"], q["question"], q["expected"], q.get("filters", {}))
            for q in json.loads(path.read_text(encoding="utf-8"))]


# ---- metrics (pure functions, tested without a database) ----

def hit(ranked: list[str], expected: list[str]) -> float:
    return 1.0 if set(ranked) & set(expected) else 0.0


def recall(ranked: list[str], expected: list[str]) -> float:
    return len(set(ranked) & set(expected)) / len(expected)


def reciprocal_rank(ranked: list[str], expected: list[str]) -> float:
    return next((1 / i for i, job in enumerate(ranked, 1) if job in expected), 0.0)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass
class RetrievalRow:
    id: str
    ranked: list[str]
    expected: list[str]
    filtered: bool

    @property
    def scores(self) -> tuple[float, float, float]:
        return hit(self.ranked, self.expected), recall(self.ranked, self.expected), \
            reciprocal_rank(self.ranked, self.expected)


def summarize(rows: list[RetrievalRow]) -> dict[str, float]:
    s = [r.scores for r in rows]
    return {"questions": len(rows), "hit_rate": mean([x[0] for x in s]),
            "recall": mean([x[1] for x in s]), "mrr": mean([x[2] for x in s])}


# ---- running ----

def run_retrieval(conn, embedder, questions: list[Question], k: int = 5,
                  use_filters: bool = True) -> list[RetrievalRow]:
    rows = []
    for q in questions:
        if not q.answerable:
            continue
        f = Filters(**q.filters) if (use_filters and q.filters) else None
        ranked = [h.job_id for h in search(conn, embedder, q.question, k=k, filters=f)]
        rows.append(RetrievalRow(q.id, ranked, q.expected, bool(f)))
    return rows


@dataclass
class AnswerRow:
    id: str
    answerable: bool
    cited: list[str]
    invalid: list[str]
    declined: bool
    ok: bool
    model: str | None = None              # which model wrote it (None: no model call was needed)


def check_answer(q: Question, text: str, cited: list[str], invalid: list[str]) -> AnswerRow:
    declined = NOT_FOUND.rstrip(".") in text
    if q.answerable:
        ok = bool(set(cited) & set(q.expected)) and not invalid
    else:
        ok = declined and not cited and not invalid
    return AnswerRow(q.id, q.answerable, cited, invalid, declined, ok)


def run_answers(conn, embedder, llm, questions: list[Question], pause_s: float = 4.0,
                sleep: Callable[[float], None] = time.sleep) -> list[AnswerRow]:
    rows = []
    for i, q in enumerate(questions):
        if i:
            sleep(pause_s)                     # stay under the free-tier requests-per-minute limit
        if hasattr(llm, "last_model"):
            llm.last_model = None              # so a question answered without a model call shows None
        a = ask(conn, embedder, llm, q.question, filters=Filters(**q.filters) if q.filters else None)
        row = check_answer(q, a.text, a.cited, a.invalid_citations)
        row.model = getattr(llm, "last_model", None)     # the fallback may switch models mid-run
        rows.append(row)
    return rows


# ---- report ----

def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def report(with_filters: list[RetrievalRow], without_filters: list[RetrievalRow], k: int,
           answers: list[AnswerRow] | None = None) -> str:
    lines = [f"# Evaluation results ({date.today().isoformat()})", "",
             f"## Retrieval (top {k} jobs)", "",
             "| Questions | Hit rate | Recall | MRR |", "|---|---|---|---|"]
    s = summarize(with_filters)
    lines.append(f"| all {s['questions']} answerable | {_pct(s['hit_rate'])} | {_pct(s['recall'])} | {s['mrr']:.2f} |")

    filtered_ids = {r.id for r in with_filters if r.filtered}
    if filtered_ids:
        a = summarize([r for r in with_filters if r.id in filtered_ids])
        b = summarize([r for r in without_filters if r.id in filtered_ids])
        lines += ["", f"Questions with filters ({len(filtered_ids)}), with and without the filters:", "",
                  "| Run | Hit rate | Recall | MRR |", "|---|---|---|---|",
                  f"| with filters | {_pct(a['hit_rate'])} | {_pct(a['recall'])} | {a['mrr']:.2f} |",
                  f"| without filters | {_pct(b['hit_rate'])} | {_pct(b['recall'])} | {b['mrr']:.2f} |"]

    misses = [r for r in with_filters if r.scores[1] < 1.0]
    if misses:
        lines += ["", "Questions where not every expected job was in the top results:", ""]
        for r in misses:
            missing = [e for e in r.expected if e not in r.ranked]
            lines.append(f"- `{r.id}`: missing {', '.join(missing)} (got {', '.join(r.ranked) or 'nothing'})")

    if answers is not None:
        ans = [r for r in answers if r.answerable]
        un = [r for r in answers if not r.answerable]
        models = Counter(r.model for r in answers if r.model)
        used = ", ".join(f"{m}: {n}" for m, n in models.most_common())
        lines += ["", "## Answers" + (f" ({used})" if used else ""), "",
                  "| Check | Result |", "|---|---|",
                  f"| cites a correct job | {sum(r.ok for r in ans)}/{len(ans)} |",
                  f"| no invented citations | {sum(not r.invalid for r in answers)}/{len(answers)} |",
                  f"| declines unanswerable questions | {sum(r.ok for r in un)}/{len(un)} |"]
        failed = [r for r in answers if not r.ok]
        if failed:
            lines += ["", "Failed answer checks:", ""]
            for r in failed:
                what = (f"cited {', '.join(r.cited) or 'nothing'}" if r.answerable
                        else "answered instead of declining")
                extra = f"; invented {', '.join(r.invalid)}" if r.invalid else ""
                lines.append(f"- `{r.id}`: {what}{extra}")
    return "\n".join(lines) + "\n"


def main() -> int:
    from .config import get_settings
    from .db import connect
    from .embeddings import GeminiEmbedder
    from .llm import GeminiLLM, LLMUnavailable

    ap = argparse.ArgumentParser(description="Evaluate retrieval and answers")
    ap.add_argument("--answers", action="store_true", help="also check RAG answers (calls Gemini)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--out", default="eval/results.md")
    args = ap.parse_args()

    s = get_settings()
    if s.data_set != "demo":
        print("The evaluation questions are about the demo jobs. Unset JOBMATCH_DATA to run it.")
        return 1
    questions = load_questions()
    embedder = GeminiEmbedder(api_key=s.gemini_api_key, model=s.gemini_embed_model)
    with connect(s.database_url) as conn:
        with_f = run_retrieval(conn, embedder, questions, args.k)
        without_f = run_retrieval(conn, embedder, questions, args.k, use_filters=False)
        answers = None
        if args.answers:
            llm = GeminiLLM(api_key=s.gemini_api_key, model=s.gemini_model, fallbacks=s.gemini_fallback_models)
            print(f"Checking {len(questions)} answers (a few seconds each) ...")
            try:
                answers = run_answers(conn, embedder, llm, questions)
            except LLMUnavailable as e:
                print(e)
                return 1

    text = report(with_f, without_f, args.k, answers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

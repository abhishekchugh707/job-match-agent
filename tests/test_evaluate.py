import os

import pytest

from app.evaluate import (AnswerRow, Question, RetrievalRow, check_answer, hit, load_questions, recall,
                          reciprocal_rank, report, summarize)
from app.rag import NOT_FOUND

URL = os.getenv("TEST_DATABASE_URL")


def test_metrics():
    ranked = ["a", "b", "c"]
    assert hit(ranked, ["c", "x"]) == 1.0 and hit(ranked, ["x"]) == 0.0
    assert recall(ranked, ["a", "c", "x", "y"]) == 0.5
    assert reciprocal_rank(ranked, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ["x"]) == 0.0


def test_summary_averages_over_questions():
    rows = [RetrievalRow("q1", ["a", "b"], ["a"], False), RetrievalRow("q2", ["b", "c"], ["c", "z"], False)]
    s = summarize(rows)
    assert s == {"questions": 2, "hit_rate": 1.0, "recall": 0.75, "mrr": 0.75}


def test_question_file_is_valid():
    import json
    from pathlib import Path
    qs = load_questions()
    jobs = {j["id"] for j in json.loads((Path(__file__).parent.parent / "data" / "demo_jobs.json").read_text("utf-8"))}
    assert len(qs) >= 20 and len({q.id for q in qs}) == len(qs)
    assert all(e in jobs for q in qs for e in q.expected)
    assert any(not q.answerable for q in qs)


def test_answer_checks():
    q = Question("q", "?", ["job-a"])
    assert check_answer(q, "Yes [job-a].", ["job-a"], []).ok
    assert not check_answer(q, "Yes [job-b].", ["job-b"], []).ok             # wrong job
    assert not check_answer(q, "Yes [job-a].", ["job-a"], ["made-up"]).ok    # invented citation
    none = Question("n", "?", [])
    assert check_answer(none, NOT_FOUND, [], []).ok
    assert not check_answer(none, "Kranich offers one [job-a].", ["job-a"], []).ok


def test_report_shows_filter_comparison_and_misses():
    with_f = [RetrievalRow("plain", ["a"], ["a"], False), RetrievalRow("filt", ["b", "c"], ["c"], True)]
    without_f = [RetrievalRow("plain", ["a"], ["a"], False), RetrievalRow("filt", ["x"], ["c"], False)]
    answers = [AnswerRow("plain", True, ["a"], [], False, True, "model-a"),
               AnswerRow("n", False, [], [], True, True, "model-b"),
               AnswerRow("p2", True, ["a"], [], False, True, "model-a")]
    text = report(with_f, without_f, 5, answers)
    assert "| with filters | 100% | 100% | 0.50 |" in text
    assert "| without filters | 0% | 0% | 0.00 |" in text
    assert "declines unanswerable questions | 1/1" in text
    assert "## Answers (model-a: 2, model-b: 1)" in text          # counts every model that answered


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
def test_runs_against_the_database():
    from app.db import connect, reset
    from app.embeddings import FakeEmbedder
    from app.evaluate import run_answers, run_retrieval
    from app.ingest import DEMO_FILE, ingest, load_jobs
    from app.llm import FakeLLM

    conn = connect(URL)
    reset(conn)
    ingest(conn, FakeEmbedder(), load_jobs(DEMO_FILE))
    try:
        qs = load_questions()
        rows = run_retrieval(conn, FakeEmbedder(), qs, k=5)
        assert len(rows) == sum(q.answerable for q in qs) and all(len(r.ranked) <= 5 for r in rows)
        filtered = next(r for r in rows if r.id == "remote-english")
        assert filtered.filtered and set(filtered.ranked) <= set(filtered.expected)   # filters are exact

        class SwitchingLLM(FakeLLM):                     # pretends the fallback kicks in after the first call
            last_model = None

            def generate(self, prompt, system=""):
                self.last_model = "main" if not self.prompts else "backup"
                return super().generate(prompt, system)

        answers = run_answers(conn, FakeEmbedder(), SwitchingLLM(NOT_FOUND), qs[:3] + qs[-1:], sleep=lambda s: None)
        assert [a.ok for a in answers] == [False, False, False, True]   # declines everything: only last passes
        assert [a.model for a in answers] == ["main", "backup", "backup", "backup"]
    finally:
        reset(conn)
        conn.close()

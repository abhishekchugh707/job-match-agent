from app.llm import FakeLLM
from app.rag import NOT_FOUND, SYSTEM, answer_from_hits, build_context, check_citations
from app.search import Hit


def hit(job_id, chunk="text", score=0.8):
    return Hit(job_id, f"Title {job_id}", f"Company {job_id}", "Berlin", chunk, score)


def test_no_hits_means_no_llm_call():
    llm = FakeLLM("should not be used")
    a = answer_from_hits("anything?", [], llm)
    assert a.text == NOT_FOUND
    assert llm.prompts == []


def test_prompt_contains_only_retrieved_postings_and_the_rules():
    llm = FakeLLM("Job A uses FastAPI [job-a].")
    answer_from_hits("Which jobs use FastAPI?", [hit("job-a", "FastAPI and PostgreSQL")], llm)
    assert '<posting id="job-a">' in llm.prompts[0]
    assert "FastAPI and PostgreSQL" in llm.prompts[0]
    assert "Question: Which jobs use FastAPI?" in llm.prompts[0]
    assert llm.systems[0] == SYSTEM


def test_chunks_of_the_same_job_are_grouped():
    ctx = build_context([hit("job-a", "tasks"), hit("job-b", "other"), hit("job-a", "requirements")])
    assert ctx.count('<posting id="job-a">') == 1
    assert "tasks\n...\nrequirements" in ctx


def test_valid_citations_are_kept():
    llm = FakeLLM("Both use Python [job-a] [job-b].")
    a = answer_from_hits("q", [hit("job-a"), hit("job-b")], llm)
    assert a.cited == ["job-a", "job-b"]
    assert a.invalid_citations == []
    assert "[job-a]" in a.text


def test_invented_citations_are_removed_and_reported():
    text, valid, invalid = check_citations("Uses Go [job-a] [made-up-job].", {"job-a"})
    assert valid == ["job-a"]
    assert invalid == ["made-up-job"]
    assert text == "Uses Go [job-a]."


def test_sources_list_each_job_once():
    a = answer_from_hits("q", [hit("job-a"), hit("job-a"), hit("job-b")], FakeLLM("x [job-a]"))
    assert [s.job_id for s in a.sources] == ["job-a", "job-b"]


def test_empty_llm_reply_falls_back_to_not_found():
    a = answer_from_hits("q", [hit("job-a")], FakeLLM(""))
    assert a.text == NOT_FOUND

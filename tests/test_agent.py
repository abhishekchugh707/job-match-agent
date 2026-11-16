from types import SimpleNamespace

from app.agent import AgentResult, GeminiAgentModel, ToolCall, Turn, run

DECLS = [{"name": "search_jobs", "description": "d", "parameters": {"type": "object", "properties": {}}}]


class ScriptedModel:
    """Plays back a fixed list of turns, and records what the agent sent back."""

    def __init__(self, *turns):
        self.turns, self.results, self.allow = list(turns), [], []

    def start(self, system, question, tools):
        self.question = question

    def step(self, allow_tools=True):
        self.allow.append(allow_tools)
        return self.turns.pop(0)

    def add_results(self, results):
        self.results.append(results)


def tools(log):
    def search_jobs(query, **kw):
        log.append(("search_jobs", query, kw))
        return {"results": [{"job_id": "job-a", "title": "A"}, {"job_id": "job-b", "title": "B"}]}

    def get_job(job_id):
        if job_id == "boom":
            raise RuntimeError("database down")
        return {"job_id": job_id, "description": "..."}

    return {"search_jobs": search_jobs, "get_job": get_job}


def test_runs_tools_then_answers_with_checked_citations():
    log = []
    m = ScriptedModel(Turn(calls=[ToolCall("search_jobs", {"query": "AI", "remote": True})]),
                      Turn(calls=[ToolCall("get_job", {"job_id": "job-a"})]),
                      Turn(text="Job A fits [job-a], not [job-z]."))
    r = run(m, tools(log), DECLS, "remote AI jobs?")
    assert isinstance(r, AgentResult)
    assert [s.tool for s in r.trace] == ["search_jobs", "get_job"]
    assert log == [("search_jobs", "AI", {"remote": True})]
    assert r.cited == ["job-a"] and r.invalid_citations == ["job-z"]
    assert "[job-z]" not in r.answer


def test_unknown_tool_and_tool_errors_go_back_to_the_model():
    m = ScriptedModel(Turn(calls=[ToolCall("delete_everything", {}), ToolCall("get_job", {"job_id": "boom"}),
                                  ToolCall("get_job", {"wrong": 1})]),
                      Turn(text="Sorry."))
    r = run(m, tools([]), DECLS, "q")
    sent = [res for _, res in m.results[0]]
    assert "unknown tool" in sent[0]["error"]
    assert "database down" in sent[1]["error"]
    assert "bad arguments" in sent[2]["error"]
    assert r.answer == "Sorry."


def test_tool_call_limit_forces_an_answer():
    looping = [Turn(calls=[ToolCall("search_jobs", {"query": f"q{i}"})]) for i in range(10)]
    m = ScriptedModel(*looping[:4], Turn(text="Best I can do [job-a]."))
    r = run(m, tools([]), DECLS, "q", max_tool_calls=3)
    assert len(r.trace) == 3
    assert r.stopped_by_limit
    assert m.allow[-1] is False                     # final step had tools switched off
    assert "limit reached" in m.results[-1][0][1]["error"]
    assert r.answer == "Best I can do [job-a]."


def test_empty_answer_gets_a_fallback():
    r = run(ScriptedModel(Turn(text="")), tools([]), DECLS, "q")
    assert "couldn't find" in r.answer


# --- the Gemini adapter, with a fake client (checks the messages it builds) ---------------------

class FakeGemini:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requests.append((list(contents), config))
        return self.responses.pop(0)


def response(text=None, calls=()):
    from google.genai import types
    parts = ([types.Part(text=text)] if text else []) + \
            [types.Part(function_call=types.FunctionCall(id=f"c{i}", name=n, args=a)) for i, (n, a) in enumerate(calls)]
    content = types.Content(role="model", parts=parts)
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)],
                           function_calls=[p.function_call for p in parts if p.function_call] or None)


def test_gemini_adapter_round_trip():
    client = FakeGemini(response(calls=[("search_jobs", {"query": "AI"})]), response(text="Done [job-a]."))
    model = GeminiAgentModel(api_key="x", model="m", client=client)
    r = run(model, tools([]), DECLS, "AI jobs?")
    assert r.answer == "Done [job-a]."
    contents, config = client.requests[1]
    assert [c.role for c in contents] == ["user", "model", "user"]       # question, tool call, tool result
    fr = contents[2].parts[0].function_response
    assert fr.name == "search_jobs" and fr.id == "c0"
    assert fr.response["result"]["results"][0]["job_id"] == "job-a"
    assert config.automatic_function_calling.disable is True
    assert config.tools[0].function_declarations[0].name == "search_jobs"


def test_gemini_adapter_switches_tools_off_when_asked():
    client = FakeGemini(response(text="Final answer."))
    model = GeminiAgentModel(api_key="x", model="m", client=client)
    model.start("sys", "q", DECLS)
    model.step(allow_tools=False)
    _, config = client.requests[0]
    assert config.tool_config.function_calling_config.mode == "NONE"
    assert model.config.tool_config is None                  # the original config is unchanged

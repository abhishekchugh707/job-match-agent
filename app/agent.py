"""A tool-calling agent: the model decides which tools to call, in which order, and when to answer.

    python -m app.agent "Compare the remote AI jobs that don't need German"
    python -m app.agent "I know Python, FastAPI and Docker. What am I missing for the Kranich job?"

The loop:
    1. Send the question (plus the tool descriptions) to the model.
    2. If the model asks for tool calls, run them and send the results back.
    3. Repeat until the model answers in text, or the tool-call limit is reached.

Safety rails: at most MAX_TOOL_CALLS tool calls; unknown tools and tool errors go back to the model
as error messages instead of crashing; every call is recorded in a trace; job ids the answer cites
are checked against the jobs the tools actually returned.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .rag import check_citations

MAX_TOOL_CALLS = 6

SYSTEM = """You help a job seeker explore job postings, using the tools provided.
Rules:
- Use the tools to find facts. Never invent jobs, companies or requirements.
- Start with search_jobs. Put hard conditions (German level, remote, city) into its filters.
- Use get_job when you need a posting's exact requirements; use skill_gap only with skills the user gave you.
- Cite the job id in square brackets after each statement about a job, e.g. [nordlicht-backend].
- If the tools return nothing useful, say so plainly.
- Be concise."""


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None


@dataclass
class Turn:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)


class AgentModel(Protocol):
    def start(self, system: str, question: str, tools: list[dict]) -> None: ...
    def step(self, allow_tools: bool = True) -> Turn: ...
    def add_results(self, results: list[tuple[ToolCall, dict]]) -> None: ...


@dataclass
class TraceStep:
    tool: str
    args: dict
    result: dict


@dataclass
class AgentResult:
    question: str
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    stopped_by_limit: bool = False
    cited: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    model: str | None = None              # which Gemini model answered (None for test models)


def _job_ids(result: Any) -> set[str]:
    """Every job id that appears anywhere in a tool result."""
    found: set[str] = set()
    if isinstance(result, dict):
        if isinstance(result.get("job_id"), str):
            found.add(result["job_id"])
        for v in result.values():
            found |= _job_ids(v)
    elif isinstance(result, list):
        for v in result:
            found |= _job_ids(v)
    return found


def run(model: AgentModel, tools: dict[str, Callable[..., dict]], declarations: list[dict], question: str,
        max_tool_calls: int = MAX_TOOL_CALLS) -> AgentResult:
    model.start(SYSTEM, question, declarations)
    trace: list[TraceStep] = []
    seen_ids: set[str] = set()
    limited = False

    while True:
        turn = model.step(allow_tools=not limited)
        if not turn.calls or limited:
            text, valid, invalid = check_citations(turn.text, seen_ids)
            return AgentResult(question, text or "I couldn't find an answer with the available tools.",
                               trace, limited, valid, invalid, getattr(model, "model_used", None))

        results = []
        for call in turn.calls:
            if len(trace) >= max_tool_calls:
                limited = True
                result = {"error": "Tool call limit reached. Answer now with what you have."}
            elif call.name not in tools:
                result = {"error": f"unknown tool '{call.name}'. Available: {sorted(tools)}"}
            else:
                try:
                    result = tools[call.name](**call.args)
                except TypeError as e:                       # wrong or missing arguments
                    result = {"error": f"bad arguments for {call.name}: {e}"}
                except Exception as e:                       # a failing tool must not crash the agent
                    result = {"error": f"{call.name} failed: {e}"}
            if not limited:
                trace.append(TraceStep(call.name, call.args, result))
                seen_ids |= _job_ids(result)
            results.append((call, result))
        model.add_results(results)


class GeminiAgentModel:
    """AgentModel on top of Gemini function calling. Keeps the conversation (including the model's
    own turns, which carry its thought signatures) so each step sees everything so far.

    Fallback: until the first answer, a busy model is skipped for the next one in the list. After that
    the conversation stays on the model that answered, because thought signatures belong to that model."""

    def __init__(self, api_key: str, model: str, client=None, attempts: int = 4, fallbacks=(),
                 sleep=None):
        import time

        from google.genai import types

        from .llm import gemini_client
        self.t, self.attempts, self.sleep = types, attempts, sleep or time.sleep
        self.models = [model, *[f for f in fallbacks if f and f != model]]
        self.model_used: str | None = None
        self.client = client or gemini_client(api_key)
        self.contents: list = []
        self.config = None

    def start(self, system: str, question: str, tools: list[dict]) -> None:
        t = self.t
        decls = [t.FunctionDeclaration(name=d["name"], description=d["description"],
                                       parameters_json_schema=d["parameters"]) for d in tools]
        self.config = t.GenerateContentConfig(
            system_instruction=system, temperature=0.2,
            tools=[t.Tool(function_declarations=decls)],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True))   # we run the loop
        self.contents = [t.Content(role="user", parts=[t.Part(text=question)])]

    def step(self, allow_tools: bool = True) -> Turn:
        from .llm import thinking_for, with_fallback
        t = self.t
        cfg = self.config
        if not allow_tools:
            cfg = cfg.model_copy(update={"tool_config": t.ToolConfig(
                function_calling_config=t.FunctionCallingConfig(mode="NONE"))})
        models = [self.model_used] if self.model_used else self.models
        self.model_used, r = with_fallback(models, lambda m: self.client.models.generate_content(
            model=m, contents=self.contents, config=cfg.model_copy(update={"thinking_config": thinking_for(t, m)})),
            self.attempts, self.sleep)
        content = r.candidates[0].content if r.candidates else None
        if content is not None:
            self.contents.append(content)
        parts = (content.parts or []) if content is not None else []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        calls = [ToolCall(fc.name, dict(fc.args or {}), fc.id) for fc in (r.function_calls or [])]
        return Turn(text.strip(), calls)

    def add_results(self, results: list[tuple[ToolCall, dict]]) -> None:
        t = self.t
        parts = [t.Part(function_response=t.FunctionResponse(id=c.id, name=c.name, response={"result": res}))
                 for c, res in results]
        self.contents.append(t.Content(role="user", parts=parts))   # Gemini API: tool results go in a user turn


def main() -> int:
    from .config import get_settings
    from .db import connect
    from .embeddings import GeminiEmbedder
    from .llm import LLMUnavailable
    from .tools import DECLARATIONS, ToolContext, bind

    ap = argparse.ArgumentParser(description="Ask the job agent")
    ap.add_argument("question")
    ap.add_argument("--max-tool-calls", type=int, default=MAX_TOOL_CALLS)
    args = ap.parse_args()

    s = get_settings()
    embedder = GeminiEmbedder(api_key=s.gemini_api_key, model=s.gemini_embed_model)
    model = GeminiAgentModel(api_key=s.gemini_api_key, model=s.gemini_model, fallbacks=s.gemini_fallback_models)
    with connect(s.database_url) as conn:
        try:
            res = run(model, bind(ToolContext(conn, embedder)), DECLARATIONS, args.question, args.max_tool_calls)
        except LLMUnavailable as e:
            print(e)
            return 1

    print(f"Model: {res.model}")
    print("Steps:")
    for i, st in enumerate(res.trace, 1):
        summary = st.result.get("error") or ", ".join(sorted(_job_ids(st.result))[:5]) or "ok"
        print(f"  {i}. {st.tool}({json.dumps(st.args, ensure_ascii=False)}) -> {summary}")
    if res.stopped_by_limit:
        print("  (tool-call limit reached)")
    print("\n" + res.answer)
    if res.invalid_citations:
        print(f"\nRemoved invented citations: {', '.join(res.invalid_citations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

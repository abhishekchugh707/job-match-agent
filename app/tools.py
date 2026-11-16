"""The tools the agent can call. Plain functions with a JSON-schema description each.

The same definitions serve the Gemini agent (step 4) and the MCP server (step 5), so a tool is
written once. Every tool returns JSON-serializable data; problems come back as {"error": ...}
instead of exceptions, so the model can read them and try something else.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .embeddings import Embedder
from .search import GERMAN_LEVELS, Filters, search

# Technologies the skill tools look for. Each entry: display name -> regex (case-insensitive unless noted).
TECH: dict[str, str] = {
    "Python": r"\bpython\b", "Java": r"\bjava\b(?!script)", "Go": r"(?-i:\bGo\b)", "Rust": r"\brust\b",
    "TypeScript": r"\btypescript\b", "JavaScript": r"\bjavascript\b", "C++": r"c\+\+", "Scala": r"\bscala\b",
    "SQL": r"\bsql\b", "FastAPI": r"\bfastapi\b", "Django": r"\bdjango\b", "Spring Boot": r"\bspring boot\b",
    "Node.js": r"\bnode\.js\b", "Next.js": r"\bnext\.js\b", "React": r"\breact\b", "Angular": r"\bangular\b",
    "PostgreSQL": r"\bpostgres(?:ql)?\b", "TimescaleDB": r"\btimescaledb\b", "Redis": r"\bredis\b",
    "Oracle": r"\boracle\b", "Snowflake": r"\bsnowflake\b", "BigQuery": r"\bbigquery\b", "dbt": r"\bdbt\b",
    "Spark": r"\bspark\b", "Airflow": r"\bairflow\b", "Kafka": r"\bkafka\b", "Flink": r"\bflink\b",
    "Iceberg": r"\biceberg\b", "Docker": r"\bdocker\b", "Kubernetes": r"\bkubernetes\b",
    "Terraform": r"\bterraform\b", "AWS": r"\baws\b", "Google Cloud": r"\b(?:gcp|google cloud)\b",
    "Azure": r"\bazure\b", "GitLab CI": r"\bgitlab\b", "GitHub Actions": r"\bgithub actions\b",
    "CI/CD": r"\bci/cd\b", "Linux": r"\blinux\b", "Bash": r"\bbash\b", "Prometheus": r"\bprometheus\b",
    "Grafana": r"\bgrafana\b", "OpenTelemetry": r"\bopentelemetry\b", "MLflow": r"\bmlflow\b",
    "PyTorch": r"\bpytorch\b", "scikit-learn": r"\bscikit-learn\b", "LightGBM": r"\blightgbm\b",
    "pandas": r"\bpandas\b", "LLMs": r"\bllms?\b", "RAG": r"\brag\b|retrieval-augmented",
    "MCP": r"\bmcp\b|model context protocol", "Vector databases": r"vector database|pgvector|qdrant",
    "ONNX": r"\bonnx\b", "TensorRT": r"\btensorrt\b", "OpenCV": r"\bopencv\b",
    "Computer vision": r"computer vision", "Time series": r"time series|zeitreihen",
    "A/B testing": r"a/b test", "Causal inference": r"causal inference", "OAuth 2.0": r"\boauth\s?2?(?:\.0)?\b",
    "OpenID Connect": r"openid connect", "JWT": r"\bjwt\b", "mTLS": r"\bmtls\b", "OWASP": r"\bowasp\b",
    "pytest": r"\bpytest\b", "Jest": r"\bjest\b", "Playwright": r"\bplaywright\b",
}
_TECH_RE = {name: re.compile(p, re.IGNORECASE) for name, p in TECH.items()}


def technologies(text: str) -> list[str]:
    return [name for name, rx in _TECH_RE.items() if rx.search(text)]


def _norm(skill: str) -> str:
    return re.sub(r"[^a-z0-9+#/.]", "", skill.lower())


_ALIASES = {_norm(n): n for n in TECH} | {"postgres": "PostgreSQL", "k8s": "Kubernetes", "gcp": "Google Cloud",
                                           "sklearn": "scikit-learn", "llm": "LLMs", "oauth2": "OAuth 2.0",
                                           "oauth": "OAuth 2.0", "githubactions": "GitHub Actions",
                                           "gitlab": "GitLab CI", "cicd": "CI/CD", "golang": "Go"}


def canonical(skill: str) -> str:
    return _ALIASES.get(_norm(skill), skill.strip())


# Knowing the left-hand skill covers the ones on the right (kept short and uncontroversial).
IMPLIES: dict[str, set[str]] = {
    "PostgreSQL": {"SQL"}, "TimescaleDB": {"PostgreSQL", "SQL"}, "Oracle": {"SQL"},
    "Snowflake": {"SQL"}, "BigQuery": {"SQL"},
    "GitLab CI": {"CI/CD"}, "GitHub Actions": {"CI/CD"},
}


def compare_skills(needed: list[str], skills: list[str]) -> dict:
    """Split a posting's technologies into matched and missing for someone with these skills.

    matched: in the posting and covered by the skills (directly or through IMPLIES)
    implied: for matches that came through IMPLIES, which of the user's skills covers them
    missing: in the posting, not covered
    not_asked_for: the user's skills the posting doesn't mention (not a gap, just not required)"""
    have = list(dict.fromkeys(canonical(x) for x in skills if x.strip()))
    via: dict[str, str] = {}
    for h in have:
        for covered in IMPLIES.get(h, ()):
            via.setdefault(covered, h)
    matched = [t for t in needed if t in have or t in via]
    return {"matched": matched,
            "implied": {t: via[t] for t in matched if t not in have},
            "missing": [t for t in needed if t not in matched],
            "not_asked_for": [h for h in have if h not in needed]}


@dataclass
class ToolContext:
    conn: Any
    embedder: Embedder


def _job_row(conn, job_id: str) -> dict | None:
    row = conn.execute("""SELECT id, company, title, location, remote, german, posted, url, description
                          FROM jobs WHERE id = %s""", (job_id,)).fetchone()
    if not row:
        return None
    keys = ("job_id", "company", "title", "location", "remote", "german", "posted", "url", "description")
    job = dict(zip(keys, row))
    job["posted"] = str(job["posted"]) if job["posted"] else None
    return job


def search_jobs(ctx: ToolContext, query: str, german: str | None = None, max_german: str | None = None,
                remote: bool | None = None, location: str | None = None, k: int = 5) -> dict:
    for name, value in (("german", german), ("max_german", max_german)):
        if value and value not in GERMAN_LEVELS:
            return {"error": f"{name} must be one of {list(GERMAN_LEVELS)}"}
    f = Filters(german=german, max_german=max_german, remote=remote, location=location)
    hits = search(ctx.conn, ctx.embedder, query, k=max(1, min(int(k), 10)), filters=f)
    return {"results": [{"job_id": h.job_id, "title": h.title, "company": h.company, "location": h.location,
                         "score": h.score, "snippet": h.chunk.split("\n", 1)[-1][:300]} for h in hits]}


def get_job(ctx: ToolContext, job_id: str) -> dict:
    job = _job_row(ctx.conn, job_id)
    return job if job else {"error": f"no job with id '{job_id}'"}


def skill_gap(ctx: ToolContext, job_id: str, skills: list[str]) -> dict:
    job = _job_row(ctx.conn, job_id)
    if not job:
        return {"error": f"no job with id '{job_id}'"}
    needed = technologies(job["description"])
    return {"job_id": job_id, "title": job["title"], "company": job["company"], "german": job["german"],
            "technologies_in_posting": needed} | compare_skills(needed, skills)


def compare_jobs(ctx: ToolContext, job_ids: list[str]) -> dict:
    if not 2 <= len(job_ids) <= 5:
        return {"error": "give between 2 and 5 job ids"}
    rows, unknown = [], []
    for jid in job_ids:
        job = _job_row(ctx.conn, jid)
        if not job:
            unknown.append(jid)
            continue
        rows.append({k: job[k] for k in ("job_id", "title", "company", "location", "remote", "german", "posted")}
                    | {"technologies": technologies(job["description"])})
    return {"jobs": rows} | ({"unknown_ids": unknown} if unknown else {})


# Descriptions are what the model reads to decide which tool to call, so they say when to use each.
DECLARATIONS: list[dict] = [
    {"name": "search_jobs",
     "description": "Semantic search over the job postings. Use it first to find candidate jobs. "
                    "Use the filters for hard conditions (German level, remote, city) instead of putting "
                    "them in the query.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "what the job is about, e.g. 'backend Python APIs'"},
         "german": {"type": "string", "enum": list(GERMAN_LEVELS), "description": "exact German level"},
         "max_german": {"type": "string", "enum": list(GERMAN_LEVELS),
                        "description": "at most this German level ('none' = no German needed)"},
         "remote": {"type": "boolean", "description": "true = remote jobs only"},
         "location": {"type": "string", "description": "city name or 'Remote'"},
         "k": {"type": "integer", "description": "number of results, 1-10 (default 5)"}},
         "required": ["query"]}},
    {"name": "get_job",
     "description": "Full text and details of one job posting. Use it when you need the exact requirements.",
     "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
    {"name": "skill_gap",
     "description": "Compare a job's technologies with a list of skills. Returns matched (including skills "
                    "covered by related ones, e.g. PostgreSQL covers SQL, see 'implied'), missing (what the job "
                    "asks for that the skills don't cover) and not_asked_for (the user's skills the job doesn't "
                    "mention; these are not gaps). Only use it when the user has told you their skills.",
     "parameters": {"type": "object", "properties": {
         "job_id": {"type": "string"},
         "skills": {"type": "array", "items": {"type": "string"}, "description": "the user's skills"}},
         "required": ["job_id", "skills"]}},
    {"name": "compare_jobs",
     "description": "Side-by-side facts for 2-5 jobs: location, remote, German level, posted date, technologies.",
     "parameters": {"type": "object", "properties": {
         "job_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["job_ids"]}},
]

FUNCTIONS: dict[str, Callable[..., dict]] = {
    "search_jobs": search_jobs, "get_job": get_job, "skill_gap": skill_gap, "compare_jobs": compare_jobs}


def bind(ctx: ToolContext) -> dict[str, Callable[..., dict]]:
    """Tools with the context filled in: name -> function(**args)."""
    return {name: (lambda f: lambda **kw: f(ctx, **kw))(fn) for name, fn in FUNCTIONS.items()}

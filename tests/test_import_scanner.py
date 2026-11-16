"""The Job Scanner import, tested on a tiny fake scanner database (no real data)."""
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from app.import_scanner import german_level, import_jobs, open_read_only, slug
from app.ingest import FIELDS
from app.rag import CITATION

NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)
LONG = "Build backend services in Python and FastAPI. " * 10


def record(**kw):
    base = {"title": "Backend Engineer", "company": "Acme GmbH", "url": "https://jobs.example/1",
            "location": "Köln, Germany", "remote": False, "german_ad": False, "flags": [],
            "posted": "2026-11-10T00:00:00+00:00", "description": LONG,
            "contacts": {"company": "Acme", "people": [{"name": "Real Person"}]}, "score": "40"}
    base.update(kw)
    return base


@pytest.fixture
def scanner_db(tmp_path):
    path = tmp_path / "jobs.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE jobs (key TEXT PRIMARY KEY, fingerprint TEXT, grp TEXT, status TEXT,
                   first_seen TEXT, last_seen TEXT, status_at TEXT, score INTEGER, data TEXT)""")
    rows = [
        ("aaaaaa1111", "main", "new", record()),
        ("bbbbbb2222", "main", "skipped", record(title="Skipped one")),
        ("cccccc3333", "low", "seen", record(title="Low match")),
        ("dddddd4444", "main", "saved", record(title="Short", description="too short")),
        ("eeeeee5555", "maybe", "applied", record(title="Old", posted="2026-08-01T00:00:00+00:00")),
        ("ffffff6666", "main", "new", record(title="Daten (m/w/d)", company="Müller & Söhne",
                                             flags="['ad written in German']", remote="True")),
        ("gggggg7777", "main", "new", "{not json"),
    ]
    for key, grp, status, data in rows:
        con.execute("INSERT INTO jobs (key, grp, status, first_seen, data) VALUES (?,?,?,?,?)",
                    (key, grp, status, "2026-11-12T08:00:00+00:00",
                     data if isinstance(data, str) else json.dumps(data)))
    con.commit()
    con.close()
    return path


def test_keeps_only_jobs_worth_asking_about(scanner_db):
    conn = open_read_only(scanner_db)
    jobs, left_out = import_jobs(conn, days=30, now=NOW)
    conn.close()
    assert [j["title"] for j in jobs] == ["Backend Engineer", "Daten (m/w/d)"]
    assert left_out == {"marked as skipped": 1, "low match": 1, "no readable description": 1,
                        "older than 30 days": 1, "unreadable record": 1}


def test_low_matches_can_be_included(scanner_db):
    conn = open_read_only(scanner_db)
    jobs, _ = import_jobs(conn, days=30, include_low=True, now=NOW)
    conn.close()
    assert "Low match" in [j["title"] for j in jobs]


def test_output_has_exactly_the_project_fields_and_no_private_data(scanner_db):
    conn = open_read_only(scanner_db)
    jobs, _ = import_jobs(conn, now=NOW)
    conn.close()
    for j in jobs:
        assert set(j) == set(FIELDS)
        assert "Real Person" not in json.dumps(j)                  # contacts never copied
        assert CITATION.fullmatch(f"[{j['id']}]")                  # ids work as citations
    german_job = next(j for j in jobs if j["title"] == "Daten (m/w/d)")
    assert german_job["german"] == "B2" and german_job["remote"] is True
    assert german_job["id"].startswith("muller-sohne-")


def test_scanner_database_is_opened_read_only(scanner_db):
    conn = open_read_only(scanner_db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM jobs")
    conn.close()


def test_missing_database_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No scanner database"):
        open_read_only(tmp_path / "nope.db")


def test_german_level_rules():
    assert german_level({"flags": ["German B2 / good level asked: 'gute Deutschkenntnisse'"]}) == "B2"
    assert german_level({"flags": "['ad written in German']"}) == "B2"
    assert german_level({"german_ad": True}) == "B2"
    assert german_level({"flags": [], "german_ad": False}) == "none"


def test_slug():
    assert slug("Zalando Marketing Services") == "zalando-marketing-services"
    assert slug("Müller & Söhne GmbH") == "muller-sohne-gmbh"
    assert slug("!!!") == "job"

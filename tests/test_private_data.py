"""The switch between demo data and your own jobs (separate database)."""
import os

import pytest

from app.config import get_settings

URL = os.getenv("TEST_DATABASE_URL")


def test_demo_is_the_default(monkeypatch):
    monkeypatch.delenv("JOBMATCH_DATA", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = get_settings()
    assert s.data_set == "demo" and s.database_url.endswith("/jobmatch")


def test_private_uses_a_separate_database(monkeypatch):
    monkeypatch.setenv("JOBMATCH_DATA", "private")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5433/jobmatch")
    monkeypatch.delenv("PRIVATE_DATABASE_URL", raising=False)
    s = get_settings()
    assert s.data_set == "private" and s.database_url.endswith("/jobmatch_private")


def test_unknown_data_set_is_rejected(monkeypatch):
    monkeypatch.setenv("JOBMATCH_DATA", "prod")
    with pytest.raises(ValueError, match="demo' or 'private"):
        get_settings()


def test_evaluation_refuses_private_data(monkeypatch, capsys):
    import sys

    from app import evaluate
    monkeypatch.setenv("JOBMATCH_DATA", "private")
    monkeypatch.setattr(sys, "argv", ["evaluate"])
    assert evaluate.main() == 1
    assert "demo jobs" in capsys.readouterr().out


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
def test_ensure_database_creates_it_once():
    import psycopg
    from psycopg.conninfo import make_conninfo

    from app.db import ensure_database
    url = make_conninfo(URL, dbname="jobmatch_test_private")
    try:
        assert ensure_database(url) == "jobmatch_test_private"
        assert ensure_database(url) == "jobmatch_test_private"          # second call: already there
        with psycopg.connect(url) as conn:
            assert conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        with psycopg.connect(make_conninfo(URL, dbname="postgres"), autocommit=True) as admin:
            admin.execute("DROP DATABASE IF EXISTS jobmatch_test_private")

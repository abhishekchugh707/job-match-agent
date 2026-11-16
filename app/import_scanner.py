"""Import jobs from a Job Scanner database (github.com/abhishekchugh707/job-scanner) into this project's format.

    python -m app.import_scanner --db path/to/jobs.db
    python -m app.import_scanner --db path/to/jobs.db --days 60 --include-low

Writes data/private/my_jobs.json (git ignores data/private/). Load it with:

    python -m app.ingest --file data/private/my_jobs.json

The scanner database is opened read-only, so this can never change it. Only the fields this
project needs are copied (title, company, location, remote, German level, date, link, description).
Everything else in a scanner record (scores, notes, LinkedIn contacts) is left behind on purpose.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "private" / "my_jobs.json"
MIN_DESCRIPTION = 200          # shorter descriptions are usually "couldn't read the page" stubs


def open_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"No scanner database at {path}")
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _as_list(value) -> list:
    """Scanner flags are stored as a list, or as the text of one."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        for parse in (json.loads, ast.literal_eval):
            try:
                out = parse(value)
                return out if isinstance(out, list) else []
            except (ValueError, SyntaxError):
                continue
    return []


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def german_level(record: dict) -> str:
    """The scanner already drops jobs that need fluent or C1 German. Of the rest, anything that asks
    for German or is written in German counts as B2 (better careful than wrong), the others as none."""
    flags = " ".join(str(f) for f in _as_list(record.get("flags"))).lower()
    if "german b2" in flags or "ad written in german" in flags or _as_bool(record.get("german_ad")):
        return "B2"
    return "none"


def slug(text: str, limit: int = 30) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:limit].strip("-") or "job"


def _date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def convert(key: str, record: dict, first_seen: str | None) -> dict:
    return {
        "id": f"{slug(record.get('company', ''))}-{slug(key, 6)}",
        "company": record.get("company", "").strip(),
        "title": record.get("title", "").strip(),
        "location": record.get("location", "").strip() or "Germany",
        "remote": _as_bool(record.get("remote")),
        "german": german_level(record),
        "posted": _date(record.get("posted")) or _date(first_seen),
        "url": record.get("apply_url") or record.get("url") or "",
        "description": (record.get("description") or "").strip(),
    }


def import_jobs(conn: sqlite3.Connection, days: int = 30, include_low: bool = False,
                now: datetime | None = None) -> tuple[list[dict], Counter]:
    """Jobs worth asking about, plus a count of why the others were left out."""
    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=days)).date().isoformat()
    kept, left_out, seen_ids = [], Counter(), set()
    for key, status, grp, first_seen, data in conn.execute(
            "SELECT key, status, grp, first_seen, data FROM jobs ORDER BY first_seen DESC"):
        if status == "skipped":
            left_out["marked as skipped"] += 1
            continue
        if grp == "low" and not include_low:
            left_out["low match"] += 1
            continue
        try:
            record = json.loads(data)
        except (TypeError, ValueError):
            left_out["unreadable record"] += 1
            continue
        job = convert(key, record, first_seen)
        if len(job["description"]) < MIN_DESCRIPTION:
            left_out["no readable description"] += 1
            continue
        if not job["title"] or not job["company"]:
            left_out["no title or company"] += 1
            continue
        if (job["posted"] or "") < cutoff:
            left_out[f"older than {days} days"] += 1
            continue
        if job["id"] in seen_ids:
            left_out["duplicate"] += 1
            continue
        seen_ids.add(job["id"])
        kept.append(job)
    return kept, left_out


def main() -> int:
    ap = argparse.ArgumentParser(description="Import jobs from a Job Scanner database")
    ap.add_argument("--db", type=Path, required=True, help="path to the scanner's jobs.db")
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--days", type=int, default=30, help="only jobs from the last N days (default 30)")
    ap.add_argument("--include-low", action="store_true", help="also import jobs the scanner rated a low match")
    args = ap.parse_args()

    conn = open_read_only(args.db)
    try:
        jobs, left_out = import_jobs(conn, args.days, args.include_low)
    finally:
        conn.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Imported {len(jobs)} jobs to {args.out}")
    for reason, n in left_out.most_common():
        print(f"  left out {n}: {reason}")
    print(f"\nNext: python -m app.ingest --file {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

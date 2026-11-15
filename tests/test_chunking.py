from app.chunking import chunk_job, split_text

JOB = {"id": "j1", "company": "Acme", "title": "Backend Engineer", "location": "Berlin",
       "remote": False, "german": "none", "posted": "2026-11-01",
       "description": "About us: we build things.\n\nTasks:\n- Build APIs\n- Write SQL\n\nRequirements:\n- Python"}


def test_first_chunk_is_overview_with_german_level():
    first = chunk_job(JOB)[0]
    assert first.index == 0
    assert "Backend Engineer at Acme, Berlin" in first.text
    assert "German not required" in first.text


def test_every_chunk_carries_the_job_header():
    for c in chunk_job(JOB):
        assert c.text.startswith("Backend Engineer at Acme")
        assert c.job_id == "j1"


def test_sections_are_packed_up_to_the_limit():
    parts = split_text(JOB["description"], max_chars=1000)
    assert len(parts) == 1                     # everything fits in one piece
    parts = split_text(JOB["description"], max_chars=35)
    assert len(parts) == 3                     # one piece per section
    assert all(len(p) <= 35 for p in parts)


def test_long_section_is_split_on_lines():
    section = "\n".join(f"- requirement number {i}" for i in range(40))
    parts = split_text(section, max_chars=200)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)
    assert "\n".join(parts).count("requirement number") == 40   # nothing lost

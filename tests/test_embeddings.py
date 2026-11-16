import math
from types import SimpleNamespace

from app.embeddings import EmbeddingCache, FakeEmbedder, GeminiEmbedder, normalize


class FakeClient:
    """Stands in for google.genai.Client: counts calls, returns fixed vectors."""

    def __init__(self):
        self.calls = 0
        self.models = self

    def embed_content(self, model, contents, config):
        self.calls += 1
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0]) for _ in contents])


def test_normalize_gives_unit_length():
    assert math.isclose(math.hypot(*normalize([3.0, 4.0])), 1.0)


def test_gemini_embedder_caches_and_normalizes(tmp_path):
    client = FakeClient()
    e = GeminiEmbedder(api_key="x", dim=2, cache=EmbeddingCache(tmp_path / "c.sqlite"), client=client)
    v1 = e.embed_documents(["a", "b"])
    assert v1 == [[0.6, 0.8], [0.6, 0.8]]
    assert client.calls == 1
    e.embed_documents(["a", "b"])              # second time: served from the cache
    assert client.calls == 1
    e.embed_query("a")                         # same text, different task type: new call
    assert client.calls == 2


def test_fake_embedder_similar_texts_score_higher():
    e = FakeEmbedder()
    q = e.embed_query("python backend fastapi")
    near = e.embed_documents(["python fastapi backend engineer"])[0]
    far = e.embed_documents(["react frontend designer"])[0]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(q, near) > dot(q, far)


DAILY = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: embed_content_free_tier_requests, "
         "quotaId: EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier")


class QuotaClient(FakeClient):
    """Answers the first request, then fails with the given error."""

    def __init__(self, error):
        super().__init__()
        self.error = error

    def embed_content(self, model, contents, config):
        if self.calls >= 1:
            self.calls += 1
            raise Exception(self.error)
        return super().embed_content(model, contents, config)


def test_daily_quota_stops_at_once_and_keeps_finished_batches(tmp_path, monkeypatch):
    import pytest

    from app import embeddings
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: pytest.fail("should not wait on a daily limit"))
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    texts = [f"chunk {i}" for i in range(embeddings.BATCH + 10)]          # two batches
    e = GeminiEmbedder(api_key="x", dim=2, cache=cache, client=QuotaClient(DAILY))
    with pytest.raises(embeddings.EmbeddingQuotaExhausted, match="saved in the cache"):
        e.embed_documents(texts)
    assert e.client.calls == 2                                             # no retries
    # the first batch is cached: a later run only needs the rest
    again = GeminiEmbedder(api_key="x", dim=2, cache=cache, client=FakeClient())
    again.embed_documents(texts)
    assert again.client.calls == 1


def test_short_rate_limit_is_still_retried(tmp_path, monkeypatch):
    from app import embeddings
    waits = []
    monkeypatch.setattr(embeddings.time, "sleep", waits.append)

    class Flaky(FakeClient):
        def embed_content(self, model, contents, config):
            self.calls += 1
            if self.calls == 1:
                raise Exception("429 RESOURCE_EXHAUSTED. Too many requests per minute")
            self.calls -= 1
            return super().embed_content(model, contents, config)

    e = GeminiEmbedder(api_key="x", dim=2, cache=EmbeddingCache(tmp_path / "c.sqlite"), client=Flaky())
    assert len(e.embed_documents(["a"])) == 1 and waits == [5]

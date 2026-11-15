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

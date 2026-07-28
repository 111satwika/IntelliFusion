"""Tests for app.embeddings.embedder.

The real sentence-transformers model is NOT loaded here - _get_model
is faked out so these tests run fast and don't depend on the model
being downloaded. The real model is exercised manually via
`python -m app.embeddings.embedder` (see module docstring) and
indirectly through the whole pipeline in practice.
"""

from app.chunking.chunker import Chunk
from app.embeddings import embedder


class _FakeModel:
    """Returns a distinct, deterministic vector per input text."""

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        import numpy as np

        return np.array([[float(len(text)), 0.0] for text in texts])


def test_embed_texts_returns_empty_list_for_empty_input():
    assert embedder.embed_texts([]) == []


def test_embed_texts_preserves_order_and_uses_the_model(monkeypatch):
    monkeypatch.setattr(embedder, "_get_model", lambda: _FakeModel())

    vectors = embedder.embed_texts(["ab", "abcd"])

    assert vectors == [[2.0, 0.0], [4.0, 0.0]]


def test_embed_chunks_returns_empty_list_for_empty_input():
    assert embedder.embed_chunks([]) == []


def test_embed_chunks_pairs_each_chunk_with_its_embedding_and_metadata(monkeypatch):
    monkeypatch.setattr(embedder, "_get_model", lambda: _FakeModel())

    chunks = [
        Chunk(content="ab", metadata={"chunk_index": 0}),
        Chunk(content="abcd", metadata={"chunk_index": 1}),
    ]

    embedded = embedder.embed_chunks(chunks)

    assert len(embedded) == 2
    assert embedded[0].content == "ab"
    assert embedded[0].embedding == [2.0, 0.0]
    assert embedded[0].metadata == {"chunk_index": 0}
    assert embedded[1].content == "abcd"
    assert embedded[1].embedding == [4.0, 0.0]

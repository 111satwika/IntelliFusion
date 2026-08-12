"""Cross-owner isolation tests for the per-user data scoping added
across app.vectorstore.store / app.retrieval.hybrid_retriever /
app.retrieval.retriever (see the "real per-user accounts" plan).

Every test here ingests near-identical content (same document_id /
image_url / repository name / start_seconds) under TWO synthetic
owners, "alice" and "bob", so a missing or wrong owner filter shows up
as one owner's private content leaking into the other's results,
rather than being accidentally masked by the two owners' data simply
looking different from each other.

Each test points Chroma at a fresh tmp_path directory (never the real
data/chroma_db/) and resets hybrid_retriever's module-global BM25
cache, matching the isolation patterns already used by test_store.py
and test_hybrid_retriever.py individually - this file combines both
since the BM25 cache is exactly the kind of hidden shared state a
storage-level fix alone wouldn't catch.
"""

import pytest

from app.embeddings.embedder import EmbeddedChunk
from app.retrieval import hybrid_retriever
from app.vectorstore import store


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CHROMA_DB_DIR", str(tmp_path))
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store, "_collections", {})
    monkeypatch.setattr(store, "_image_collection", None)
    monkeypatch.setattr(store, "_audio_clip_collection", None)
    hybrid_retriever._bm25_index_by_owner_kb = {}
    yield store
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store, "_collections", {})
    monkeypatch.setattr(store, "_image_collection", None)
    monkeypatch.setattr(store, "_audio_clip_collection", None)
    hybrid_retriever._bm25_index_by_owner_kb = {}


def _md_chunk(owner: str, content: str, document_id: str = "shared.md", chunk_index: int = 0) -> EmbeddedChunk:
    return EmbeddedChunk(
        content=content,
        embedding=[1.0, 0.0],
        metadata={
            "source_type": "markdown",
            "document_id": document_id,
            "chunk_index": chunk_index,
            "owner": owner,
        },
    )


def _github_chunk(owner: str, content: str, repository: str = "shared-org/shared-repo") -> EmbeddedChunk:
    return EmbeddedChunk(
        content=content,
        embedding=[1.0, 0.0],
        metadata={
            "source_type": "code",
            "document_id": "src/main.py",
            "chunk_index": 0,
            "repository": repository,
            "owner": owner,
        },
    )


# ---------- Storage-id / count isolation ----------


def test_identical_document_id_across_owners_does_not_collide(isolated_store):
    isolated_store.add_embedded_chunks(
        [
            _md_chunk("alice", "Alice's private notes about the roadmap."),
            _md_chunk("bob", "Bob's private notes about the roadmap."),
        ]
    )

    # Both stored under the shared document_id - an owner-unaware id
    # scheme would have upserted one over the other, leaving count == 1.
    assert isolated_store.count(kb="markdown") == 2

    alice_docs = isolated_store.list_documents("markdown", "alice")
    bob_docs = isolated_store.list_documents("markdown", "bob")
    assert [d["id"] for d in alice_docs] == ["shared.md"]
    assert [d["id"] for d in bob_docs] == ["shared.md"]


def test_identical_image_url_across_owners_does_not_collide(isolated_store):
    shared_url = "https://example.com/diagram.png"
    isolated_store.add_image_chunks(
        [
            {
                "embedding": [1.0, 0.0],
                "content": "Alice's diagram",
                "metadata": {"image_url": shared_url, "alt_text": "alice's version", "owner": "alice"},
            },
            {
                "embedding": [0.0, 1.0],
                "content": "Bob's diagram",
                "metadata": {"image_url": shared_url, "alt_text": "bob's version", "owner": "bob"},
            },
        ]
    )

    assert isolated_store.image_count() == 2


def test_identical_document_id_and_start_seconds_across_owners_does_not_collide(isolated_store):
    isolated_store.add_audio_clip_chunks(
        [
            {
                "embedding": [1.0, 0.0],
                "content": "Alice's clip",
                "metadata": {"document_id": "meeting.mp3", "start_seconds": 0.0, "owner": "alice"},
            },
            {
                "embedding": [0.0, 1.0],
                "content": "Bob's clip",
                "metadata": {"document_id": "meeting.mp3", "start_seconds": 0.0, "owner": "bob"},
            },
        ]
    )

    assert isolated_store.audio_clip_count() == 2


# ---------- Dense search isolation ----------


def test_dense_search_never_returns_another_owners_chunk(isolated_store):
    isolated_store.add_embedded_chunks(
        [
            _md_chunk("alice", "Alice's private notes about the roadmap.", chunk_index=0),
            _md_chunk("bob", "Bob's private notes about the roadmap.", chunk_index=0),
        ]
    )

    alice_hits = isolated_store.query_embedding([1.0, 0.0], top_k=10, where={"owner": "alice"}, kb="markdown")
    bob_hits = isolated_store.query_embedding([1.0, 0.0], top_k=10, where={"owner": "bob"}, kb="markdown")

    assert len(alice_hits) == 1
    assert alice_hits[0]["metadata"]["owner"] == "alice"
    assert len(bob_hits) == 1
    assert bob_hits[0]["metadata"]["owner"] == "bob"


def test_image_dense_search_never_returns_another_owners_image(isolated_store):
    shared_url = "https://example.com/diagram.png"
    isolated_store.add_image_chunks(
        [
            {
                "embedding": [1.0, 0.0],
                "content": "Alice's diagram",
                "metadata": {"image_url": shared_url, "alt_text": "alice's version", "owner": "alice"},
            },
            {
                "embedding": [0.0, 1.0],
                "content": "Bob's diagram",
                "metadata": {"image_url": shared_url, "alt_text": "bob's version", "owner": "bob"},
            },
        ]
    )

    alice_hits = isolated_store.query_image_embedding([1.0, 0.0], top_k=10, where={"owner": "alice"})

    assert len(alice_hits) == 1
    assert alice_hits[0]["metadata"]["owner"] == "alice"


def test_audio_dense_search_never_returns_another_owners_clip(isolated_store):
    isolated_store.add_audio_clip_chunks(
        [
            {
                "embedding": [1.0, 0.0],
                "content": "Alice's clip",
                "metadata": {"document_id": "meeting.mp3", "start_seconds": 0.0, "owner": "alice"},
            },
            {
                "embedding": [0.0, 1.0],
                "content": "Bob's clip",
                "metadata": {"document_id": "meeting.mp3", "start_seconds": 0.0, "owner": "bob"},
            },
        ]
    )

    alice_hits = isolated_store.query_audio_clip_embedding([1.0, 0.0], top_k=10, where={"owner": "alice"})

    assert len(alice_hits) == 1
    assert alice_hits[0]["metadata"]["owner"] == "alice"


# ---------- BM25 search + cache-contents isolation ----------


def test_bm25_search_never_returns_another_owners_chunk(isolated_store):
    isolated_store.add_embedded_chunks(
        [
            _md_chunk("alice", "quokkas are excellent swimmers", chunk_index=0),
            _md_chunk("bob", "quokkas are excellent swimmers", chunk_index=0),
        ]
    )

    alice_hits = hybrid_retriever._bm25_search("quokkas swimmers", "markdown", top_k=10, where=None, owner="alice")

    assert len(alice_hits) == 1
    assert alice_hits[0]["metadata"]["owner"] == "alice"


def test_bm25_cache_contents_never_hold_another_owners_raw_text(isolated_store):
    # The structural fix this guards: _build_bm25_index fetches with
    # where={"owner": owner} at BUILD time, not just at search time -
    # so another owner's chunk text should never even enter the cached
    # index's .contents/.metadatas in the first place.
    isolated_store.add_embedded_chunks(
        [
            _md_chunk("alice", "alice's confidential quarterly numbers", chunk_index=0),
            _md_chunk("bob", "bob's confidential quarterly numbers", chunk_index=0),
        ]
    )

    hybrid_retriever._bm25_search("quarterly numbers", "markdown", top_k=10, where=None, owner="alice")

    cached = hybrid_retriever._bm25_index_by_owner_kb[("alice", "markdown")]
    assert cached is not None
    assert all(metadata["owner"] == "alice" for metadata in cached.metadatas)
    assert not any("bob" in content for content in cached.contents)


def test_bm25_cache_is_keyed_independently_per_owner(isolated_store):
    isolated_store.add_embedded_chunks(
        [
            _md_chunk("alice", "alice content", chunk_index=0),
            _md_chunk("bob", "bob content", chunk_index=0),
        ]
    )

    hybrid_retriever._bm25_search("content", "markdown", top_k=10, where=None, owner="alice")
    hybrid_retriever._bm25_search("content", "markdown", top_k=10, where=None, owner="bob")

    assert set(hybrid_retriever._bm25_index_by_owner_kb.keys()) == {("alice", "markdown"), ("bob", "markdown")}
    alice_cached = hybrid_retriever._bm25_index_by_owner_kb[("alice", "markdown")]
    bob_cached = hybrid_retriever._bm25_index_by_owner_kb[("bob", "markdown")]
    assert alice_cached.contents == ["alice content"]
    assert bob_cached.contents == ["bob content"]


# ---------- Parent-child resolution isolation ----------


def _web_parent_child_chunks(owner: str, document_id: str = "shared-page") -> list[EmbeddedChunk]:
    parent = EmbeddedChunk(
        content=f"{owner}'s full paragraph about onboarding.",
        embedding=[1.0, 0.0],
        metadata={
            "source_type": "web",
            "document_id": document_id,
            "chunk_index": 0,
            "chunk_role": "parent",
            "owner": owner,
        },
    )
    child = EmbeddedChunk(
        content=f"{owner}'s onboarding sentence.",
        embedding=[1.0, 0.0],
        metadata={
            "source_type": "web",
            "document_id": document_id,
            "chunk_index": 1,
            "chunk_role": "child",
            "parent_chunk_index": 0,
            "owner": owner,
        },
    )
    return [parent, child]


def test_parent_child_search_never_resolves_another_owners_parent(isolated_store):
    isolated_store.add_embedded_chunks([
        *_web_parent_child_chunks("alice"),
        *_web_parent_child_chunks("bob"),
    ])

    alice_hits = hybrid_retriever._parent_child_search(
        [1.0, 0.0], "web", top_k=10, where=None, owner="alice"
    )

    assert len(alice_hits) == 1
    assert alice_hits[0]["metadata"]["owner"] == "alice"
    assert "alice" in alice_hits[0]["content"]
    assert "bob" not in alice_hits[0]["content"]


def test_fetch_parents_never_returns_another_owners_parent_even_given_the_key(isolated_store):
    # Defense-in-depth check: even if a caller somehow obtained the
    # (document_id, chunk_index) key of another owner's parent (e.g. a
    # bug upstream), _fetch_parents' own independent owner clause must
    # still refuse to resolve it.
    isolated_store.add_embedded_chunks([
        *_web_parent_child_chunks("alice"),
        *_web_parent_child_chunks("bob"),
    ])

    resolved = hybrid_retriever._fetch_parents("web", [("shared-page", 0)], owner="alice")

    assert ("shared-page", 0) in resolved
    assert resolved[("shared-page", 0)]["metadata"]["owner"] == "alice"


# ---------- Delete isolation ----------


def test_delete_document_never_touches_another_owners_identically_named_document(isolated_store):
    isolated_store.add_embedded_chunks(
        [
            _md_chunk("alice", "Alice's private notes.", chunk_index=0),
            _md_chunk("bob", "Bob's private notes.", chunk_index=0),
        ]
    )

    deleted = isolated_store.delete_document("shared.md", "alice", kb="markdown")

    assert deleted == 1
    assert isolated_store.list_documents("markdown", "alice") == []
    assert len(isolated_store.list_documents("markdown", "bob")) == 1


def test_delete_repository_never_touches_another_owners_identically_named_repository(isolated_store):
    isolated_store.add_embedded_chunks(
        [
            _github_chunk("alice", "def alice_fn(): ..."),
            _github_chunk("bob", "def bob_fn(): ..."),
        ]
    )

    deleted = isolated_store.delete_repository("shared-org/shared-repo", "alice")

    assert deleted == 1
    assert isolated_store.list_documents("github", "alice") == []
    assert len(isolated_store.list_documents("github", "bob")) == 1

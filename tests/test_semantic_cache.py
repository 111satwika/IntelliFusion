"""Tests for app.generation.semantic_cache - see its module docstring."""

import pytest

from app.generation import semantic_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    """Module-global state (the cache list + the enabled flag) must
    not leak between tests regardless of pass/fail, mirroring the
    autouse-fixture pattern already used for llm_generator's/router's
    own module-global caches this session."""
    semantic_cache.clear_cache()
    semantic_cache.set_enabled(False)
    yield
    semantic_cache.clear_cache()
    semantic_cache.set_enabled(False)


def _chunk(document_id: str, chunk_index: int, content: str) -> dict:
    return {"content": content, "metadata": {"document_id": document_id, "chunk_index": chunk_index}}


_CHUNKS_A = [_chunk("doc1", 0, "Hybrid retrieval combines dense and BM25 search.")]


def test_disabled_by_default_never_hits_even_with_a_perfect_match():
    # store_answer() is itself a no-op while disabled (see the next
    # test), so an entry is forced in directly via enabled=True first,
    # THEN the module is switched back to its real default (disabled)
    # to prove a lookup against that existing, perfectly-matching entry
    # still misses.
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("q", [1.0, 0.0], _CHUNKS_A, "web", None, "cached answer")
    semantic_cache.set_enabled(False)

    result = semantic_cache.find_cached_answer([1.0, 0.0], _CHUNKS_A, "web", None)

    assert result is None


def test_store_answer_is_a_noop_while_disabled():
    semantic_cache.store_answer("q", [1.0, 0.0], _CHUNKS_A, "web", None, "cached answer")

    semantic_cache.set_enabled(True)
    result = semantic_cache.find_cached_answer([1.0, 0.0], _CHUNKS_A, "web", None)

    assert result is None  # nothing was ever actually stored


def test_hit_when_both_similarity_and_overlap_pass():
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("original question", [1.0, 0.0], _CHUNKS_A, "web", None, "cached answer")

    result = semantic_cache.find_cached_answer([1.0, 0.0], _CHUNKS_A, "web", None)

    assert result == "cached answer"


def test_miss_on_kb_mismatch_even_with_perfect_similarity_and_overlap():
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("q", [1.0, 0.0], _CHUNKS_A, "web", None, "cached answer")

    result = semantic_cache.find_cached_answer([1.0, 0.0], _CHUNKS_A, "github", None)

    assert result is None


def test_miss_on_repository_mismatch_even_with_perfect_similarity_and_overlap():
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("q", [1.0, 0.0], _CHUNKS_A, "github", "owner/repo-a", "cached answer")

    result = semantic_cache.find_cached_answer([1.0, 0.0], _CHUNKS_A, "github", "owner/repo-b")

    assert result is None


def test_miss_when_question_similarity_is_low_even_with_full_chunk_overlap():
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("q", [1.0, 0.0], _CHUNKS_A, "web", None, "cached answer")

    # Orthogonal vector -> cosine similarity 0.0, well below threshold.
    result = semantic_cache.find_cached_answer([0.0, 1.0], _CHUNKS_A, "web", None)

    assert result is None


def test_miss_when_chunk_overlap_is_low_even_with_perfect_question_similarity():
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("q", [1.0, 0.0], _CHUNKS_A, "web", None, "cached answer")

    unrelated_chunks = [_chunk("doc2", 0, "Something about a completely different topic.")]
    result = semantic_cache.find_cached_answer([1.0, 0.0], unrelated_chunks, "web", None)

    assert result is None


def test_content_hash_invalidates_a_stale_id_match():
    # Same (document_id, chunk_index) as _CHUNKS_A, but the CONTENT has
    # changed (e.g. the source was re-ingested/edited) - see module
    # docstring: chunk identity folds in a content hash specifically so
    # a stale id-only match can't serve outdated evidence as current.
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("q", [1.0, 0.0], _CHUNKS_A, "web", None, "cached answer")

    edited_chunks = [_chunk("doc1", 0, "This content has been completely rewritten since caching.")]
    result = semantic_cache.find_cached_answer([1.0, 0.0], edited_chunks, "web", None)

    assert result is None


def test_empty_chunks_on_lookup_returns_none_without_error():
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("q", [1.0, 0.0], _CHUNKS_A, "web", None, "cached answer")

    result = semantic_cache.find_cached_answer([1.0, 0.0], [], "web", None)

    assert result is None


def test_store_answer_noops_on_empty_answer():
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("q", [1.0, 0.0], _CHUNKS_A, "web", None, "")

    result = semantic_cache.find_cached_answer([1.0, 0.0], _CHUNKS_A, "web", None)

    assert result is None


def test_store_answer_noops_on_empty_chunks():
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("q", [1.0, 0.0], [], "web", None, "an answer")

    assert len(semantic_cache._cache) == 0


def test_enabled_override_takes_precedence_over_module_flag():
    semantic_cache.set_enabled(True)
    semantic_cache.store_answer("q", [1.0, 0.0], _CHUNKS_A, "web", None, "cached answer")

    # Module flag is ON, but an explicit enabled=False override must win.
    result = semantic_cache.find_cached_answer([1.0, 0.0], _CHUNKS_A, "web", None, enabled=False)

    assert result is None


def test_fifo_eviction_beyond_maxsize(monkeypatch):
    monkeypatch.setattr(semantic_cache, "_SEMANTIC_CACHE_MAXSIZE", 2)
    semantic_cache.set_enabled(True)

    for i in range(3):
        chunks = [_chunk(f"doc{i}", 0, f"unique content number {i}")]
        semantic_cache.store_answer(f"q{i}", [1.0, 0.0], chunks, "web", None, f"answer {i}")

    # The first entry (doc0) should have been evicted; the last two remain.
    assert len(semantic_cache._cache) == 2
    stored_answers = {entry.answer for entry in semantic_cache._cache}
    assert stored_answers == {"answer 1", "answer 2"}

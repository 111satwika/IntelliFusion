"""Tests for app.retrieval.hybrid_retriever.

The hybrid PDF retriever = dense retrieval + BM25 → Reciprocal Rank
Fusion → cross-encoder rerank. Each stage is unit-tested here in
isolation (faking the vector store, faking the cross-encoder, faking
the BM25 index build) so this file never loads a real model or touches
a real vector store on disk.
"""

import pytest

from app.retrieval import hybrid_retriever


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """Every test starts from a clean cache slate so one test's cached
    BM25 index or cross-encoder can't leak into another's setup."""
    hybrid_retriever._bm25_index_by_owner_kb = {}
    hybrid_retriever._cross_encoder = None
    yield
    hybrid_retriever._bm25_index_by_owner_kb = {}
    hybrid_retriever._cross_encoder = None


# ---- _bm25_tokenize --------------------------------------------------


def test_bm25_tokenize_lowercases_and_strips_punctuation():
    tokens = hybrid_retriever._bm25_tokenize("Hello, World! 2024-Q3.")
    assert tokens == ["hello", "world", "2024", "q3"]


def test_bm25_tokenize_drops_single_char_tokens():
    """Single-char tokens are noise (stray letters/punctuation debris),
    excluded to match the >1-char guard the tokenizer documents."""
    tokens = hybrid_retriever._bm25_tokenize("A B ab cd")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "ab" in tokens
    assert "cd" in tokens


# ---- _reciprocal_rank_fusion -----------------------------------------


def _hit(doc_id: str, chunk_index: int, content: str = "x") -> dict:
    return {
        "content": content,
        "metadata": {"document_id": doc_id, "chunk_index": chunk_index},
    }


def test_rrf_gives_highest_score_to_document_in_both_lists_at_top():
    """A document ranked #1 in both retrievers should beat one ranked
    #1 in only a single retriever."""
    a = _hit("doc-a", 0)
    b = _hit("doc-b", 0)
    c = _hit("doc-c", 0)

    fused = hybrid_retriever._reciprocal_rank_fusion(
        [
            [a, b, c],  # dense: a > b > c
            [a, c, b],  # bm25:  a > c > b
        ]
    )

    # Sorted by rrf_score descending.
    assert fused[0]["metadata"]["document_id"] == "doc-a"


def test_rrf_boosts_document_in_both_lists_even_when_neither_is_first():
    """A doc ranked #3 in both lists should beat one ranked #1 in only
    one list - the whole point of RRF's k damping."""
    top_only = _hit("top-only", 0)
    both = _hit("both", 0)
    filler1 = _hit("f1", 0)
    filler2 = _hit("f2", 0)

    fused = hybrid_retriever._reciprocal_rank_fusion(
        [
            [top_only, filler1, both],   # dense: top_only #1, both #3
            [filler2, filler1, both],    # bm25: both #3, top_only absent
        ]
    )

    fused_ids = [h["metadata"]["document_id"] for h in fused]
    assert fused_ids.index("both") < fused_ids.index("top-only")


def test_rrf_dedups_by_document_id_and_chunk_index():
    """The same (document_id, chunk_index) key from two retrievers
    should collapse into one fused entry, not two."""
    same_a = _hit("doc-a", 0, content="original")
    same_a_dup = _hit("doc-a", 0, content="duplicate")
    other = _hit("doc-b", 0)

    fused = hybrid_retriever._reciprocal_rank_fusion(
        [
            [same_a, other],
            [same_a_dup],
        ]
    )

    doc_ids = [h["metadata"]["document_id"] for h in fused]
    assert doc_ids.count("doc-a") == 1
    assert doc_ids.count("doc-b") == 1


def test_rrf_handles_hits_with_missing_document_id():
    """Chunks with no document_id metadata (rare synthetic case) should
    still participate in fusion via a content-based fallback key,
    rather than all collapsing onto one (None, None) key."""
    no_id_1 = {"content": "alpha", "metadata": {}}
    no_id_2 = {"content": "beta", "metadata": {}}

    fused = hybrid_retriever._reciprocal_rank_fusion([[no_id_1, no_id_2]])
    assert len(fused) == 2


# ---- _matches_where --------------------------------------------------


def test_matches_where_flat_equality():
    metadata = {"repository": "owner/repo", "content_type": "general"}
    assert hybrid_retriever._matches_where(metadata, {"repository": "owner/repo"})
    assert not hybrid_retriever._matches_where(metadata, {"repository": "other/repo"})


def test_matches_where_in_operator():
    metadata = {"content_type": "table_row"}
    assert hybrid_retriever._matches_where(
        metadata, {"content_type": {"$in": ["table", "table_row"]}}
    )
    assert not hybrid_retriever._matches_where(
        metadata, {"content_type": {"$in": ["class", "method"]}}
    )


def test_matches_where_and_wrapper():
    metadata = {"repository": "owner/repo", "content_type": "general"}
    where = {
        "$and": [
            {"repository": "owner/repo"},
            {"content_type": {"$in": ["general"]}},
        ]
    }
    assert hybrid_retriever._matches_where(metadata, where)


def test_matches_where_none_is_always_true():
    """A None filter (the dense retriever's unfiltered mode) should
    trivially match every chunk."""
    assert hybrid_retriever._matches_where({"anything": "here"}, None)


# ---- _bm25_search ----------------------------------------------------


def _prime_bm25_index(monkeypatch, contents, metadatas, owner="local"):
    """Bypass the real collection.get() call by stuffing a prebuilt
    BM25 index into the module-level cache."""
    from rank_bm25 import BM25Okapi
    tokenized = [hybrid_retriever._bm25_tokenize(doc) for doc in contents]
    index = hybrid_retriever._BM25Index(contents, metadatas, BM25Okapi(tokenized), tokenized)
    hybrid_retriever._bm25_index_by_owner_kb[(owner, "pdf")] = index


def test_bm25_search_ranks_by_term_frequency(monkeypatch):
    """A doc that repeats the query term should rank above one that
    mentions it once, all else equal."""
    _prime_bm25_index(
        monkeypatch,
        contents=[
            "revenue revenue revenue quarterly report",
            "revenue mentioned once here",
            "unrelated content about weather",
        ],
        metadatas=[{"document_id": "a", "chunk_index": 0}] * 3,
    )
    for i, m in enumerate(hybrid_retriever._bm25_index_by_owner_kb[("local", "pdf")].metadatas):
        m["chunk_index"] = i

    hits = hybrid_retriever._bm25_search("revenue", kb="pdf", top_k=3, where=None, owner="local")
    assert hits[0]["content"].startswith("revenue revenue revenue")


def test_bm25_search_skips_zero_score_docs(monkeypatch):
    """BM25 gives 0 to docs sharing no query tokens; those should be
    filtered out so they don't pollute the fused shortlist."""
    _prime_bm25_index(
        monkeypatch,
        contents=[
            "revenue and profit",
            "totally unrelated tokens like widget sprocket",
        ],
        metadatas=[{"document_id": "a", "chunk_index": 0}, {"document_id": "b", "chunk_index": 0}],
    )

    hits = hybrid_retriever._bm25_search("revenue", kb="pdf", top_k=5, where=None, owner="local")
    document_ids = [h["metadata"]["document_id"] for h in hits]
    assert "a" in document_ids
    assert "b" not in document_ids


def test_bm25_search_honors_where_filter(monkeypatch):
    """BM25 must apply the same metadata filter the dense retriever
    would, so fused results stay consistent with the caller's scope
    (e.g. repository=owner/foo)."""
    _prime_bm25_index(
        monkeypatch,
        contents=["revenue report", "revenue overview"],
        metadatas=[
            {"document_id": "a", "chunk_index": 0, "repository": "owner/foo"},
            {"document_id": "b", "chunk_index": 0, "repository": "owner/bar"},
        ],
    )

    hits = hybrid_retriever._bm25_search(
        "revenue", kb="pdf", top_k=5, where={"repository": "owner/foo"}, owner="local"
    )
    assert [h["metadata"]["repository"] for h in hits] == ["owner/foo"]


def test_bm25_search_returns_empty_when_index_missing(monkeypatch):
    """No chunks yet → BM25 build returns None → search short-circuits
    to []."""
    monkeypatch.setattr(hybrid_retriever, "_build_bm25_index", lambda kb, owner: None)
    hits = hybrid_retriever._bm25_search("anything", kb="pdf", top_k=5, where=None, owner="local")
    assert hits == []


# ---- invalidate_bm25_cache -------------------------------------------


def test_invalidate_bm25_cache_drops_specific_kb_for_all_owners():
    hybrid_retriever._bm25_index_by_owner_kb = {
        ("local", "pdf"): "sentinel", ("alice", "pdf"): "sentinel2", ("local", "docx"): "keep"
    }
    hybrid_retriever.invalidate_bm25_cache("pdf")
    assert ("local", "pdf") not in hybrid_retriever._bm25_index_by_owner_kb
    assert ("alice", "pdf") not in hybrid_retriever._bm25_index_by_owner_kb
    assert hybrid_retriever._bm25_index_by_owner_kb.get(("local", "docx")) == "keep"


def test_invalidate_bm25_cache_specific_owner_and_kb():
    hybrid_retriever._bm25_index_by_owner_kb = {("local", "pdf"): "a", ("alice", "pdf"): "b"}
    hybrid_retriever.invalidate_bm25_cache("pdf", owner="local")
    assert ("local", "pdf") not in hybrid_retriever._bm25_index_by_owner_kb
    assert hybrid_retriever._bm25_index_by_owner_kb.get(("alice", "pdf")) == "b"


def test_invalidate_bm25_cache_none_drops_everything():
    hybrid_retriever._bm25_index_by_owner_kb = {("local", "pdf"): "a", ("local", "docx"): "b"}
    hybrid_retriever.invalidate_bm25_cache(None)
    assert hybrid_retriever._bm25_index_by_owner_kb == {}


# ---- retrieve_pdf_hybrid (end-to-end wiring) ------------------------


def test_retrieve_pdf_hybrid_runs_all_three_stages(monkeypatch):
    """dense_search + bm25_search must both be called, their outputs
    fused, and the fused shortlist reranked by the cross-encoder."""
    calls: dict[str, int] = {"dense": 0, "bm25": 0, "cross_encoder": 0}

    def fake_dense(query_vector, kb, top_k, where):
        calls["dense"] += 1
        return [
            {"content": "dense-a", "metadata": {"document_id": "a", "chunk_index": 0}},
            {"content": "dense-b", "metadata": {"document_id": "b", "chunk_index": 0}},
        ]

    def fake_bm25(query_text, kb, top_k, where, owner):
        calls["bm25"] += 1
        return [
            {"content": "bm25-b", "metadata": {"document_id": "b", "chunk_index": 0}},
            {"content": "bm25-c", "metadata": {"document_id": "c", "chunk_index": 0}},
        ]

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            calls["cross_encoder"] += 1
            # Highest CE score to whichever pair mentions "b" - a
            # dumb rule, just enough to verify rerank order actually
            # depends on the CE's output rather than upstream RRF.
            return [10.0 if "b" in text else 1.0 for _, text in pairs]

    monkeypatch.setattr(hybrid_retriever, "_dense_search", fake_dense)
    monkeypatch.setattr(hybrid_retriever, "_bm25_search", fake_bm25)
    monkeypatch.setattr(hybrid_retriever, "_get_cross_encoder", lambda: FakeCE())

    results = hybrid_retriever.retrieve_pdf_hybrid(
        "b query",
        query_vector=[0.1, 0.2, 0.3],
        top_k=2,
        owner="local",
        where=None,
    )

    assert calls == {"dense": 1, "bm25": 1, "cross_encoder": 1}
    assert results[0]["metadata"]["document_id"] == "b"
    assert "cross_encoder_score" in results[0]


def test_retrieve_pdf_hybrid_respects_top_k(monkeypatch):
    """Final result length is clamped to top_k regardless of pool
    sizes upstream."""
    monkeypatch.setattr(
        hybrid_retriever,
        "_dense_search",
        lambda *a, **kw: [
            {"content": f"c{i}", "metadata": {"document_id": f"d{i}", "chunk_index": 0}}
            for i in range(20)
        ],
    )
    monkeypatch.setattr(hybrid_retriever, "_bm25_search", lambda *a, **kw: [])

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            return list(range(len(pairs)))

    monkeypatch.setattr(hybrid_retriever, "_get_cross_encoder", lambda: FakeCE())

    results = hybrid_retriever.retrieve_pdf_hybrid(
        "query", query_vector=[0.0], top_k=3, owner="local", where=None
    )
    assert len(results) == 3


def test_retrieve_pdf_hybrid_handles_empty_kb(monkeypatch):
    """Empty PDF KB → dense returns [], BM25 returns [] → RRF empty →
    cross-encoder never called → clean empty result."""
    monkeypatch.setattr(hybrid_retriever, "_dense_search", lambda *a, **kw: [])
    monkeypatch.setattr(hybrid_retriever, "_bm25_search", lambda *a, **kw: [])

    ce_called = {"n": 0}

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            ce_called["n"] += 1
            return [0.0] * len(pairs)

    monkeypatch.setattr(hybrid_retriever, "_get_cross_encoder", lambda: FakeCE())

    results = hybrid_retriever.retrieve_pdf_hybrid(
        "query", query_vector=[0.0], top_k=5, owner="local", where=None
    )
    assert results == []
    assert ce_called["n"] == 0


# ---- retrieve_hybrid multi-KB dispatch -------------------------------


@pytest.mark.parametrize("kb", ["pdf", "docx", "markdown"])
def test_retrieve_hybrid_supports_every_hybrid_kb(monkeypatch, kb):
    """DOCX and Markdown share the same pipeline as PDF - the same
    hybrid entry point should work for all three, threading the kb
    argument through to the underlying dense + BM25 calls."""
    captured_kbs: dict[str, str] = {}

    def fake_dense(query_vector, kb_arg, top_k, where):
        captured_kbs["dense"] = kb_arg
        return [{"content": "d", "metadata": {"document_id": "x", "chunk_index": 0}}]

    def fake_bm25(query_text, kb_arg, top_k, where, owner):
        captured_kbs["bm25"] = kb_arg
        return [{"content": "b", "metadata": {"document_id": "y", "chunk_index": 0}}]

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            return [1.0] * len(pairs)

    monkeypatch.setattr(hybrid_retriever, "_dense_search", fake_dense)
    monkeypatch.setattr(hybrid_retriever, "_bm25_search", fake_bm25)
    monkeypatch.setattr(hybrid_retriever, "_get_cross_encoder", lambda: FakeCE())

    results = hybrid_retriever.retrieve_hybrid(
        "q", query_vector=[0.0], kb=kb, top_k=5, owner="local", where=None
    )
    assert captured_kbs == {"dense": kb, "bm25": kb}
    assert len(results) > 0


def test_hybrid_kbs_registry_includes_all_seven():
    """Guardrail: if this set drifts (e.g. a refactor drops docx by
    accident), the retriever's per-KB dispatch would silently fall
    back to plain dense - fail fast here instead. github and web are
    included because they layer parent-child retrieval on top of the
    same dense+BM25+cross-encoder ensemble. audio/video are hybrid
    (BM25 catches exact spoken terms/names well) but deliberately NOT
    in _PARENT_CHILD_KBS - see that set's own docstring."""
    assert hybrid_retriever._HYBRID_KBS == {"pdf", "docx", "markdown", "github", "web", "audio", "video"}


def test_parent_child_kbs_registry():
    """github and web store parent+child chunks and get the
    small-to-big sentence-window retriever fused into RRF as a third
    ranked list - none of the other KBs do."""
    assert hybrid_retriever._PARENT_CHILD_KBS == {"github", "web"}


# ---- Parent-child search --------------------------------------------


def test_add_parent_role_filter_leaves_non_parent_child_kb_untouched():
    """PDF/DOCX/Markdown don't tag chunks with chunk_role, so the
    filter helper must NOT inject a parent-role clause for them or
    every dense query would return zero hits."""
    assert hybrid_retriever._add_parent_role_filter(None, "pdf") is None
    assert hybrid_retriever._add_parent_role_filter(
        {"repository": "x"}, "pdf"
    ) == {"repository": "x"}


def test_add_parent_role_filter_injects_role_clause_for_github():
    """A None where against github should become just the role clause."""
    assert hybrid_retriever._add_parent_role_filter(None, "github") == {
        "chunk_role": "parent"
    }


def test_add_parent_role_filter_combines_with_existing_where():
    """A flat where + role clause should compose under $and so Chroma
    treats them as a conjunction, not an ambiguous single filter."""
    combined = hybrid_retriever._add_parent_role_filter(
        {"repository": "owner/x"}, "github"
    )
    assert combined == {
        "$and": [{"repository": "owner/x"}, {"chunk_role": "parent"}]
    }


def test_add_parent_role_filter_extends_existing_and_clause():
    """An existing $and shouldn't get nested - the role clause should
    be appended to the same array so the resulting filter stays flat."""
    combined = hybrid_retriever._add_parent_role_filter(
        {"$and": [{"a": 1}, {"b": 2}]}, "web"
    )
    assert combined == {
        "$and": [{"a": 1}, {"b": 2}, {"chunk_role": "parent"}]
    }


def test_dense_search_scopes_to_parent_role_for_github(monkeypatch):
    """Github's dense search must filter to chunk_role=parent so it
    doesn't return sentence fragments alongside their parents."""
    captured = {}

    def fake_query_embedding(vector, top_k, where=None, kb=None):
        captured["where"] = where
        captured["kb"] = kb
        return []

    monkeypatch.setattr(hybrid_retriever, "query_embedding", fake_query_embedding)
    hybrid_retriever._dense_search(
        [0.0], kb="github", top_k=10, where={"repository": "o/r"}
    )

    assert captured["kb"] == "github"
    assert captured["where"] == {
        "$and": [{"repository": "o/r"}, {"chunk_role": "parent"}]
    }


def test_dense_search_leaves_pdf_where_unchanged(monkeypatch):
    """PDF chunks have no chunk_role tag - injecting one would drop
    every result."""
    captured = {}

    def fake_query_embedding(vector, top_k, where=None, kb=None):
        captured["where"] = where
        return []

    monkeypatch.setattr(hybrid_retriever, "query_embedding", fake_query_embedding)
    hybrid_retriever._dense_search(
        [0.0], kb="pdf", top_k=10, where={"repository": "o/r"}
    )

    assert captured["where"] == {"repository": "o/r"}


def test_parent_child_search_returns_empty_for_non_parent_child_kb(monkeypatch):
    """Only github/web store child chunks. Calling parent-child on any
    other KB should short-circuit to [] without hitting the store."""

    def unexpected(*a, **kw):
        raise AssertionError("query_embedding must not be called for non-PC KBs")

    monkeypatch.setattr(hybrid_retriever, "query_embedding", unexpected)
    result = hybrid_retriever._parent_child_search(
        [0.0], kb="pdf", top_k=10, where=None, owner="local"
    )
    assert result == []


def test_parent_child_search_dedupes_children_by_parent_index(monkeypatch):
    """Multiple child hits pointing at the same parent should collapse
    to ONE parent hit - not one per matched sentence - so the RRF
    input from this retriever gives each parent a single rank
    position."""

    def fake_query_embedding(vector, top_k, where=None, kb=None):
        # Both children point at the same parent (doc-a chunk_index=5).
        # A third child points at a different parent.
        return [
            {
                "content": "sentence 1 of parent A",
                "metadata": {
                    "document_id": "doc-a",
                    "chunk_index": 100,
                    "parent_chunk_index": 5,
                    "chunk_role": "child",
                },
                "similarity": 0.9,
            },
            {
                "content": "sentence 2 of parent A",
                "metadata": {
                    "document_id": "doc-a",
                    "chunk_index": 101,
                    "parent_chunk_index": 5,
                    "chunk_role": "child",
                },
                "similarity": 0.8,
            },
            {
                "content": "sentence 1 of parent B",
                "metadata": {
                    "document_id": "doc-b",
                    "chunk_index": 200,
                    "parent_chunk_index": 7,
                    "chunk_role": "child",
                },
                "similarity": 0.7,
            },
        ]

    class FakeCollection:
        def get(self, where=None, include=None):
            # Return a synthetic parent for whichever doc was asked for.
            doc_id = where["$and"][0]["document_id"]
            return {
                "documents": [f"full parent text for {doc_id}"],
                "metadatas": [
                    {
                        "document_id": doc_id,
                        "chunk_index": 5 if doc_id == "doc-a" else 7,
                        "chunk_role": "parent",
                    }
                ],
            }

    monkeypatch.setattr(hybrid_retriever, "query_embedding", fake_query_embedding)
    monkeypatch.setattr(hybrid_retriever, "_get_collection", lambda kb: FakeCollection())

    hits = hybrid_retriever._parent_child_search(
        [0.0], kb="github", top_k=10, where=None, owner="local"
    )

    # Two unique parents from three child hits.
    assert len(hits) == 2
    doc_ids = [h["metadata"]["document_id"] for h in hits]
    assert doc_ids == ["doc-a", "doc-b"]
    # First-ranked child's similarity is what feeds RRF ordering; make
    # sure the observability field is populated on the parent hit.
    assert hits[0]["matched_child_content"] == "sentence 1 of parent A"
    assert hits[0]["matched_child_similarity"] == 0.9
    # Parents carry the FULL parent content, not the child's fragment.
    assert hits[0]["content"] == "full parent text for doc-a"


def test_parent_child_search_forwards_where_filter(monkeypatch):
    """Caller-provided where (e.g. repository scope) must combine with
    the chunk_role=child clause so the child search stays scoped to
    the caller's intent."""
    captured = {}

    def fake_query_embedding(vector, top_k, where=None, kb=None):
        captured["where"] = where
        return []

    monkeypatch.setattr(hybrid_retriever, "query_embedding", fake_query_embedding)
    hybrid_retriever._parent_child_search(
        [0.0], kb="web", top_k=10, where={"source_type": "web"}, owner="local"
    )

    assert captured["where"] == {
        "$and": [{"source_type": "web"}, {"chunk_role": "child"}]
    }


def test_retrieve_hybrid_fuses_three_lists_for_github(monkeypatch):
    """For parent-child KBs, retrieve_hybrid must add parent-child hits
    as a third RRF input on top of dense + BM25."""
    calls: dict[str, int] = {"dense": 0, "bm25": 0, "parent_child": 0}
    captured_rrf_inputs: dict[str, int] = {}

    def fake_dense(query_vector, kb, top_k, where):
        calls["dense"] += 1
        return [{"content": "d", "metadata": {"document_id": "d1", "chunk_index": 0}}]

    def fake_bm25(query_text, kb, top_k, where, owner):
        calls["bm25"] += 1
        return [{"content": "b", "metadata": {"document_id": "b1", "chunk_index": 0}}]

    def fake_parent_child(query_vector, kb, top_k, where, owner):
        calls["parent_child"] += 1
        return [{"content": "pc", "metadata": {"document_id": "pc1", "chunk_index": 0}}]

    original_rrf = hybrid_retriever._reciprocal_rank_fusion

    def spy_rrf(ranked_lists, k=hybrid_retriever._RRF_K):
        captured_rrf_inputs["count"] = len(ranked_lists)
        return original_rrf(ranked_lists, k)

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            return [1.0] * len(pairs)

    monkeypatch.setattr(hybrid_retriever, "_dense_search", fake_dense)
    monkeypatch.setattr(hybrid_retriever, "_bm25_search", fake_bm25)
    monkeypatch.setattr(hybrid_retriever, "_parent_child_search", fake_parent_child)
    monkeypatch.setattr(hybrid_retriever, "_reciprocal_rank_fusion", spy_rrf)
    monkeypatch.setattr(hybrid_retriever, "_get_cross_encoder", lambda: FakeCE())

    hybrid_retriever.retrieve_hybrid(
        "q", query_vector=[0.0], kb="github", top_k=5, owner="local", where=None
    )

    assert calls == {"dense": 1, "bm25": 1, "parent_child": 1}
    # Three ranked lists went into RRF: dense + BM25 + parent-child.
    assert captured_rrf_inputs["count"] == 3


def test_retrieve_hybrid_uses_two_lists_for_pdf(monkeypatch):
    """PDF is NOT a parent-child KB - parent-child search must NOT
    fire, and RRF sees only two ranked lists."""
    called = {"parent_child": 0}
    captured_rrf_inputs: dict[str, int] = {}

    monkeypatch.setattr(
        hybrid_retriever,
        "_dense_search",
        lambda *a, **kw: [{"content": "d", "metadata": {"document_id": "x", "chunk_index": 0}}],
    )
    monkeypatch.setattr(hybrid_retriever, "_bm25_search", lambda *a, **kw: [])

    def unexpected_pc(*a, **kw):
        called["parent_child"] += 1
        return []

    original_rrf = hybrid_retriever._reciprocal_rank_fusion

    def spy_rrf(ranked_lists, k=hybrid_retriever._RRF_K):
        captured_rrf_inputs["count"] = len(ranked_lists)
        return original_rrf(ranked_lists, k)

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            return [1.0] * len(pairs)

    monkeypatch.setattr(hybrid_retriever, "_parent_child_search", unexpected_pc)
    monkeypatch.setattr(hybrid_retriever, "_reciprocal_rank_fusion", spy_rrf)
    monkeypatch.setattr(hybrid_retriever, "_get_cross_encoder", lambda: FakeCE())

    hybrid_retriever.retrieve_hybrid(
        "q", query_vector=[0.0], kb="pdf", top_k=5, owner="local", where=None
    )

    assert called["parent_child"] == 0
    assert captured_rrf_inputs["count"] == 2


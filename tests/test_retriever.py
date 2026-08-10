"""Tests for app.retrieval.retriever.

The embedding model and vector store are faked out here (via
monkeypatch) so this test only checks retriever's own wiring - that it
embeds the query with the same embedder used at ingestion time and
forwards top_k correctly - without the cost of loading a real model or
touching a real vector store.
"""

import pytest

from app.retrieval import retriever
from app.routing.router import RouteDecision


@pytest.fixture(autouse=True)
def _default_general_route(monkeypatch):
    """
    classify_route() calls the real embedding model to score its
    semantic-routing exemplars (see app.routing.router) - default
    every test here to the plain "general" (today's unscoped) route
    unless a test explicitly overrides this fixture's monkeypatch
    (later monkeypatch.setattr calls in a test body win), so the vast
    majority of retriever tests - which aren't testing routing at all
    - never load the real model.

    Also default list_populated_kbs() (used as the fallback when a
    RouteDecision carries no explicit .kbs signal) to "markdown" - an
    ordinary hybrid KB - rather than letting it fall through to the
    real vector store on disk.

    Every real KB now goes through either app.retrieval.github_adaptive
    (for "github") or app.retrieval.hybrid_retriever.retrieve_hybrid
    (for pdf/docx/markdown/web, and github's cross-KB slice used to
    before this fix - see retrieve()'s module docstring) - there is no
    more per-route dense-only dispatch to mock query_embedding
    against, so every test here mocks retrieve_hybrid directly.
    """
    monkeypatch.setattr(
        retriever, "classify_route", lambda query_text: RouteDecision(["general"], "default", {})
    )
    monkeypatch.setattr(retriever, "list_populated_kbs", lambda: ["markdown"])


def test_retrieve_embeds_query_and_forwards_to_vector_store(monkeypatch):
    captured = {}

    def fake_embed_texts(texts):
        captured["texts"] = texts
        return [[0.1, 0.2, 0.3]]

    def fake_retrieve_hybrid(query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None):
        captured["vector"] = query_vector
        captured["top_k"] = top_k
        return [{"content": "hello", "metadata": {}, "distance": 0.1, "similarity": 0.9}]

    monkeypatch.setattr(retriever, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retriever, "retrieve_hybrid", fake_retrieve_hybrid)

    results = retriever.retrieve("what is RAG", top_k=2)

    assert captured["texts"] == ["what is RAG"]
    assert captured["vector"] == [0.1, 0.2, 0.3]
    # retrieve_hybrid manages its own internal candidate pools
    # (_DENSE_POOL_SIZE/_BM25_POOL_SIZE/_RERANK_POOL_SIZE) - retrieve()
    # just forwards the caller's real top_k straight through.
    assert captured["top_k"] == 2
    assert results == [{"content": "hello", "metadata": {}, "distance": 0.1, "similarity": 0.9}]


def test_retrieve_passes_no_where_filter_by_default(monkeypatch):
    captured = {}

    def fake_retrieve_hybrid(query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None):
        captured["where"] = where
        return []

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever, "retrieve_hybrid", fake_retrieve_hybrid)

    retriever.retrieve("a question")

    assert captured["where"] is None


def test_retrieve_scopes_to_a_repository_when_given(monkeypatch):
    captured = {}

    def fake_retrieve_hybrid(query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None):
        captured["where"] = where
        return []

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever, "retrieve_hybrid", fake_retrieve_hybrid)

    retriever.retrieve("a question", repository="owner/repo")

    assert captured["where"] == {"repository": "owner/repo"}


def test_retrieve_runs_an_additional_query_when_the_table_route_matches(monkeypatch):
    # A query routed to both "table" and "general" should run TWO
    # separate retrieve_hybrid calls for the same KB - one scoped to
    # table content types, one unscoped - since these are genuinely
    # separate retrievers being combined, not a replacement (see
    # retrieve()'s docstring on the "table" route).
    calls = []

    def fake_retrieve_hybrid(query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None):
        calls.append(where)
        return []

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(
        retriever,
        "classify_route",
        lambda query_text: RouteDecision(["table", "general"], "rule", {}, kbs=["markdown"]),
    )

    retriever.retrieve("what does this table show?")

    assert len(calls) == 2
    assert {"content_type": {"$in": ["table", "table_row"]}} in calls
    assert None in calls


def test_retrieve_combines_repository_scope_with_the_table_route_filter(monkeypatch):
    captured_wheres = []

    def fake_retrieve_hybrid(query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None):
        captured_wheres.append(where)
        return []

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(retriever, "classify_route", lambda query_text: RouteDecision(["table"], "rule", {}))

    retriever.retrieve("what is in this table?", repository="owner/repo")

    assert {
        "$and": [{"repository": "owner/repo"}, {"content_type": {"$in": ["table", "table_row"]}}]
    } in captured_wheres


def test_retrieve_merges_and_dedupes_candidates_from_the_table_and_general_routes(monkeypatch):
    # The same chunk showing up in both the table-scoped query and the
    # unscoped "general" query (entirely plausible - a table chunk is
    # part of the general corpus too) should only appear once.
    shared_hit = {
        "content": "| A | B |\n| --- | --- |\n| 1 | 2 |",
        "metadata": {"document_id": "doc.md", "chunk_index": 0, "content_type": "table"},
        "distance": 0.1,
        "similarity": 0.9,
    }
    only_in_general = {
        "content": "just prose",
        "metadata": {"document_id": "readme.md", "chunk_index": 0},
        "distance": 0.3,
        "similarity": 0.7,
    }

    def fake_retrieve_hybrid(query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None):
        if where is not None:
            return [shared_hit]
        return [shared_hit, only_in_general]

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(
        retriever, "classify_route", lambda query_text: RouteDecision(["table", "general"], "rule", {})
    )

    results = retriever.retrieve("what's in the table", top_k=5)

    assert len(results) == 2
    assert shared_hit in results
    assert only_in_general in results


def test_retrieve_skips_the_image_route_entirely_for_text_search(monkeypatch):
    # "image" isn't a text-collection content_type filter at all - it
    # should never trigger an extra retrieve_hybrid call on its own.
    calls = []

    def fake_retrieve_hybrid(query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None):
        calls.append(where)
        return []

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(
        retriever, "classify_route", lambda query_text: RouteDecision(["general", "image"], "rule", {})
    )

    retriever.retrieve("show me a screenshot of the dashboard")

    assert calls == [None]


def test_retrieve_default_top_k_is_five(monkeypatch):
    captured = {}

    def fake_retrieve_hybrid(query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever, "retrieve_hybrid", fake_retrieve_hybrid)

    retriever.retrieve("a question")

    assert captured["top_k"] == 5


def test_retrieve_prefers_readme_intro_chunk_for_generic_overview_question(monkeypatch):
    # Reproduces a real bug: for a vague "what is this repository
    # about" question, plain embedding similarity can rank an unrelated
    # file's boilerplate (e.g. a CHANGELOG's "this project adheres to
    # Semantic Versioning" intro) above the README's actual project
    # description, since the boilerplate happens to share more surface
    # wording with the vague query. The README's own opening chunk
    # should win regardless of raw similarity.
    changelog_hit = {
        "content": "All notable changes to this project will be documented in this file.",
        "metadata": {"file_path": "CHANGELOG.md", "section": "Changelog"},
        "distance": 0.65,
        "similarity": 0.35,
    }
    readme_subsection_hit = {
        "content": "pip install python-dotenv",
        "metadata": {"file_path": "README.md", "section": "python-dotenv > Getting Started"},
        "distance": 0.9,
        "similarity": 0.1,
    }
    readme_intro_hit = {
        "content": "Python-dotenv reads key-value pairs from a .env file.",
        "metadata": {"file_path": "README.md", "section": "python-dotenv"},
        "distance": 0.95,
        "similarity": 0.05,
    }

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(
        retriever,
        "retrieve_hybrid",
        lambda query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None: [
            changelog_hit, readme_subsection_hit, readme_intro_hit,
        ],
    )

    results = retriever.retrieve("what is the repository about", top_k=1)

    assert results == [readme_intro_hit]


def test_retrieve_does_not_boost_readme_for_unrelated_questions(monkeypatch):
    # The README-overview boost should only kick in for generic
    # "about the project/repo" style questions - a specific, unrelated
    # question should still rank purely on similarity/title overlap.
    readme_hit = {
        "content": "Python-dotenv reads key-value pairs from a .env file.",
        "metadata": {"file_path": "README.md", "section": "python-dotenv"},
        "distance": 0.9,
        "similarity": 0.1,
    }
    specific_hit = {
        "content": "load_dotenv(override=True) overrides existing environment variables.",
        "metadata": {"file_path": "README.md", "section": "python-dotenv > File format > Variable expansion"},
        "distance": 0.05,
        "similarity": 0.95,
    }

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(
        retriever,
        "retrieve_hybrid",
        lambda query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None: [
            readme_hit, specific_hit,
        ],
    )

    results = retriever.retrieve("does load_dotenv override existing environment variables?", top_k=1)

    assert results == [specific_hit]


def test_retrieve_replaces_a_retrieved_table_row_with_the_complete_table(monkeypatch):
    # A table with 4 rows but only 1 row makes it into top_k: the
    # retrieved row should be swapped for the complete table so the
    # LLM sees all 4 rows, not just the one that happened to rank.
    row_hit = {
        "content": "| Role | Access |\n| --- | --- |\n| Admin | Yes |",
        "metadata": {"content_type": "table_row", "table_id": "doc.md::table_0"},
        "distance": 0.1,
        "similarity": 0.9,
    }
    full_table = {
        "content": (
            "| Role | Access |\n| --- | --- |\n"
            "| Admin | Yes |\n| Viewer | No |\n| Dev | No |\n| Release | Conditional |"
        ),
        "metadata": {"content_type": "table", "table_id": "doc.md::table_0"},
        "distance": None,
        "similarity": None,
    }

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(
        retriever,
        "retrieve_hybrid",
        lambda query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None: [row_hit],
    )
    monkeypatch.setattr(
        retriever, "get_table_chunk", lambda table_id: full_table if table_id == "doc.md::table_0" else None
    )

    results = retriever.retrieve("which role can deploy to production?", top_k=3)

    assert len(results) == 1
    assert results[0]["content"] == full_table["content"]
    assert "Release" in results[0]["content"]


def test_retrieve_does_not_duplicate_the_table_when_it_is_already_a_direct_hit(monkeypatch):
    table_hit = {
        "content": "whole table",
        "metadata": {"content_type": "table", "table_id": "doc.md::table_0"},
        "distance": 0.05,
        "similarity": 0.95,
    }
    row_hit = {
        "content": "one row",
        "metadata": {"content_type": "table_row", "table_id": "doc.md::table_0"},
        "distance": 0.2,
        "similarity": 0.8,
    }

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(
        retriever,
        "retrieve_hybrid",
        lambda query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None: [
            table_hit, row_hit,
        ],
    )

    def fail_if_called(table_id):
        raise AssertionError("get_table_chunk should not be called when the table is already a direct hit")

    monkeypatch.setattr(retriever, "get_table_chunk", fail_if_called)

    results = retriever.retrieve("a question", top_k=3)

    # The row is dropped (already represented by the direct table hit),
    # not duplicated.
    assert results == [table_hit]


def test_retrieve_leaves_non_table_chunks_untouched(monkeypatch):
    paragraph_hit = {
        "content": "just a paragraph",
        "metadata": {"content_type": "paragraph"},
        "distance": 0.1,
        "similarity": 0.9,
    }

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(
        retriever,
        "retrieve_hybrid",
        lambda query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None: [paragraph_hit],
    )

    results = retriever.retrieve("a question", top_k=3)

    assert results == [paragraph_hit]


def test_retrieve_replaces_a_retrieved_method_with_the_complete_class(monkeypatch):
    # A class with several methods but only 1 method makes it into
    # top_k: the retrieved method should be swapped for the complete
    # class so the LLM sees the constructor and sibling methods too,
    # not just the one method that happened to rank.
    method_hit = {
        "content": "[Method: Foo.bar]\n\ndef bar(self):\n    return 1",
        "metadata": {"content_type": "method", "class_id": "foo.py::class_0"},
        "distance": 0.1,
        "similarity": 0.9,
    }
    full_class = {
        "content": "[Class: Foo]\n\nclass Foo:\n    def __init__(self):\n        ...\n\n    def bar(self):\n        return 1",
        "metadata": {"content_type": "class", "class_id": "foo.py::class_0"},
        "distance": None,
        "similarity": None,
    }

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(
        retriever,
        "retrieve_hybrid",
        lambda query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None: [method_hit],
    )
    monkeypatch.setattr(
        retriever, "get_class_chunk", lambda class_id: full_class if class_id == "foo.py::class_0" else None
    )

    results = retriever.retrieve("what does bar return?", top_k=3)

    assert len(results) == 1
    assert results[0]["content"] == full_class["content"]
    assert "__init__" in results[0]["content"]


def test_retrieve_does_not_duplicate_the_class_when_it_is_already_a_direct_hit(monkeypatch):
    class_hit = {
        "content": "whole class",
        "metadata": {"content_type": "class", "class_id": "foo.py::class_0"},
        "distance": 0.05,
        "similarity": 0.95,
    }
    method_hit = {
        "content": "one method",
        "metadata": {"content_type": "method", "class_id": "foo.py::class_0"},
        "distance": 0.2,
        "similarity": 0.8,
    }

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(
        retriever,
        "retrieve_hybrid",
        lambda query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None: [
            class_hit, method_hit,
        ],
    )

    def fail_if_called(class_id):
        raise AssertionError("get_class_chunk should not be called when the class is already a direct hit")

    monkeypatch.setattr(retriever, "get_class_chunk", fail_if_called)

    results = retriever.retrieve("a question", top_k=3)

    # The method is dropped (already represented by the direct class
    # hit), not duplicated.
    assert results == [class_hit]


def test_retrieve_leaves_non_method_code_chunks_untouched(monkeypatch):
    function_hit = {
        "content": "[Function: baz]\n\ndef baz():\n    ...",
        "metadata": {"content_type": "function"},
        "distance": 0.1,
        "similarity": 0.9,
    }

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(
        retriever,
        "retrieve_hybrid",
        lambda query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None: [function_hit],
    )

    results = retriever.retrieve("a question", top_k=3)

    assert results == [function_hit]


def test_retrieve_promotes_a_title_matching_chunk_over_a_higher_similarity_one(monkeypatch):
    # A short, keyword-dense but unrelated chunk (higher raw cosine
    # similarity) should be outranked by a chunk whose section heading
    # is an almost exact lexical match for the query, once reranked -
    # this is the automation-rules "irrelevant docs" bug: a small
    # embedding model can rank a coincidentally keyword-heavy but
    # off-topic chunk above the chunk that is actually about what was
    # asked.
    unrelated_hit = {
        "content": "unrelated but keyword-dense text",
        "metadata": {"content_type": "paragraph", "section": "Integration with Wrike"},
        "distance": 0.35,
        "similarity": 0.65,
    }
    title_match_hit = {
        "content": "the actual matching content",
        "metadata": {
            "content_type": "paragraph",
            "section": "Assign the people working on a Task to its User Story when the User Story is set",
        },
        "distance": 0.45,
        "similarity": 0.55,
    }

    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(
        retriever,
        "retrieve_hybrid",
        lambda query_text, query_vector, kb, top_k, where=None, variants=None, hyde_answer=None, keywords=None: [
            unrelated_hit, title_match_hit,
        ],
    )

    results = retriever.retrieve(
        "Assign the people working on a Task to its User Story when the User Story is set",
        top_k=1,
    )

    assert results == [title_match_hit]


def test_retrieve_images_embeds_query_with_clip_and_forwards_to_image_collection(monkeypatch):
    captured = {}

    def fake_embed_text_for_image_search(text):
        captured["text"] = text
        return [0.1, 0.2]

    def fake_query_image_embedding(vector, top_k, where=None):
        captured["vector"] = vector
        captured["top_k"] = top_k
        return [
            {
                "content": "Authentication flow diagram",
                "metadata": {"image_url": "https://example.com/auth.png"},
                "distance": 0.1,
                "similarity": 0.9,
            }
        ]

    monkeypatch.setattr(retriever, "embed_text_for_image_search", fake_embed_text_for_image_search)
    monkeypatch.setattr(retriever, "query_image_embedding", fake_query_image_embedding)

    results = retriever.retrieve_images("how does authentication work?", top_k=2)

    assert captured["text"] == "how does authentication work?"
    assert captured["vector"] == [0.1, 0.2]
    # retrieve_images() now fetches (effectively) the entire image
    # collection rather than a similarity-based pool, since a fixed
    # pool size can still miss the correct image for longer,
    # natural-language-wrapped queries (see _ALL_IMAGES_POOL_SIZE).
    assert captured["top_k"] == retriever._ALL_IMAGES_POOL_SIZE
    assert results[0]["content"] == "Authentication flow diagram"


def test_retrieve_images_default_top_k_is_three(monkeypatch):
    captured = {}

    def fake_query_image_embedding(vector, top_k, where=None):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(retriever, "embed_text_for_image_search", lambda text: [0.0])
    monkeypatch.setattr(retriever, "query_image_embedding", fake_query_image_embedding)

    retriever.retrieve_images("a question")

    # Candidate pool is _ALL_IMAGES_POOL_SIZE regardless of top_k.
    assert captured["top_k"] == retriever._ALL_IMAGES_POOL_SIZE


def test_retrieve_images_promotes_an_alt_text_matching_image_over_a_higher_similarity_one(monkeypatch):
    # Mirrors test_retrieve_promotes_a_title_matching_chunk_over_a_higher_similarity_one:
    # CLIP visual similarity alone can rank a visually-similar but
    # off-topic screenshot above the screenshot whose alt_text is an
    # almost exact lexical match for the question.
    unrelated_hit = {
        "content": "some generic UI screenshot",
        "metadata": {"image_url": "https://example.com/generic.png", "alt_text": "epics and bugs"},
        "distance": 0.35,
        "similarity": 0.65,
    }
    alt_text_match_hit = {
        "content": "the actual matching screenshot",
        "metadata": {
            "image_url": "https://example.com/set_feature.png",
            "alt_text": "When a Bug is assigned to a User Story, inherit User Story's Feature",
        },
        "distance": 0.45,
        "similarity": 0.55,
    }

    monkeypatch.setattr(retriever, "embed_text_for_image_search", lambda text: [0.0])
    monkeypatch.setattr(
        retriever, "query_image_embedding", lambda vector, top_k, where=None: [unrelated_hit, alt_text_match_hit]
    )

    results = retriever.retrieve_images(
        "When a Bug is assigned to a User Story, inherit User Story's Feature",
        top_k=1,
    )

    assert results == [alt_text_match_hit]


def test_retrieve_audio_clips_embeds_query_with_clap_and_forwards_to_audio_collection(monkeypatch):
    captured = {}

    def fake_embed_text_for_audio_search(text):
        captured["text"] = text
        return [0.1, 0.2]

    def fake_query_audio_clip_embedding(vector, top_k, where=None):
        captured["vector"] = vector
        captured["top_k"] = top_k
        return [
            {
                "content": "Clip from meeting.mp3 at 00:00",
                "metadata": {"document_id": "meeting.mp3", "start_seconds": 0.0},
                "distance": 0.1,
                "similarity": 0.9,
            }
        ]

    monkeypatch.setattr(retriever, "embed_text_for_audio_search", fake_embed_text_for_audio_search)
    monkeypatch.setattr(retriever, "query_audio_clip_embedding", fake_query_audio_clip_embedding)

    results = retriever.retrieve_audio_clips("find audio that sounds like rain", top_k=2)

    assert captured["text"] == "find audio that sounds like rain"
    assert captured["vector"] == [0.1, 0.2]
    # Unlike retrieve_images(), no wide-pool override - top_k passes
    # straight through (see retrieve_audio_clips's own docstring for
    # why the rerank treatment isn't copied here).
    assert captured["top_k"] == 2
    assert results[0]["content"] == "Clip from meeting.mp3 at 00:00"


def test_retrieve_audio_clips_default_top_k_is_three(monkeypatch):
    captured = {}

    def fake_query_audio_clip_embedding(vector, top_k, where=None):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(retriever, "embed_text_for_audio_search", lambda text: [0.0])
    monkeypatch.setattr(retriever, "query_audio_clip_embedding", fake_query_audio_clip_embedding)

    retriever.retrieve_audio_clips("a question")

    assert captured["top_k"] == 3


def test_retrieve_audio_clips_scopes_to_kb_when_given(monkeypatch):
    # Regression test for a real bug: without this filter, a question
    # asked about one specific ingested video surfaced an unrelated
    # audio clip from a completely different file, since the
    # audio-clip collection has no other notion of "which KB" a clip
    # belongs to (see retrieve_audio_clips's own docstring).
    captured = {}

    def fake_query_audio_clip_embedding(vector, top_k, where=None):
        captured["where"] = where
        return []

    monkeypatch.setattr(retriever, "embed_text_for_audio_search", lambda text: [0.0])
    monkeypatch.setattr(retriever, "query_audio_clip_embedding", fake_query_audio_clip_embedding)

    retriever.retrieve_audio_clips("a question", kb="video")

    assert captured["where"] == {"source_kb": "video"}


def test_retrieve_audio_clips_applies_no_filter_for_cross_kb_chat(monkeypatch):
    captured = {}

    def fake_query_audio_clip_embedding(vector, top_k, where=None):
        captured["where"] = where
        return []

    monkeypatch.setattr(retriever, "embed_text_for_audio_search", lambda text: [0.0])
    monkeypatch.setattr(retriever, "query_audio_clip_embedding", fake_query_audio_clip_embedding)

    retriever.retrieve_audio_clips("a question", kb=None)

    assert captured["where"] is None


def test_retrieve_images_scopes_to_kb_when_given(monkeypatch):
    captured = {}

    def fake_query_image_embedding(vector, top_k, where=None):
        captured["where"] = where
        return []

    monkeypatch.setattr(retriever, "embed_text_for_image_search", lambda text: [0.0])
    monkeypatch.setattr(retriever, "query_image_embedding", fake_query_image_embedding)

    retriever.retrieve_images("a question", kb="video")

    assert captured["where"] == {"source_kb": "video"}


def test_retrieve_images_applies_no_filter_for_cross_kb_chat(monkeypatch):
    captured = {}

    def fake_query_image_embedding(vector, top_k, where=None):
        captured["where"] = where
        return []

    monkeypatch.setattr(retriever, "embed_text_for_image_search", lambda text: [0.0])
    monkeypatch.setattr(retriever, "query_image_embedding", fake_query_image_embedding)

    retriever.retrieve_images("a question", kb=None)

    assert captured["where"] is None

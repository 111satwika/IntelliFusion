"""Tests for cli.ask() - the pipeline wiring itself.

retrieve, build_prompt, and generate_answer are all faked out here so
this test only verifies cli.ask() calls them in the right order with
the right arguments, and returns whatever generate_answer produces -
it does not exercise the real pipeline (that's covered by each
module's own tests, plus manual end-to-end runs).

Fake `retrieve` chunks must be dicts with "content"/"metadata" keys,
not bare strings: cli._images_referenced_in_chunks (called from every
ask_with_vision path) introspects that shape looking for "[Image: ...]
(url)" markers in chunk content.
"""

import pytest

import cli
from app.routing.router import RouteDecision


@pytest.fixture(autouse=True)
def _default_image_route(monkeypatch):
    """
    classify_route() calls the real embedding model to score its
    semantic-routing exemplars (see app.routing.router). Default every
    test here to a decision that INCLUDES "image" (a test body's own
    later monkeypatch.setattr call wins), so the many existing
    image-retrieval-focused tests below - none of which are testing
    routing itself - keep calling retrieve_images() unconditionally
    exactly as they did before routing existed, and never load the
    real model.
    """
    monkeypatch.setattr(cli, "classify_route", lambda query_text: RouteDecision(["general", "image"], "rule", {}))


def test_ask_wires_retrieve_build_prompt_and_generate_answer_in_order(monkeypatch):
    calls = []

    def fake_retrieve(query_text, top_k, repository=None):
        calls.append(("retrieve", query_text, top_k))
        return ["chunk1", "chunk2"]

    def fake_build_prompt(query_text, chunks):
        calls.append(("build_prompt", query_text, chunks))
        return "THE PROMPT"

    def fake_generate_answer(prompt):
        calls.append(("generate_answer", prompt))
        return "THE ANSWER"

    monkeypatch.setattr(cli, "retrieve", fake_retrieve)
    monkeypatch.setattr(cli, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(cli, "generate_answer", fake_generate_answer)

    result = cli.ask("What is this repository about?", top_k=5)

    assert calls == [
        ("retrieve", "What is this repository about?", 5),
        ("build_prompt", "What is this repository about?", ["chunk1", "chunk2"]),
        ("generate_answer", "THE PROMPT"),
    ]
    assert result == "THE ANSWER"


def test_ask_default_top_k_is_three(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None, kb=None, **_: captured.setdefault("top_k", top_k) or [])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks, **_: "PROMPT")
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "ANSWER")

    cli.ask("a question")

    assert captured["top_k"] == 3


def test_ask_with_vision_uses_text_only_generator_when_no_relevant_images(monkeypatch):
    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None, kb=None, **_: [{"content": "chunk1", "metadata": {}}])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks, **_: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k, kb=None: [{"similarity": 0.05, "metadata": {"image_url": "u"}}])
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "TEXT ANSWER")
    monkeypatch.setattr(
        cli, "generate_answer_with_images", lambda prompt, urls: (_ for _ in ()).throw(AssertionError("should not be called"))
    )

    answer, image_hits = cli.ask_with_vision("a question", use_vision=True)

    assert answer == "TEXT ANSWER"
    assert image_hits == []


def test_ask_with_vision_defaults_to_text_only_even_with_relevant_images(monkeypatch):
    # use_vision defaults to False: on CPU-only setups, vision calls
    # reliably time out and starve the text-only fallback of CPU too
    # (see /memories/repo/rag-platform-notes.md) - so the vision model
    # must never be called for a plain (non-code) image unless a
    # caller explicitly opts in.
    relevant_hit = {"similarity": 0.9, "metadata": {"image_url": "https://example.com/a.png"}}

    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None, kb=None, **_: [{"content": "chunk1", "metadata": {}}])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks, **_: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k, kb=None: [relevant_hit])
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "TEXT ANSWER")
    monkeypatch.setattr(
        cli, "generate_answer_with_images", lambda prompt, urls: (_ for _ in ()).throw(AssertionError("should not be called"))
    )
    monkeypatch.setattr(
        cli, "get_chunk_by_document_id", lambda doc_id: {"metadata": {"content_type": "image_ocr"}}
    )

    answer, image_hits = cli.ask_with_vision("a question")

    assert answer == "TEXT ANSWER"
    assert image_hits == [relevant_hit]  # still returned for display, just not sent to the vision model


def test_ask_with_vision_auto_uses_vision_for_code_screenshot_even_without_use_vision(monkeypatch):
    # A relevant image already classified as a code screenshot at
    # ingestion time (content_type == "image_code") should be sent to
    # the vision model automatically, even with use_vision left at its
    # default False - this is exactly the case (verbatim code shown as
    # pixels) where plain OCR text is most likely to be inaccurate.
    relevant_hit = {"similarity": 0.9, "metadata": {"image_url": "https://example.com/code.png"}}
    calls = []

    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None, kb=None, **_: [{"content": "chunk1", "metadata": {}}])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks, **_: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k, kb=None: [relevant_hit])
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: (_ for _ in ()).throw(AssertionError("should not be called")))
    monkeypatch.setattr(
        cli, "get_chunk_by_document_id", lambda doc_id: {"metadata": {"content_type": "image_code"}}
    )

    def fake_generate_with_images(prompt, image_urls):
        calls.append((prompt, image_urls))
        return "VISION ANSWER"

    monkeypatch.setattr(cli, "generate_answer_with_images", fake_generate_with_images)

    answer, image_hits = cli.ask_with_vision("a question")

    assert answer == "VISION ANSWER"
    assert image_hits == [relevant_hit]
    assert calls == [("PROMPT", ["https://example.com/code.png"])]


def test_ask_with_vision_uses_vision_generator_when_relevant_images_found(monkeypatch):
    relevant_hit = {"similarity": 0.9, "metadata": {"image_url": "https://example.com/a.png"}}
    calls = []

    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None, kb=None, **_: [{"content": "chunk1", "metadata": {}}])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks, **_: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k, kb=None: [relevant_hit])
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: (_ for _ in ()).throw(AssertionError("should not be called")))

    def fake_generate_with_images(prompt, image_urls):
        calls.append((prompt, image_urls))
        return "VISION ANSWER"

    monkeypatch.setattr(cli, "generate_answer_with_images", fake_generate_with_images)

    answer, image_hits = cli.ask_with_vision("a question", use_vision=True)

    assert answer == "VISION ANSWER"
    assert image_hits == [relevant_hit]
    assert calls == [("PROMPT", ["https://example.com/a.png"])]


def test_ask_with_vision_falls_back_to_text_when_image_retrieval_fails(monkeypatch):
    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None, kb=None, **_: [{"content": "chunk1", "metadata": {}}])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks, **_: "PROMPT")

    def fake_retrieve_images(q, top_k, kb=None):
        raise RuntimeError("CLIP model unavailable")

    monkeypatch.setattr(cli, "retrieve_images", fake_retrieve_images)
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "TEXT ANSWER")

    answer, image_hits = cli.ask_with_vision("a question", use_vision=True)

    assert answer == "TEXT ANSWER"
    assert image_hits == []


def test_ask_with_vision_falls_back_to_text_when_vision_generation_fails(monkeypatch):
    relevant_hit = {"similarity": 0.9, "metadata": {"image_url": "https://example.com/a.png"}}

    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None, kb=None, **_: [{"content": "chunk1", "metadata": {}}])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks, **_: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k, kb=None: [relevant_hit])
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "TEXT ANSWER")

    def fake_generate_with_images(prompt, image_urls):
        raise TimeoutError("vision model too slow")

    monkeypatch.setattr(cli, "generate_answer_with_images", fake_generate_with_images)

    answer, image_hits = cli.ask_with_vision("a question", use_vision=True)

    assert answer == "TEXT ANSWER"
    assert image_hits == [relevant_hit]  # still reported as used, even though the fallback answer is text-only


def test_prepare_context_uses_contextualized_query_for_retrieval_but_original_for_prompt(monkeypatch):
    # Conversation memory (see app.retrieval.contextualize): retrieval
    # must search for the RESOLVED standalone form of a follow-up, but
    # the LLM prompt must still show the user's ORIGINAL wording (plus
    # history) - the two should never receive the same value here.
    from app.retrieval.contextualize import ContextualizeResult

    calls = []
    fake_result = ContextualizeResult(
        original="what about tests for that?", resolved="what tests exist for hybrid retrieval?",
        used_llm=True, fell_back=False,
    )
    monkeypatch.setattr(cli, "contextualize_query", lambda *a, **kw: fake_result)
    monkeypatch.setattr(
        cli, "retrieve",
        lambda q, top_k, repository=None, kb=None, **_: calls.append(("retrieve", q)) or [{"content": "c", "metadata": {}}],
    )
    monkeypatch.setattr(
        cli, "build_prompt",
        lambda q, chunks, **kw: calls.append(("build_prompt", q, kw.get("history"))) or "PROMPT",
    )
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k, kb=None: [])

    history = [{"question": "How does hybrid retrieval work?", "answer": "Dense + BM25."}]
    prompt, _images, _urls, _chunks, _crag, contextualization, _cached_answer, _audio_clip_hits = cli._prepare_context_and_images(
        "what about tests for that?", top_k=3, use_vision=False, repository=None, kb=None,
        history=history, conversation_memory_enabled=True,
    )

    assert ("retrieve", "what tests exist for hybrid retrieval?") in calls
    assert ("build_prompt", "what about tests for that?", history) in calls
    assert contextualization is fake_result


def _stub_prepare_context_deps(monkeypatch, chunks=None):
    """Common stubbing for the semantic-cache tests below: a minimal
    retrieve/build_prompt/retrieve_images/embed_texts setup so
    _prepare_context_and_images can run end-to-end without touching
    the real embedding model or vector store."""
    chunks = chunks if chunks is not None else [{"content": "c", "metadata": {"document_id": "d", "chunk_index": 0}}]
    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None, kb=None, **_: chunks)
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks, **kw: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k, kb=None: [])
    monkeypatch.setattr(cli, "embed_texts", lambda texts: [[1.0, 0.0]])
    return chunks


def test_prepare_context_returns_cached_answer_on_a_semantic_cache_hit(monkeypatch):
    _stub_prepare_context_deps(monkeypatch)
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: True)
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: "a cached answer")

    *_rest, cached_answer, _audio_clip_hits = cli._prepare_context_and_images(
        "a question", top_k=3, use_vision=False, repository=None, kb=None,
    )

    assert cached_answer == "a cached answer"


def test_prepare_context_cached_answer_is_none_on_a_semantic_cache_miss(monkeypatch):
    _stub_prepare_context_deps(monkeypatch)
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: True)
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: None)

    *_rest, cached_answer, _audio_clip_hits = cli._prepare_context_and_images(
        "a question", top_k=3, use_vision=False, repository=None, kb=None,
    )

    assert cached_answer is None


def test_prepare_context_never_looks_up_semantic_cache_when_history_is_non_empty(monkeypatch):
    # Gate 1 (see app.generation.semantic_cache's module docstring): a
    # thread's prior turns get rendered into the prompt regardless of
    # any cache decision, so a mid-conversation question must never
    # even attempt a lookup.
    _stub_prepare_context_deps(monkeypatch)
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: True)
    calls = []
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: calls.append(1) or "should never be used")

    history = [{"question": "earlier question", "answer": "earlier answer"}]
    *_rest, cached_answer, _audio_clip_hits = cli._prepare_context_and_images(
        "a question", top_k=3, use_vision=False, repository=None, kb=None, history=history,
    )

    assert cached_answer is None
    assert not calls


def test_prepare_context_never_looks_up_semantic_cache_when_vision_images_are_attached(monkeypatch):
    # Gate 3: a code-screenshot hit makes vision_image_urls non-empty
    # even with use_vision=False (auto-enabled for code screenshots -
    # see _is_code_screenshot) - the semantic cache has no visibility
    # into which image was analyzed, so it must never fire here either.
    _stub_prepare_context_deps(monkeypatch)
    code_screenshot_hit = {"similarity": 0.9, "metadata": {"image_url": "https://example.com/code.png"}}
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k, kb=None: [code_screenshot_hit])
    monkeypatch.setattr(cli, "get_chunk_by_document_id", lambda doc_id: {"metadata": {"content_type": "image_code"}})
    monkeypatch.setattr(cli, "classify_route", lambda query_text: RouteDecision(["general", "image"], "rule", {}))
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: True)
    calls = []
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: calls.append(1) or "should never be used")

    *_rest, cached_answer, _audio_clip_hits = cli._prepare_context_and_images(
        "a question", top_k=3, use_vision=False, repository=None, kb=None,
    )

    assert cached_answer is None
    assert not calls


def test_prepare_context_never_looks_up_semantic_cache_when_disabled(monkeypatch):
    _stub_prepare_context_deps(monkeypatch)
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: False)
    calls = []
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: calls.append(1) or "should never be used")

    *_rest, cached_answer, _audio_clip_hits = cli._prepare_context_and_images(
        "a question", top_k=3, use_vision=False, repository=None, kb=None,
    )

    assert cached_answer is None
    assert not calls


def test_ask_with_vision_uses_cached_answer_and_skips_generation(monkeypatch):
    _stub_prepare_context_deps(monkeypatch)
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: True)
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: "a cached answer")
    monkeypatch.setattr(
        cli, "generate_answer", lambda prompt: (_ for _ in ()).throw(AssertionError("should not be called"))
    )
    monkeypatch.setattr(
        cli, "generate_answer_with_images", lambda prompt, urls: (_ for _ in ()).throw(AssertionError("should not be called"))
    )

    answer, _images = cli.ask_with_vision("a question")

    assert answer == "a cached answer"


def test_ask_with_vision_stores_answer_only_on_a_genuine_miss(monkeypatch):
    _stub_prepare_context_deps(monkeypatch)
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: True)
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "a fresh answer")
    store_calls = []
    monkeypatch.setattr(cli.semantic_cache, "store_answer", lambda *a, **kw: store_calls.append(a))

    cli.ask_with_vision("a question")

    assert len(store_calls) == 1
    assert store_calls[0][-1] == "a fresh answer"  # answer is the last positional arg


def test_ask_with_vision_does_not_store_on_a_cache_hit(monkeypatch):
    _stub_prepare_context_deps(monkeypatch)
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: True)
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: "a cached answer")
    store_calls = []
    monkeypatch.setattr(cli.semantic_cache, "store_answer", lambda *a, **kw: store_calls.append(a))

    cli.ask_with_vision("a question")

    assert not store_calls


def test_ask_with_vision_stream_yields_cached_answer_as_one_chunk_and_reports_used_cache(monkeypatch):
    _stub_prepare_context_deps(monkeypatch)
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: True)
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: "a cached answer")
    monkeypatch.setattr(
        cli, "generate_answer_stream", lambda prompt: (_ for _ in ()).throw(AssertionError("should not be called"))
    )

    token_iter, _images, _chunks, _crag, _ctx, used_semantic_cache, _audio_clip_hits = cli.ask_with_vision_stream("a question")

    assert list(token_iter) == ["a cached answer"]
    assert used_semantic_cache is True


def test_ask_with_vision_stream_reports_used_cache_false_on_a_miss(monkeypatch):
    _stub_prepare_context_deps(monkeypatch)
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: True)
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "generate_answer_stream", lambda prompt: iter(["fresh ", "answer"]))
    monkeypatch.setattr(cli.semantic_cache, "store_answer", lambda *a, **kw: None)

    token_iter, _images, _chunks, _crag, _ctx, used_semantic_cache, _audio_clip_hits = cli.ask_with_vision_stream("a question")

    assert list(token_iter) == ["fresh ", "answer"]
    assert used_semantic_cache is False


def test_ask_with_vision_stream_stores_answer_after_streaming_completes_on_a_miss(monkeypatch):
    _stub_prepare_context_deps(monkeypatch)
    monkeypatch.setattr(cli.semantic_cache, "is_enabled", lambda: True)
    monkeypatch.setattr(cli.semantic_cache, "find_cached_answer", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "generate_answer_stream", lambda prompt: iter(["fresh ", "answer"]))
    store_calls = []
    monkeypatch.setattr(cli.semantic_cache, "store_answer", lambda *a, **kw: store_calls.append(a))

    token_iter, *_rest = cli.ask_with_vision_stream("a question")
    list(token_iter)  # drain the generator so the post-stream store hook actually runs

    assert len(store_calls) == 1
    assert store_calls[0][-1] == "fresh answer"


def test_ask_with_vision_skips_image_retrieval_when_query_is_not_routed_to_images(monkeypatch):
    # Query routing (see app.routing.router): when the query doesn't
    # match the "image" route at all, retrieve_images() should never
    # even be called - it uses a separate CLIP model/vector space, so
    # skipping it entirely for clearly non-visual questions avoids that
    # extra cost.
    monkeypatch.setattr(cli, "classify_route", lambda query_text: RouteDecision(["general"], "default", {}))
    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None, kb=None, **_: [{"content": "chunk1", "metadata": {}}])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks, **_: "PROMPT")
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "TEXT ANSWER")

    def fail_if_called(query_text, top_k, kb=None):
        raise AssertionError("retrieve_images should not be called when the image route isn't matched")

    monkeypatch.setattr(cli, "retrieve_images", fail_if_called)

    answer, image_hits = cli.ask_with_vision("what is this repository about")

    assert answer == "TEXT ANSWER"
    assert image_hits == []

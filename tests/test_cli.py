"""Tests for cli.ask() - the pipeline wiring itself.

retrieve, build_prompt, and generate_answer are all faked out here so
this test only verifies cli.ask() calls them in the right order with
the right arguments, and returns whatever generate_answer produces -
it does not exercise the real pipeline (that's covered by each
module's own tests, plus manual end-to-end runs).
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

    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None: captured.setdefault("top_k", top_k) or [])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks: "PROMPT")
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "ANSWER")

    cli.ask("a question")

    assert captured["top_k"] == 3


def test_ask_with_vision_uses_text_only_generator_when_no_relevant_images(monkeypatch):
    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None: ["chunk1"])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k: [{"similarity": 0.05, "metadata": {"image_url": "u"}}])
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

    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None: ["chunk1"])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k: [relevant_hit])
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

    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None: ["chunk1"])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k: [relevant_hit])
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

    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None: ["chunk1"])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k: [relevant_hit])
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
    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None: ["chunk1"])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks: "PROMPT")

    def fake_retrieve_images(q, top_k):
        raise RuntimeError("CLIP model unavailable")

    monkeypatch.setattr(cli, "retrieve_images", fake_retrieve_images)
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "TEXT ANSWER")

    answer, image_hits = cli.ask_with_vision("a question", use_vision=True)

    assert answer == "TEXT ANSWER"
    assert image_hits == []


def test_ask_with_vision_falls_back_to_text_when_vision_generation_fails(monkeypatch):
    relevant_hit = {"similarity": 0.9, "metadata": {"image_url": "https://example.com/a.png"}}

    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None: ["chunk1"])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks: "PROMPT")
    monkeypatch.setattr(cli, "retrieve_images", lambda q, top_k: [relevant_hit])
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "TEXT ANSWER")

    def fake_generate_with_images(prompt, image_urls):
        raise TimeoutError("vision model too slow")

    monkeypatch.setattr(cli, "generate_answer_with_images", fake_generate_with_images)

    answer, image_hits = cli.ask_with_vision("a question", use_vision=True)

    assert answer == "TEXT ANSWER"
    assert image_hits == [relevant_hit]  # still reported as used, even though the fallback answer is text-only


def test_ask_with_vision_skips_image_retrieval_when_query_is_not_routed_to_images(monkeypatch):
    # Query routing (see app.routing.router): when the query doesn't
    # match the "image" route at all, retrieve_images() should never
    # even be called - it uses a separate CLIP model/vector space, so
    # skipping it entirely for clearly non-visual questions avoids that
    # extra cost.
    monkeypatch.setattr(cli, "classify_route", lambda query_text: RouteDecision(["general"], "default", {}))
    monkeypatch.setattr(cli, "retrieve", lambda q, top_k, repository=None: ["chunk1"])
    monkeypatch.setattr(cli, "build_prompt", lambda q, chunks: "PROMPT")
    monkeypatch.setattr(cli, "generate_answer", lambda prompt: "TEXT ANSWER")

    def fail_if_called(query_text, top_k):
        raise AssertionError("retrieve_images should not be called when the image route isn't matched")

    monkeypatch.setattr(cli, "retrieve_images", fail_if_called)

    answer, image_hits = cli.ask_with_vision("what is this repository about")

    assert answer == "TEXT ANSWER"
    assert image_hits == []

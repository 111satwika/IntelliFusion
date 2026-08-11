"""Tests for app.routing.router.

Semantic routing embeds the query + a handful of route exemplars with
the real sentence-transformers model (no store/network dependency, but
not free either) - these tests accept that one-time model load cost
rather than faking it out, since the whole point of this module is the
actual embedding-similarity behavior, not just wiring.

KB semantic scoring is corpus-derived (see module docstring), so it
DOES touch app.vectorstore.store - most tests below still exercise it
against the real (possibly empty, possibly populated) project store,
same as before, but a few isolate the scoring logic itself by
monkeypatching sample_kb_embeddings with controlled vectors, so those
specific assertions don't depend on what happens to be ingested.
"""

import pytest

from app.routing import router
from app.routing.router import RouteDecision, classify_route


@pytest.fixture(autouse=True)
def _reset_kb_routing_cache():
    """Every test that monkeypatches sample_kb_embeddings must not leak
    its fake samples into a later test - clear the cache before AND
    after, so a failed assertion mid-test still leaves things clean
    (unlike a manual cleanup call at the end of the test body, which a
    raised AssertionError would skip)."""
    router._kb_sample_embeddings_cache = None
    yield
    router._kb_sample_embeddings_cache = None


def test_rule_based_keyword_routes_to_code():
    decision = classify_route("show me the function definition for parse_config")

    assert "code" in decision.routes
    assert "general" in decision.routes  # always included as a safety net


def test_rule_based_keyword_routes_to_table():
    decision = classify_route("what values are in this table?")

    assert "table" in decision.routes
    assert "general" in decision.routes


def test_rule_based_keyword_routes_to_image():
    decision = classify_route("show me a screenshot of the dashboard")

    assert "image" in decision.routes
    assert "general" in decision.routes


def test_rule_based_keyword_routes_to_sound():
    decision = classify_route("find a recording that sounds like a car engine")

    assert "sound" in decision.routes
    assert "general" in decision.routes


def test_semantic_routing_catches_a_sound_paraphrase_without_keywords():
    # No literal "sounds like"/"audio clip" keyword here, but this is
    # semantically an acoustic-similarity question.
    decision = classify_route("is there a clip with laughter in it")

    assert decision.method in {"semantic", "rule+semantic"}
    assert "sound" in decision.routes or decision.scores["sound"] > 0.0


def test_unmatched_query_falls_back_to_general_only():
    decision = classify_route("hello there")

    assert decision.routes == ["general"]
    assert decision.method == "default"


def test_semantic_routing_catches_a_code_paraphrase_without_keywords():
    # No literal "code"/"function"/"method" keyword here, but this is
    # semantically a code question - the semantic router (not the
    # keyword router) should still catch it.
    decision = classify_route("how does the login process work internally")

    assert decision.method in {"semantic", "rule+semantic"}
    assert "code" in decision.routes or decision.scores["code"] > 0.0


def test_route_decision_equality_and_repr():
    a = RouteDecision(["general"], "default", {"code": 0.1})
    b = RouteDecision(["general"], "default", {"code": 0.1})

    assert a == b
    assert "general" in repr(a)


def test_no_kb_signal_leaves_kbs_empty(monkeypatch):
    # No explicit source mentioned at all - kbs should be empty,
    # leaving the "which KB(s) to search" decision to the caller (see
    # app.retrieval.retriever, which falls back to every populated KB).
    # Mocked out (unlike most tests in this file - see module
    # docstring) because this assertion is a strict "no KB scored
    # above threshold": against the REAL corpus, "hello there" could
    # occasionally draw a random sample that happens to score >= the
    # threshold purely by chance, making this specific test flaky in a
    # way the softer "in decision.routes"-style assertions elsewhere in
    # this file are not.
    monkeypatch.setattr(router, "sample_kb_embeddings", lambda kb, limit=40: [])

    decision = classify_route("hello there")

    assert decision.kbs == []


def test_rule_based_keyword_routes_to_the_pdf_kb():
    decision = classify_route("what does the pdf report say about revenue")

    assert "pdf" in decision.kbs


def test_rule_based_keyword_routes_to_the_github_kb():
    decision = classify_route("how does the github repository handle authentication")

    assert "github" in decision.kbs


def test_rule_based_keyword_routes_to_the_docx_kb():
    decision = classify_route("what does the word document say about onboarding")

    assert "docx" in decision.kbs


def test_rule_based_keyword_routes_to_the_web_kb():
    decision = classify_route("what does the documentation site say about setup")

    assert "web" in decision.kbs


def test_rule_based_keyword_routes_to_the_audio_kb():
    decision = classify_route("what does the podcast say about pricing")

    assert "audio" in decision.kbs


def test_rule_based_keyword_routes_to_the_video_kb():
    decision = classify_route("what does the screen recording show about setup")

    assert "video" in decision.kbs


def test_semantic_kb_scoring_uses_corpus_samples_not_hardcoded_exemplars(monkeypatch):
    # A query vector identical to one of "web"'s sampled chunk vectors
    # should score a perfect 1.0 for "web" and 0.0 for every KB with no
    # sample at all - proving the score comes from the injected sample,
    # not any hand-written exemplar text.
    def fake_sample(kb, limit=40):
        if kb == "web":
            return [[1.0, 0.0], [0.0, 1.0]]
        return []

    monkeypatch.setattr(router, "sample_kb_embeddings", fake_sample)

    scores = router._semantic_kbs([1.0, 0.0])

    assert scores["web"] == 1.0
    assert scores["markdown"] == 0.0
    assert scores["pdf"] == 0.0


def test_semantic_kb_scoring_caches_samples_across_calls(monkeypatch):
    calls = []

    def fake_sample(kb, limit=40):
        calls.append(kb)
        return [[1.0, 0.0]]

    monkeypatch.setattr(router, "sample_kb_embeddings", fake_sample)

    router._semantic_kbs([1.0, 0.0])
    first_call_count = len(calls)
    router._semantic_kbs([0.0, 1.0])

    assert len(calls) == first_call_count  # second call was served from cache, no re-sampling


def test_invalidate_kb_routing_cache_forces_resample(monkeypatch):
    calls = []

    def fake_sample(kb, limit=40):
        calls.append(kb)
        return [[1.0, 0.0]]

    monkeypatch.setattr(router, "sample_kb_embeddings", fake_sample)

    router._semantic_kbs([1.0, 0.0])
    router.invalidate_kb_routing_cache("web")
    router._semantic_kbs([1.0, 0.0])

    # Every KB got re-sampled after invalidation (invalidate always
    # clears the whole cache - see its own docstring for why).
    assert calls.count("web") == 2


def test_semantic_kb_scoring_treats_empty_sample_as_zero_score(monkeypatch):
    monkeypatch.setattr(router, "sample_kb_embeddings", lambda kb, limit=40: [])

    scores = router._semantic_kbs([1.0, 0.0])

    assert all(score == 0.0 for score in scores.values())


def test_a_dominant_kb_excludes_weaker_kbs_that_only_cleared_the_floor(monkeypatch):
    # Reproduces the real "what is automation rule" case: web scored
    # ~0.46 while markdown/docx/github all cleared the 0.2 absolute
    # floor by coincidence (~0.2-0.3) without being genuine contenders.
    # docx here scores 0.3 - comfortably above the 0.2 floor on its
    # own, but well below 70% of web's 1.0 - so it should be dropped
    # from .kbs even though .kb_scores still reports it. embed_texts is
    # monkeypatched too so the query vector is a known [1.0, 0.0],
    # matching what classify_route() actually feeds into _semantic_kbs
    # (unlike test_semantic_kb_scoring_* above, which call _semantic_kbs
    # directly and can pass a controlled vector without this step).
    monkeypatch.setattr(router, "embed_texts", lambda texts: [[1.0, 0.0]])

    def fake_sample(kb, limit=40):
        if kb == "web":
            return [[1.0, 0.0]]
        if kb == "docx":
            return [[0.3, 0.9539392014169456]]  # cosine(query, this) == 0.3
        return []

    monkeypatch.setattr(router, "sample_kb_embeddings", fake_sample)

    decision = classify_route("what is automation rule")

    assert decision.kbs == ["web"]
    assert decision.kb_scores["docx"] == pytest.approx(0.3, abs=0.01)


def test_close_contenders_are_both_kept(monkeypatch):
    # github scores 0.85 against web's 1.0 - well within the 70%
    # relative margin, so both are genuine contenders and neither
    # should be dropped.
    monkeypatch.setattr(router, "embed_texts", lambda texts: [[1.0, 0.0]])

    def fake_sample(kb, limit=40):
        if kb == "web":
            return [[1.0, 0.0]]
        if kb == "github":
            return [[0.85, 0.5268046013283738]]  # cosine(query, this) == 0.85
        return []

    monkeypatch.setattr(router, "sample_kb_embeddings", fake_sample)

    decision = classify_route("what is automation rule")

    assert set(decision.kbs) == {"web", "github"}


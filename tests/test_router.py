"""Tests for app.routing.router.

Semantic routing embeds the query + a handful of route exemplars with
the real sentence-transformers model (no store/network dependency, but
not free either) - these tests accept that one-time model load cost
rather than faking it out, since the whole point of this module is the
actual embedding-similarity behavior, not just wiring.
"""

from app.routing.router import RouteDecision, classify_route


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


def test_no_kb_signal_leaves_kbs_empty():
    # No explicit source mentioned at all - kbs should be empty,
    # leaving the "which KB(s) to search" decision to the caller (see
    # app.retrieval.retriever, which falls back to every populated KB).
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


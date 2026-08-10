"""Tests for app.retrieval.github_adaptive - specifically the new
"activity" intent branch (see that module's docstring for why it
exists: issue/PR chunks routinely lose the similarity contest against
hundreds of code chunks in plain hybrid search).

No test file existed for this module before. classify_github_intent
and the retrieval primitives (_dense_search/_cross_encoder_rerank/
retrieve_hybrid) are monkeypatched so these tests exercise only the
dispatch logic in retrieve_github_adaptive, not the real embedding
model or vector store.
"""

import app.retrieval.github_adaptive as github_adaptive
from app.routing.github_intent import GitHubIntent


def _activity_decision():
    return GitHubIntent(intent="activity", method="rule", scores={"activity": 0.9})


def test_activity_intent_filters_by_content_type(monkeypatch):
    monkeypatch.setattr(github_adaptive, "classify_github_intent", lambda q: _activity_decision())
    monkeypatch.setattr(github_adaptive, "embed_texts", lambda texts: [[1.0, 0.0]])

    captured_where = {}

    def fake_dense_search(query_vector, kb, pool_size, where):
        captured_where["where"] = where
        return [{"content": "a pr", "metadata": {"content_type": "pull_request"}}]

    monkeypatch.setattr(github_adaptive, "_dense_search", fake_dense_search)
    monkeypatch.setattr(github_adaptive, "_cross_encoder_rerank", lambda q, hits, top_k: hits)

    chunks, decision = github_adaptive.retrieve_github_adaptive(
        "what are the latest pull requests and issues", top_k=3, repository="owner/repo",
    )

    assert decision.intent == "activity"
    assert len(chunks) == 1
    assert captured_where["where"] == {
        "$and": [
            {"repository": "owner/repo"},
            {"content_type": {"$in": ["issue", "pull_request", "discussion"]}},
        ]
    }


def test_activity_intent_falls_back_to_hybrid_when_no_activity_chunks(monkeypatch):
    monkeypatch.setattr(github_adaptive, "classify_github_intent", lambda q: _activity_decision())
    monkeypatch.setattr(github_adaptive, "embed_texts", lambda texts: [[1.0, 0.0]])
    monkeypatch.setattr(github_adaptive, "_dense_search", lambda *a, **kw: [])

    hybrid_calls = []

    def fake_run_hybrid(query_text, query_vector, top_k, where, *, query_transform_enabled=None):
        hybrid_calls.append(where)
        return [{"content": "some code", "metadata": {}}]

    monkeypatch.setattr(github_adaptive, "_run_hybrid", fake_run_hybrid)

    chunks, decision = github_adaptive.retrieve_github_adaptive(
        "what are the latest pull requests and issues", top_k=3, repository="owner/repo",
    )

    assert len(hybrid_calls) == 1
    assert chunks == [{"content": "some code", "metadata": {}}]


def test_activity_intent_without_repository_still_filters_by_content_type(monkeypatch):
    # No repository scope - the content_type filter alone (no
    # $and-wrapped repository clause) should still apply.
    monkeypatch.setattr(github_adaptive, "classify_github_intent", lambda q: _activity_decision())
    monkeypatch.setattr(github_adaptive, "embed_texts", lambda texts: [[1.0, 0.0]])

    captured_where = {}

    def fake_dense_search(query_vector, kb, pool_size, where):
        captured_where["where"] = where
        return [{"content": "a pr", "metadata": {"content_type": "pull_request"}}]

    monkeypatch.setattr(github_adaptive, "_dense_search", fake_dense_search)
    monkeypatch.setattr(github_adaptive, "_cross_encoder_rerank", lambda q, hits, top_k: hits)

    github_adaptive.retrieve_github_adaptive("what pull requests are open", top_k=3, repository=None)

    assert captured_where["where"] == {"content_type": {"$in": ["issue", "pull_request", "discussion"]}}


def test_non_activity_intent_is_unaffected(monkeypatch):
    explanation_decision = GitHubIntent(intent="explanation", method="rule", scores={"explanation": 0.9})
    monkeypatch.setattr(github_adaptive, "classify_github_intent", lambda q: explanation_decision)
    monkeypatch.setattr(github_adaptive, "embed_texts", lambda texts: [[1.0, 0.0]])

    captured_where = {}

    def fake_dense_search(query_vector, kb, pool_size, where):
        captured_where["where"] = where
        return [{"content": "explains stuff", "metadata": {}}]

    monkeypatch.setattr(github_adaptive, "_dense_search", fake_dense_search)
    monkeypatch.setattr(github_adaptive, "_cross_encoder_rerank", lambda q, hits, top_k: hits)

    chunks, decision = github_adaptive.retrieve_github_adaptive(
        "how does the retry logic work", top_k=3, repository="owner/repo",
    )

    assert decision.intent == "explanation"
    # No content_type filter leaked into an unrelated intent's where clause.
    assert captured_where["where"] == {"repository": "owner/repo"}

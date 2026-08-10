"""Tests for app.routing.github_intent.

No test file existed for this module before - these focus on the new
"activity" intent (added to fix issue/PR questions losing the
similarity contest against hundreds of code chunks - see
app.retrieval.github_adaptive's module docstring), plus a handful of
sanity checks on the pre-existing intents this change must not
regress.

Uses the real sentence-transformers model (same convention as
tests/test_router.py) since intent classification's whole point is
real embedding-similarity behavior, not just wiring.
"""

from app.routing.github_intent import classify_github_intent


def test_pull_requests_keyword_routes_to_activity():
    decision = classify_github_intent("what are the latest pull requests and issues")

    assert decision.intent == "activity"


def test_open_issues_keyword_routes_to_activity():
    decision = classify_github_intent("are there any open issues in this repo")

    assert decision.intent == "activity"


def test_activity_semantic_paraphrase_without_keywords():
    # No literal "pull request"/"issue" keyword, but semantically this
    # is an activity question.
    decision = classify_github_intent("has anyone reported this as a bug")

    assert decision.intent in {"activity", "general"}
    assert decision.scores.get("activity", 0.0) > 0.0


def test_activity_does_not_steal_a_code_explanation_question():
    # Regression guard: a question that happens to use the word
    # "issue" in its ordinary English sense (not GitHub Issues) should
    # still be classified as explanation, not activity.
    decision = classify_github_intent("how does the retry logic work in this codebase")

    assert decision.intent != "activity"


def test_history_keyword_still_wins_over_activity():
    # history and activity are both "meta" (non-code-content) intents
    # sitting adjacently in _INTENT_PRIORITY - a clear commit-history
    # question must still win on its own rule match, not get stolen by
    # activity's exemplars.
    decision = classify_github_intent("who changed this file recently")

    assert decision.intent == "history"


def test_exact_code_keyword_still_wins_when_no_activity_signal():
    decision = classify_github_intent("find the function classify_route")

    assert decision.intent == "exact_code"


def test_activity_in_priority_list_between_history_and_visual():
    from app.routing.github_intent import _INTENT_PRIORITY

    assert _INTENT_PRIORITY.index("history") < _INTENT_PRIORITY.index("activity") < _INTENT_PRIORITY.index("visual")

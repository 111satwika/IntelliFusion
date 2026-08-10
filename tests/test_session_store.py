"""Tests for session_store.py's conversation-history functions
(get_history/append_history/clear_history) - see its module docstring.

Scoped to the NEW history functions only, not retrofitting coverage
for the pre-existing pending-document/settings functions (a
pre-existing gap, out of scope here - see the conversation-memory
plan's "explicitly out of scope" section).
"""

import concurrent.futures

import pytest

import session_store


@pytest.fixture(autouse=True)
def _isolated_tracker(tmp_path, monkeypatch):
    """Point the tracker file/lock at a per-test tmp_path so tests
    never read/write the real data/pending_documents.json."""
    tracker_path = tmp_path / "pending_documents.json"
    monkeypatch.setattr(session_store, "_PENDING_TRACKER_PATH", tracker_path)
    monkeypatch.setattr(session_store, "_PENDING_TRACKER_LOCK_PATH", tracker_path.with_suffix(".json.lock"))


def test_get_history_empty_by_default():
    assert session_store.get_history("some-session", "__cross__") == []


def test_append_history_records_a_turn():
    session_store.append_history("s1", "__cross__", "What is RAG?", "Retrieval-Augmented Generation.")

    history = session_store.get_history("s1", "__cross__")

    assert len(history) == 1
    assert history[0]["question"] == "What is RAG?"
    assert history[0]["answer"] == "Retrieval-Augmented Generation."
    assert "at" in history[0]


def test_append_history_caps_at_max_turns():
    for i in range(session_store._MAX_HISTORY_TURNS + 5):
        session_store.append_history("s1", "__cross__", f"question {i}", f"answer {i}")

    history = session_store.get_history("s1", "__cross__")

    assert len(history) == session_store._MAX_HISTORY_TURNS
    # Oldest turns dropped first - the most recent ones survive.
    assert history[-1]["question"] == f"question {session_store._MAX_HISTORY_TURNS + 4}"
    assert history[0]["question"] == "question 5"


def test_history_is_isolated_per_thread():
    session_store.append_history("s1", "markdown", "md question", "md answer")
    session_store.append_history("s1", "__cross__", "cross question", "cross answer")

    assert len(session_store.get_history("s1", "markdown")) == 1
    assert len(session_store.get_history("s1", "__cross__")) == 1
    assert session_store.get_history("s1", "markdown")[0]["question"] == "md question"


def test_history_is_isolated_per_session():
    session_store.append_history("s1", "__cross__", "s1 question", "s1 answer")
    session_store.append_history("s2", "__cross__", "s2 question", "s2 answer")

    assert session_store.get_history("s1", "__cross__")[0]["question"] == "s1 question"
    assert session_store.get_history("s2", "__cross__")[0]["question"] == "s2 question"


def test_clear_history_empties_one_thread_only():
    session_store.append_history("s1", "markdown", "md question", "md answer")
    session_store.append_history("s1", "__cross__", "cross question", "cross answer")

    session_store.clear_history("s1", "markdown")

    assert session_store.get_history("s1", "markdown") == []
    assert len(session_store.get_history("s1", "__cross__")) == 1


def test_clear_history_on_nonexistent_session_is_a_no_op():
    session_store.clear_history("never-existed", "__cross__")  # must not raise
    assert session_store.get_history("never-existed", "__cross__") == []


def test_default_settings_include_conversation_memory_disabled():
    settings = session_store.get_settings("brand-new-session")
    assert settings["conversation_memory_enabled"] is False


def test_concurrent_append_history_loses_no_turns():
    """Mirrors the live concurrency proof already run for the pending
    tracker - many threads racing to append_history() on the SAME
    session/thread must not lose any writes to the shared FileLock."""

    def add(i):
        session_store.append_history("s1", "__cross__", f"q{i}", f"a{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(add, range(20)))

    history = session_store.get_history("s1", "__cross__")
    # Capped at _MAX_HISTORY_TURNS, but every append must have taken
    # the lock and been durably recorded before being trimmed - i.e.
    # the final list is exactly the last N by count, never fewer than
    # the cap due to a lost write.
    assert len(history) == session_store._MAX_HISTORY_TURNS

"""Tests for app.retrieval.contextualize - see its module docstring."""

import json

from app.retrieval import contextualize
from app.retrieval.contextualize import contextualize_query


def test_disabled_skips_llm_call_entirely(monkeypatch):
    called = []
    monkeypatch.setattr(contextualize, "_call_ollama_json", lambda *a, **kw: called.append(1))

    history = [{"question": "What is RAG?", "answer": "Retrieval-Augmented Generation."}]
    result = contextualize_query("what about that?", history, enabled=False)

    assert result.original == "what about that?"
    assert result.resolved == "what about that?"
    assert result.used_llm is False
    assert not called


def test_no_history_skips_llm_call(monkeypatch):
    called = []
    monkeypatch.setattr(contextualize, "_call_ollama_json", lambda *a, **kw: called.append(1))

    result = contextualize_query("what about that?", [], enabled=True)

    assert result.resolved == "what about that?"
    assert result.used_llm is False
    assert not called


def test_standalone_looking_question_skips_llm_call(monkeypatch):
    called = []
    monkeypatch.setattr(contextualize, "_call_ollama_json", lambda *a, **kw: called.append(1))

    history = [{"question": "What is RAG?", "answer": "Retrieval-Augmented Generation."}]
    # Long, no pronouns/continuation openers - reads as fully standalone.
    question = "How does hybrid retrieval combine dense and sparse search results?"
    result = contextualize_query(question, history, enabled=True)

    assert result.resolved == question
    assert result.used_llm is False
    assert not called


def test_pronoun_bearing_followup_resolves_via_llm(monkeypatch):
    monkeypatch.setattr(
        contextualize, "_call_ollama_json",
        lambda *a, **kw: json.dumps({"standalone_question": "What tests exist for hybrid retrieval?"}),
    )

    history = [{"question": "How does hybrid retrieval work?", "answer": "It combines dense and BM25 search."}]
    result = contextualize_query("what about tests for that?", history, enabled=True)

    assert result.used_llm is True
    assert result.fell_back is False
    assert result.resolved == "What tests exist for hybrid retrieval?"
    assert result.original == "what about tests for that?"


def test_short_followup_triggers_llm_even_without_pronoun(monkeypatch):
    monkeypatch.setattr(
        contextualize, "_call_ollama_json",
        lambda *a, **kw: json.dumps({"standalone_question": "resolved"}),
    )
    history = [{"question": "q", "answer": "a"}]

    result = contextualize_query("and tests?", history, enabled=True)

    assert result.used_llm is True


def test_ollama_exception_falls_back_to_original(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(contextualize, "_call_ollama_json", boom)
    history = [{"question": "q", "answer": "a"}]

    result = contextualize_query("what about that?", history, enabled=True)

    assert result.resolved == "what about that?"
    assert result.used_llm is True
    assert result.fell_back is True


def test_malformed_json_falls_back_to_original(monkeypatch):
    monkeypatch.setattr(contextualize, "_call_ollama_json", lambda *a, **kw: "not json at all")
    history = [{"question": "q", "answer": "a"}]

    result = contextualize_query("what about that?", history, enabled=True)

    assert result.resolved == "what about that?"
    assert result.fell_back is True


def test_empty_response_falls_back_to_original(monkeypatch):
    monkeypatch.setattr(contextualize, "_call_ollama_json", lambda *a, **kw: "")
    history = [{"question": "q", "answer": "a"}]

    result = contextualize_query("what about that?", history, enabled=True)

    assert result.resolved == "what about that?"
    assert result.fell_back is True


def test_empty_question_returns_immediately_without_history_check():
    result = contextualize_query("", [{"question": "q", "answer": "a"}], enabled=True)
    assert result.original == ""
    assert result.resolved == ""
    assert result.used_llm is False


def test_code_fenced_json_response_is_parsed(monkeypatch):
    fenced = "```json\n" + json.dumps({"standalone_question": "resolved question"}) + "\n```"
    monkeypatch.setattr(contextualize, "_call_ollama_json", lambda *a, **kw: fenced)
    history = [{"question": "q", "answer": "a"}]

    result = contextualize_query("what about that?", history, enabled=True)

    assert result.resolved == "resolved question"
    assert result.fell_back is False


def test_enabled_none_falls_back_to_process_global_default():
    # Module-level default is off (see contextualize.is_enabled()) -
    # enabled=None (the default) must resolve to that, not to True.
    assert contextualize.is_enabled() is False
    history = [{"question": "q", "answer": "a"}]

    result = contextualize_query("what about that?", history)  # enabled omitted

    assert result.resolved == "what about that?"
    assert result.used_llm is False

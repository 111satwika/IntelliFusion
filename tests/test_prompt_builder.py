"""Tests for app.prompting.prompt_builder."""

from app.prompting import prompt_builder
from app.prompting.prompt_builder import build_prompt


def _chunk(content: str, section: str = "Some Section", similarity: float = 0.9) -> dict:
    return {"content": content, "metadata": {"section": section}, "similarity": similarity}


def test_build_prompt_includes_instructions_context_note_and_question():
    chunks = [_chunk("RAG stands for Retrieval-Augmented Generation.")]

    prompt = build_prompt("What is RAG?", chunks)

    assert "Answer the question using only the context provided below" in prompt
    assert "this repository's own" in prompt
    assert "Question: What is RAG?" in prompt
    assert prompt.strip().endswith("Answer:")


def test_build_prompt_context_note_lists_actual_source_file_names():
    chunks = [
        {"content": "a", "metadata": {"file_name": "README.md"}, "similarity": 0.9},
        {"content": "b", "metadata": {"file_name": "sample.pdf", "page_number": 2}, "similarity": 0.8},
    ]

    prompt = build_prompt("A question?", chunks)

    assert "README.md, sample.pdf" in prompt


def test_build_prompt_context_note_falls_back_when_no_file_name_present():
    chunks = [_chunk("Some content.")]

    prompt = build_prompt("A question?", chunks)

    assert "this repository's own files" in prompt


def test_build_prompt_instructs_the_model_not_to_hedge_on_a_single_item():
    # Regression test for the false-hedge bug (see module docstring):
    # a small local model answered "I don't have enough information...
    # does not contain details about OTHER pull requests or issues"
    # when the context genuinely contained exactly one matching item.
    # Pins the specific worked-example wording that was actually needed
    # to fix it (two prior, more abstract wordings did NOT work when
    # tested live against the real model - see module docstring), so a
    # future edit that waters this back down to something abstract
    # doesn't silently reintroduce the same live-verified failure.
    chunks = [_chunk("Some content.")]

    prompt = build_prompt("What are the latest pull requests and issues?", chunks)

    assert "EVEN IF that is only a single item" in prompt
    assert "Worked example" in prompt
    assert "I don't have enough information. The context does not contain details about other pull requests or issues." in prompt


def test_build_prompt_labels_each_source_with_section_and_similarity():
    chunks = [_chunk("Some content.", section="Intro", similarity=0.8321)]

    prompt = build_prompt("A question?", chunks)

    assert "[Source 1 — section: 'Intro', similarity: 0.8321]" in prompt
    assert "Some content." in prompt


def test_build_prompt_labels_pdf_source_with_file_name_and_page_number():
    chunks = [
        {
            "content": "PDF page content.",
            "metadata": {"file_name": "sample.pdf", "page_number": 3},
            "similarity": 0.7047,
        }
    ]

    prompt = build_prompt("A question?", chunks)

    assert "[Source 1 — sample.pdf, page 3, similarity: 0.7047]" in prompt


def test_build_prompt_always_includes_first_chunk_even_if_it_exceeds_budget():
    huge_chunk = _chunk("word " * 5000)

    prompt = build_prompt("A question?", [huge_chunk], max_context_tokens=10)

    assert "word" in prompt


def test_build_prompt_drops_later_chunks_once_token_budget_is_exceeded():
    chunks = [_chunk(f"chunk number {i} " * 20) for i in range(5)]

    prompt = build_prompt("A question?", chunks, max_context_tokens=15)

    assert "[Source 1" in prompt
    assert "[Source 5" not in prompt


def test_build_prompt_uses_all_chunks_when_well_under_budget():
    chunks = [_chunk("short"), _chunk("also short")]

    prompt = build_prompt("A question?", chunks, max_context_tokens=1000)

    assert "[Source 1" in prompt
    assert "[Source 2" in prompt


def test_count_tokens_handles_empty_string():
    assert prompt_builder._count_tokens("") == 0


def test_build_prompt_omits_history_block_when_absent():
    chunks = [_chunk("content")]

    prompt = build_prompt("A question?", chunks)

    assert "Conversation so far:" not in prompt


def test_build_prompt_omits_history_block_when_empty_list():
    chunks = [_chunk("content")]

    prompt = build_prompt("A question?", chunks, history=[])

    assert "Conversation so far:" not in prompt


def test_build_prompt_includes_history_block_when_present():
    chunks = [_chunk("content")]
    history = [{"question": "What is RAG?", "answer": "Retrieval-Augmented Generation."}]

    prompt = build_prompt("What about hybrid search?", chunks, history=history)

    assert "Conversation so far:" in prompt
    assert "Q: What is RAG?" in prompt
    assert "A: Retrieval-Augmented Generation." in prompt
    # History block appears before the Context/Question, and the
    # ORIGINAL question (not a contextualized rewrite) is what's asked.
    assert prompt.index("Conversation so far:") < prompt.index("Context:")
    assert "Question: What about hybrid search?" in prompt


def test_build_prompt_history_drops_oldest_turns_first_over_budget():
    history = [
        {"question": f"Question {i}", "answer": f"Answer number {i} " * 20}
        for i in range(5)
    ]

    prompt = build_prompt("Latest question?", [], history=history, max_history_tokens=15)

    assert "Question 4" in prompt  # most recent turn survives
    assert "Question 0" not in prompt  # oldest turn dropped first


def test_format_history_keeps_chronological_order():
    history = [
        {"question": "first", "answer": "a1"},
        {"question": "second", "answer": "a2"},
    ]

    text = prompt_builder._format_history(history, max_history_tokens=1000)

    assert text.index("first") < text.index("second")

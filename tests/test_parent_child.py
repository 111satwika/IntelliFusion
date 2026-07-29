"""Tests for app.chunking.parent_child.

Covers the sentence-window splitter, the parent-child chunk expander,
and the chunk_with_parent_child() drop-in wrapper that ingest.py and
the Streamlit UI both call in place of chunk_document. The point of
this test file is to lock in the invariants the hybrid_retriever's
parent-child stage relies on:

- Parents keep their original chunk_index.
- Children get their own chunk_index starting AFTER the last parent.
- Every child carries parent_chunk_index pointing at its parent.
- chunk_role tagging is applied to both.
- Documents whose source_type isn't in _PARENT_CHILD_SOURCE_TYPES
  pass through untouched (no children generated).
- Chunks whose content_type is already fine-grained (class/method/
  function/table/table_row) get chunk_role=parent but NO children,
  so we don't fragment a Python method into 40-word slices.
"""

from app.chunking.chunker import Chunk
from app.chunking.parent_child import (
    _PARENT_CHILD_KBS,
    _PARENT_CHILD_SOURCE_TYPES,
    _SKIP_CHILDREN_CONTENT_TYPES,
    _sentence_windows,
    chunk_with_parent_child,
    expand_to_parent_child,
)
from app.ingestion.loader import Document


# ---- _sentence_windows -----------------------------------------------


def test_sentence_windows_returns_empty_list_for_empty_text():
    assert _sentence_windows("") == []
    assert _sentence_windows("   ") == []


def test_sentence_windows_returns_single_window_for_short_text():
    """A short paragraph (well under the target word budget) fits
    into exactly one window - we don't split for the sake of
    splitting."""
    text = "This is a very short paragraph. It has two sentences."
    windows = _sentence_windows(text)
    assert windows == ["This is a very short paragraph. It has two sentences."]


def test_sentence_windows_never_splits_within_a_sentence():
    """A single sentence longer than the word target still becomes
    one window - the splitter aggregates BY sentences and never
    cuts inside one."""
    long_sentence = "word " * 200 + "end."
    windows = _sentence_windows(long_sentence)
    assert len(windows) == 1
    assert "end." in windows[0]


def test_sentence_windows_starts_a_new_window_when_budget_would_be_exceeded():
    """Aggregating another sentence that would push the running word
    count past the target should trigger a new window. The exact
    number of windows depends on target size, so this test only
    asserts multiplicity for a clearly-multi-window input."""
    sentence = "word " * 30 + "one."
    text = " ".join([sentence] * 5)
    windows = _sentence_windows(text)
    assert len(windows) >= 2


def test_sentence_windows_emits_below_min_only_when_it_is_the_only_content():
    """A single tiny fragment (< min words) still gets emitted rather
    than lost - we'd rather have a small child chunk than no child
    coverage for the parent at all."""
    windows = _sentence_windows("Yes.")
    assert windows == ["Yes."]


# ---- expand_to_parent_child ------------------------------------------


def _parent(chunk_index: int, content: str, extra_metadata: dict | None = None) -> Chunk:
    metadata = {"chunk_index": chunk_index, "document_id": "doc-a"}
    if extra_metadata:
        metadata.update(extra_metadata)
    return Chunk(content=content, metadata=metadata)


def test_expand_returns_parents_first_then_children():
    """Output layout is [all parents, all children] - parents are
    emitted contiguously so downstream code that iterates parents
    only can slice by len(input_chunks)."""
    parents_in = [
        _parent(0, "First sentence. Second sentence. Third sentence."),
        _parent(1, "Another paragraph here. With more content."),
    ]
    result = expand_to_parent_child(parents_in)

    assert result[0].metadata["chunk_role"] == "parent"
    assert result[1].metadata["chunk_role"] == "parent"
    assert all(c.metadata["chunk_role"] == "child" for c in result[2:])


def test_expand_preserves_original_parent_chunk_index():
    """Parents keep their input chunk_index so existing dense/BM25
    metadata references (e.g. from another retriever's dedup key)
    stay valid."""
    parents_in = [_parent(0, "A. B."), _parent(1, "C. D.")]
    result = expand_to_parent_child(parents_in)
    parent_indices = [c.metadata["chunk_index"] for c in result if c.metadata["chunk_role"] == "parent"]
    assert parent_indices == [0, 1]


def test_expand_reassigns_child_chunk_index_to_avoid_collisions():
    """Children get chunk_index values STARTING after the last parent
    so (document_id, chunk_index) tuples stay globally unique across
    parents+children in one collection."""
    parents_in = [_parent(0, "A. B."), _parent(1, "C. D.")]
    result = expand_to_parent_child(parents_in)
    children = [c for c in result if c.metadata["chunk_role"] == "child"]
    child_indices = [c.metadata["chunk_index"] for c in children]
    # Every child index must be >= number of parents (which is 2).
    assert all(idx >= 2 for idx in child_indices)
    # Indices are unique and sequential.
    assert child_indices == list(range(2, 2 + len(children)))


def test_expand_tags_every_child_with_parent_chunk_index():
    """The parent-child retriever uses parent_chunk_index to resolve
    a child hit back to its parent - every child must carry it."""
    parents_in = [_parent(0, "A. B."), _parent(1, "C. D.")]
    result = expand_to_parent_child(parents_in)
    for chunk in result:
        if chunk.metadata["chunk_role"] == "child":
            assert "parent_chunk_index" in chunk.metadata
            assert chunk.metadata["parent_chunk_index"] in {0, 1}


def test_expand_deep_copies_parent_metadata_so_child_edits_do_not_leak():
    """A child's metadata dict must not share references with its
    parent's - otherwise a downstream mutation on a child (e.g. adding
    a match score) would silently corrupt the parent's metadata."""
    original_metadata = {"chunk_index": 0, "document_id": "doc-a", "tags": ["x"]}
    parents_in = [Chunk(content="A. B.", metadata=original_metadata)]
    result = expand_to_parent_child(parents_in)

    child_tags = [
        c.metadata["tags"] for c in result if c.metadata["chunk_role"] == "child"
    ]
    for tags in child_tags:
        tags.append("mutated")

    assert original_metadata["tags"] == ["x"]


def test_expand_skips_children_for_structured_content_types():
    """A code AST method or a table row is already fine-grained;
    splitting it into 40-word slices would just fragment structural
    units. It still gets chunk_role=parent so dense/BM25 index it,
    but no children are generated."""
    parents_in = [
        _parent(0, "def foo():\n    return 1", extra_metadata={"content_type": "method"}),
        _parent(1, "| a | b |\n| 1 | 2 |", extra_metadata={"content_type": "table_row"}),
        _parent(
            2,
            "This is prose. It has multiple sentences. Suitable for splitting.",
            extra_metadata={"content_type": "paragraph"},
        ),
    ]
    result = expand_to_parent_child(parents_in)
    children = [c for c in result if c.metadata["chunk_role"] == "child"]
    # Only the paragraph should have produced children.
    assert len(children) >= 1
    for child in children:
        assert child.metadata["parent_chunk_index"] == 2


def test_expand_skip_registry_covers_expected_content_types():
    """Guardrail: if this registry drifts, code AST chunks would
    start getting fragmented into arbitrary sentence windows."""
    assert _SKIP_CHILDREN_CONTENT_TYPES == {
        "class",
        "method",
        "function",
        "table",
        "table_row",
    }


# ---- chunk_with_parent_child -----------------------------------------


def _fake_document(source_type: str) -> Document:
    return Document(
        content="First sentence. Second sentence.",
        metadata={"file_name": "x", "source_type": source_type},
    )


def test_chunk_with_parent_child_expands_for_web_source(monkeypatch):
    """A document whose source_type is in _PARENT_CHILD_SOURCE_TYPES
    ("code" or "web") gets parent-child expansion at ingest time."""
    from app.chunking import parent_child as pc

    fake_parents = [_parent(0, "First sentence. Second sentence.")]
    monkeypatch.setattr(pc, "chunk_document", lambda doc, **kw: fake_parents)

    result = chunk_with_parent_child(_fake_document("web"))

    roles = {c.metadata.get("chunk_role") for c in result}
    assert "parent" in roles
    assert "child" in roles


def test_chunk_with_parent_child_expands_for_code_source(monkeypatch):
    from app.chunking import parent_child as pc

    fake_parents = [_parent(0, "First sentence. Second sentence.")]
    monkeypatch.setattr(pc, "chunk_document", lambda doc, **kw: fake_parents)

    result = chunk_with_parent_child(_fake_document("code"))

    roles = {c.metadata.get("chunk_role") for c in result}
    assert "child" in roles


def test_chunk_with_parent_child_leaves_pdf_untouched(monkeypatch):
    """PDF/Markdown/DOCX documents pass through chunk_document
    directly - no chunk_role tag, no children generated."""
    from app.chunking import parent_child as pc

    fake_parents = [_parent(0, "First sentence. Second sentence.")]
    monkeypatch.setattr(pc, "chunk_document", lambda doc, **kw: fake_parents)

    result = chunk_with_parent_child(_fake_document("pdf"))

    assert result == fake_parents
    assert all("chunk_role" not in c.metadata for c in result)


def test_chunk_with_parent_child_forwards_chunk_kwargs(monkeypatch):
    """chunk_size / chunk_overlap should thread through to the
    underlying chunk_document so callers don't have to know which
    function they're really configuring."""
    from app.chunking import parent_child as pc

    captured = {}

    def spy_chunk_document(doc, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(pc, "chunk_document", spy_chunk_document)

    chunk_with_parent_child(_fake_document("pdf"), chunk_size=500, chunk_overlap=50)

    assert captured == {"chunk_size": 500, "chunk_overlap": 50}


# ---- Registry sanity checks ------------------------------------------


def test_parent_child_source_types_registry():
    """Guardrail: adding another parent-child KB (say, "notion") should
    require updating this set explicitly, not silently opt in."""
    assert _PARENT_CHILD_SOURCE_TYPES == {"code", "web"}


def test_parent_child_kbs_registry():
    """The KB-name set (used by hybrid_retriever) must stay in sync
    with the source_type set (used at ingest time)."""
    assert _PARENT_CHILD_KBS == {"github", "web"}

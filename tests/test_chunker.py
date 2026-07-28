"""Tests for app.chunking.chunker."""

import pytest

from app.chunking.chunker import chunk_document
from app.ingestion.loader import Document


def _doc(content: str) -> Document:
    return Document(content=content, metadata={"file_name": "test.md"})


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_document(_doc("# Title\n\nSome text."), chunk_size=50, chunk_overlap=50)


def test_python_documents_are_dispatched_to_the_ast_chunker(monkeypatch):
    captured = {}

    def fake_chunk_python_document(document, chunk_size, chunk_overlap):
        captured["document"] = document
        captured["chunk_size"] = chunk_size
        captured["chunk_overlap"] = chunk_overlap
        return ["sentinel-result"]

    monkeypatch.setattr(
        "app.chunking.code_chunker.chunk_python_document", fake_chunk_python_document
    )

    python_doc = Document(content="def foo():\n    pass\n", metadata={"file_name": "foo.py", "language": "python"})
    result = chunk_document(python_doc, chunk_size=150, chunk_overlap=30)

    assert result == ["sentinel-result"]
    assert captured["document"] is python_doc
    assert captured["chunk_size"] == 150
    assert captured["chunk_overlap"] == 30


def test_non_python_documents_are_not_dispatched_to_the_ast_chunker():
    # A .md document (no "language" metadata, or language != "python")
    # should go through the normal Markdown/generic chunking path, not
    # be treated as Python source.
    chunks = chunk_document(_doc("# Title\n\nSome text."), chunk_size=150, chunk_overlap=10)

    assert len(chunks) >= 1
    assert chunks[0].metadata.get("content_type") != "class"


def test_each_chunk_is_prefixed_with_its_section_breadcrumb():
    content = "# Section One\n\nParagraph in section one.\n\n# Section Two\n\nParagraph in section two."
    chunks = chunk_document(_doc(content), chunk_size=150, chunk_overlap=10)

    assert len(chunks) == 2
    assert "[Section: Section One]" in chunks[0].content
    assert "[Section: Section Two]" in chunks[1].content
    assert chunks[0].metadata["section"] == "Section One"
    assert chunks[1].metadata["section"] == "Section Two"


def test_code_block_is_kept_atomic_even_when_larger_than_chunk_size():
    code_lines = "\n".join(f"line {i}" for i in range(50))
    content = f"# Section\n\n```\n{code_lines}\n```"

    chunks = chunk_document(_doc(content), chunk_size=10, chunk_overlap=2)

    code_chunks = [c for c in chunks if "block_types" in c.metadata and "code" in c.metadata["block_types"]]
    assert len(code_chunks) == 1
    assert "```" in code_chunks[0].content
    assert "line 49" in code_chunks[0].content


def test_image_caption_before_code_is_not_split_into_its_own_runt_chunk():
    # Regression test: an "[Image: ...] (url)" caption immediately
    # followed by a code block used to get flushed as its own
    # standalone chunk (with no code) whenever the code didn't fit
    # alongside it - creating a near-duplicate, code-less chunk that
    # shared the same section breadcrumb as (and could outrank) the
    # real code chunk in similarity search. The caption should always
    # be folded into the following code chunk instead.
    #
    # chunk_size is picked large enough that the short caption alone
    # never trips the (unrelated) oversized-paragraph splitting path,
    # but small enough that the code block still overflows it.
    code_lines = "\n".join(f"line {i}" for i in range(50))
    content = (
        "# Section\n\n"
        "[Image: shot] (https://example.com/i.png)\n\n"
        f"```\n{code_lines}\n```"
    )

    chunks = chunk_document(_doc(content), chunk_size=30, chunk_overlap=5)

    assert len(chunks) == 1
    assert "[Image: shot]" in chunks[0].content
    assert "```" in chunks[0].content
    assert "line 49" in chunks[0].content


def test_horizontal_rules_are_dropped_not_kept_as_paragraphs():
    content = "# Section\n\nSome text.\n\n---\n\nMore text."
    chunks = chunk_document(_doc(content), chunk_size=150, chunk_overlap=10)

    combined = "\n".join(c.content for c in chunks)
    assert "---" not in combined


def test_chunk_index_is_sequential_across_sections():
    content = "# One\n\nFirst.\n\n# Two\n\nSecond.\n\n# Three\n\nThird."
    chunks = chunk_document(_doc(content), chunk_size=150, chunk_overlap=10)

    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_table_produces_both_whole_table_and_per_row_chunks():
    content = (
        "# RBAC\n\n"
        "| Role | Deploy to Production |\n"
        "| --- | --- |\n"
        "| Viewer | No |\n"
        "| Administrator | Yes |"
    )
    chunks = chunk_document(_doc(content), chunk_size=150, chunk_overlap=10)

    whole_table_chunks = [c for c in chunks if c.metadata.get("content_type") == "table"]
    row_chunks = [c for c in chunks if c.metadata.get("content_type") == "table_row"]

    # One whole-table chunk containing every row...
    assert len(whole_table_chunks) == 1
    assert "Viewer" in whole_table_chunks[0].content
    assert "Administrator" in whole_table_chunks[0].content

    # ...plus one additional chunk per data row, each self-describing.
    assert len(row_chunks) == 2
    assert "Viewer" in row_chunks[0].content and "Administrator" not in row_chunks[0].content
    assert "Administrator" in row_chunks[1].content and "Viewer" not in row_chunks[1].content
    for row_chunk in row_chunks:
        assert "| Role | Deploy to Production |" in row_chunk.content

    # Chunk indices stay sequential across the whole-table + row chunks.
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_table_row_chunks_share_table_id_with_their_whole_table_chunk():
    # table_id is what lets retrieval trade a retrieved row for the
    # complete table (see app.retrieval.retriever._complete_partial_tables).
    content = (
        "# RBAC\n\n"
        "| Role | Deploy to Production |\n"
        "| --- | --- |\n"
        "| Viewer | No |\n"
        "| Administrator | Yes |"
    )
    chunks = chunk_document(_doc(content), chunk_size=150, chunk_overlap=10)

    whole_table_chunks = [c for c in chunks if c.metadata.get("content_type") == "table"]
    row_chunks = [c for c in chunks if c.metadata.get("content_type") == "table_row"]

    table_id = whole_table_chunks[0].metadata["table_id"]
    assert table_id
    for row_chunk in row_chunks:
        assert row_chunk.metadata["table_id"] == table_id


def test_two_separate_tables_get_distinct_table_ids():
    content = (
        "# Section One\n\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n\n"
        "# Section Two\n\n"
        "| C | D |\n"
        "| --- | --- |\n"
        "| 3 | 4 |"
    )
    chunks = chunk_document(_doc(content), chunk_size=150, chunk_overlap=10)

    table_ids = [c.metadata["table_id"] for c in chunks if c.metadata.get("content_type") == "table"]

    assert len(table_ids) == 2
    assert table_ids[0] != table_ids[1]

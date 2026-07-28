"""Tests for app.chunking.code_chunker (AST-based Python chunking)."""

from app.chunking.code_chunker import chunk_python_document
from app.ingestion.loader import Document


def _py_doc(content: str) -> Document:
    return Document(content=content, metadata={"file_name": "foo.py", "language": "python"})


def test_top_level_function_becomes_its_own_chunk():
    content = "def add(a, b):\n    return a + b\n"

    chunks = chunk_python_document(_py_doc(content))

    function_chunks = [c for c in chunks if c.metadata.get("content_type") == "function"]
    assert len(function_chunks) == 1
    assert "def add(a, b):" in function_chunks[0].content
    assert function_chunks[0].metadata["function_name"] == "add"
    assert function_chunks[0].metadata["section"] == "add"


def test_class_produces_one_whole_class_chunk_and_one_chunk_per_method():
    content = (
        "class Foo:\n"
        "    def __init__(self):\n"
        "        self.value = 1\n\n"
        "    def bar(self):\n"
        "        return self.value\n"
    )

    chunks = chunk_python_document(_py_doc(content))

    class_chunks = [c for c in chunks if c.metadata.get("content_type") == "class"]
    method_chunks = [c for c in chunks if c.metadata.get("content_type") == "method"]

    assert len(class_chunks) == 1
    assert "class Foo:" in class_chunks[0].content
    assert "def __init__" in class_chunks[0].content
    assert "def bar" in class_chunks[0].content
    assert class_chunks[0].metadata["section"] == "Foo"

    assert len(method_chunks) == 2
    method_names = {c.metadata["function_name"] for c in method_chunks}
    assert method_names == {"__init__", "bar"}

    bar_chunk = next(c for c in method_chunks if c.metadata["function_name"] == "bar")
    assert bar_chunk.metadata["section"] == "Foo.bar"
    assert bar_chunk.metadata["class_name"] == "Foo"


def test_methods_share_the_same_class_id_as_their_parent_class():
    content = (
        "class Foo:\n"
        "    def a(self):\n"
        "        pass\n\n"
        "    def b(self):\n"
        "        pass\n"
    )

    chunks = chunk_python_document(_py_doc(content))

    class_chunk = next(c for c in chunks if c.metadata.get("content_type") == "class")
    method_chunks = [c for c in chunks if c.metadata.get("content_type") == "method"]

    class_id = class_chunk.metadata["class_id"]
    assert class_id
    assert all(c.metadata["class_id"] == class_id for c in method_chunks)


def test_module_level_code_is_captured_as_leftover_chunks():
    content = (
        "import os\n\n"
        "CONSTANT = 42\n\n"
        "def foo():\n"
        "    pass\n"
    )

    chunks = chunk_python_document(_py_doc(content))

    leftover_chunks = [c for c in chunks if c.metadata.get("content_type") == "module"]
    assert leftover_chunks
    assert any("import os" in c.content for c in leftover_chunks)
    assert any("CONSTANT" in c.content for c in leftover_chunks)


def test_syntax_error_falls_back_to_generic_chunking():
    content = "def broken(:\n    this is not valid python\n"

    chunks = chunk_python_document(_py_doc(content))

    # Falls back to the generic chunker rather than raising - so we
    # still get at least one chunk, just without code-specific
    # content_types like "class"/"method"/"function".
    assert len(chunks) >= 1
    assert all(c.metadata.get("content_type") not in {"class", "method", "function"} for c in chunks)


def test_multiple_top_level_functions_each_get_their_own_chunk():
    content = (
        "def foo():\n"
        "    pass\n\n"
        "def bar():\n"
        "    pass\n"
    )

    chunks = chunk_python_document(_py_doc(content))

    function_chunks = [c for c in chunks if c.metadata.get("content_type") == "function"]
    assert len(function_chunks) == 2
    names = {c.metadata["function_name"] for c in function_chunks}
    assert names == {"foo", "bar"}

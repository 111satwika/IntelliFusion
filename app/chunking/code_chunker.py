"""
AST-based, structure-aware chunking for Python source code.

Responsibility: split a Python source file's Document into logical
code units (classes, methods, top-level functions) instead of
arbitrary character/token windows, so retrieval can return "the
authenticate_user function" as one complete, correctly-scoped chunk
rather than an arbitrary fragment that just happened to fit
chunk_size.

Design (mirrors app.chunking.chunker's Markdown-table "Strategy 4"
pattern exactly, applied to code instead of tables - see that
module's docstring for the original):
- A class is chunked as a PARENT: its ENTIRE source (class line,
  docstring, every method) always becomes one atomic, dedicated chunk
  (content_type="class") - the code equivalent of a "whole table"
  chunk - never split internally regardless of chunk_size, since
  splitting a class definition arbitrarily would destroy the very
  structure this chunker exists to preserve.
- ALSO, one CHILD chunk per method inside that class
  (content_type="method"), each carrying a class_id matching its
  parent class chunk's own id - the code equivalent of a table's
  per-row chunks. This lets a precise question like "how does
  authenticate_user work?" retrieve just that one method (instead of
  competing, in one shared embedding, against every other method in
  the class), while app.retrieval.retriever._complete_partial_classes()
  can still trade a retrieved method for its complete parent class
  when that gives more complete context (the constructor, shared
  instance state, sibling methods) - exactly like
  _complete_partial_tables() trades a table row for the whole table.
- A top-level function (not inside any class) is its own atomic
  chunk (content_type="function") - already a complete, self-
  contained unit on its own, so no separate parent/child split is
  needed for it, unlike a class's methods.
- Every chunk's "section" metadata is set to a breadcrumb - just the
  class name for a class chunk, "ClassName.method_name" for a method,
  or the bare function name for a top-level function - mirroring the
  heading breadcrumb app.chunking.chunker uses for Markdown, so
  app.retrieval.retriever's existing lexical-overlap rerank (which
  reads the "section" field) works identically for code as it already
  does for prose/tables, with no retrieval-side changes needed.
- Any code NOT inside a class or top-level function (module
  docstring, imports, top-level constants, an
  `if __name__ == "__main__":` block, etc.) is packed into ordinary
  "module"-level chunk(s) by re-running it through the EXISTING
  generic chunker (app.chunking.chunker.chunk_document), so nothing
  is silently dropped just because it isn't a class/function - the
  lines already claimed by a class/function chunk are blanked out
  first (see _extract_uncovered_lines) so they aren't duplicated.
- A class/method/function's exact source text is recovered via
  ast.get_source_segment() against the file's ORIGINAL text - never
  reconstructed from the parsed AST - so formatting, inline comments,
  and exact whitespace are preserved verbatim, the same "reproduce
  content verbatim" principle used everywhere else in this project.
- A file that fails to parse (ast.parse() raises SyntaxError - e.g.
  a Python 2-only file, a template with a misleading .py extension,
  or a genuinely broken file) falls back to the ordinary generic
  chunker for that ONE file, logged as a warning, rather than failing
  its ingestion outright - the same "one bad source can't break
  everything else" resilience pattern used by every loader in
  app.ingestion.loader.
- Only Python is AST-parsed today (the stdlib `ast` module only knows
  Python's grammar). Every other language GitHub ingestion recognizes
  (JavaScript, TypeScript, Java, Go, JSON, YAML, ...) still falls back
  to the generic chunker - a disclosed limitation, matching the
  project plan's own "later explore Tree-sitter for multi-language
  parsing" note for a future version.
"""

import ast
import logging

from app.chunking.chunker import Chunk
from app.chunking.chunker import chunk_document as _generic_chunk_document
from app.ingestion.loader import Document

logger = logging.getLogger(__name__)


def _non_python_metadata(document_metadata: dict) -> dict:
    """
    A copy of document_metadata with "language" removed, for documents
    handed off to the generic chunker (syntax-error fallback, or
    module-level leftover code).

    Without this, chunk_document() would see metadata["language"] ==
    "python" on the handed-off Document and dispatch it straight back
    to chunk_python_document() - infinite recursion, since a
    syntax-broken file fails to parse the same way every time, and
    leftover module-level source (imports/constants) has no
    classes/functions to consume, so nothing would ever change
    between calls.
    """
    return {key: value for key, value in document_metadata.items() if key != "language"}


def _build_class_id(document_metadata: dict, chunk_index: int) -> str:
    """
    Stable identifier shared by a whole-class chunk and every one of
    its per-method chunks, mirroring
    app.chunking.chunker._build_table_id's document_id + index scheme.
    """
    document_id = document_metadata.get("document_id") or document_metadata.get("file_name", "document")
    return f"{document_id}::class_{chunk_index}"


def _qualified_name(class_name: str | None, function_name: str | None) -> str:
    """Breadcrumb for a code chunk's "section" metadata - see module docstring."""
    if class_name and function_name:
        return f"{class_name}.{function_name}"
    return class_name or function_name or ""


def _node_line_range(node: ast.AST) -> set[int]:
    """Every source line number (1-indexed, inclusive) a node's text spans."""
    end_line = node.end_lineno or node.lineno
    return set(range(node.lineno, end_line + 1))


def _chunk_function(
    node: "ast.FunctionDef | ast.AsyncFunctionDef",
    source: str,
    document_metadata: dict,
    chunk_index: int,
    class_name: str | None,
    class_id: str | None = None,
) -> tuple[Chunk, set[int]]:
    """
    Build one atomic chunk for a single function/method, keeping its
    exact source text (see module docstring). Returns the chunk plus
    the set of source line numbers it consumed, so the caller can
    track what's already been accounted for.
    """
    function_source = ast.get_source_segment(source, node) or ""
    breadcrumb = _qualified_name(class_name, node.name)

    metadata = dict(document_metadata)
    metadata["chunk_index"] = chunk_index
    metadata["content_type"] = "method" if class_name else "function"
    metadata["class_name"] = class_name
    metadata["function_name"] = node.name
    metadata["section"] = breadcrumb
    if class_id:
        metadata["class_id"] = class_id

    label = "Method" if class_name else "Function"
    content = f"[{label}: {breadcrumb}]\n\n{function_source}"
    return Chunk(content=content, metadata=metadata), _node_line_range(node)


def _chunk_class(
    node: ast.ClassDef,
    source: str,
    document_metadata: dict,
    start_chunk_index: int,
) -> tuple[list[Chunk], set[int]]:
    """
    Build the parent whole-class chunk plus one child chunk per method
    inside it (see module docstring's "Strategy 4 for code" design).
    Returns (chunks, covered_line_numbers).
    """
    class_source = ast.get_source_segment(source, node)
    if class_source is None:
        return [], set()

    class_chunk_index = start_chunk_index
    class_id = _build_class_id(document_metadata, class_chunk_index)

    class_metadata = dict(document_metadata)
    class_metadata["chunk_index"] = class_chunk_index
    class_metadata["content_type"] = "class"
    class_metadata["class_name"] = node.name
    class_metadata["function_name"] = None
    class_metadata["section"] = node.name
    class_metadata["class_id"] = class_id

    chunks = [Chunk(content=f"[Class: {node.name}]\n\n{class_source}", metadata=class_metadata)]
    covered_lines = _node_line_range(node)

    method_index = class_chunk_index + 1
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_chunk, method_lines = _chunk_function(
                child, source, document_metadata, method_index, class_name=node.name, class_id=class_id
            )
            chunks.append(method_chunk)
            covered_lines.update(method_lines)
            method_index += 1

    return chunks, covered_lines


def _extract_uncovered_lines(source: str, covered_lines: set[int]) -> str:
    """
    Blank out every source line already claimed by a class/function
    chunk, keeping everything else (module docstring, imports,
    top-level constants, `if __name__ == "__main__":`, etc.) at its
    original line number so it can still be fed through the ordinary
    generic chunker without duplicating anything already chunked
    above.
    """
    lines = source.split("\n")
    return "\n".join(
        "" if line_number in covered_lines else line for line_number, line in enumerate(lines, start=1)
    )


def chunk_python_document(
    document: Document,
    chunk_size: int = 150,
    chunk_overlap: int = 30,
) -> list[Chunk]:
    """
    AST-based chunking entry point for a Python source Document (see
    module docstring for the class/method parent-child design).

    Falls back to app.chunking.chunker.chunk_document (the ordinary
    generic, character/token-window chunker) for the WHOLE file if
    the source doesn't parse as valid Python at all.

    Args:
        document: A Document whose content is Python source code.
        chunk_size: Passed through to the generic chunker for any
            module-level leftover code (see _extract_uncovered_lines).
            Class/method/function chunks are always atomic and ignore
            this, same as a Markdown table chunk ignores it.
        chunk_overlap: Passed through the same way.

    Returns:
        A list of Chunk objects: one per class (parent) plus one per
        method inside it (child), one per top-level function, and
        zero or more module-level chunks for anything else.
    """
    source = document.content
    file_label = document.metadata.get("file_path", document.metadata.get("file_name", "document"))

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        logger.warning(
            "Could not parse '%s' as Python (%s) - falling back to generic chunking.",
            file_label,
            error,
        )
        fallback_document = Document(content=source, metadata=_non_python_metadata(document.metadata))
        return _generic_chunk_document(fallback_document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Chunk] = []
    chunk_index = 0
    covered_lines: set[int] = set()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_chunks, consumed = _chunk_class(node, source, document.metadata, chunk_index)
            chunks.extend(class_chunks)
            chunk_index += len(class_chunks)
            covered_lines.update(consumed)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_chunk, consumed = _chunk_function(
                node, source, document.metadata, chunk_index, class_name=None
            )
            chunks.append(function_chunk)
            chunk_index += 1
            covered_lines.update(consumed)

    leftover_source = _extract_uncovered_lines(source, covered_lines)
    if leftover_source.strip():
        leftover_document = Document(content=leftover_source, metadata=_non_python_metadata(document.metadata))
        leftover_chunks = _generic_chunk_document(
            leftover_document, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        for chunk in leftover_chunks:
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata.setdefault("content_type", "module")
            chunk_index += 1
        chunks.extend(leftover_chunks)

    logger.info(
        "AST-chunked '%s' into %d chunk(s) (classes/methods/functions/module-level)",
        file_label,
        len(chunks),
    )
    return chunks

"""
Chunking for the RAG pipeline.

Responsibility: split a Document's content into smaller chunks. No
embedding happens here.

Strategy used: heading-aware, structure-aware Markdown chunking.

The document is first split into structural blocks: headings, fenced
code blocks / ASCII diagrams, Markdown tables, horizontal rules, and
plain paragraphs. Horizontal rules (e.g. "---") are dropped, since they
carry no semantic content and would otherwise be miscounted as
"paragraph" blocks.

Blocks are then grouped into SECTIONS along heading boundaries (a new
section starts at every heading; a "breadcrumb" tracks the current
heading hierarchy, e.g. "7. Chunking Strategy Comparison"). Each
chunk is prefixed with its section's breadcrumb, so retrieval always
knows which part of the document a chunk came from.

Within a section, blocks are greedily packed into chunks up to
chunk_size. Code blocks/diagrams are ATOMIC (never split internally).
Tables are handled with TWO representations, kept side by side: the
whole table is always forced into its own dedicated chunk (atomic,
never combined with surrounding text - good for comparison-style
questions like "list all roles"), AND one additional chunk is
generated per data row, each paired with the header row so it is
self-describing on its own (good for precise single-value questions
like "which role can deploy to production?"). Neither representation
replaces the other. Every row chunk carries a `table_id` matching its
parent whole-table chunk's own id, so retrieval can trade a retrieved
row for the complete table when that gives more complete context (see
app.retrieval.retriever._complete_partial_tables) - this is what fixes
the case where a table has more rows than top_k, so only some of its
rows would otherwise be retrieved. When a section needs more than one
chunk, the
last chunk_overlap words of the previous chunk are carried forward
into the next one, so context is not lost at chunk boundaries (this
restores overlap between packed blocks, not just within the paragraph
fallback).

Chunk size/overlap are measured in TOKENS if the optional `tiktoken`
package is installed (matches how real embedding/LLM models measure
limits); otherwise this falls back to word count, which is only an
approximation.

Known, disclosed limitations (not fully solved here):
- This is a hand-written line-based parser, not a full CommonMark
  parser. Nested lists, blockquotes, and inline HTML are treated as
  plain paragraph text rather than being parsed structurally.
- The paragraph-fallback splitter (for a single oversized paragraph)
  still slices by word windows even when token counting is active,
  since slicing exactly on token boundaries would require re-encoding
  each candidate window.

(See data/raw/README.md section 7 for a comparison of chunking strategies.)
"""

import logging
import re
from dataclasses import dataclass, field

from app.ingestion.loader import Document, PAGE_MARKER_PREFIX, PAGE_MARKER_SUFFIX, _IMAGE_MARKER_RE

logger = logging.getLogger(__name__)

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
# Matches the invisible per-page marker load_pdf_document() inserts
# (e.g. "\x00PAGE:3\x00") so a chunk's page_number can be recovered
# even though the whole PDF is one continuous Document. Never appears
# in Markdown/DOCX content, since only load_pdf_document() emits it.
_PAGE_MARKER_RE = re.compile(
    rf"^{re.escape(PAGE_MARKER_PREFIX)}(\d+){re.escape(PAGE_MARKER_SUFFIX)}$"
)

_tiktoken_encoder = None
_tiktoken_checked = False


@dataclass
class Chunk:
    """A smaller piece of a Document, ready to be embedded."""

    content: str
    metadata: dict = field(default_factory=dict)


def _count_units(text: str) -> int:
    """
    Count size units for a piece of text.

    Uses real token counts via tiktoken if it's installed (pip install
    tiktoken), since that matches how embedding/LLM models actually
    measure limits. Falls back to a word count approximation if
    tiktoken is not available.
    """
    global _tiktoken_encoder, _tiktoken_checked

    if not _tiktoken_checked:
        try:
            import tiktoken

            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _tiktoken_encoder = None
        _tiktoken_checked = True

    if _tiktoken_encoder is not None:
        return len(_tiktoken_encoder.encode(text))
    return len(text.split())


def _using_tokens() -> bool:
    """Whether _count_units is currently measuring tokens (vs. words)."""
    _count_units("")  # ensure the tiktoken probe has run
    return _tiktoken_encoder is not None


def _split_into_blocks(text: str) -> list[tuple[str, str]]:
    """
    Split raw Markdown text into (block_type, block_text) tuples.

    block_type is one of: "heading", "code", "table", "paragraph",
    "page_marker" (an invisible PDF page-boundary signal - see
    _PAGE_MARKER_RE - carrying the page number as block_text, never
    real content).
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        joined = "\n".join(paragraph_buffer).strip()
        if joined:
            blocks.append(("paragraph", joined))
        paragraph_buffer.clear()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Invisible PDF page-boundary marker (see load_pdf_document) -
        # a page-tracking signal, never real content, so it is kept
        # out of paragraph text entirely rather than flushed into one.
        page_marker_match = _PAGE_MARKER_RE.match(line.strip())
        if page_marker_match:
            flush_paragraph()
            blocks.append(("page_marker", page_marker_match.group(1)))
            i += 1
            continue

        # Fenced code block / ASCII diagram — atomic, preserve verbatim.
        if line.strip().startswith("```"):
            flush_paragraph()
            code_lines = [line]
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                code_lines.append(lines[i])  # closing fence
                i += 1
            blocks.append(("code", "\n".join(code_lines)))
            continue

        # Horizontal rule (thematic break) — carries no semantic content,
        # so it is dropped rather than folded into a "paragraph" block.
        if _HR_RE.match(line):
            flush_paragraph()
            i += 1
            continue

        # Heading — starts a new section (handled later in grouping).
        if _HEADING_RE.match(line):
            flush_paragraph()
            blocks.append(("heading", line.strip()))
            i += 1
            continue

        # Markdown table — only if a valid header + separator row pair,
        # so a stray line with pipes in it isn't mistaken for a table.
        if (
            _TABLE_ROW_RE.match(line)
            and i + 1 < n
            and _TABLE_SEPARATOR_RE.match(lines[i + 1])
        ):
            flush_paragraph()
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", "\n".join(table_lines)))
            continue

        # Blank line marks a paragraph boundary.
        if line.strip() == "":
            flush_paragraph()
            i += 1
            continue

        paragraph_buffer.append(line)
        i += 1

    flush_paragraph()
    return blocks


def _group_into_sections(
    blocks: list[tuple[str, str]],
) -> list[dict]:
    """
    Group blocks into sections along heading boundaries.

    Each section is {"breadcrumb": "1. Title > 1.2 Subtitle", "blocks": [...]}.
    The breadcrumb tracks the current heading hierarchy (higher-level
    headings are dropped once a same-or-higher-level heading appears).
    """
    sections: list[dict] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    current_blocks: list[tuple[str, str]] = []

    def breadcrumb_str() -> str:
        return " > ".join(title for _level, title in heading_stack)

    def start_new_section() -> None:
        nonlocal current_blocks
        if current_blocks:
            sections.append(
                {"breadcrumb": breadcrumb_str(), "blocks": current_blocks}
            )
        current_blocks = []

    for block_type, block_text in blocks:
        if block_type == "heading":
            start_new_section()
            match = _HEADING_RE.match(block_text)
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_blocks.append((block_type, block_text))
        else:
            current_blocks.append((block_type, block_text))

    start_new_section()
    return sections


def _word_split_with_overlap(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """
    Fallback fixed-size word splitter, used only for a single oversized
    paragraph. Operates on word windows regardless of whether token
    counting is active elsewhere (see module docstring limitation).
    """
    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    pieces = []
    start = 0
    while start < len(words):
        pieces.append(" ".join(words[start : start + chunk_size]))
        start += step
    return pieces


def _build_table_id(document_metadata: dict, chunk_index: int, page_number: int | None = None) -> str:
    """
    Build a stable identifier shared by a whole-table chunk and every
    one of its per-row chunks, so retrieval can look the complete table
    back up given just one retrieved row (see
    app.retrieval.retriever._complete_partial_tables). Mirrors the same
    document_id + page_number + index scheme app.vectorstore.store
    uses for its own chunk ids, which keeps it unique across documents
    and, for multi-page PDFs, across pages too.

    page_number is the page this specific table chunk was packed from
    (computed during chunking from the PDF's page markers - see
    _pack_section), not a document-level attribute, since a whole PDF
    is now loaded as a single Document with no one page of its own.
    """
    document_id = document_metadata.get("document_id") or document_metadata.get(
        "file_name", "document"
    )
    if page_number is not None:
        return f"{document_id}::page_{page_number}::table_{chunk_index}"
    return f"{document_id}::table_{chunk_index}"


def _table_row_chunks(
    table_text: str,
    breadcrumb: str,
    document_metadata: dict,
    start_chunk_index: int,
    table_id: str,
    page_number: int | None = None,
) -> list[Chunk]:
    """
    Build one additional chunk per data row of a Markdown table, each
    paired with the table's header row so the row stays self-describing
    on its own.

    This is "Strategy 4" for table chunking: keep BOTH a whole-table
    chunk (built separately in _pack_section, good for comparison-style
    questions like "list all roles") AND per-row chunks (built here,
    good for precise single-value questions like "which role can
    deploy to production?") instead of picking only one representation.

    Each row chunk carries the same table_id as its parent whole-table
    chunk, so retrieval can trade a retrieved row for the complete
    table when that gives the model more complete context (see
    app.retrieval.retriever).
    """
    lines = table_text.split("\n")
    if len(lines) < 3:
        return []  # just a header + separator row, no data rows to split out

    header_line, separator_line, *data_lines = lines
    row_chunks: list[Chunk] = []

    for offset, data_line in enumerate(data_lines):
        row_table = "\n".join([header_line, separator_line, data_line])
        body = f"[Section: {breadcrumb}]\n\n{row_table}" if breadcrumb else row_table

        metadata = dict(document_metadata)
        metadata["chunk_index"] = start_chunk_index + offset
        metadata["block_types"] = ["table_row"]
        metadata["section"] = breadcrumb
        metadata["content_type"] = "table_row"
        metadata["table_id"] = table_id
        if page_number is not None:
            metadata["page_number"] = page_number
        row_chunks.append(Chunk(content=body, metadata=metadata))

    return row_chunks


def _is_pure_image_caption(parts: list[str], types: list[str]) -> bool:
    """
    True if every currently pending part is nothing but an
    "[Image: <alt text>] (<url>)" marker line (see loader._describe_image_tag)
    - i.e. there is no other prose/content pending. Such captions carry
    no unique retrievable text of their own (the image is already
    separately indexed via CLIP + OCR/vision extraction), so they
    should never be flushed as their own standalone chunk right before
    an atomic code block - only ever folded into it.
    """
    return bool(parts) and all(
        block_type in ("paragraph", "overlap") and _IMAGE_MARKER_RE.fullmatch(part.strip())
        for part, block_type in zip(parts, types)
    )


def _pack_section(
    section: dict,
    chunk_size: int,
    chunk_overlap: int,
    start_chunk_index: int,
    document_metadata: dict,
) -> list[Chunk]:
    """
    Pack one section's blocks into one or more chunks, prefixing each
    chunk with the section breadcrumb and carrying forward overlap text
    between consecutive chunks of the same section.

    Also tracks PDF page markers (see _PAGE_MARKER_RE) as blocks are
    consumed, and attaches page_number (the first page a chunk's
    content came from) - plus page_range, if a chunk happens to span
    more than one page's blocks - to that chunk's metadata. Markdown
    and DOCX content never contains these markers, so current_page
    simply stays None and no page metadata is added for those sources.
    """
    breadcrumb = section["breadcrumb"]
    blocks = section["blocks"]

    chunks: list[Chunk] = []
    chunk_index = start_chunk_index

    current_parts: list[str] = []
    current_types: list[str] = []
    current_units = 0
    current_pages: list[int] = []
    carry_over_text = ""
    carry_over_page: int | None = None
    current_page: int | None = None

    def build_content(body: str) -> str:
        if breadcrumb and body:
            return f"[Section: {breadcrumb}]\n\n{body}"
        if breadcrumb:
            return f"[Section: {breadcrumb}]"
        return body

    def flush(prepare_carry_over: bool) -> None:
        nonlocal current_parts, current_types, current_units, current_pages
        nonlocal carry_over_text, carry_over_page, chunk_index

        if not current_parts:
            return

        body = "\n\n".join(current_parts)
        metadata = dict(document_metadata)
        metadata["chunk_index"] = chunk_index
        metadata["block_types"] = list(current_types)
        metadata["section"] = breadcrumb

        pages_in_chunk = sorted({page for page in current_pages if page is not None})
        if pages_in_chunk:
            metadata["page_number"] = pages_in_chunk[0]
            if len(pages_in_chunk) > 1:
                metadata["page_range"] = f"{pages_in_chunk[0]}-{pages_in_chunk[-1]}"

        chunks.append(Chunk(content=build_content(body), metadata=metadata))
        chunk_index += 1

        if prepare_carry_over:
            words = body.split()
            carry_over_text = " ".join(words[-chunk_overlap:]) if words else ""
            carry_over_page = pages_in_chunk[-1] if pages_in_chunk else None
        else:
            carry_over_text = ""
            carry_over_page = None

        current_parts = []
        current_types = []
        current_units = 0
        current_pages = []

    for block_type, block_text in blocks:
        if block_type == "page_marker":
            current_page = int(block_text)
            continue

        if block_type == "heading":
            # Heading text itself is folded into the breadcrumb; skip
            # re-adding it as body content.
            continue

        block_units = _count_units(block_text)

        if block_type == "code":
            # Only flush the pending preamble as its OWN chunk if it is
            # not PURELY "[Image: ...] (url)" caption line(s). A code
            # sample is very often immediately preceded by a screenshot
            # of that same code, and flushing the caption alone would
            # store a near-empty, low-value chunk that duplicates the
            # section breadcrumb without the code - and since carry-over
            # would repeat that same short text in the very next (code)
            # chunk anyway, the standalone flush adds nothing but a
            # misleading, denser "header-only" competitor that can
            # outrank the real code chunk in similarity search (a URL
            # tokenizes into many small tokens, so this caption is often
            # NOT actually small in token count, even though it carries
            # no unique retrievable content - the image itself is
            # already separately indexed via CLIP + OCR/vision). Folding
            # it directly into the code chunk instead (even if that
            # makes the chunk oversized, same as any solo oversized code
            # block already is) keeps the caption and its code together.
            if (
                current_units + block_units > chunk_size
                and current_parts
                and not _is_pure_image_caption(current_parts, current_types)
            ):
                flush(prepare_carry_over=True)
                if carry_over_text:
                    current_parts.append(carry_over_text)
                    current_types.append("overlap")
                    current_units += _count_units(carry_over_text)
                    current_pages.append(carry_over_page)
            current_parts.append(block_text)
            current_types.append(block_type)
            current_units += block_units
            current_pages.append(current_page)
            if current_units > chunk_size:
                # Oversized atomic block (or oversized after folding in
                # a short preamble): still kept whole, own chunk.
                flush(prepare_carry_over=False)
        elif block_type == "table":
            # A table is ALWAYS forced into its own dedicated chunk
            # (never combined with surrounding paragraphs), so the
            # whole-table representation stays self-contained - good
            # for comparison-style questions ("list all roles").
            if current_parts:
                flush(prepare_carry_over=True)
                if carry_over_text:
                    current_parts.append(carry_over_text)
                    current_types.append("overlap")
                    current_units += _count_units(carry_over_text)
                    current_pages.append(carry_over_page)
            current_parts.append(block_text)
            current_types.append(block_type)
            current_units += block_units
            current_pages.append(current_page)
            flush(prepare_carry_over=False)
            chunks[-1].metadata["content_type"] = "table"
            table_page_number = chunks[-1].metadata.get("page_number")
            table_id = _build_table_id(
                document_metadata, chunks[-1].metadata["chunk_index"], table_page_number
            )
            chunks[-1].metadata["table_id"] = table_id

            # ALSO emit one chunk per data row (paired with the header),
            # so a single row can be retrieved precisely without
            # competing against the rest of the table's embedding -
            # good for specific-value questions ("which role can deploy
            # to production?"). Both representations are kept side by
            # side rather than choosing only one (Strategy 4). Sharing
            # table_id with the whole-table chunk lets retrieval trade a
            # retrieved row for the complete table when useful.
            row_chunks = _table_row_chunks(
                block_text, breadcrumb, document_metadata, chunk_index, table_id, table_page_number
            )
            chunks.extend(row_chunks)
            chunk_index += len(row_chunks)
        else:
            if block_units > chunk_size:
                flush(prepare_carry_over=False)
                for piece in _word_split_with_overlap(
                    block_text, chunk_size, chunk_overlap
                ):
                    metadata = dict(document_metadata)
                    metadata["chunk_index"] = chunk_index
                    metadata["block_types"] = ["paragraph_fallback"]
                    metadata["section"] = breadcrumb
                    if current_page is not None:
                        metadata["page_number"] = current_page
                    chunks.append(
                        Chunk(content=build_content(piece), metadata=metadata)
                    )
                    chunk_index += 1
            else:
                if current_units + block_units > chunk_size and current_parts:
                    flush(prepare_carry_over=True)
                    if carry_over_text:
                        current_parts.append(carry_over_text)
                        current_types.append("overlap")
                        current_units += _count_units(carry_over_text)
                        current_pages.append(carry_over_page)
                current_parts.append(block_text)
                current_types.append(block_type)
                current_units += block_units
                current_pages.append(current_page)

    flush(prepare_carry_over=False)
    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = 150,
    chunk_overlap: int = 30,
) -> list[Chunk]:
    """
    Split a Document's content into heading-aware, structure-aware chunks.

    Documents whose metadata marks them as Python source
    (metadata["language"] == "python", set by
    app.ingestion.loader.load_github_repository for GitHub-ingested
    .py files) are instead routed to
    app.chunking.code_chunker.chunk_python_document(), which parses
    the file's AST and returns class/method/function-level chunks
    (see that module's docstring) instead of this function's
    Markdown-oriented heading/table/paragraph packing - imported
    lazily, here, to avoid a circular import (code_chunker imports
    this module's chunk_document/Chunk for its own fallback path).

    Args:
        document: The Document to split.
        chunk_size: Target maximum size per chunk (tokens if tiktoken
            is installed, otherwise approximate word count).
        chunk_overlap: Amount repeated between consecutive chunks
            within the same section, so context is not lost at
            chunk boundaries.

    Returns:
        A list of Chunk objects, each carrying the parent document's
        metadata plus chunk_index, block_types, and section breadcrumb.

    Raises:
        ValueError: if chunk_overlap >= chunk_size.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})"
        )

    if document.metadata.get("language") == "python":
        from app.chunking.code_chunker import chunk_python_document

        return chunk_python_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    blocks = _split_into_blocks(document.content)
    sections = _group_into_sections(blocks)

    all_chunks: list[Chunk] = []
    next_index = 0
    for section in sections:
        section_chunks = _pack_section(
            section, chunk_size, chunk_overlap, next_index, document.metadata
        )
        all_chunks.extend(section_chunks)
        next_index += len(section_chunks)

    logger.info(
        "Chunked '%s' into %d chunks across %d sections (chunk_size=%d, chunk_overlap=%d)",
        document.metadata.get("file_name", "document"),
        len(all_chunks),
        len(sections),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks


if __name__ == "__main__":
    import sys

    from app.ingestion.loader import (
        load_docx_document,
        load_markdown_document,
        load_pdf_document,
    )

    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/README.md"
    if file_path.lower().endswith(".pdf"):
        documents = [load_pdf_document(file_path)]
    elif file_path.lower().endswith(".docx"):
        documents = [load_docx_document(file_path)]
    else:
        documents = [load_markdown_document(file_path)]

    unit_label = "tokens" if _using_tokens() else "words (tiktoken not installed)"
    print(f"Sizing unit: {unit_label}")

    for doc in documents:
        chunks = chunk_document(doc, chunk_size=150, chunk_overlap=30)
        page_label = (
            f" (page {doc.metadata['page_number']}/{doc.metadata['total_pages']})"
            if "page_number" in doc.metadata
            else ""
        )
        print(f"\n{'#' * 70}")
        print(f"Document: {doc.metadata.get('file_name')}{page_label} -> {len(chunks)} chunk(s)")
        print("#" * 70)

        for c in chunks:
            types = c.metadata.get("block_types", [])
            section = c.metadata.get("section", "")
            content_type = c.metadata.get("content_type", "")
            units = _count_units(c.content)
            extra = f" | content_type: {content_type}" if content_type else ""
            page = c.metadata.get("page_range") or c.metadata.get("page_number")
            page_info = f" | page: {page}" if page is not None else ""
            print(f"\n### Chunk {c.metadata['chunk_index']} "
                  f"| {unit_label}: {units} | section: '{section}' | blocks: {types}{extra}{page_info}")
            print("-" * 70)
            print(c.content)
            print("=" * 70)

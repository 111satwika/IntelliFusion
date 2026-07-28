"""
Document loading for the RAG pipeline.

Responsibility: turn a file on disk into one or more normalized
Document objects (content + metadata). No chunking, no embedding
happens here.

Design:
- Every loader function returns the SAME Document shape (content +
  metadata dict), regardless of source file format. Nothing downstream
  (chunker, embedder, store, retriever) needs to know or care whether
  a Document came from a .md file or a .pdf file - this is what makes
  the pipeline "multi-source": one shape in, from many formats.
- A Markdown file is one logical document -> load_markdown_document()
  returns a single Document.
- A PDF is internally page-based (pdfplumber reads it page by page),
  but load_pdf_document() still returns ONE Document for the whole
  file - all pages concatenated - the same "one continuous stream"
  treatment used for DOCX (see below), rather than one Document per
  page. This keeps every source type consistent at the Document
  level: nothing downstream ever has to reason about "which
  Document/page did this chunk come from", only "which section of
  this file". Unlike DOCX, a PDF page IS a real, stored structural
  unit (pdfplumber can report an exact page for any text), so this
  page number is still worth keeping - rather than lose it, each
  page's text is prefixed with an invisible marker
  (PAGE_MARKER_PREFIX/PAGE_MARKER_SUFFIX) before the pages are joined.
  app.chunking.chunker reads these markers off while splitting the
  content into blocks/sections, and attaches the page(s) a chunk's
  content came from as page_number (and page_range, if a chunk's
  packed blocks happen to straddle a page boundary) - restoring
  per-chunk page citations without reintroducing a per-page Document
  split. A table detected as continuing from the previous page (see
  _is_table_continuation) still has its rows merged back into the page
  where it started, before the pages are joined - this step remains
  necessary even with one Document per file, since chunker.py's
  table-block detection needs the table's lines to be contiguous, and
  merely concatenating pages does not automatically re-join a table's
  two page-fragments into one.
- PDF binary parsing (compression, fonts, cross-reference tables) is
  far too complex to hand-roll, unlike the manual cosine-similarity
  implementation used earlier in this project - so this wraps the
  `pdfplumber` library in one function, the same "one place per
  concern" pattern used for the embedding model in embedder.py.
- pdfplumber can detect table structure (rows/columns), unlike plain
  text extraction. Detected tables are serialized as GitHub-flavored
  Markdown tables ("| cell | cell |" rows) inside the page's content.
  This is a deliberate reuse of an existing capability: chunker.py
  already treats "| ... |" rows as an ATOMIC block that is never split
  internally - so PDF tables get the same protection as Markdown
  tables, without adding any table-specific logic to the chunker.
  Prose text on the page (outside detected table regions) is kept as
  plain paragraph text. Multi-column layouts and headers/footers are
  still not understood as structured elements beyond this.
- DOCX is loaded the same way: as ONE continuous Document, never
  split per page. Unlike a PDF page (a fixed structural unit baked
  into the file), a Word document has no reliable stored concept of
  "page" at all - page breaks are a side effect of rendering
  (margins, fonts, zoom level), recomputed every time the file is
  opened, not data the .docx file guarantees - so there was never a
  reliable per-page citation to trade away here in the first place.
  Headings (Word's "Heading N" paragraph styles) are converted to
  "#"-prefixed Markdown heading lines, and tables reuse
  _table_to_markdown() - the same two conventions chunker.py already
  understands from Markdown - so chunker.py needs zero DOCX-specific
  code, exactly like the PDF case.
- A website is loaded the same way as DOCX: ONE continuous Document
  per URL, never split per section. `requests` fetches the raw HTML
  and `BeautifulSoup` (bs4) parses it - hand-rolling an HTML parser
  would be as impractical as hand-rolling PDF binary parsing, so this
  wraps `bs4` the same "one place per concern" way pdfplumber/docx are
  wrapped. Boilerplate tags (script/style/nav/header/footer/aside) are
  dropped before extraction, and an <article>/<main> element is
  preferred over the full <body> when present, to bias toward the
  page's actual content over navigation/ads. `<h1>`-`<h6>` become
  "#"-prefixed Markdown heading lines and `<table>` elements reuse
  _table_to_markdown() - the same two conventions chunker.py already
  understands - so, again, chunker.py needs zero web-specific code.
- crawl_website() discovers and loads a whole documentation site from
  one seed URL, instead of requiring every page's exact URL up front
  in data/urls.txt. It deliberately does NOT crawl an entire domain -
  it only follows links that stay under the seed URL's own path
  (e.g. a seed of /docs/en/product/hub only follows links under
  /docs/en/product/hub/..., never sibling docs or the rest of the
  site) and stops after max_pages, so it can't runaway-scrape an
  entire unrelated site by accident. Link discovery reads <a href>
  tags from the FULL page (before boilerplate stripping), since a
  documentation site's navigation menu/sidebar is usually exactly
  where the list of other pages to crawl lives - load_website_document
  strips that same nav for content-extraction purposes, but that
  happens afterward, on a per-page basis, and doesn't affect link
  discovery. Each in-scope page is turned into a Document via the same
  extraction logic as load_website_document (in fact they share it -
  see _build_website_document), so a crawled page is indistinguishable
  downstream from one loaded individually.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import docx
import pdfplumber
import requests
from bs4 import BeautifulSoup
from docx.table import Table as DocxTable

logger = logging.getLogger(__name__)

# Invisible marker inserted at the start of each PDF page's text (see
# load_pdf_document) so app.chunking.chunker can still recover which
# physical page a chunk's content came from, even though the whole PDF
# is loaded as ONE continuous Document rather than one Document per
# page. Uses NUL-delimited sentinels that will never appear in real
# extracted text. chunker.py strips these markers out of chunk content
# before it ever reaches embedding/the LLM - they exist purely as a
# page-boundary signal read off during chunking, not as content.
PAGE_MARKER_PREFIX = "\x00PAGE:"
PAGE_MARKER_SUFFIX = "\x00"


@dataclass
class Document:
    """Unified document representation used across the whole pipeline."""

    content: str
    metadata: dict = field(default_factory=dict)


def load_markdown_document(file_path: str) -> Document:
    """
    Load a single Markdown file from disk and return it as a Document.

    Args:
        file_path: Path to the .md file to load.

    Returns:
        Document with the file's raw text as `content` and basic
        file metadata attached.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        logger.error("Document not found: %s", file_path)
        raise FileNotFoundError(f"Document not found: {file_path}")

    raw_text = path.read_text(encoding="utf-8")

    # Normalize Windows line endings so chunking later sees consistent input.
    normalized_text = raw_text.replace("\r\n", "\n")

    metadata = {
        "source_type": "markdown",
        "file_name": path.name,
        "file_path": str(path),
        "document_id": path.name,
    }

    logger.info(
        "Loaded document '%s' (%d characters)", path.name, len(normalized_text)
    )
    return Document(content=normalized_text, metadata=metadata)


def _clean_cell(cell) -> str:
    """Normalize a single table cell into one Markdown-safe text line."""
    text = (cell or "").strip().replace("\n", " ")
    return text.replace("|", "\\|")  # escape so it isn't mistaken for a column separator


def _table_to_markdown(table_rows: list[list]) -> str:
    """
    Serialize a pdfplumber-extracted table (list of rows, each a list of
    cell strings) into a GitHub-flavored Markdown table.

    This is what lets chunker.py's existing table-detection regex treat
    a PDF table as a single atomic block, the same as a hand-written
    Markdown table - no chunker changes needed.
    """
    header, *data_rows = [[_clean_cell(cell) for cell in row] for row in table_rows]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in data_rows)
    return "\n".join(lines)


def _extract_prose_text(page: pdfplumber.page.Page, table_bboxes: list) -> str:
    """Extract a page's plain prose text, excluding any table regions."""
    if not table_bboxes:
        return (page.extract_text() or "").strip()

    def outside_tables(obj: dict) -> bool:
        return not any(
            bbox[0] <= obj["x0"] and obj["x1"] <= bbox[2]
            and bbox[1] <= obj["top"] and obj["bottom"] <= bbox[3]
            for bbox in table_bboxes
        )

    return (page.filter(outside_tables).extract_text() or "").strip()


def _is_table_continuation(prev_rows: list[list], next_rows: list[list]) -> bool:
    """
    Heuristic: a table at the very start of a page is treated as a
    continuation of the previous page's last table when they have the
    same column count AND the same header row text. Most PDF table
    generators - including fpdf2's own multi-page table support used
    for this project's test fixtures - repeat the header row on every
    page a table spans onto, so a matching header is a reliable signal
    that this is the same table continuing, not a new, unrelated one.

    Known limitation: a continued table that does NOT repeat its
    header row (no header row at all on the continuation page) will
    not be detected by this heuristic and will be treated as a
    separate table instead.
    """
    if not prev_rows or not next_rows:
        return False

    prev_header = [_clean_cell(cell) for cell in prev_rows[0]]
    next_header = [_clean_cell(cell) for cell in next_rows[0]]
    return prev_header == next_header


def load_pdf_document(file_path: str) -> Document:
    """
    Load a single PDF file from disk and return the WHOLE file as ONE
    Document (all pages concatenated), matching how load_docx_document()
    treats a .docx file as one continuous stream rather than one
    Document per page.

    Args:
        file_path: Path to the .pdf file to load.

    Returns:
        A single Document whose content is every page's text (in page
        order) joined together. Metadata includes source_type,
        file_name, file_path, document_id, and total_pages. Each
        page's text is prefixed with an invisible page marker (see
        PAGE_MARKER_PREFIX/PAGE_MARKER_SUFFIX) so app.chunking.chunker
        can still attach an accurate page_number to each chunk it
        produces, without reintroducing a per-page Document split here.
        Tables detected on a page are embedded in the content as
        Markdown tables (see _table_to_markdown).

        A table split across a page boundary (continuing without a new
        header, but repeating the same header row - see
        _is_table_continuation) is reconstructed BEFORE chunking: its
        continuation rows are merged into the table on the page where
        it started. This merge step still matters even though the
        whole file becomes one Document: without it, chunker.py would
        still see the continuation as a second, separate Markdown
        table later in the same text (its table-block detection needs
        contiguous "| ... |" lines, and the two page-fragments are not
        automatically re-joined into one just by concatenating pages),
        producing the same "two incomplete fragments" problem removing
        the per-page Document split alone does not fix.

        Pages with no extractable text (e.g. a scanned image page with
        no OCR) contribute nothing to the content, rather than leaving
        a stray blank section.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        logger.error("Document not found: %s", file_path)
        raise FileNotFoundError(f"Document not found: {file_path}")

    with pdfplumber.open(str(path)) as pdf:
        total_pages = len(pdf.pages)

        # Pass 1: extract prose text and RAW table rows per page. Tables
        # are kept as structured data (not yet serialized to Markdown)
        # so a continuation across a page boundary can be detected and
        # merged before anything is turned into final content.
        page_prose: list[str] = []
        page_tables: list[list[list[list]]] = []

        for page in pdf.pages:
            tables = page.find_tables()
            table_bboxes = [table.bbox for table in tables]

            page_prose.append(_extract_prose_text(page, table_bboxes))
            page_tables.append(
                [
                    table.extract()
                    for table in tables
                    if table.extract() and len(table.extract()) > 1
                ]
            )

        # Pass 2: merge a page's FIRST table into the previous page's
        # LAST table when it looks like a continuation.
        for page_index in range(1, total_pages):
            if not page_tables[page_index] or not page_tables[page_index - 1]:
                continue

            first_table_this_page = page_tables[page_index][0]
            last_table_prev_page = page_tables[page_index - 1][-1]

            if _is_table_continuation(last_table_prev_page, first_table_this_page):
                _header, *continued_rows = first_table_this_page
                last_table_prev_page.extend(continued_rows)
                page_tables[page_index].pop(0)
                logger.info(
                    "Merged %d continued table row(s) from page %d into the "
                    "table that started on page %d",
                    len(continued_rows),
                    page_index + 1,
                    page_index,
                )

        # Pass 3: serialize each page's final prose + table set, then
        # join every page into ONE continuous string - no per-page
        # Document boundary, matching load_docx_document(). Each page's
        # text is prefixed with an invisible marker line so chunker.py
        # can recover which physical page a chunk came from and attach
        # it as page_number metadata, restoring per-chunk page citations
        # even though the file is loaded as one Document.
        page_texts: list[str] = []
        for index in range(total_pages):
            parts = [page_prose[index]] if page_prose[index] else []
            parts.extend(_table_to_markdown(rows) for rows in page_tables[index])
            page_text = "\n\n".join(parts).strip()
            if page_text:
                marker = f"{PAGE_MARKER_PREFIX}{index + 1}{PAGE_MARKER_SUFFIX}"
                page_texts.append(f"{marker}\n\n{page_text}")

    content = "\n\n".join(page_texts)

    metadata = {
        "source_type": "pdf",
        "file_name": path.name,
        "file_path": str(path),
        "document_id": path.name,
        "total_pages": total_pages,
    }

    logger.info(
        "Loaded PDF '%s': %d page(s), %d character(s)",
        path.name,
        total_pages,
        len(content),
    )
    return Document(content=content, metadata=metadata)


_HEADING_STYLE_RE = re.compile(r"^Heading (\d+)$")


def _heading_level(style_name: str) -> int | None:
    """
    Map a Word paragraph style name to a Markdown heading level
    (1-6), or None if the style isn't a heading.

    "Title" is treated as a top-level (H1) heading. "Heading N" styles
    map directly to level N, capped at 6 (Markdown's deepest level).
    """
    if style_name == "Title":
        return 1
    match = _HEADING_STYLE_RE.match(style_name or "")
    return min(int(match.group(1)), 6) if match else None


def load_docx_document(file_path: str) -> Document:
    """
    Load a single .docx file from disk and return it as ONE Document
    (never split per page - see module docstring for why).

    Paragraphs using a "Heading N" style become "#"-prefixed Markdown
    heading lines (so chunker.py's existing heading-aware section
    grouping applies unchanged, exactly as it does for README.md).
    Tables are serialized with _table_to_markdown(), the same helper
    load_pdf_document() uses, so chunker.py's table-atomicity handling
    applies here too with zero DOCX-specific code. Paragraphs and
    tables are walked in true document order (docx.iter_inner_content),
    so the resulting content preserves the original reading order.

    Args:
        file_path: Path to the .docx file to load.

    Returns:
        Document with the file's text (headings + prose + tables,
        each table as a Markdown table) as `content`, and metadata
        matching the same shape used by the other loaders
        (source_type, file_name, file_path, document_id).

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        logger.error("Document not found: %s", file_path)
        raise FileNotFoundError(f"Document not found: {file_path}")

    word_document = docx.Document(str(path))

    parts: list[str] = []
    for block in word_document.iter_inner_content():
        if isinstance(block, DocxTable):
            table_rows = [[cell.text for cell in row.cells] for row in block.rows]
            if len(table_rows) > 1:
                parts.append(_table_to_markdown(table_rows))
            continue

        text = block.text.strip()
        if not text:
            continue

        style_name = block.style.name if block.style else ""
        heading_level = _heading_level(style_name)
        parts.append(f"{'#' * heading_level} {text}" if heading_level else text)

    content = "\n\n".join(parts)

    metadata = {
        "source_type": "docx",
        "file_name": path.name,
        "file_path": str(path),
        "document_id": path.name,
    }

    logger.info("Loaded DOCX '%s' (%d characters)", path.name, len(content))
    return Document(content=content, metadata=metadata)


_HTML_HEADING_TAG_RE = re.compile(r"^h([1-6])$")

# Boilerplate elements that never contain the actual page content, so
# they're dropped entirely before extraction rather than risking their
# text (nav links, cookie banners, footer boilerplate) polluting chunks.
_HTML_TAGS_TO_STRIP = ["script", "style", "noscript", "nav", "header", "footer", "aside"]

# The element tags load_website_document() walks in document order.
# Deliberately excludes <div>/<span> (too generic - would just re-walk
# every other tag's text a second time as an untyped block) and list
# items are treated as plain paragraph lines rather than a distinct
# block type, since chunker.py has no list-specific handling to feed.
# "img" is included so meaningful images (see _describe_image_tag) are
# captured inline, in document order, right where they appear relative
# to the surrounding text - preserving the image/text relationship
# instead of extracting images as separate, disconnected entries.
# "pre" is included so literal code samples (rendered as <pre>/<pre><code>
# by doc sites, e.g. IBM Docs) are captured as real text instead of being
# silently dropped - previously MISSING from this list meant every fenced
# code block on every crawled page was invisible to find_all(), which is
# why questions asking for exact code from a doc page returned nothing
# (or a fabricated answer) even though the code was genuinely on the page
# and not just an unlabelled screenshot (see /memories/repo/rag-platform-notes.md).
_HTML_CONTENT_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "table", "li", "img", "pre"]


def _html_table_to_rows(table_tag) -> list[list[str]]:
    """Extract a <table> element's rows/cells as plain text, one list per <tr>."""
    rows = []
    for row in table_tag.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if cells:
            rows.append([cell.get_text(" ", strip=True) for cell in cells])
    return rows


def _describe_image_tag(img_tag, page_url: str) -> str | None:
    """
    Turn an <img> element into a searchable inline text marker, or None
    if the image is decorative and should be skipped entirely.

    Approach: use the image's own alt text rather than a vision model
    (see the module-level note in crawl_website()'s docstring area for
    why) - simple, needs no new model/dependency, and works as long as
    the site's alt text is meaningful, which is the common case for
    documentation sites (accessibility-conscious authors write
    descriptive alt text for diagrams/screenshots).

    Decorative-image heuristic: an empty/missing alt attribute is the
    standard HTML convention for "this image carries no information,
    skip it" (e.g. `alt=""` on a logo or spacer) - this is exactly the
    signal authors use to mark images that assistive tech should
    ignore, so it doubles as a reliable "don't bother embedding this"
    signal here too. A whitespace-only alt is treated the same way.

    Returns a single line like "[Image: <alt text>] (<absolute url>)"
    so it flows through chunker.py as ordinary paragraph text - no
    chunker/embedder/store changes needed, same "reuse the existing
    text pipeline" pattern used for headings/tables elsewhere in this
    file. The absolute URL is kept so the source image can still be
    cited/opened, even though only its alt text is what's searchable.
    """
    alt_text = (img_tag.get("alt") or "").strip()
    if not alt_text:
        return None

    src = img_tag.get("src")
    if not src:
        return f"[Image: {alt_text}]"

    absolute_url = urljoin(page_url, src)
    return f"[Image: {alt_text}] ({absolute_url})"


# Matches the "[Image: <alt text>] (<url>)" marker _describe_image_tag()
# embeds inline into a website Document's content, so it can be parsed
# back out into structured records post hoc (see extract_image_records).
_IMAGE_MARKER_RE = re.compile(r"^\[Image: (.+)\] \((\S+)\)$", re.MULTILINE)


def extract_image_records(document: Document) -> list[dict]:
    """
    Parse the "[Image: <alt text>] (<url>)" markers _build_website_document()
    embeds inline (see _describe_image_tag) back out of an already-loaded
    Document's content, as structured records ready for multimodal
    (image) embedding/storage.

    This is a deliberately separate, opt-in step from normal text
    chunking/embedding (see app.chunking.chunker.chunk_document) rather
    than something folded into it automatically: an image needs a
    completely different embedding model and vector space (CLIP, via
    app.embeddings.image_embedder) than the all-MiniLM-L6-v2 text
    embeddings chunk_document()'s output feeds into, and its own Chroma
    collection (see app.vectorstore.store's separate rag_image_chunks
    collection) - see app.ingestion.ingest for how the two paths are
    combined during ingestion.

    Returns:
        A list of dicts, one per image marker found, each with
        "image_url", "alt_text", "page_url", and "page_title" (the
        page's file_name metadata, for a human-readable citation).
        A Document with no image markers - which is every non-web
        Document, since only _build_website_document() ever emits this
        marker format - returns an empty list.
    """
    return [
        {
            "image_url": image_url,
            "alt_text": alt_text,
            "page_url": document.metadata.get("url"),
            "page_title": document.metadata.get("file_name"),
        }
        for alt_text, image_url in _IMAGE_MARKER_RE.findall(document.content)
    ]



def _fetch_html(url: str, timeout: int) -> BeautifulSoup:
    """
    Fetch a URL and parse it as HTML. Raises
    requests.exceptions.RequestException if the page can't be fetched
    (network error, timeout, non-2xx status, etc.) - shared by
    load_website_document() and crawl_website().

    This never executes JavaScript, so a page whose real content is
    only rendered client-side (a single-page app) comes back as
    whatever the server sent - often an empty shell. Use
    _fetch_html_with_browser() (via render_js=True on
    load_website_document()/crawl_website()) for such sites instead.
    """
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAGIngestBot/1.0)"},
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _fetch_html_with_browser(url: str, timeout: int) -> BeautifulSoup:
    """
    Fetch a URL by rendering it in a real headless Chromium browser
    (via Playwright), then parse the fully-rendered DOM as HTML.

    Unlike _fetch_html(), this actually executes the page's
    JavaScript before reading its markup, so it works on
    JavaScript-rendered single-page-app sites (e.g. IBM Docs) whose
    raw HTTP response is just an empty shell (a <div id="app">) with
    no server-rendered content or links - _fetch_html() would report
    0 characters and 0 links for such a page no matter how it's
    parsed. It's meaningfully slower/heavier than _fetch_html() (spins
    up a Chromium instance per call), so it's opt-in only (render_js=
    True on load_website_document()/crawl_website()), never the
    default fetch path.

    Requires the Playwright Python package AND its Chromium browser
    binary to be installed (`pip install playwright` then
    `playwright install chromium`) - imported lazily here so neither
    is required unless a caller actually asks for render_js=True.

    Raises whatever Playwright raises on navigation failure/timeout
    (e.g. playwright.sync_api.Error, playwright.sync_api.TimeoutError).
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright_session:
        browser = playwright_session.chromium.launch()
        try:
            page = browser.new_page(user_agent="Mozilla/5.0 (compatible; RAGIngestBot/1.0)")
            page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
            html = page.content()
        finally:
            browser.close()

    return BeautifulSoup(html, "html.parser")


def _build_website_document(url: str, soup: BeautifulSoup) -> Document:
    """
    Turn an already-fetched page's soup into ONE Document (never split
    per section - see module docstring for why, same treatment as
    DOCX). Mutates `soup` in place (boilerplate tags are decomposed).

    `<h1>`-`<h6>` become "#"-prefixed Markdown heading lines and
    `<table>` elements are serialized with _table_to_markdown() - the
    same two conventions load_docx_document()/load_pdf_document() use
    - so chunker.py needs zero web-specific code to chunk this content
    correctly. Boilerplate tags (script/style/nav/header/footer/aside)
    are stripped before extraction. An <article> or <main> element is
    preferred over the full <body> when present, since it's more
    likely to be the page's actual content rather than navigation/ads.

    `<img>` elements with meaningful alt text become an inline
    "[Image: <alt text>] (<absolute url>)" marker (see
    _describe_image_tag), kept in document order right alongside the
    surrounding paragraphs - so a diagram sitting between two
    paragraphs stays between the same two paragraphs in the extracted
    text, preserving the image/text relationship instead of splitting
    it out separately. Decorative images (empty/missing alt) are
    skipped entirely. This only makes the image's alt text searchable,
    not the image's actual visual content - see module-level notes on
    why a vision-model description step was intentionally left out of
    this first pass.

    `<pre>` elements (literal code samples, e.g. `<pre><code>...` blocks
    common on documentation sites) become an inline Markdown fenced
    code block ("```\\n...\\n```"), preserving their original
    whitespace/newlines - so chunker.py's existing atomic-code-block
    handling picks them up automatically, no chunker changes needed.

    Returns:
        Document with the page's text (headings + prose + tables) as
        `content`, and metadata with source_type="web", url,
        document_id (the URL itself, so re-ingesting the same URL
        upserts instead of duplicating), and file_name (the page's
        <title>, falling back to the URL if the page has none).
    """
    for tag in soup.find_all(_HTML_TAGS_TO_STRIP):
        tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup

    parts: list[str] = []
    for element in root.find_all(_HTML_CONTENT_TAGS):
        # Skip anything nested inside a <table> we've already/will
        # handle via the "table" branch below, so a heading or
        # paragraph living inside a table cell isn't extracted twice.
        if element.name != "table" and element.find_parent("table") is not None:
            continue

        if element.name == "table":
            table_rows = _html_table_to_rows(element)
            if len(table_rows) > 1:
                parts.append(_table_to_markdown(table_rows))
            continue

        if element.name == "img":
            image_marker = _describe_image_tag(element, url)
            if image_marker:
                parts.append(image_marker)
            continue

        if element.name == "pre":
            # get_text() with NO separator/strip collapsing (unlike the
            # generic branch below) - a <pre> block's whitespace/newlines
            # are significant (indentation, line breaks), so collapsing
            # them to single spaces would destroy the code's structure.
            code_text = element.get_text().strip("\n")
            if not code_text.strip():
                continue
            parts.append(f"```\n{code_text}\n```")
            continue

        text = element.get_text(" ", strip=True)
        if not text:
            continue

        heading_match = _HTML_HEADING_TAG_RE.match(element.name)
        if heading_match:
            parts.append(f"{'#' * int(heading_match.group(1))} {text}")
        elif element.name == "li":
            parts.append(f"- {text}")
        else:
            parts.append(text)

    content = "\n\n".join(parts)

    title = soup.title.get_text(strip=True) if soup.title else ""

    metadata = {
        "source_type": "web",
        "url": url,
        "file_name": title or url,
        "document_id": url,
    }

    logger.info("Loaded website '%s' (%d characters)", url, len(content))
    return Document(content=content, metadata=metadata)


def load_website_document(url: str, timeout: int = 10, render_js: bool = False) -> Document:
    """
    Fetch a single web page and return it as ONE Document. See
    _build_website_document() for the extraction rules. Use
    crawl_website() instead if you want to discover and load an
    entire documentation site from one seed URL.

    Args:
        url: The web page URL to fetch.
        timeout: Request timeout in seconds.
        render_js: If True, fetch via a headless browser
            (_fetch_html_with_browser) instead of a plain HTTP request,
            so JavaScript single-page-app sites (whose real content
            only exists after JS runs) yield actual content instead of
            an empty page. Slower - only set this for sites that
            genuinely need it.

    Raises:
        requests.exceptions.RequestException: if the page can't be
            fetched (network error, timeout, non-2xx status, etc.) and
            render_js is False.
        Exception: whatever Playwright raises on navigation failure/
            timeout if render_js is True.
    """
    soup = _fetch_html_with_browser(url, timeout) if render_js else _fetch_html(url, timeout)
    return _build_website_document(url, soup)


def _extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """
    Collect every <a href> on a page as absolute, fragment-stripped
    URLs (so "/page#section" and "/page" are treated as the same
    page). Reads the FULL soup (not content-stripped), since a docs
    site's nav/sidebar - which load_website_document deliberately
    excludes from page CONTENT - is usually where the list of other
    pages to crawl actually lives.
    """
    links = []
    for anchor in soup.find_all("a", href=True):
        absolute_url = urljoin(base_url, anchor["href"])
        links.append(urldefrag(absolute_url).url)
    return links


def _is_in_crawl_scope(url: str, scope_netloc: str, scope_path_prefix: str) -> bool:
    """
    True if `url` is on the same host as the seed URL AND its path is
    the seed's path or nested under it (e.g. seed path "/docs/foo"
    matches "/docs/foo" and "/docs/foo/bar", but not "/docs/foobar" or
    a different host) - this is what keeps crawl_website() scoped to
    one documentation site instead of the whole domain.
    """
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or parts.netloc != scope_netloc:
        return False
    path = parts.path.rstrip("/")
    return path == scope_path_prefix or path.startswith(scope_path_prefix + "/")


_IBM_DOCS_NETLOC = "www.ibm.com"
_IBM_DOCS_PATH_PREFIX = "/docs/en/"


def _is_ibm_docs_url(url: str) -> bool:
    """True if `url` is a page under IBM's documentation platform (ibm.com/docs/en/...)."""
    parts = urlparse(url)
    return parts.netloc == _IBM_DOCS_NETLOC and parts.path.startswith(_IBM_DOCS_PATH_PREFIX)


def _ibm_docs_product_path(url: str) -> str:
    """
    Extract the product path IBM Docs' own API uses to identify a
    site, e.g. "targetprocess/tp-dev-hub/saas" from
    "https://www.ibm.com/docs/en/targetprocess/tp-dev-hub/saas"
    (query string/fragment, if any, are ignored).
    """
    return urlparse(url).path[len(_IBM_DOCS_PATH_PREFIX) :].rstrip("/")


def _fetch_ibm_docs_toc(product_path: str, timeout: int) -> dict:
    """
    Fetch an IBM Docs site's full table of contents as a nested JSON
    tree, via the same internal API IBM Docs' own frontend calls to
    render its sidebar (see crawl_ibm_docs() docstring for why this
    exists). Raises requests.exceptions.RequestException on failure.
    """
    url = f"https://www.ibm.com/docs/api/v1/toc/{product_path}?lang=en"
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAGIngestBot/1.0)"},
    )
    response.raise_for_status()
    return response.json()


def _flatten_ibm_docs_topics(node: dict) -> list[dict]:
    """
    Depth-first flatten of one _fetch_ibm_docs_toc() tree node (and
    all its descendants) into a flat list of {"topicId", "label"}
    dicts, in table-of-contents order. A node without its own
    "topicId" (a pure grouping header with no content page of its
    own) contributes nothing itself but its children still are - so
    this only ever returns nodes that actually have fetchable content.
    """
    topics: list[dict] = []
    if node.get("topicId"):
        topics.append({"topicId": node["topicId"], "label": node.get("label")})
    for child in node.get("topics", []):
        topics.extend(_flatten_ibm_docs_topics(child))
    return topics


def _fetch_ibm_docs_topic_html(product_path: str, topic_id: str, timeout: int) -> str:
    """
    Fetch one topic's content as a clean HTML fragment
    (<main><article>...</article></main>, no <head>/nav/sidebar) via
    the same internal API IBM Docs' own frontend calls when a user
    navigates to a topic. Raises
    requests.exceptions.RequestException on failure.
    """
    url = (
        f"https://www.ibm.com/docs/api/v1/content/{product_path}"
        f"?topic={topic_id}&parsebody=true&lang=en"
    )
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAGIngestBot/1.0)"},
    )
    response.raise_for_status()
    return response.text


def crawl_ibm_docs(seed_url: str, timeout: int = 10) -> list[Document]:
    """
    Crawl an IBM Docs product/documentation site (any page under
    ibm.com/docs/en/...) using IBM's own public TOC + content REST
    API, instead of rendering pages in a browser.

    Why this exists: IBM Docs pages are a client-side-routed
    single-page app whose raw HTML is an empty shell, and even a
    headless browser (_fetch_html_with_browser / render_js=True)
    can't reliably render a specific topic's content on a direct/cold
    page load - the app's own router only hydrates the generic
    hub/landing view for such visits, no matter how long you wait for
    the network to go idle. IBM Docs' own frontend sidesteps this by
    calling two internal APIs instead of depending on a fresh page
    render: /docs/api/v1/toc/{product_path} (the whole site's table of
    contents, as a nested JSON tree - see _fetch_ibm_docs_toc) and
    /docs/api/v1/content/{product_path}?topic={topicId} (that topic's
    actual content, as a clean HTML fragment - see
    _fetch_ibm_docs_topic_html). Calling these two endpoints directly
    is faster and far more reliable than simulating clicks in a
    browser, and needs no JavaScript execution/browser at all.

    Args:
        seed_url: Any page under an IBM Docs product, e.g.
            "https://www.ibm.com/docs/en/targetprocess/tp-dev-hub/saas".
            Only its https://www.ibm.com/docs/en/{product_path}
            portion is used - query string/fragment are ignored, so
            it doesn't matter which specific topic's URL is given.
        timeout: Per-request timeout in seconds.

    Returns:
        One Document per topic in the site's table of contents (same
        shape as load_website_document()), in table-of-contents order.
        A topic whose content fails to fetch is skipped with a
        warning rather than aborting the rest of the crawl, same as
        an unreachable page in crawl_website().
    """
    product_path = _ibm_docs_product_path(seed_url)
    toc = _fetch_ibm_docs_toc(product_path, timeout)
    topics = _flatten_ibm_docs_topics(toc.get("toc", {}))

    documents: list[Document] = []
    for topic in topics:
        topic_id = topic["topicId"]
        page_url = f"https://www.ibm.com/docs/en/{product_path}?topic={topic_id}"

        try:
            html = _fetch_ibm_docs_topic_html(product_path, topic_id, timeout)
        except requests.exceptions.RequestException as error:
            logger.warning("Skipping IBM Docs topic '%s' (%s): %s", topic_id, page_url, error)
            continue

        soup = BeautifulSoup(html, "html.parser")
        document = _build_website_document(page_url, soup)
        # The content fragment has no <title>, so _build_website_document()
        # falls back to the URL for file_name - use the TOC's own label
        # instead, since it's the actual human-readable topic name.
        if topic.get("label") and document.metadata["file_name"] == page_url:
            document.metadata["file_name"] = topic["label"]
        documents.append(document)

    logger.info(
        "Crawled %d IBM Docs topic(s) from '%s' via TOC+content API",
        len(documents),
        product_path,
    )
    return documents


def crawl_website(
    seed_url: str, max_pages: int = 30, timeout: int = 10, render_js: bool = False
) -> list[Document]:
    """
    Discover and load every page of a documentation site reachable
    from `seed_url` by following in-scope links, breadth-first,
    instead of requiring every page's exact URL listed individually.

    IBM Docs sites (ibm.com/docs/en/...) are special-cased and
    delegated to crawl_ibm_docs() instead, regardless of render_js -
    see its docstring for why (their public TOC+content API is more
    reliable than rendering pages, even in a real browser).

    "In-scope" means: same host as seed_url, and a path equal to or
    nested under seed_url's own path (see _is_in_crawl_scope) - so
    this deliberately does NOT crawl an entire domain, only the
    sub-site the seed URL belongs to. Crawling stops once max_pages
    have been loaded, as a hard cap against runaway crawls. A page
    that fails to fetch (network error, timeout, non-2xx status, or -
    when render_js=True - a browser navigation failure) is skipped
    with a warning rather than aborting the whole crawl, same as an
    unreachable URL in data/urls.txt.

    Args:
        seed_url: Starting page. Defines the crawl's host+path scope.
        max_pages: Maximum number of pages to fetch and return.
        timeout: Per-request timeout in seconds.
        render_js: If True, fetch every page via a headless browser
            (_fetch_html_with_browser) instead of a plain HTTP request
            - needed for JavaScript single-page-app documentation
            sites, whose raw HTML has no server-rendered content OR
            links to discover other pages from. Slower per page since
            it spins up a browser page navigation instead of a plain
            HTTP request - only set this for sites that need it.

    Returns:
        One Document per successfully-fetched in-scope page (same
        shape as load_website_document()), in the order discovered.
    """
    if _is_ibm_docs_url(seed_url):
        return crawl_ibm_docs(seed_url, timeout=timeout)

    seed_parts = urlparse(seed_url)
    scope_netloc = seed_parts.netloc
    scope_path_prefix = seed_parts.path.rstrip("/")

    seed_url = urldefrag(seed_url).url
    queue: list[str] = [seed_url]
    visited: set[str] = set()
    documents: list[Document] = []

    while queue and len(documents) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            soup = _fetch_html_with_browser(url, timeout) if render_js else _fetch_html(url, timeout)
        except Exception as error:
            logger.warning("Skipping unreachable page '%s' during crawl: %s", url, error)
            continue

        for link in _extract_links(soup, url):
            if link not in visited and _is_in_crawl_scope(link, scope_netloc, scope_path_prefix):
                queue.append(link)

        documents.append(_build_website_document(url, soup))

    logger.info(
        "Crawled %d page(s) from seed '%s' (scope: %s%s)",
        len(documents),
        seed_url,
        scope_netloc,
        scope_path_prefix or "/",
    )
    return documents


_GITHUB_API_BASE = "https://api.github.com"

# Repo ingestion caps: a repository can have thousands of files and
# some (generated bundles, datasets, binaries) are huge - without a
# limit, ingesting "any GitHub repo" the user names could take an
# unbounded amount of time/memory. These defaults are generous enough
# for a typical project's actual source/docs while still bounding
# worst-case cost; callers can override either for a specific repo.
_DEFAULT_MAX_FILES = 300
_DEFAULT_MAX_FILE_BYTES = 200_000

# Directories whose contents are virtually never useful for answering
# questions about a codebase (dependencies, build output, caches) and
# can easily dwarf a repo's actual source in file count.
_EXCLUDED_DIR_NAMES = {
    "node_modules", ".git", "dist", "build", "vendor", "venv", ".venv",
    "env", "__pycache__", "target", ".idea", ".vscode", "coverage",
    ".next", ".pytest_cache", "bin", "obj", "out", ".mypy_cache",
    ".tox", "egg-info",
}

# Machine-generated lockfiles: large, low-signal, and not something a
# user would ever ask a question about.
_EXCLUDED_FILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Cargo.lock", "Gemfile.lock", "composer.lock",
}

# Extension -> language, used for metadata only (no AST/code-aware
# chunking yet - see app.chunking.chunker's module docstring; that is
# a later, separate step). Kept close to the project's planned
# language list (README/Markdown, Python, JavaScript, TypeScript,
# Java, Go, JSON, YAML, config files) plus a few more common ones,
# rather than trying to be an exhaustive list of every language.
_LANGUAGE_BY_EXTENSION = {
    ".md": "markdown", ".markdown": "markdown", ".rst": "text", ".txt": "text",
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".json": "json",
    ".yml": "yaml", ".yaml": "yaml",
    ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".rs": "rust",
    ".kt": "kotlin",
    ".swift": "swift",
    ".sh": "shell",
    ".sql": "sql",
}

# Config-style files with a well-known name but no (or a misleading)
# extension.
_LANGUAGE_BY_FILENAME = {
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
    "README": "text",
    "LICENSE": "text",
}


def _infer_language(path: str) -> str | None:
    """
    Best-effort language name for a repo file path, used only as
    metadata (see _LANGUAGE_BY_EXTENSION's docstring). None means
    "not a recognized/ingestible source or doc file" - callers use
    that to skip the file entirely.
    """
    name = Path(path).name
    if name in _LANGUAGE_BY_FILENAME:
        return _LANGUAGE_BY_FILENAME[name]
    return _LANGUAGE_BY_EXTENSION.get(Path(path).suffix.lower())


def _is_ingestible_path(path: str, size: int, max_file_bytes: int) -> bool:
    """
    True if a GitHub tree entry is worth downloading and ingesting:
    not inside an excluded directory (_EXCLUDED_DIR_NAMES), not a
    known lockfile (_EXCLUDED_FILE_NAMES), within max_file_bytes (the
    GitHub tree API already reports each blob's size, so this can be
    checked WITHOUT downloading the file first), and of a recognized
    language/doc type (_infer_language).
    """
    if any(part in _EXCLUDED_DIR_NAMES for part in Path(path).parts):
        return False
    if Path(path).name in _EXCLUDED_FILE_NAMES:
        return False
    if size > max_file_bytes:
        return False
    return _infer_language(path) is not None


def _parse_github_repo_url(repo_url: str) -> tuple[str, str, str | None]:
    """
    Parse a GitHub repository reference into (owner, repo, branch).

    Accepts, so the user can paste whatever form they have handy
    rather than needing to know a specific format:
      - "owner/repo"
      - "https://github.com/owner/repo"
      - "https://github.com/owner/repo.git"
      - "https://github.com/owner/repo/tree/<branch>"

    branch is None if the reference doesn't specify one - the caller
    (load_github_repository) then resolves the repo's actual default
    branch via the GitHub API, rather than hardcoding a guess like
    "main" that would break for repos still using "master" (or any
    other default).

    Raises:
        ValueError: if an owner/repo pair can't be parsed out.
    """
    text = repo_url.strip()
    text = re.sub(r"^https?://github\.com/", "", text)
    if text.endswith(".git"):
        text = text[: -len(".git")]
    parts = [part for part in text.strip("/").split("/") if part]

    if len(parts) < 2:
        raise ValueError(f"Could not parse a GitHub owner/repo from: {repo_url!r}")

    owner, repo = parts[0], parts[1]
    branch = "/".join(parts[3:]) if len(parts) >= 4 and parts[2] == "tree" else None
    return owner, repo, branch


def _github_headers() -> dict:
    """
    Auth header for GitHub API requests, if a GITHUB_TOKEN environment
    variable is set. Unauthenticated requests to api.github.com are
    capped at 60/hour, which a repo with many files can exhaust just
    listing its tree; an optional token raises that to 5000/hour.
    Public repos still work fine with no token at all - this is only
    ever a rate-limit improvement, never a requirement.
    """
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def load_github_repository(
    repo_url: str,
    branch: str | None = None,
    max_files: int = _DEFAULT_MAX_FILES,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    timeout: int = 30,
) -> list[Document]:
    """
    Load every ingestible file from a public GitHub repository into
    one Document per file - the GitHub counterpart to
    load_markdown_document/load_pdf_document/load_docx_document, but
    for a whole repository at once rather than one local file.

    Unlike this project's other sources, a GitHub repo is not
    pre-listed under raw_dir or urls.txt: the whole point is that ANY
    repo the user names can be ingested on demand (see
    app.ingestion.ingest.ingest_github_repo, cli.py's --ingest-github
    flag, and app_ui.py's sidebar), so this function takes the repo
    reference directly as an argument instead.

    How it works:
      1. Parse the repo reference into (owner, repo, branch) - see
         _parse_github_repo_url(). If no branch was specified, resolve
         the repo's actual default branch via GET /repos/{owner}/{repo}
         rather than guessing "main"/"master".
      2. List the ENTIRE file tree in one call via GitHub's Git Trees
         API (GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1)
         instead of a real `git clone` - this avoids needing the git
         binary or downloading the repo's full history, and the tree
         response already includes each blob's size in bytes, which is
         needed for step 3 anyway.
      3. Filter the tree down to files worth ingesting (see
         _is_ingestible_path: skips dependency/build/cache directories,
         lockfiles, oversized files, and unrecognized extensions), then
         cap the result at max_files (by path order) so a huge
         monorepo can't make ingestion unbounded.
      4. Download each remaining file's raw content directly from
         raw.githubusercontent.com (a CDN, not subject to the same
         strict api.github.com rate limit) and decode it as UTF-8. A
         file that fails to download or isn't valid UTF-8 text (most
         likely a binary file _infer_language's extension check let
         through, e.g. a misnamed asset) is skipped with a warning
         rather than failing the whole repo's ingestion - the same
         "one bad source can't break everything else" pattern used by
         every other loader in this module.

    Each returned Document's metadata includes: source_type="code",
    repository ("owner/repo"), branch, commit_hash (the tree's
    resolved sha), file_path, file_name, language (see
    _infer_language), and document_id ("owner/repo:path" - stable
    across re-ingestion, so re-ingesting the same repo/branch upserts
    the same chunks instead of duplicating them, exactly like every
    other source's document_id).

    Args:
        repo_url: Any of the forms _parse_github_repo_url() accepts.
        branch: Branch/ref to ingest. Overrides any branch parsed out
            of repo_url. None = the repo's own default branch.
        max_files: Cap on how many files to ingest (by path order),
            after filtering. A warning is logged if this cap is hit.
        max_file_bytes: Skip any single file larger than this.
        timeout: Per-request timeout in seconds.

    Raises:
        ValueError: if repo_url doesn't parse into an owner/repo.
        requests.exceptions.RequestException: if the GitHub API itself
            is unreachable or returns an error (e.g. repo not found,
            rate-limited) - this is NOT caught here, since without a
            file tree there is nothing partial to ingest, unlike a
            single unreachable file within an otherwise-listable repo.
    """
    owner, repo, url_branch = _parse_github_repo_url(repo_url)
    branch = branch or url_branch
    headers = _github_headers()

    if branch is None:
        repo_response = requests.get(f"{_GITHUB_API_BASE}/repos/{owner}/{repo}", headers=headers, timeout=timeout)
        repo_response.raise_for_status()
        branch = repo_response.json()["default_branch"]

    tree_response = requests.get(
        f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
        headers=headers,
        timeout=timeout,
    )
    tree_response.raise_for_status()
    tree_data = tree_response.json()
    commit_hash = tree_data.get("sha", branch)

    if tree_data.get("truncated"):
        logger.warning(
            "GitHub tree listing for %s/%s@%s was truncated by the API (repo too large) - "
            "not every file could be considered for ingestion.",
            owner, repo, branch,
        )

    candidates = [
        entry
        for entry in tree_data.get("tree", [])
        if entry.get("type") == "blob" and _is_ingestible_path(entry["path"], entry.get("size", 0), max_file_bytes)
    ]

    if len(candidates) > max_files:
        logger.warning(
            "GitHub repo %s/%s has %d ingestible file(s); only the first %d (by path order) "
            "will be ingested. Pass a higher max_files to ingest more.",
            owner, repo, len(candidates), max_files,
        )
        candidates = candidates[:max_files]

    documents: list[Document] = []
    for entry in candidates:
        path = entry["path"]
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        try:
            file_response = requests.get(raw_url, timeout=timeout)
            file_response.raise_for_status()
            content = file_response.content.decode("utf-8")
        except (requests.exceptions.RequestException, UnicodeDecodeError) as error:
            logger.warning("Skipping unreadable GitHub file '%s': %s", path, error)
            continue

        metadata = {
            "source_type": "code",
            "repository": f"{owner}/{repo}",
            "branch": branch,
            "commit_hash": commit_hash,
            "file_path": path,
            "file_name": Path(path).name,
            "language": _infer_language(path),
            "document_id": f"{owner}/{repo}:{path}",
        }
        documents.append(Document(content=content, metadata=metadata))

    logger.info(
        "Loaded %d file(s) from GitHub repo %s/%s@%s",
        len(documents), owner, repo, branch,
    )
    return documents


if __name__ == "__main__":
    doc = load_markdown_document("data/raw/README.md")
    print(f"Loaded {doc.metadata['file_name']}")
    print(f"Character count: {len(doc.content)}")
    print("--- First 200 characters ---")
    print(doc.content[:200])

    pdf_doc = load_pdf_document("data/raw/rag_ingestion_sample.pdf")
    print(f"\nLoaded {pdf_doc.metadata['file_name']} ({pdf_doc.metadata['total_pages']} page(s))")
    print(f"Character count: {len(pdf_doc.content)}")
    print("--- First 200 characters ---")
    print(pdf_doc.content[:200])


"""
Multi-source ingestion orchestrator for the RAG pipeline.

Responsibility: discover source files under a directory, load each one
into Document(s) using whichever loader matches its file type, then
chunk, embed, and store all of them into one unified index. This is
the single place that ties every source type together - no
source-type-specific parsing or chunking logic lives here.

Design:
- _LOADERS_BY_EXTENSION maps a file extension to the loader function
  that knows how to read it. Adding a new source type means adding one
  entry here, pointing at a loader that returns a Document or
  list[Document] - chunk_with_parent_child(), embed_chunks(), and
  add_embedded_chunks() are already source-agnostic (they only look at
  Document.content/metadata), so nothing else needs to change.
- A loader may return a single Document (e.g. one Markdown file, one
  PDF, or one DOCX - all now single, continuous Documents) or a list
  of Documents, for any future source type that genuinely needs
  per-unit splitting. Both shapes are normalized into a flat list
  here before chunking, so the rest of the pipeline never needs to
  know which case it's in.
- Re-running ingestion is safe/idempotent: add_embedded_chunks()
  upserts by a stable id derived from document_id (+ page_number, for
  multi-page sources) + chunk_index, so re-ingesting the same files
  overwrites the same chunks instead of duplicating them. This holds
  for URLs too, since load_website_document() sets document_id to the
  URL itself.
- Websites aren't files under raw_dir (there's no path to scan by
  extension), so they're listed instead, one per line, in URLS_FILE
  (data/urls.txt), with an optional prefix selecting how each line is
  fetched:
    - "https://..."          -> single page, load_website_document()
    - "crawl:https://..."    -> documentation-site seed URL, expanded
                                 by crawl_website() into every in-scope
                                 page reachable from it (see loader.py's
                                 crawl_website() docstring for what
                                 "in-scope" means) - the recommended
                                 way to index a whole docs site, rather
                                 than blindly scraping every page or
                                 hand-listing each one. IBM Docs sites
                                 (ibm.com/docs/en/...) are auto-detected
                                 here and crawled via IBM's own
                                 TOC+content REST API instead (see
                                 crawl_ibm_docs()), since their pages
                                 are a single-page app that even a
                                 headless browser can't reliably render
                                 topic-specific content for.
    - "js:https://..."       -> single page, rendered in a headless
                                 browser first (render_js=True) - for
                                 JavaScript single-page-app pages whose
                                 raw HTML has no real content.
    - "crawl-js:https://..." -> same as "crawl:" but every page is
                                 rendered in a headless browser too -
                                 for JS single-page-app documentation
                                 sites whose raw HTML has neither
                                 content nor discoverable links without
                                 running its JavaScript (unless the
                                 site is an IBM Docs site, in which
                                 case "crawl:" auto-detection above
                                 already handles it more reliably).
  Either way, the result is folded into the same flat Document list as
  every file-based source before chunking. A URL/page that fails to
  fetch (network error, timeout, 404, etc., or - for "js:"/"crawl-js:"
  - a browser navigation failure) is skipped with a warning, same as
  an unsupported file extension, so one dead link can't break
  ingestion of everything else.
"""

import logging
from pathlib import Path

import requests

from app.chunking.parent_child import chunk_with_parent_child
from app.embeddings.embedder import EmbeddedChunk, embed_chunks, embed_texts
from app.embeddings.image_embedder import download_image, embed_images
from app.generation.llm_generator import generate_vision_text
from app.ingestion.loader import (
    Document,
    crawl_website,
    extract_image_records,
    load_docx_document,
    load_github_repository,
    load_markdown_document,
    load_pdf_document,
    load_website_document,
)
from app.ocr.image_classifier import CODE, classify_extracted_text
from app.ocr.image_ocr import combine_ocr_and_vision, extract_text_from_image
from app.ocr.prompts import CODE_EXTRACTION_PROMPT
from app.vectorstore.store import (
    _iter_collections,
    add_embedded_chunks,
    add_image_chunks,
    count,
    image_count,
)

logger = logging.getLogger(__name__)

DATA_RAW_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "raw")
URLS_FILE = str(Path(__file__).resolve().parent.parent.parent / "data" / "urls.txt")

_LOADERS_BY_EXTENSION = {
    ".md": load_markdown_document,
    ".pdf": load_pdf_document,
    ".docx": load_docx_document,
}


def _load_all_documents(raw_dir: str) -> list[Document]:
    """
    Discover every supported file directly under raw_dir and load each
    one into Document(s) via the matching loader, flattened into a
    single list. Files with an unsupported extension are skipped (with
    a warning) rather than raising, so one unrelated file can't break
    ingestion of everything else.
    """
    documents: list[Document] = []

    for path in sorted(Path(raw_dir).iterdir()):
        if not path.is_file():
            continue

        loader = _LOADERS_BY_EXTENSION.get(path.suffix.lower())
        if loader is None:
            logger.warning("Skipping unsupported file type: %s", path.name)
            continue

        result = loader(str(path))
        if isinstance(result, list):
            documents.extend(result)
        else:
            documents.append(result)

    return documents


def _read_urls(urls_file: str) -> list[str]:
    """
    Read one URL per line from urls_file. Blank lines and lines
    starting with "#" (comments) are ignored. Returns an empty list if
    the file doesn't exist, since having no URLs configured is a
    normal, expected state, not an error.
    """
    path = Path(urls_file)
    if not path.exists():
        return []

    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            urls.append(stripped)
    return urls


def _load_all_websites(urls_file: str) -> list[Document]:
    """
    Load every URL/seed listed in urls_file into Document(s), honoring
    the "crawl:"/"js:"/"crawl-js:" prefixes described in the module
    docstring. A URL/page that fails to fetch is skipped (with a
    warning) rather than raising, so one dead/unreachable link can't
    break ingestion of every other source. The render_js=True paths
    can raise non-requests exceptions (Playwright navigation errors),
    so those are caught with a broad Exception rather than
    requests.exceptions.RequestException.
    """
    documents: list[Document] = []

    for entry in _read_urls(urls_file):
        if entry.startswith("crawl-js:"):
            seed_url = entry[len("crawl-js:") :].strip()
            documents.extend(crawl_website(seed_url, render_js=True))
            continue

        if entry.startswith("crawl:"):
            seed_url = entry[len("crawl:") :].strip()
            documents.extend(crawl_website(seed_url))
            continue

        if entry.startswith("js:"):
            url = entry[len("js:") :].strip()
            try:
                documents.append(load_website_document(url, render_js=True))
            except Exception as error:
                logger.warning("Skipping unreachable URL '%s': %s", url, error)
            continue

        try:
            documents.append(load_website_document(entry))
        except requests.exceptions.RequestException as error:
            logger.warning("Skipping unreachable URL '%s': %s", entry, error)

    return documents


def _already_extracted_image_urls() -> set[str]:
    """
    Return the set of image_urls that already have a text chunk stored
    (content_type in {"image_code", "image_ocr"}) across every text
    KB. Used by _ingest_images() to skip re-downloading, re-OCR'ing,
    and (crucially) re-running the slow llava vision pass on images
    whose text is already extracted from a prior ingest run.

    Called once per _ingest_images() invocation (i.e. once per
    document). Cost is one metadata-only get() per text KB with no
    similarity search - fast (milliseconds) even with thousands of
    chunks, and orders of magnitude cheaper than a single wasted llava
    call.
    """
    urls: set[str] = set()
    for collection in _iter_collections():
        for content_type in ("image_code", "image_ocr"):
            result = collection.get(
                where={"content_type": content_type},
                include=["metadatas"],
            )
            for metadata in result["metadatas"]:
                url = metadata.get("image_url")
                if url:
                    urls.add(url)
    return urls


def _ingest_images(document: Document) -> int:
    """
    Extract, download (once), embed (via CLIP), and extract text from
    every image found in a single Document's content (see
    app.ingestion.loader.extract_image_records), then store:
      1. the CLIP embedding, in the image collection (as before), for
         image similarity search/display; and
      2. IF any meaningful text was read out of the image's pixels,
         that text as an ORDINARY extra text chunk in the normal text
         collection, so it is embedded with the regular
         all-MiniLM-L6-v2 text model and becomes retrievable/promptable
         exactly like any other chunk (never as a raw image embedding -
         a text embedding of the ACTUAL extracted content is what
         retrieval/generation need). This closes the "the code exists
         on the page but only as a picture" gap.

    Text extraction is a two-stage pipeline per image:
      a. Traditional OCR (see app.ocr.image_ocr.extract_text_from_image)
         always runs first - cheap, and usually good enough for plain
         screenshots/diagrams with a little text.
      b. The OCR'd text is then classified (see
         app.ocr.image_classifier.classify_extracted_text) as CODE or
         OTHER, based on whether it looks like source code. ONLY for
         images classified CODE, a local vision model (llava) is ALSO
         run, using a strict, transcription-only extraction prompt
         (app.ocr.prompts.CODE_EXTRACTION_PROMPT - deliberately not a
         generic "describe this image" question, which would invite the
         model to explain/infer/"correct" the code instead of
         transcribing it exactly). The two outputs are then combined
         (see app.ocr.image_ocr.combine_ocr_and_vision) into one final
         block of extracted content, rather than trusting either engine
         alone.
    Gating the (expensive, CPU-bound, tens-of-seconds-per-image) vision
    call to CODE-classified images only keeps ingestion runtime roughly
    proportional to how much actual code exists in the corpus, instead
    of multiplying every single image's cost - while still directly
    targeting the exact problem this pipeline exists to solve (code
    rendered only as a screenshot). A vision call that fails/times out
    is logged and skipped (falls back to OCR-only text for that image)
    rather than failing the whole document's ingestion.

    A no-op (returns 0) for any Document with no image markers, which
    today is every non-web Document - so calling this for every source
    type unconditionally is safe/cheap rather than needing a
    source_type check here.

    Each image URL is downloaded only ONCE (the same decoded image is
    reused for CLIP embedding, OCR, AND vision extraction, rather than
    fetching the same URL repeatedly for different concerns). An image
    that fails to download/decode is silently skipped (already logged
    by download_image) rather than failing the whole document's
    ingestion, matching the "one bad source can't break everything
    else" pattern used throughout this module. Duplicate image URLs on
    the same page (e.g. an icon reused inline and in a gallery) are
    deduped up front (keeping the last occurrence) since both the
    image chunk and the extracted-text chunk below are id'd by
    image_url.

    Returns:
        How many images were actually embedded and stored.
    """
    records = extract_image_records(document)
    if not records:
        return 0

    records_by_url = {record["image_url"]: record for record in records}

    # Skip images that already have a text chunk (image_code/image_ocr)
    # from a prior ingest run. This makes a re-crawl idempotent AND
    # cheap: without this check every re-ingest would re-download every
    # image, re-OCR it, re-classify it, and (worst) re-run llava on
    # every code screenshot from scratch - the same multi-hour vision
    # pass whose result is already in the store. New/decorative images
    # (no text chunk yet) still run the full pipeline.
    already_done = _already_extracted_image_urls()
    skipped_count = 0
    if already_done:
        pending_records = {
            url: record for url, record in records_by_url.items() if url not in already_done
        }
        skipped_count = len(records_by_url) - len(pending_records)
        records_by_url = pending_records
    if not records_by_url:
        if skipped_count:
            logger.info(
                "All %d image(s) on this page were already extracted in a prior run - skipping",
                skipped_count,
            )
        return 0

    images_by_url = {}
    for image_url in records_by_url:
        image = download_image(image_url)
        if image is not None:
            images_by_url[image_url] = image

    if not images_by_url:
        return 0

    valid_urls = list(images_by_url.keys())
    embeddings = embed_images([images_by_url[url] for url in valid_urls])

    image_chunks = [
        {
            "content": records_by_url[url]["alt_text"],
            "embedding": embedding,
            "metadata": records_by_url[url],
        }
        for url, embedding in zip(valid_urls, embeddings)
    ]
    add_image_chunks(image_chunks)

    extracted_chunks: list[EmbeddedChunk] = []
    code_image_count = 0
    for url in valid_urls:
        image = images_by_url[url]
        ocr_text = extract_text_from_image(image)
        image_type = classify_extracted_text(ocr_text)

        final_text = ocr_text
        if image_type == CODE:
            code_image_count += 1
            try:
                vision_text = generate_vision_text(image, CODE_EXTRACTION_PROMPT)
            except Exception as error:
                logger.warning("Vision extraction failed for '%s' (keeping OCR-only text): %s", url, error)
                vision_text = ""
            final_text = combine_ocr_and_vision(ocr_text, vision_text)

        if not final_text:
            continue

        record = records_by_url[url]
        content_type = "image_code" if image_type == CODE else "image_ocr"
        extracted_chunks.append(
            EmbeddedChunk(
                content=f"[Text extracted from image ({image_type}): {record['alt_text']}]\n{final_text}",
                embedding=[],
                metadata={
                    "document_id": url,
                    "chunk_index": 0,
                    "file_name": record.get("page_title"),
                    "url": record.get("page_url"),
                    "section": record.get("alt_text"),
                    "content_type": content_type,
                    "image_url": url,
                    "source_type": "web",
                },
            )
        )

    if extracted_chunks:
        vectors = embed_texts([chunk.content for chunk in extracted_chunks])
        for chunk, vector in zip(extracted_chunks, vectors):
            chunk.embedding = vector
        add_embedded_chunks(extracted_chunks)
        logger.info(
            "Extracted text from %d/%d image(s) (%d classified as code, vision-cross-checked; "
            "%d skipped as already-extracted from prior run) and stored as additional text chunk(s)",
            len(extracted_chunks),
            len(valid_urls),
            code_image_count,
            skipped_count,
        )
    elif skipped_count:
        logger.info(
            "Skipped %d image(s) on this page as already-extracted from a prior run "
            "(no new text chunks to add)",
            skipped_count,
        )

    return len(image_chunks)


def ingest_all(
    raw_dir: str = DATA_RAW_DIR,
    urls_file: str = URLS_FILE,
    chunk_size: int = 150,
    chunk_overlap: int = 30,
) -> int:
    """
    Load every supported source file under raw_dir plus every URL
    listed in urls_file, then chunk, embed, and store all of them into
    the vector store. Any images found in a website Document (see
    app.ingestion.loader.extract_image_records) are additionally
    embedded with a separate multimodal (CLIP) model and stored in
    their own collection (see app.embeddings.image_embedder,
    app.vectorstore.store's rag_image_chunks collection) - a parallel
    path alongside, not a replacement for, ordinary text chunk
    storage.

    Args:
        raw_dir: Directory to scan for source files.
        urls_file: Path to a text file listing one URL per line (blank
            lines and "#" comments ignored). Missing file = no URLs.
        chunk_size: Passed through to chunk_with_parent_child for every source.
        chunk_overlap: Passed through to chunk_with_parent_child for every source.

    Returns:
        The total number of text chunks stored across all sources
        (images stored are reported separately in the log line, and
        available via app.vectorstore.store.image_count()).
    """
    documents = _load_all_documents(raw_dir) + _load_all_websites(urls_file)
    logger.info(
        "Discovered %d document(s)/page(s) across all sources (files in %s, URLs in %s)",
        len(documents),
        raw_dir,
        urls_file,
    )

    total_chunks = 0
    total_images = 0
    for document in documents:
        chunks = chunk_with_parent_child(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embedded_chunks = embed_chunks(chunks)
        add_embedded_chunks(embedded_chunks)
        total_chunks += len(chunks)
        total_images += _ingest_images(document)

    logger.info(
        "Ingestion complete: stored %d chunk(s) and %d image(s) total across %d "
        "document(s)/page(s). Collection now has %d chunk(s), %d image(s).",
        total_chunks,
        total_images,
        len(documents),
        count(),
        image_count(),
    )
    return total_chunks


def ingest_github_repo(
    repo_url: str,
    branch: str | None = None,
    chunk_size: int = 150,
    chunk_overlap: int = 30,
    max_files: int = 300,
) -> int:
    """
    Ingest a single GitHub repository on demand: any repo the caller
    names, not just ones pre-listed under raw_dir/urls_file the way
    ingest_all()'s sources are. This is what powers cli.py's
    --ingest-github flag and app_ui.py's sidebar "ingest a GitHub
    repo" input, so a user can point the platform at an arbitrary
    public repo at any time without editing a config file and
    re-running full ingestion.

    Loads every ingestible file in the repo (see
    app.ingestion.loader.load_github_repository - source, README/
    Markdown, JSON/YAML, etc., filtered by extension, size, and
    excluded directories/lockfiles), then chunks, embeds, and stores
    each file exactly like every other source's Document, so a
    GitHub-sourced chunk is retrievable and citable the same way as a
    PDF/DOCX/website chunk.

    Re-running this for the same repo (e.g. after it's been updated)
    is safe/idempotent for the same reason as every other source:
    each Document's document_id ("owner/repo:path", see
    load_github_repository) makes chunk ids stable, so re-ingesting
    overwrites the same chunks instead of duplicating them.

    Args:
        repo_url: A GitHub repo reference (see
            app.ingestion.loader._parse_github_repo_url for accepted
            forms, e.g. "https://github.com/owner/repo" or
            "owner/repo").
        branch: Branch to ingest. None = the repo's default branch.
        chunk_size: Passed through to chunk_with_parent_child.
        chunk_overlap: Passed through to chunk_with_parent_child.
        max_files: Cap on how many files to ingest from the repo.

    Returns:
        The number of text chunks stored.
    """
    documents = load_github_repository(repo_url, branch=branch, max_files=max_files)
    logger.info("Discovered %d file(s) in GitHub repo '%s'", len(documents), repo_url)

    total_chunks = 0
    for document in documents:
        chunks = chunk_with_parent_child(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embedded_chunks = embed_chunks(chunks)
        add_embedded_chunks(embedded_chunks)
        total_chunks += len(chunks)

    logger.info(
        "GitHub ingestion complete for '%s': stored %d chunk(s) across %d file(s). "
        "Collection now has %d chunk(s).",
        repo_url,
        total_chunks,
        len(documents),
        count(),
    )
    return total_chunks


if __name__ == "__main__":
    from app.logging_config import configure_logging

    configure_logging()

    stored = ingest_all()
    print(f"Ingested and stored {stored} chunk(s) from '{DATA_RAW_DIR}' and '{URLS_FILE}'.")
    print(f"Collection now has {count()} chunk(s) total.")

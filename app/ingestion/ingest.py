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
import os
from pathlib import Path

import requests
from PIL import Image

from app.chunking.parent_child import chunk_with_parent_child
from app.embeddings.audio_embedder import embed_audio_clips
from app.embeddings.embedder import EmbeddedChunk, embed_chunks, embed_texts
from app.embeddings.image_embedder import download_image, embed_images
from app.graph.code_graph import build_repository_graph, save_repository_graph
from app.ingestion.loader import (
    Document,
    _DEFAULT_FRAME_INTERVAL_SECONDS,
    _DEFAULT_MAX_FRAMES,
    _parse_github_repo_url,
    _sample_audio_clips,
    _sample_video_frames,
    _video_duration_seconds,
    crawl_website,
    extract_image_records,
    load_audio_document,
    load_docx_document,
    load_github_discussions,
    load_github_issues,
    load_github_pull_requests,
    load_github_repository,
    load_markdown_document,
    load_pdf_document,
    load_video_document,
    load_website_document,
)
from app.media import media_store
from app.ocr.frame_analysis import analyze_frame
from app.ocr.image_classifier import CODE
from app.vectorstore.store import (
    _iter_collections,
    add_audio_clip_chunks,
    add_embedded_chunks,
    add_image_chunks,
    audio_clip_count,
    count,
    image_count,
)

logger = logging.getLogger(__name__)

DATA_RAW_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "raw")
URLS_FILE = str(Path(__file__).resolve().parent.parent.parent / "data" / "urls.txt")

# Opt-in (default off): unlike video-frame CLIP embedding (always-on -
# see _ingest_video_frames), audio CLAP embedding is a brand-new heavy
# model that shouldn't silently start downloading a large checkpoint
# and adding per-file ingestion cost for every existing deployment the
# moment this ships. Read at import time, matching WHISPER_MODEL_SIZE's
# precedent in app.ingestion.loader.
_ENABLE_AUDIO_SIMILARITY_SEARCH = os.environ.get("ENABLE_AUDIO_SIMILARITY_SEARCH", "") == "1"

_LOADERS_BY_EXTENSION = {
    ".md": load_markdown_document,
    ".pdf": load_pdf_document,
    ".docx": load_docx_document,
    ".mp3": load_audio_document,
    ".wav": load_audio_document,
    ".m4a": load_audio_document,
    ".flac": load_audio_document,
    ".ogg": load_audio_document,
    ".mp4": load_video_document,
    ".mov": load_video_document,
    ".mkv": load_video_document,
    ".avi": load_video_document,
    ".webm": load_video_document,
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

    Text extraction per image is delegated to
    app.ocr.frame_analysis.analyze_frame() - the same OCR-first,
    vision-only-for-CODE-classified-images pipeline used to be inline
    here, extracted into its own module so app.ingestion.loader's
    video ingestion (sampled frames have no Document/webpage/CLIP
    context to hang this logic off of) can reuse the identical gate
    instead of a second, independently-drifting copy. See that
    module's docstring for exactly why each stage exists (OCR-first,
    the CODE-only vision gate, combining both engines' output).

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
            # "origin"/"source_kb" written explicitly (not left
            # implicit) so metadata.get("origin", "web_image") is a
            # safe read pattern for every consumer of the image
            # collection, now that video frames (see
            # _ingest_video_frames) can also land in it with
            # origin="video_frame". source_kb is what
            # app.retrieval.retriever.retrieve_images filters by so a
            # question scoped to one KB tab doesn't match images from
            # a completely different source (see that function's
            # docstring for why this matters).
            "metadata": {**records_by_url[url], "origin": "web_image", "source_kb": "web"},
        }
        for url, embedding in zip(valid_urls, embeddings)
    ]
    add_image_chunks(image_chunks)

    extracted_chunks: list[EmbeddedChunk] = []
    code_image_count = 0
    for url in valid_urls:
        image = images_by_url[url]
        analysis = analyze_frame(image)
        final_text = analysis.text
        if analysis.image_type == CODE:
            code_image_count += 1

        if not final_text:
            continue

        record = records_by_url[url]
        content_type = "image_code" if analysis.image_type == CODE else "image_ocr"
        extracted_chunks.append(
            EmbeddedChunk(
                content=f"[Text extracted from image ({analysis.image_type}): {record['alt_text']}]\n{final_text}",
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
                    # This chunk bypasses chunk_with_parent_child (it's
                    # a standalone OCR/vision extraction, not part of a
                    # parent/child pair), but app.retrieval.hybrid_
                    # retriever's dense search and BM25 index for the
                    # "web"/"github" KBs both hard-filter to
                    # chunk_role="parent" - without this field, these
                    # chunks were silently excluded from both retrieval
                    # legs despite being stored. "parent" is correct
                    # here: this chunk is complete/standalone, exactly
                    # what that filter is meant to admit.
                    "chunk_role": "parent",
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


def _ingest_video_frames(document: Document, file_path: str) -> int:
    """
    Embed every sampled frame of a video with CLIP (visual similarity
    search - see app.embeddings.image_embedder), storing each into the
    SAME image collection website images already use
    (app.vectorstore.store.add_image_chunks) - a video frame is just
    an image in that same vector space, so a second collection would
    only fragment search for no benefit (see
    app.embeddings.audio_embedder's module docstring for why AUDIO, by
    contrast, needs a genuinely separate collection).

    This is a SEPARATE pass from load_video_document's own frame
    sampling (which extracts on-screen TEXT via OCR/vision, already
    merged into `document.content`) - re-decoding the video a second
    time here is a deliberate, accepted tradeoff (see the "native
    embeddings" plan) rather than threading CLIP embedding through the
    text-extraction loader path and blurring the loader/ingest
    boundary every other source type (including images) already
    respects: loader.py only ever decodes, ingest.py is uniformly
    where embed_*/add_*_chunks calls happen.

    Always runs (not gated by an env var) - the OCR/vision frame-text
    step this pass duplicates the decode of is ALREADY unconditional
    and far more expensive per frame (see app.ocr.frame_analysis), so
    gating only the comparatively cheap CLIP embedding step here would
    be an inconsistent half-measure.

    Frames have no real external URL the way a web image does - each
    sampled frame is saved as a small JPEG thumbnail
    (app.media.media_store.save_frame_thumbnail) and THAT servable
    "/media/frames/..." URL is what gets stored as image_url (still
    add_image_chunks()'s upsert id, unchanged).

    Returns:
        How many frames were embedded and stored.
    """
    document_id = document.metadata.get("document_id")
    duration = _video_duration_seconds(file_path)
    interval = _DEFAULT_FRAME_INTERVAL_SECONDS
    if duration and _DEFAULT_MAX_FRAMES:
        interval = max(_DEFAULT_FRAME_INTERVAL_SECONDS, duration / _DEFAULT_MAX_FRAMES)

    timestamps: list[float] = []
    frames: list[Image.Image] = []
    for timestamp, frame in _sample_video_frames(file_path, interval, duration):
        timestamps.append(timestamp)
        frames.append(frame)

    if not frames:
        return 0

    embeddings = embed_images(frames)

    image_chunks = []
    for timestamp, frame, embedding in zip(timestamps, frames, embeddings):
        thumbnail_url = media_store.save_frame_thumbnail(document_id, timestamp, frame)
        image_chunks.append(
            {
                "content": f"Frame from {document_id} at {_format_timestamp_label(timestamp)}",
                "embedding": embedding,
                "metadata": {
                    "image_url": thumbnail_url,
                    "alt_text": f"Frame from {document_id} at {_format_timestamp_label(timestamp)}",
                    "origin": "video_frame",
                    "video_document_id": document_id,
                    "timestamp_seconds": timestamp,
                    # See app.retrieval.retriever.retrieve_images's
                    # docstring - this is what a per-KB "video" chat
                    # scopes on, so frames from a different KB (there
                    # is none today, but source_kb is the general
                    # mechanism) or web images never leak in.
                    "source_kb": "video",
                },
            }
        )

    add_image_chunks(image_chunks)
    logger.info("Embedded and stored %d video frame(s) for CLIP visual search from '%s'", len(image_chunks), document_id)
    return len(image_chunks)


def _format_timestamp_label(seconds: float) -> str:
    """"MM:SS" - mirrors loader.py's own _format_timestamp, duplicated
    here rather than imported since it's a private helper of that
    module and this is a display-only label, not a parsed value."""
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _ingest_audio_clips(document: Document, file_path: str) -> int:
    """
    Embed every sampled 10s window of an audio/video file's audio
    track with CLAP (acoustic similarity search - see
    app.embeddings.audio_embedder), storing into the genuinely
    separate rag_audio_clip_chunks collection (unlike video frames,
    audio has no existing embedding space to reuse - CLAP's vector
    space has nothing in common with CLIP's or MiniLM's).

    No-op (returns 0, does no work at all - not even sampling) unless
    ENABLE_AUDIO_SIMILARITY_SEARCH=1 - see that flag's own comment for
    why this path is opt-in while video-frame CLIP embedding isn't.

    Saves ONE full copy of the source file (app.media.media_store.save_media_copy)
    per document, not one file per embedded clip - every clip hit
    references that same file plus its own start_seconds/end_seconds,
    and the UI seeks into it for playback rather than needing ~dozens
    of tiny per-clip audio files on disk.

    Returns:
        How many clips were embedded and stored.
    """
    if not _ENABLE_AUDIO_SIMILARITY_SEARCH:
        return 0

    document_id = document.metadata.get("document_id")
    clips = list(_sample_audio_clips(file_path))
    if not clips:
        return 0

    starts = [start for start, _end, _waveform in clips]
    ends = [end for _start, end, _waveform in clips]
    waveforms = [waveform for _start, _end, waveform in clips]

    embeddings = embed_audio_clips(waveforms)
    source_url = media_store.save_media_copy(document_id, file_path)

    clip_chunks = [
        {
            "content": f"Clip from {document_id} at {_format_timestamp_label(start)}",
            "embedding": embedding,
            "metadata": {
                "document_id": document_id,
                "start_seconds": start,
                "end_seconds": end,
                "source_audio_url": source_url,
                # "audio" or "video" (document.metadata["source_type"] -
                # the file this clip's audio track came from). What
                # app.retrieval.retriever.retrieve_audio_clips filters
                # by so a question scoped to one KB tab doesn't surface
                # a clip from a completely different ingested file -
                # see that function's docstring for why this matters.
                "source_kb": document.metadata.get("source_type"),
            },
        }
        for start, end, embedding in zip(starts, ends, embeddings)
    ]

    add_audio_clip_chunks(clip_chunks)
    logger.info("Embedded and stored %d audio clip(s) for CLAP acoustic search from '%s'", len(clip_chunks), document_id)
    return len(clip_chunks)


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
    total_audio_clips = 0
    for document in documents:
        chunks = chunk_with_parent_child(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embedded_chunks = embed_chunks(chunks)
        add_embedded_chunks(embedded_chunks)
        total_chunks += len(chunks)
        total_images += _ingest_images(document)
        # Native visual/acoustic similarity search (see
        # _ingest_video_frames/_ingest_audio_clips) - file_path is
        # already set by load_video_document/load_audio_document
        # themselves, so no extra path-tracking is needed here. Both
        # "audio" and "video" source types have an audio track to
        # sample for CLAP.
        source_type = document.metadata.get("source_type")
        if source_type == "video":
            total_images += _ingest_video_frames(document, document.metadata["file_path"])
        if source_type in ("audio", "video"):
            total_audio_clips += _ingest_audio_clips(document, document.metadata["file_path"])

    logger.info(
        "Ingestion complete: stored %d chunk(s), %d image(s), %d audio clip(s) total across %d "
        "document(s)/page(s). Collection now has %d chunk(s), %d image(s), %d audio clip(s).",
        total_chunks,
        total_images,
        total_audio_clips,
        len(documents),
        count(),
        image_count(),
        audio_clip_count(),
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

    # Build the per-repo GraphRAG code graph AFTER chunks are stored:
    # this way a graph on disk always corresponds to a repository
    # actually present in the vector store, so
    # app.graph.graph_retrieval never traverses to a document_id that
    # has no chunks behind it. Reuses the exact same Documents (with
    # their file_path/repository metadata) that just got chunked, so
    # graph nodes and chunk metadata stay in lockstep. Non-Python
    # files are silently skipped by build_repository_graph itself
    # (see its docstring).
    if documents:
        repository = documents[0].metadata.get("repository")
        if repository:
            file_pairs = [
                (doc.metadata["file_path"], doc.content)
                for doc in documents
                if doc.metadata.get("file_path")
            ]
            graph = build_repository_graph(repository, file_pairs)
            save_repository_graph(graph, repository)

    logger.info(
        "GitHub ingestion complete for '%s': stored %d chunk(s) across %d file(s). "
        "Collection now has %d chunk(s).",
        repo_url,
        total_chunks,
        len(documents),
        count(),
    )
    return total_chunks


def ingest_github_activity(
    repo_url: str,
    include_issues: bool = True,
    include_prs: bool = True,
    include_discussions: bool = True,
    max_items: int = 200,
) -> dict:
    """
    Ingest a repository's Issues, Pull Requests, and/or Discussions -
    the conversational counterpart to ingest_github_repo() (which
    ingests source/doc FILES). Each kind lands in the same "github" KB
    as the repo's code (see app.ingestion.loader.load_github_issues/
    load_github_pull_requests/load_github_discussions - all three use
    source_type="code" so app.vectorstore.store routes them there,
    distinguished from code chunks and from each other via
    content_type="issue"/"pull_request"/"discussion"), so a question
    like "what issues mention the login bug" is answered from the same
    hybrid retrieval path as any other GitHub-KB question, with zero
    changes needed to retrieval/routing.

    Every kind is independently toggleable because Discussions require
    a GITHUB_TOKEN (see load_github_discussions) while issues/PRs work
    token-optionally - a caller without a token can still ingest
    issues/PRs by leaving include_discussions=False.

    Re-running this for the same repo is safe/idempotent for the same
    reason as ingest_github_repo(): each Document's document_id
    ("owner/repo:issue:<n>" etc.) makes chunk ids stable, so
    re-ingesting overwrites the same chunks instead of duplicating
    them. Deletion is also already covered for free:
    app.vectorstore.store.delete_repository() removes every chunk
    matching the `repository` metadata field regardless of
    content_type, so deleting a repo already removes its issues/PRs/
    discussions along with its code.

    Args:
        repo_url: A GitHub repo reference (see
            app.ingestion.loader._parse_github_repo_url for accepted
            forms).
        include_issues, include_prs, include_discussions: Which kinds
            to fetch and ingest.
        max_items: Cap on how many of EACH kind to fetch (passed
            through as max_items to each loader).

    Returns:
        {"repository": "owner/repo", "issues": N, "pull_requests": N,
        "discussions": N, "chunks": total_chunks_stored} - counts are
        0 for any kind that was left disabled.
    """
    owner, repo, _branch = _parse_github_repo_url(repo_url)
    repository = f"{owner}/{repo}"

    counts = {"issues": 0, "pull_requests": 0, "discussions": 0}
    total_chunks = 0

    if include_issues:
        documents = load_github_issues(owner, repo, max_items=max_items)
        counts["issues"] = len(documents)
        for document in documents:
            chunks = chunk_with_parent_child(document)
            add_embedded_chunks(embed_chunks(chunks))
            total_chunks += len(chunks)

    if include_prs:
        documents = load_github_pull_requests(owner, repo, max_items=max_items)
        counts["pull_requests"] = len(documents)
        for document in documents:
            chunks = chunk_with_parent_child(document)
            add_embedded_chunks(embed_chunks(chunks))
            total_chunks += len(chunks)

    if include_discussions:
        documents = load_github_discussions(owner, repo, max_items=max_items)
        counts["discussions"] = len(documents)
        for document in documents:
            chunks = chunk_with_parent_child(document)
            add_embedded_chunks(embed_chunks(chunks))
            total_chunks += len(chunks)

    logger.info(
        "GitHub activity ingestion complete for '%s': %d issue(s), %d PR(s), %d discussion(s), "
        "%d chunk(s) stored.",
        repository, counts["issues"], counts["pull_requests"], counts["discussions"], total_chunks,
    )
    return {"repository": repository, **counts, "chunks": total_chunks}


if __name__ == "__main__":
    from app.logging_config import configure_logging

    configure_logging()

    stored = ingest_all()
    print(f"Ingested and stored {stored} chunk(s) from '{DATA_RAW_DIR}' and '{URLS_FILE}'.")
    print(f"Collection now has {count()} chunk(s) total.")

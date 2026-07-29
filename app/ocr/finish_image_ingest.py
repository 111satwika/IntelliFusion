"""
Resume-safe maintenance script: for every image whose CLIP embedding
is already in the image collection but that has NO text chunk yet in
the corresponding text KB, run the missing OCR + classify + (optional)
vision-model pass and store the resulting text chunk - so a
long-running ingest that was interrupted mid-way through
_ingest_images can be finished later, incrementally, without
re-crawling anything.

Why this exists (separate from app.ingestion.ingest AND from
app.ocr.upgrade_code_images):
  - app.ingestion.ingest runs the full crawl -> chunk -> embed -> OCR
    -> vision pipeline as one long transaction. If the user Ctrl+C's
    mid-run, every image whose _ingest_images loop iteration hadn't
    finished is left with a CLIP embedding but no text chunk.
  - app.ocr.upgrade_code_images ONLY re-processes chunks that already
    have an "image_ocr" text chunk (it upgrades OCR-only text chunks
    to vision-cross-checked "image_code" ones). It cannot help images
    that never got an OCR chunk in the first place.
  - This script fills that gap: it scans the CLIP image collection,
    diffs against the text collection's image_url set, and only
    processes images that have no text chunk yet. Every processed
    image is upserted immediately, so re-running after a Ctrl+C picks
    up exactly where the previous run stopped - no wasted work.

Gating: by default, the (slow, CPU-bound) vision-model call runs only
for CODE-classified images, matching what _ingest_images does. Pass
--skip-vision to run OCR-only (fast: ~1-3s per image) and skip llava
entirely - a good first pass that gets every image's text into the
index quickly, after which you can optionally re-run without
--skip-vision (or use upgrade_code_images) to add the vision
cross-check for code screenshots later.

Usage:
    python -m app.ocr.finish_image_ingest              # OCR + vision on code images
    python -m app.ocr.finish_image_ingest --skip-vision  # OCR-only, no llava
    python -m app.ocr.finish_image_ingest --dry-run    # show what would be done
"""

import argparse
import logging
import time

from app.embeddings.embedder import EmbeddedChunk, embed_texts
from app.embeddings.image_embedder import download_image
from app.generation.llm_generator import generate_vision_text
from app.ocr.image_classifier import CODE, classify_extracted_text
from app.ocr.image_ocr import combine_ocr_and_vision, extract_text_from_image
from app.ocr.prompts import CODE_EXTRACTION_PROMPT
from app.vectorstore.store import (
    _get_image_collection,
    _iter_collections,
    add_embedded_chunks,
)

logger = logging.getLogger(__name__)


def _existing_image_urls_in_text_kbs() -> set[str]:
    """
    Return the set of image_urls that ALREADY have a text chunk stored
    (content_type in {"image_code", "image_ocr"}) across every text
    KB. These are the images we should skip.

    We check every KB (not just "web") because image_ocr/image_code
    chunks are keyed by image_url via metadata, and future sources
    might store them under different KBs too.
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


def _all_clip_image_records() -> list[dict]:
    """
    Return every image record currently in the CLIP image collection,
    as a list of dicts with "image_url", "alt_text", "page_url", and
    "page_title" (whatever the ingest loop stored).
    """
    collection = _get_image_collection()
    result = collection.get(include=["metadatas"])
    records: list[dict] = []
    for image_url, metadata in zip(result["ids"], result["metadatas"]):
        record = dict(metadata) if metadata else {}
        record.setdefault("image_url", image_url)
        records.append(record)
    return records


def _process_one_image(record: dict, skip_vision: bool) -> bool:
    """
    Run the missing OCR + classify + (optional) vision pass on a
    single image record and upsert the resulting text chunk. Returns
    True if a text chunk was written, False if the image had no
    extractable text (decorative image with empty OCR) or failed to
    download.
    """
    image_url = record["image_url"]

    image = download_image(image_url)
    if image is None:
        logger.warning("Could not download '%s' - skipping", image_url)
        return False

    ocr_text = extract_text_from_image(image)
    image_type = classify_extracted_text(ocr_text)

    final_text = ocr_text
    if image_type == CODE and not skip_vision:
        try:
            vision_text = generate_vision_text(image, CODE_EXTRACTION_PROMPT)
        except Exception as error:
            logger.warning(
                "Vision extraction failed for '%s' (keeping OCR-only text): %s",
                image_url,
                error,
            )
            vision_text = ""
        final_text = combine_ocr_and_vision(ocr_text, vision_text)

    if not final_text:
        return False

    content_type = "image_code" if image_type == CODE and not skip_vision else "image_ocr"
    alt_text = record.get("alt_text", "") or record.get("section", "")
    content = f"[Text extracted from image ({image_type}): {alt_text}]\n{final_text}"

    metadata = {
        "document_id": image_url,
        "chunk_index": 0,
        "file_name": record.get("page_title") or record.get("file_name"),
        "url": record.get("page_url") or record.get("url"),
        "section": alt_text,
        "content_type": content_type,
        "image_url": image_url,
        "source_type": record.get("source_type", "web"),
    }

    [vector] = embed_texts([content])
    add_embedded_chunks([EmbeddedChunk(content=content, embedding=vector, metadata=metadata)])
    return True


def finish_image_ingest(skip_vision: bool = False, dry_run: bool = False) -> dict:
    """
    Main entry point. Diff CLIP image collection vs text KBs, process
    each missing image, upsert its text chunk.

    Args:
        skip_vision: If True, never invoke llava - OCR-only pass. Much
            faster; code images will still be classified as CODE but
            stored with OCR-only text (content_type=image_ocr instead
            of image_code). Use upgrade_code_images afterwards to layer
            in the vision cross-check for those.
        dry_run: If True, don't download anything or write to the
            store - just report how many candidates exist.

    Returns:
        A dict summarizing the run: total_clip, already_done,
        candidates, processed, written, skipped_empty, download_failed.
    """
    clip_records = _all_clip_image_records()
    already_done = _existing_image_urls_in_text_kbs()

    candidates = [r for r in clip_records if r["image_url"] not in already_done]

    summary = {
        "total_clip": len(clip_records),
        "already_done": len(already_done),
        "candidates": len(candidates),
        "processed": 0,
        "written": 0,
        "skipped_empty": 0,
        "download_failed": 0,
    }

    logger.info(
        "CLIP images=%d, already-text-chunked=%d, candidates=%d",
        summary["total_clip"],
        summary["already_done"],
        summary["candidates"],
    )

    if dry_run or not candidates:
        return summary

    start = time.monotonic()
    for i, record in enumerate(candidates, start=1):
        image_url = record["image_url"]
        try:
            wrote = _process_one_image(record, skip_vision=skip_vision)
        except KeyboardInterrupt:
            logger.warning("Interrupted after processing %d/%d candidate(s)", i - 1, len(candidates))
            raise
        except Exception as error:
            logger.warning("Unexpected error on '%s' (skipping): %s", image_url, error)
            wrote = False

        summary["processed"] += 1
        if wrote:
            summary["written"] += 1
        else:
            # We can't cheaply tell empty-OCR from download-failed here
            # since _process_one_image already logged; count both together.
            summary["skipped_empty"] += 1

        elapsed = time.monotonic() - start
        rate = summary["processed"] / elapsed if elapsed > 0 else 0.0
        remaining = len(candidates) - i
        eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
        logger.info(
            "[%d/%d] %s '%s' | written=%d elapsed=%.1fs ETA=%.1fmin",
            i,
            len(candidates),
            "wrote" if wrote else "skip",
            image_url,
            summary["written"],
            elapsed,
            eta_min,
        )

    logger.info(
        "Finish-image-ingest complete: candidates=%d, written=%d, skipped=%d",
        summary["candidates"],
        summary["written"],
        summary["skipped_empty"],
    )
    return summary


if __name__ == "__main__":
    from app.logging_config import configure_logging

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip the (slow) llava vision-model call; OCR-only pass.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many images would be processed without doing anything.",
    )
    args = parser.parse_args()

    configure_logging()

    result = finish_image_ingest(skip_vision=args.skip_vision, dry_run=args.dry_run)

    print("\nSummary:")
    for key, value in result.items():
        print(f"  {key}: {value}")

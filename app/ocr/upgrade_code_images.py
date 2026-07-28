"""
One-off maintenance script: upgrade already-ingested "image_ocr"
chunks whose text looks like source code, by adding a vision-model
cross-check - without re-running the full (multi-hour) crawl+chunk+
embed+OCR ingestion pipeline.

Why this exists (separate from app.ingestion.ingest): a full
re-ingestion of the whole corpus takes multiple hours, and the vision
model (llava) is itself CPU-slow - re-running everything just to layer
in vision cross-checking for code-classified images would throw away
all the already-completed OCR work for no reason. Instead, this
script:
  1. Fetches every already-stored "image_ocr" chunk (see
     app.vectorstore.store.get_chunks_by_content_type) - this covers
     chunks stored by any ingestion run to date, since content_type was
     always "image_ocr" before app.ingestion.ingest's classification
     step was added.
  2. Re-classifies each one's underlying OCR text
     (app.ocr.image_classifier.classify_extracted_text).
  3. For CODE-classified chunks only, re-downloads the source image,
     runs the vision model with the strict extraction prompt
     (app.ocr.prompts.CODE_EXTRACTION_PROMPT), combines the two
     outputs (app.ocr.image_ocr.combine_ocr_and_vision), and UPSERTS
     the SAME chunk id in place (same document_id/chunk_index the
     chunk already carries, recovered from its own stored metadata) -
     no new records are created and no other chunk is touched.

Gating the expensive vision call to code-classified chunks only (not
every image_ocr chunk) keeps this script's runtime roughly proportional
to how much actual code exists in the corpus.

Usage: python -m app.ocr.upgrade_code_images
"""

import logging

from app.embeddings.embedder import EmbeddedChunk, embed_texts
from app.embeddings.image_embedder import download_image
from app.generation.llm_generator import generate_vision_text
from app.ocr.image_classifier import CODE, classify_extracted_text
from app.ocr.image_ocr import combine_ocr_and_vision
from app.ocr.prompts import CODE_EXTRACTION_PROMPT
from app.vectorstore.store import add_embedded_chunks, get_chunks_by_content_type

logger = logging.getLogger(__name__)


def _strip_label(content: str) -> str:
    """
    Strip the "[Text extracted ... from image: ...]\\n" label this
    pipeline prepends to every extracted-text chunk, returning just the
    raw extracted text underneath - what classify_extracted_text/
    combine_ocr_and_vision actually need to see, not the label.
    """
    _, separator, rest = content.partition("]\n")
    return rest if separator else content


def upgrade_code_images() -> int:
    """
    Scan every stored "image_ocr" chunk and, for any whose text
    classifies as code, add a vision-model cross-check in place.

    Returns:
        How many chunks were upgraded (vision cross-check added).
    """
    chunks = get_chunks_by_content_type("image_ocr")
    logger.info("Found %d existing image_ocr chunk(s) to check", len(chunks))

    upgraded = 0
    for chunk in chunks:
        ocr_text = _strip_label(chunk["content"])
        if classify_extracted_text(ocr_text) != CODE:
            continue

        image_url = chunk["metadata"].get("image_url")
        if not image_url:
            continue

        image = download_image(image_url)
        if image is None:
            logger.warning("Could not re-download '%s' (skipping upgrade)", image_url)
            continue

        try:
            vision_text = generate_vision_text(image, CODE_EXTRACTION_PROMPT)
        except Exception as error:
            logger.warning("Vision extraction failed for '%s' (skipping upgrade): %s", image_url, error)
            continue

        combined_text = combine_ocr_and_vision(ocr_text, vision_text)
        alt_text = chunk["metadata"].get("section", "")
        content = f"[Text extracted from image (code): {alt_text}]\n{combined_text}"

        metadata = dict(chunk["metadata"])
        metadata["content_type"] = "image_code"

        [vector] = embed_texts([content])
        add_embedded_chunks([EmbeddedChunk(content=content, embedding=vector, metadata=metadata)])
        upgraded += 1
        logger.info("Upgraded '%s' (%d/%d checked so far)", image_url, upgraded, len(chunks))

    logger.info("Upgrade complete: %d/%d image_ocr chunk(s) upgraded to image_code", upgraded, len(chunks))
    return upgraded


if __name__ == "__main__":
    from app.logging_config import configure_logging

    configure_logging()

    upgraded_count = upgrade_code_images()
    print(f"Upgraded {upgraded_count} code-classified image chunk(s) with a vision-model cross-check.")

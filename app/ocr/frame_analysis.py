"""
Frame-level OCR+vision analysis, shared by video ingestion and website
image ingestion.

Responsibility: given ONE already-decoded image (a video frame or a
downloaded web image), extract whatever text is meaningfully present in
it - via cheap OCR always, and via the local vision model ONLY when the
OCR text looks like code - and return the combined result as a plain
string. No downloading, embedding, chunking, storage, or Document/
webpage-specific concerns happen here; this is the single reusable
"analyze one image" step both app.ingestion.ingest._ingest_images
(website images) and app.ingestion.loader.load_video_document (sampled
video frames) call into, so the OCR->classify->(conditionally)vision
gate exists in exactly one place instead of two independently-drifting
copies.

Design: identical to the per-image pipeline _ingest_images ran inline
before this was extracted - see app.ocr.image_ocr/app.ocr.image_classifier
for why each step exists (OCR-first, code-only vision gating, combining
both engines' output rather than picking one). Extracted here because a
video frame has no Document/webpage/CLIP/dedup context to hang the old
inline version off of - it's genuinely just "one image in, one string
out."
"""

import logging
from dataclasses import dataclass

from PIL import Image

from app.generation.llm_generator import generate_vision_text
from app.ocr.image_classifier import CODE, classify_extracted_text
from app.ocr.image_ocr import combine_ocr_and_vision, extract_text_from_image
from app.ocr.prompts import CODE_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    """
    text:       The final extracted text (OCR-only, OCR+vision
                combined, or "" if nothing meaningful was found) - what
                every caller actually wants.
    image_type: image_classifier.CODE or .OTHER - kept alongside text
                (rather than making the caller re-run classification)
                for callers that need it for their own metadata, e.g.
                app.ingestion.ingest._ingest_images tagging a chunk
                content_type="image_code" vs "image_ocr". Video frame
                ingestion (app.ingestion.loader.load_video_document)
                ignores this field - a frame's text is interleaved
                into the transcript with no separate content_type
                distinction.
    """

    text: str
    image_type: str


def analyze_frame(image: Image.Image) -> FrameAnalysis:
    """
    Run the OCR->classify->(conditionally)vision pipeline on one
    already-decoded image.

    Cheap OCR (see app.ocr.image_ocr.extract_text_from_image) always
    runs first. Only when the OCR text is classified CODE (see
    app.ocr.image_classifier) does the slow local vision model
    additionally run, using the same strict transcription-only prompt
    image ingestion already uses - gating the expensive call this way
    keeps a batch of frames' total analysis cost roughly proportional
    to how much actual code/text appears on screen, instead of paying
    a ~10-60s vision call for every single sampled frame regardless of
    content.

    A vision-model failure (timeout, Ollama not running, etc.) is
    logged and swallowed, falling back to OCR-only text for that frame
    - matching _ingest_images's prior behavior - so one bad frame can't
    break the rest of a video's ingestion.

    Returns:
        FrameAnalysis(text="", image_type=OTHER) if OCR found nothing
        and the (empty) text wasn't classified CODE.
    """
    ocr_text = extract_text_from_image(image)
    image_type = classify_extracted_text(ocr_text)

    final_text = ocr_text
    if image_type == CODE:
        try:
            vision_text = generate_vision_text(image, CODE_EXTRACTION_PROMPT)
        except Exception as error:  # noqa: BLE001 - a bad frame must not break the rest of ingestion
            logger.warning("Vision extraction failed for a frame (keeping OCR-only text): %s", error)
            vision_text = ""
        final_text = combine_ocr_and_vision(ocr_text, vision_text)

    return FrameAnalysis(text=final_text, image_type=image_type)

"""
OCR (text-in-image) extraction for the RAG pipeline.

Responsibility: given an already-downloaded image, extract any text
rendered inside it (e.g. a code snippet shown as a documentation
screenshot) as a plain string. No downloading, embedding, chunking, or
storage happens here - this module only reads pixels and returns text.

Why this exists: app.embeddings.image_embedder's CLIP embedding only
captures an image's *visual* similarity to a text query well enough
for search/display - it cannot reproduce what text is actually written
inside the image's pixels. Some ingested documentation pages (e.g. IBM
Docs automation-rule "Examples" pages) show the actual code as a
syntax-highlighted screenshot rather than as real HTML text, so
without OCR that code is invisible to the text-only retriever/LLM -
the RAG pipeline could find the RIGHT page and image, but had no way
to answer "write me that code" because the code text itself was never
captured anywhere as searchable/promptable text. app.ingestion.ingest
runs this at ingestion time and, when meaningful text is found, stores
it as an ordinary extra text chunk (content_type="image_ocr") so it is
retrieved and reasoned over exactly like any other text - no changes
needed anywhere else in the pipeline (chunker/embedder/store/retriever/
prompt_builder).

Provider used: EasyOCR - a pure-Python, PyTorch-based OCR engine (no
system binary to install, unlike Tesseract), fitting this project's
"no external services, everything runs locally" design already used by
embedder.py/image_embedder.py. Runs on CPU by default.

Design:
- extract_text_from_image() is the single entry point other modules
  should use, matching the "one place per concern" pattern used
  throughout this project.
- A short OCR result (below _MIN_OCR_CHARACTERS) is treated as "no
  meaningful text found" and normalized to "" rather than kept as a
  noisy one-or-two-character false positive - most decorative
  icons/photos produce a handful of junk characters, not real words.
- Any OCR failure (corrupt image, model load failure, etc.) is logged
  and swallowed, returning "" - a bad image can't break ingestion of
  everything else, matching the same defensive pattern used by
  image_embedder.py's _download_image().
"""

import logging

from PIL import Image

logger = logging.getLogger(__name__)

_MIN_OCR_CHARACTERS = 8

_reader = None


def _get_reader():
    """
    Lazily create the EasyOCR reader (English), so importing this
    module is cheap and the (one-time) OCR model download only happens
    when text extraction is actually attempted.
    """
    global _reader
    if _reader is None:
        import easyocr

        logger.info("Loading local OCR model (EasyOCR, English)...")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("OCR model loaded.")
    return _reader


def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract any text rendered inside an image (e.g. a code screenshot),
    one recognized line of text per output line, top-to-bottom as
    EasyOCR detects it.

    Args:
        image: An already-downloaded/decoded Pillow image (see
            app.embeddings.image_embedder.download_image).

    Returns:
        The extracted text, or "" if no meaningful text was found (or
        OCR failed) - never raises.
    """
    try:
        import numpy as np

        reader = _get_reader()
        lines = reader.readtext(np.array(image), detail=0, paragraph=True)
        text = "\n".join(line.strip() for line in lines if line.strip())
    except Exception as error:
        logger.warning("OCR failed on image (skipping): %s", error)
        return ""

    if len(text) < _MIN_OCR_CHARACTERS:
        return ""
    return text


def combine_ocr_and_vision(ocr_text: str, vision_text: str) -> str:
    """
    Merge traditional-OCR output (this module) and vision-model output
    (see app.generation.llm_generator.generate_vision_text, using
    app.ocr.prompts.CODE_EXTRACTION_PROMPT) for the SAME image into one
    final block of extracted content, for a code screenshot where both
    engines were run.

    Neither engine is silently discarded in favor of the other: OCR is
    typically more character-exact for real code (it doesn't
    "helpfully" fill in gaps), while the vision model can sometimes
    recover context OCR's line-by-line reading loses (e.g. code
    structure/order) - and there's no ground truth available here to
    algorithmically decide which one is "more correct" for any given
    character. Both are kept, clearly labeled, so the downstream
    retrieval/generation step (see app.prompting.prompt_builder) can
    let the LLM cross-reference them itself when answering a question
    (e.g. trusting a token that agrees in both over one that only
    appears in one output) instead of losing information up front.

    Returns:
        Just whichever of the two is non-empty, if only one is; both,
        combined and labeled, if both are; "" if neither found
        anything.
    """
    if not ocr_text:
        return vision_text
    if not vision_text:
        return ocr_text
    return f"OCR-extracted text:\n{ocr_text}\n\nVision-model-extracted text (cross-check):\n{vision_text}"


if __name__ == "__main__":
    import sys

    from app.embeddings.image_embedder import download_image

    if len(sys.argv) < 2:
        print("Usage: python -m app.ocr.image_ocr <image_url>")
        raise SystemExit(1)

    fetched_image = download_image(sys.argv[1])
    if fetched_image is None:
        print("Could not download that image.")
        raise SystemExit(1)

    extracted = extract_text_from_image(fetched_image)
    print(f"Extracted {len(extracted)} character(s):\n{extracted or '(no meaningful text found)'}")

"""
Lightweight heuristic classifier for OCR-extracted image text.

Responsibility: given the text traditional OCR already read out of an
image (see app.ocr.image_ocr), decide whether that image is *likely*
to be a source-code screenshot - worth the extra cost of also asking a
vision model to cross-check it - versus a plain UI screenshot/diagram,
where OCR alone (or nothing at all) is good enough.

Design:
- No image-classification model is used here - only a regex-pattern
  count over the OCR text string. This is cheap (no extra model load,
  no extra inference call) and, in practice, code screenshots reliably
  contain several code-only keywords/symbols (`const`, `function`,
  `=>`, `{`/`}`/`;`, etc.) that plain-English UI screenshots or
  diagrams essentially never do.
- Deliberately conservative (requires _CODE_PATTERN_THRESHOLD distinct
  pattern hits, not just one) so a stray brace or the word "class" in
  an unrelated sentence doesn't misclassify a normal screenshot as
  "code" and trigger an unnecessary vision-model call.
- No OCR text at all (empty string) is classified "other", not "code"
  - a code screenshot that OCR completely failed to read anything from
    would be extremely unusual; far more likely it's a diagram/photo
    with no legible text, or a decorative image (though decorative
    images are already filtered out earlier, in
    app.ingestion.loader._describe_image_tag, via empty alt text).
"""

import re

_CODE_PATTERNS = [
    r"\bconst\b",
    r"\bfunction\b",
    r"\breturn\b",
    r"\bclass\b",
    r"\bdef\b",
    r"\bimport\b",
    r"\bawait\b",
    r"=>",
    r"[{};]",
    r"\bif\s*\(",
    r"\bqueryAsync\b",
    r"\bpayload\b",
    r"\bcommand\b",
]
_CODE_PATTERN_THRESHOLD = 3

CODE = "code"
OTHER = "other"


def classify_extracted_text(ocr_text: str) -> str:
    """
    Returns CODE if ocr_text matches at least _CODE_PATTERN_THRESHOLD
    distinct code-like patterns (keywords/symbols), else OTHER.
    """
    if not ocr_text:
        return OTHER

    hits = sum(1 for pattern in _CODE_PATTERNS if re.search(pattern, ocr_text))
    return CODE if hits >= _CODE_PATTERN_THRESHOLD else OTHER

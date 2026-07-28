"""Tests for app.ocr.image_classifier."""

from app.ocr import image_classifier
from app.ocr.image_classifier import CODE, OTHER, classify_extracted_text


def test_classify_extracted_text_returns_other_for_empty_text():
    assert classify_extracted_text("") == OTHER


def test_classify_extracted_text_returns_other_for_plain_prose():
    text = "Click the Settings icon in the top right corner to open preferences."

    assert classify_extracted_text(text) == OTHER


def test_classify_extracted_text_returns_code_for_javascript_snippet():
    text = (
        "const request = context.GetEntity();\n"
        "if (request.EntityState.Name === 'Closed') {\n"
        "  await api.queryAsync(payload);\n"
        "  return true;\n"
        "}"
    )

    assert classify_extracted_text(text) == CODE


def test_classify_extracted_text_requires_multiple_pattern_hits():
    # Only one weak signal (a brace) - should not be enough on its own.
    text = "Configuration {default}"

    assert classify_extracted_text(text) == OTHER


def test_code_pattern_threshold_is_respected(monkeypatch):
    monkeypatch.setattr(image_classifier, "_CODE_PATTERN_THRESHOLD", 1)

    assert classify_extracted_text("class Foo") == CODE

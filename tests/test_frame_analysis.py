"""Tests for app.ocr.frame_analysis.

The real EasyOCR/Ollama models are NOT used here - extract_text_from_image
and generate_vision_text are monkeypatched directly on the frame_analysis
module (the same "monkeypatch at the point of use" convention already
used in tests/test_ingest.py), so this only verifies analyze_frame()'s
own OCR->classify->(conditionally)vision->combine branching.
"""

from app.ocr import frame_analysis
from app.ocr.image_classifier import CODE, OTHER

_CODE_LIKE_TEXT = "const request = context.GetEntity();\nif (request.EntityState.Name === 'Closed') {\n  return true;\n}"


def test_analyze_frame_returns_ocr_only_text_when_not_code(monkeypatch):
    monkeypatch.setattr(frame_analysis, "extract_text_from_image", lambda image: "just a caption")

    result = frame_analysis.analyze_frame(object())

    assert result.text == "just a caption"
    assert result.image_type == OTHER


def test_analyze_frame_returns_empty_when_ocr_finds_nothing(monkeypatch):
    monkeypatch.setattr(frame_analysis, "extract_text_from_image", lambda image: "")

    result = frame_analysis.analyze_frame(object())

    assert result.text == ""
    assert result.image_type == OTHER


def test_analyze_frame_runs_vision_and_combines_for_code_classified_text(monkeypatch):
    monkeypatch.setattr(frame_analysis, "extract_text_from_image", lambda image: _CODE_LIKE_TEXT)
    monkeypatch.setattr(frame_analysis, "generate_vision_text", lambda image, instruction: "vision transcription")
    monkeypatch.setattr(frame_analysis, "combine_ocr_and_vision", lambda ocr_text, vision_text: "COMBINED")

    result = frame_analysis.analyze_frame(object())

    assert result.text == "COMBINED"
    assert result.image_type == CODE


def test_analyze_frame_falls_back_to_ocr_only_when_vision_fails(monkeypatch):
    monkeypatch.setattr(frame_analysis, "extract_text_from_image", lambda image: _CODE_LIKE_TEXT)

    def failing_vision(image, instruction):
        raise TimeoutError("vision model too slow")

    monkeypatch.setattr(frame_analysis, "generate_vision_text", failing_vision)

    result = frame_analysis.analyze_frame(object())

    # combine_ocr_and_vision(ocr_text, "") returns ocr_text unchanged
    # (see app.ocr.image_ocr.combine_ocr_and_vision) - not monkeypatched
    # here, so this exercises the real combine function too.
    assert result.text == _CODE_LIKE_TEXT
    assert result.image_type == CODE


def test_analyze_frame_passes_the_code_extraction_prompt_to_vision(monkeypatch):
    monkeypatch.setattr(frame_analysis, "extract_text_from_image", lambda image: _CODE_LIKE_TEXT)
    captured = {}

    def fake_vision(image, instruction):
        captured["instruction"] = instruction
        return ""

    monkeypatch.setattr(frame_analysis, "generate_vision_text", fake_vision)

    frame_analysis.analyze_frame(object())

    assert captured["instruction"] == frame_analysis.CODE_EXTRACTION_PROMPT

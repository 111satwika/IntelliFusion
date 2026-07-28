"""Tests for app.ocr.image_ocr.

The real EasyOCR model is NOT used here - _get_reader is faked out so
these tests run fast and don't depend on a model download. The real
model is exercised manually via
`python -m app.ocr.image_ocr <image_url>` (see module docstring) and
indirectly through the whole ingestion pipeline in practice.
"""

from PIL import Image

from app.ocr import image_ocr


class _FakeReader:
    def __init__(self, lines):
        self._lines = lines

    def readtext(self, image_array, detail=0, paragraph=True):
        return self._lines


def test_extract_text_from_image_returns_joined_lines(monkeypatch):
    monkeypatch.setattr(image_ocr, "_get_reader", lambda: _FakeReader(["def close_request():", "return True"]))

    text = image_ocr.extract_text_from_image(Image.new("RGB", (10, 10)))

    assert text == "def close_request():\nreturn True"


def test_extract_text_from_image_returns_empty_string_for_short_result(monkeypatch):
    # Below _MIN_OCR_CHARACTERS - treated as noise (e.g. a decorative icon).
    monkeypatch.setattr(image_ocr, "_get_reader", lambda: _FakeReader(["ok"]))

    assert image_ocr.extract_text_from_image(Image.new("RGB", (10, 10))) == ""


def test_extract_text_from_image_returns_empty_string_for_no_text(monkeypatch):
    monkeypatch.setattr(image_ocr, "_get_reader", lambda: _FakeReader([]))

    assert image_ocr.extract_text_from_image(Image.new("RGB", (10, 10))) == ""


def test_extract_text_from_image_swallows_ocr_failure(monkeypatch):
    class _BrokenReader:
        def readtext(self, *args, **kwargs):
            raise RuntimeError("model failed")

    monkeypatch.setattr(image_ocr, "_get_reader", lambda: _BrokenReader())

    assert image_ocr.extract_text_from_image(Image.new("RGB", (10, 10))) == ""


def test_combine_ocr_and_vision_returns_vision_text_when_ocr_empty():
    assert image_ocr.combine_ocr_and_vision("", "vision result") == "vision result"


def test_combine_ocr_and_vision_returns_ocr_text_when_vision_empty():
    assert image_ocr.combine_ocr_and_vision("ocr result", "") == "ocr result"


def test_combine_ocr_and_vision_returns_empty_when_both_empty():
    assert image_ocr.combine_ocr_and_vision("", "") == ""


def test_combine_ocr_and_vision_labels_and_combines_both():
    combined = image_ocr.combine_ocr_and_vision("const x = 1;", "const x = 1;\nreturn x;")

    assert "OCR-extracted text:" in combined
    assert "const x = 1;" in combined
    assert "Vision-model-extracted text (cross-check):" in combined
    assert "return x;" in combined


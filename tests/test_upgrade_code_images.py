"""Tests for app.ocr.upgrade_code_images.

All I/O (store lookup, image download, vision model, embedding) is
faked out - these tests only verify the classify -> vision -> combine
-> upsert-in-place logic.
"""

from app.ocr import upgrade_code_images as upgrade


def _code_like_ocr_text():
    return "const request = context.GetEntity();\nif (request.EntityState.Name === 'Closed') {\n  return true;\n}"


def test_strip_label_removes_bracketed_prefix():
    content = "[Text extracted via OCR from image: Example]\nsome code here"

    assert upgrade._strip_label(content) == "some code here"


def test_strip_label_returns_content_unchanged_when_no_label():
    assert upgrade._strip_label("no label here") == "no label here"


def test_upgrade_code_images_upgrades_code_classified_chunks(monkeypatch):
    stored_chunks = [
        {
            "id": "https://example.com/code.png::chunk_0",
            "content": f"[Text extracted via OCR from image: Example]\n{_code_like_ocr_text()}",
            "metadata": {
                "document_id": "https://example.com/code.png",
                "chunk_index": 0,
                "image_url": "https://example.com/code.png",
                "section": "Example",
                "content_type": "image_ocr",
            },
        }
    ]
    calls = []
    fake_image = object()

    monkeypatch.setattr(upgrade, "get_chunks_by_content_type", lambda content_type: stored_chunks)
    monkeypatch.setattr(upgrade, "download_image", lambda url: fake_image)
    monkeypatch.setattr(upgrade, "generate_vision_text", lambda image, instruction: "vision-transcribed code")
    monkeypatch.setattr(upgrade, "combine_ocr_and_vision", lambda ocr_text, vision_text: "COMBINED")
    monkeypatch.setattr(upgrade, "embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(upgrade, "add_embedded_chunks", lambda chunks: calls.append(chunks))

    upgraded = upgrade.upgrade_code_images()

    assert upgraded == 1
    stored = calls[0][0]
    assert stored.metadata["content_type"] == "image_code"
    assert stored.metadata["document_id"] == "https://example.com/code.png"
    assert "COMBINED" in stored.content


def test_upgrade_code_images_skips_non_code_chunks(monkeypatch):
    stored_chunks = [
        {
            "id": "https://example.com/icon.png::chunk_0",
            "content": "[Text extracted via OCR from image: Icon]\nSettings",
            "metadata": {
                "document_id": "https://example.com/icon.png",
                "chunk_index": 0,
                "image_url": "https://example.com/icon.png",
                "section": "Icon",
                "content_type": "image_ocr",
            },
        }
    ]
    calls = []

    monkeypatch.setattr(upgrade, "get_chunks_by_content_type", lambda content_type: stored_chunks)
    monkeypatch.setattr(upgrade, "add_embedded_chunks", lambda chunks: calls.append(chunks))

    upgraded = upgrade.upgrade_code_images()

    assert upgraded == 0
    assert calls == []


def test_upgrade_code_images_skips_chunk_when_vision_extraction_fails(monkeypatch):
    stored_chunks = [
        {
            "id": "https://example.com/code.png::chunk_0",
            "content": f"[Text extracted via OCR from image: Example]\n{_code_like_ocr_text()}",
            "metadata": {
                "document_id": "https://example.com/code.png",
                "chunk_index": 0,
                "image_url": "https://example.com/code.png",
                "section": "Example",
                "content_type": "image_ocr",
            },
        }
    ]
    calls = []

    def failing_vision(image, instruction):
        raise TimeoutError("too slow")

    monkeypatch.setattr(upgrade, "get_chunks_by_content_type", lambda content_type: stored_chunks)
    monkeypatch.setattr(upgrade, "download_image", lambda url: object())
    monkeypatch.setattr(upgrade, "generate_vision_text", failing_vision)
    monkeypatch.setattr(upgrade, "add_embedded_chunks", lambda chunks: calls.append(chunks))

    upgraded = upgrade.upgrade_code_images()

    assert upgraded == 0
    assert calls == []


def test_upgrade_code_images_skips_chunk_when_image_undownloadable(monkeypatch):
    stored_chunks = [
        {
            "id": "https://example.com/code.png::chunk_0",
            "content": f"[Text extracted via OCR from image: Example]\n{_code_like_ocr_text()}",
            "metadata": {
                "document_id": "https://example.com/code.png",
                "chunk_index": 0,
                "image_url": "https://example.com/code.png",
                "section": "Example",
                "content_type": "image_ocr",
            },
        }
    ]
    calls = []

    monkeypatch.setattr(upgrade, "get_chunks_by_content_type", lambda content_type: stored_chunks)
    monkeypatch.setattr(upgrade, "download_image", lambda url: None)
    monkeypatch.setattr(upgrade, "add_embedded_chunks", lambda chunks: calls.append(chunks))

    upgraded = upgrade.upgrade_code_images()

    assert upgraded == 0
    assert calls == []

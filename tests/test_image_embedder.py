"""Tests for app.embeddings.image_embedder.

The real CLIP model and real HTTP calls are NOT used here - _get_model
and _download_image are faked out so these tests run fast and don't
depend on a model download or network access. The real model is
exercised manually via `python -m app.embeddings.image_embedder <url>`
(see module docstring) and indirectly through the whole pipeline in
practice.
"""

from PIL import Image

from app.embeddings import image_embedder


class _FakeClipModel:
    """Returns a distinct, deterministic vector per input (image or text)."""

    def encode(self, inputs, convert_to_numpy=True, show_progress_bar=False):
        import numpy as np

        def vector_for(item):
            if isinstance(item, Image.Image):
                return [float(item.size[0]), 0.0]
            return [0.0, float(len(item))]

        return np.array([vector_for(item) for item in inputs])


def test_embed_image_urls_returns_empty_dict_for_empty_input():
    assert image_embedder.embed_image_urls([]) == {}


def test_embed_image_urls_downloads_embeds_and_maps_by_url(monkeypatch):
    fake_image = Image.new("RGB", (10, 10))

    def fake_download(url, timeout=10):
        return fake_image if url == "https://example.com/good.png" else None

    monkeypatch.setattr(image_embedder, "_download_image", fake_download)
    monkeypatch.setattr(image_embedder, "_get_model", lambda: _FakeClipModel())

    result = image_embedder.embed_image_urls(
        ["https://example.com/good.png", "https://example.com/broken.png"]
    )

    assert list(result.keys()) == ["https://example.com/good.png"]
    assert result["https://example.com/good.png"] == [10.0, 0.0]


def test_embed_image_urls_returns_empty_dict_when_all_downloads_fail(monkeypatch):
    monkeypatch.setattr(image_embedder, "_download_image", lambda url, timeout=10: None)
    monkeypatch.setattr(image_embedder, "_get_model", lambda: _FakeClipModel())

    result = image_embedder.embed_image_urls(["https://example.com/broken.png"])

    assert result == {}


def test_embed_text_for_image_search_uses_the_clip_model(monkeypatch):
    monkeypatch.setattr(image_embedder, "_get_model", lambda: _FakeClipModel())

    vector = image_embedder.embed_text_for_image_search("auth diagram")

    assert vector == [0.0, float(len("auth diagram"))]


def test_download_image_returns_none_on_request_failure(monkeypatch):
    class _FailingSession:
        def get(self, *args, **kwargs):
            raise ConnectionError("boom")

    monkeypatch.setattr(image_embedder.requests, "get", _FailingSession().get)

    assert image_embedder._download_image("https://example.com/x.png") is None


def test_download_image_public_wrapper_delegates_to_private_downloader(monkeypatch):
    fake_image = Image.new("RGB", (5, 5))
    monkeypatch.setattr(image_embedder, "_download_image", lambda url, timeout=10: fake_image)

    assert image_embedder.download_image("https://example.com/good.png") is fake_image


def test_embed_images_returns_empty_list_for_empty_input():
    assert image_embedder.embed_images([]) == []


def test_embed_images_embeds_pre_downloaded_images(monkeypatch):
    monkeypatch.setattr(image_embedder, "_get_model", lambda: _FakeClipModel())

    vectors = image_embedder.embed_images([Image.new("RGB", (10, 10))])

    assert vectors == [[10.0, 0.0]]

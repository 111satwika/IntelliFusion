"""Tests for app.generation.llm_generator.

requests.post is faked out here so these tests run without a live
Ollama server - they only verify generate_answer() builds the correct
request payload and parses/handles the response correctly.
"""

import pytest
import requests

from app.generation import llm_generator


class _FakeResponse:
    def __init__(self, json_data, raise_exc=None):
        self._json_data = json_data
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._json_data


class _FakeGetResponse:
    """Fakes requests.get()'s response shape (raise_for_status + .content)."""

    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def test_generate_answer_sends_expected_payload_and_returns_stripped_text(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"response": "  The answer.  "})

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    answer = llm_generator.generate_answer("some prompt")

    assert captured["url"] == llm_generator._OLLAMA_URL
    assert captured["json"] == {
        "model": llm_generator._DEFAULT_MODEL,
        "prompt": "some prompt",
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096},
    }
    assert captured["timeout"] == 600
    assert answer == "The answer."


def test_generate_answer_uses_provided_model_and_temperature(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse({"response": "ok"})

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    llm_generator.generate_answer("prompt", model="qwen2.5:7b-instruct", temperature=0.5)

    assert captured["json"]["model"] == "qwen2.5:7b-instruct"
    assert captured["json"]["options"]["temperature"] == 0.5


def test_generate_answer_raises_connection_error_when_ollama_unreachable(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.exceptions.ConnectionError("no server")

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    with pytest.raises(requests.exceptions.ConnectionError):
        llm_generator.generate_answer("prompt")


def test_generate_answer_raises_http_error_on_bad_status(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse({}, raise_exc=requests.exceptions.HTTPError("bad status"))

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    with pytest.raises(requests.exceptions.HTTPError):
        llm_generator.generate_answer("prompt")


def test_generate_answer_with_images_sends_expected_chat_payload(monkeypatch):
    captured = {}

    def fake_get(url, timeout, headers):
        return _FakeGetResponse(b"fakebytes")

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"message": {"role": "assistant", "content": "  The code is X.  "}})

    monkeypatch.setattr(llm_generator.requests, "get", fake_get)
    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    answer = llm_generator.generate_answer_with_images("some prompt", ["https://example.com/a.png"])

    assert captured["url"] == llm_generator._OLLAMA_CHAT_URL
    assert captured["json"]["model"] == llm_generator._DEFAULT_VISION_MODEL
    assert captured["json"]["messages"][0]["content"] == "some prompt"
    assert captured["json"]["messages"][0]["images"] == ["ZmFrZWJ5dGVz"]  # base64("fakebytes")
    assert captured["json"]["options"]["temperature"] == 0.0
    assert answer == "The code is X."


def test_generate_answer_with_images_skips_undownloadable_images(monkeypatch):
    captured = {}

    def fake_get(url, timeout, headers):
        raise ConnectionError("boom")

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr(llm_generator.requests, "get", fake_get)
    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    llm_generator.generate_answer_with_images("prompt", ["https://example.com/broken.png"])

    assert "images" not in captured["json"]["messages"][0]


def test_generate_answer_with_images_uses_provided_model(monkeypatch):
    captured = {}

    monkeypatch.setattr(llm_generator.requests, "get", lambda url, timeout, headers: _FakeGetResponse(b"x"))

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    llm_generator.generate_answer_with_images("prompt", ["https://example.com/a.png"], model="custom-vision-model")

    assert captured["json"]["model"] == "custom-vision-model"


def test_generate_vision_text_sends_expected_chat_payload(monkeypatch):
    from PIL import Image

    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"message": {"content": "  extracted code  "}})

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    text = llm_generator.generate_vision_text(Image.new("RGB", (4, 4)), "Extract the code.")

    assert captured["url"] == llm_generator._OLLAMA_CHAT_URL
    assert captured["json"]["model"] == llm_generator._DEFAULT_VISION_MODEL
    assert captured["json"]["messages"][0]["content"] == "Extract the code."
    assert len(captured["json"]["messages"][0]["images"]) == 1
    assert captured["json"]["options"]["temperature"] == 0.0
    assert captured["timeout"] == 300
    assert text == "extracted code"


def test_generate_vision_text_uses_provided_model_and_temperature(monkeypatch):
    from PIL import Image

    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    llm_generator.generate_vision_text(
        Image.new("RGB", (4, 4)), "Describe.", model="custom-vision-model", temperature=0.3
    )

    assert captured["json"]["model"] == "custom-vision-model"
    assert captured["json"]["options"]["temperature"] == 0.3


def test_generate_vision_text_raises_connection_error_when_ollama_unreachable(monkeypatch):
    from PIL import Image

    def fake_post(url, json, timeout):
        raise requests.exceptions.ConnectionError("no server")

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    with pytest.raises(requests.exceptions.ConnectionError):
        llm_generator.generate_vision_text(Image.new("RGB", (4, 4)), "Extract.")

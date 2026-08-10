"""Tests for app.generation.llm_generator.

requests.post is faked out here so these tests run without a live
Ollama server - they only verify generate_answer() builds the correct
request payload and parses/handles the response correctly.
"""

import json

import pytest
import requests

from app.generation import llm_generator


@pytest.fixture(autouse=True)
def _clear_answer_cache():
    """The answer cache is module-global state (see llm_generator's
    module docstring) - clear it before AND after every test so a
    cache hit populated by one test can never silently change another
    test's assertions, regardless of run order."""
    llm_generator.clear_answer_cache()
    yield
    llm_generator.clear_answer_cache()


class _FakeResponse:
    def __init__(self, json_data, raise_exc=None):
        self._json_data = json_data
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._json_data


class _FakeStreamResponse:
    """Fakes the NDJSON streaming response shape generate_answer_stream
    actually consumes (response.iter_lines(decode_unicode=True)), one
    line per token plus a final {"done": true} line - not the plain
    single-JSON .json() shape stream=False would return."""

    def __init__(self, tokens, raise_exc=None):
        self._tokens = tokens
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def iter_lines(self, decode_unicode=True):
        for token in self._tokens:
            yield json.dumps({"response": token, "done": False})
        yield json.dumps({"done": True})


class _FakeGetResponse:
    """Fakes requests.get()'s response shape (raise_for_status + .content)."""

    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def test_generate_answer_sends_expected_payload_and_returns_stripped_text(monkeypatch):
    captured = {}

    def fake_post(url, json, stream, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["stream"] = stream
        captured["timeout"] = timeout
        return _FakeStreamResponse(["  The", " answer.  "])

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    answer = llm_generator.generate_answer("some prompt")

    assert captured["url"] == llm_generator._OLLAMA_URL
    assert captured["json"] == {
        "model": llm_generator._DEFAULT_MODEL,
        "prompt": "some prompt",
        "stream": True,
        "options": {"temperature": 0.0, "num_ctx": 4096},
    }
    assert captured["stream"] is True
    assert captured["timeout"] == 300
    assert answer == "The answer."


def test_generate_answer_uses_provided_model_and_temperature(monkeypatch):
    captured = {}

    def fake_post(url, json, stream, timeout):
        captured["json"] = json
        return _FakeStreamResponse(["ok"])

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    llm_generator.generate_answer("prompt", model="qwen2.5:7b-instruct", temperature=0.5)

    assert captured["json"]["model"] == "qwen2.5:7b-instruct"
    assert captured["json"]["options"]["temperature"] == 0.5


def test_generate_answer_raises_connection_error_when_ollama_unreachable(monkeypatch):
    def fake_post(url, json, stream, timeout):
        raise requests.exceptions.ConnectionError("no server")

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    with pytest.raises(requests.exceptions.ConnectionError):
        llm_generator.generate_answer("prompt")


def test_generate_answer_raises_http_error_on_bad_status(monkeypatch):
    def fake_post(url, json, stream, timeout):
        return _FakeStreamResponse([], raise_exc=requests.exceptions.HTTPError("bad status"))

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


# ---------- Answer cache ----------


def test_repeat_call_with_temperature_zero_is_a_cache_hit_and_skips_ollama(monkeypatch):
    calls = []

    def fake_post(url, json, stream, timeout):
        calls.append(1)
        return _FakeStreamResponse(["The answer."])

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    first = llm_generator.generate_answer("same prompt")
    second = llm_generator.generate_answer("same prompt")

    assert first == second == "The answer."
    assert len(calls) == 1  # Ollama only actually called once


def test_different_prompt_is_a_cache_miss(monkeypatch):
    calls = []

    def fake_post(url, json, stream, timeout):
        calls.append(json["prompt"])
        return _FakeStreamResponse([f"Answer to: {json['prompt']}"])

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    llm_generator.generate_answer("prompt A")
    llm_generator.generate_answer("prompt B")

    assert calls == ["prompt A", "prompt B"]


def test_nonzero_temperature_is_never_cached(monkeypatch):
    calls = []

    def fake_post(url, json, stream, timeout):
        calls.append(1)
        return _FakeStreamResponse(["answer"])

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    llm_generator.generate_answer("same prompt", temperature=0.5)
    llm_generator.generate_answer("same prompt", temperature=0.5)

    assert len(calls) == 2  # never served from cache


def test_different_model_is_a_cache_miss_even_for_the_same_prompt(monkeypatch):
    calls = []

    def fake_post(url, json, stream, timeout):
        calls.append(json["model"])
        return _FakeStreamResponse(["answer"])

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    llm_generator.generate_answer("same prompt", model="model-a")
    llm_generator.generate_answer("same prompt", model="model-b")

    assert calls == ["model-a", "model-b"]


def test_failed_generation_is_not_cached(monkeypatch):
    call_count = 0

    def fake_post(url, json, stream, timeout):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeStreamResponse([], raise_exc=requests.exceptions.HTTPError("bad status"))
        return _FakeStreamResponse(["recovered answer"])

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    with pytest.raises(requests.exceptions.HTTPError):
        llm_generator.generate_answer("same prompt")

    # The failed attempt must not have poisoned the cache - a retry
    # should still actually call Ollama, not replay a cached failure
    # (there's nothing to replay - it never got that far).
    answer = llm_generator.generate_answer("same prompt")
    assert answer == "recovered answer"
    assert call_count == 2


def test_generate_answer_stream_cache_hit_yields_one_chunk(monkeypatch):
    calls = []

    def fake_post(url, json, stream, timeout):
        calls.append(1)
        return _FakeStreamResponse(["The", " full", " answer."])

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    list(llm_generator.generate_answer_stream("same prompt"))  # populate cache
    second_chunks = list(llm_generator.generate_answer_stream("same prompt"))

    assert second_chunks == ["The full answer."]
    assert len(calls) == 1


def test_clear_answer_cache_forces_a_fresh_ollama_call(monkeypatch):
    calls = []

    def fake_post(url, json, stream, timeout):
        calls.append(1)
        return _FakeStreamResponse(["answer"])

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    llm_generator.generate_answer("same prompt")
    llm_generator.clear_answer_cache()
    llm_generator.generate_answer("same prompt")

    assert len(calls) == 2


def test_answer_cache_evicts_least_recently_used_beyond_maxsize(monkeypatch):
    monkeypatch.setattr(llm_generator, "_ANSWER_CACHE_MAXSIZE", 2)

    def fake_post(url, json, stream, timeout):
        return _FakeStreamResponse([f"answer for {json['prompt']}"])

    monkeypatch.setattr(llm_generator.requests, "post", fake_post)

    llm_generator.generate_answer("prompt 1")
    llm_generator.generate_answer("prompt 2")
    llm_generator.generate_answer("prompt 3")  # should evict "prompt 1"

    assert llm_generator._answer_cache_key("prompt 1", llm_generator._DEFAULT_MODEL, 0.0) not in llm_generator._answer_cache
    assert llm_generator._answer_cache_key("prompt 3", llm_generator._DEFAULT_MODEL, 0.0) in llm_generator._answer_cache

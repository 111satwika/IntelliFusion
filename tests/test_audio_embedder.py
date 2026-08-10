"""Tests for app.embeddings.audio_embedder.

The real CLAP model is NOT used here - _get_model and _get_processor
are faked out so these tests run fast and don't depend on a model
download. The real model is exercised manually via
`python -m app.embeddings.audio_embedder <query text>` (see module
docstring) and indirectly through the whole pipeline in practice -
already live-verified against a real speech recording during
implementation (real cosine similarities: ~0.45 for a paraphrase of
what was actually said, ~0.19 for a generic-but-related description,
~-0.18 for something unrelated like "a dog barking").
"""

import numpy as np
import torch

from app.embeddings import audio_embedder


class _FakeOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class _FakeClapModel:
    """Returns a distinct, deterministic vector per input clip/text."""

    def get_audio_features(self, **inputs):
        # One row per clip in the batch, value derived from the clip's
        # own length so different clips embed differently.
        clips = inputs["clip_lengths"]
        return _FakeOutput(torch.tensor([[float(n), 0.0] for n in clips]))

    def get_text_features(self, **inputs):
        texts = inputs["texts"]
        return _FakeOutput(torch.tensor([[0.0, float(len(t))] for t in texts]))


class _FakeClapProcessor:
    """Passes through just enough of the real ClapProcessor's shape for
    the fake model above to key off of - real audio/text arrays never
    reach a real model in these tests."""

    def __call__(self, audio=None, text=None, sampling_rate=None, return_tensors=None, padding=None):
        if audio is not None:
            return {"clip_lengths": [len(clip) for clip in audio]}
        return {"texts": list(text)}


def test_embed_audio_clips_returns_empty_list_for_empty_input():
    assert audio_embedder.embed_audio_clips([]) == []


def test_embed_audio_clips_returns_one_vector_per_clip(monkeypatch):
    monkeypatch.setattr(audio_embedder, "_get_model", lambda: _FakeClapModel())
    monkeypatch.setattr(audio_embedder, "_get_processor", lambda: _FakeClapProcessor())

    clips = [np.zeros(100), np.zeros(200)]
    vectors = audio_embedder.embed_audio_clips(clips)

    assert len(vectors) == 2
    assert vectors[0] == [100.0, 0.0]
    assert vectors[1] == [200.0, 0.0]


def test_embed_text_for_audio_search_uses_the_same_clap_model(monkeypatch):
    monkeypatch.setattr(audio_embedder, "_get_model", lambda: _FakeClapModel())
    monkeypatch.setattr(audio_embedder, "_get_processor", lambda: _FakeClapProcessor())

    vector = audio_embedder.embed_text_for_audio_search("hello")

    assert vector == [0.0, 5.0]

"""
Native audio embedding for acoustic similarity search.

Responsibility: convert an already-decoded audio waveform, or a text
query meant to search for audio, into a vector in the SAME embedding
space - so a text query like "sounds like a car engine" can be compared
against stored audio-clip embeddings by plain cosine similarity. No
decoding, chunking, retrieval, or storage happens here.

Why this exists: the platform's existing audio/video ingestion
(app.ingestion.loader.load_audio_document/load_video_document)
transcribes speech to TEXT (via Whisper) and embeds that text with the
ordinary all-MiniLM-L6-v2 model - which only ever answers "what was
SAID", never "what does this SOUND like" (music, tone, background
noise, non-speech audio). This module is a second, independent
retrieval path for that latter question - added alongside, not
replacing, the transcript pipeline.

Design:
- Provider used: CLAP (Contrastive Language-Audio Pretraining, LAION),
  via `transformers.ClapModel`/`ClapProcessor` - reuses this project's
  EXISTING transformers/torch dependency (already installed for
  sentence-transformers' text/CLIP models) rather than adding a
  separate, heavier audio-ML dependency tree (e.g. the standalone
  `laion-clap` package and its own dependencies) not otherwise needed
  here.
- Unlike app.embeddings.image_embedder's CLIP wrapper (a single
  SentenceTransformer object bundling model+preprocessing), raw
  ClapModel/ClapProcessor are two separate objects - two lazy
  singletons, not one.
- This MUST stay a separate model (and, in app.vectorstore.store, a
  separate collection) from BOTH app.embeddings.embedder (MiniLM text)
  AND app.embeddings.image_embedder (CLIP images/text) - CLAP's vector
  space has nothing in common with either. An audio clip embedded here
  can only ever be meaningfully compared against OTHER vectors from
  THIS model.
- embed_audio_clips() takes ALREADY-DECODED, already-resampled mono
  float32 waveforms at CLAP_SAMPLE_RATE (48kHz - CLAP's required
  input rate), mirroring embed_images() taking already-decoded PIL
  images - decoding/segmentation is app.ingestion.loader's job (see
  _sample_audio_clips), not this module's.
- Two real API quirks in the installed transformers version, verified
  live against a real audio file before writing this module (not
  guessed): ClapProcessor's audio keyword argument is `audio=`, not
  the more commonly-documented `audios=` (raises ValueError if used);
  and get_audio_features()/get_text_features() return a
  BaseModelOutputWithPooling, not a bare tensor - the actual embedding
  is `.pooler_output`.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48_000

_model = None
_processor = None


def _get_model():
    """Lazily load the local CLAP model, so importing this module is
    cheap and the (one-time, several-hundred-MB) model download only
    happens when an audio embedding call is actually attempted."""
    global _model
    if _model is None:
        from transformers import ClapModel

        logger.info("Loading local acoustic (CLAP) model '%s'...", _CLAP_MODEL_NAME)
        _model = ClapModel.from_pretrained(_CLAP_MODEL_NAME)
        logger.info("Acoustic model loaded.")
    return _model


def _get_processor():
    """Lazily load CLAP's processor (feature extraction + tokenizer) - kept
    separate from _get_model() since transformers.ClapModel/ClapProcessor
    are two independent objects, unlike sentence-transformers' bundled
    CLIP wrapper (see module docstring)."""
    global _processor
    if _processor is None:
        from transformers import ClapProcessor

        _processor = ClapProcessor.from_pretrained(_CLAP_MODEL_NAME)
    return _processor


def embed_audio_clips(clips: list[np.ndarray]) -> list[list[float]]:
    """
    Embed a batch of already-decoded, already-resampled (48kHz, mono,
    float32) audio waveforms with the local CLAP model.

    Args:
        clips: Decoded waveforms (e.g. from
            app.ingestion.loader._sample_audio_clips).

    Returns:
        A list of embedding vectors, in the same order as `clips`.
    """
    if not clips:
        return []

    import torch

    model = _get_model()
    processor = _get_processor()
    inputs = processor(audio=clips, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        vectors = model.get_audio_features(**inputs).pooler_output
    logger.info("Embedded %d audio clip(s) (dimension=%d)", len(clips), vectors.shape[1])
    return vectors.tolist()


def embed_text_for_audio_search(query_text: str) -> list[float]:
    """
    Embed a natural-language query with the SAME CLAP model used for
    stored audio clips, so it can be compared against them by cosine
    similarity (see module docstring - only works because both sides
    use this one model; never mix this with
    app.embeddings.embedder's or app.embeddings.image_embedder's
    vectors).
    """
    import torch

    model = _get_model()
    processor = _get_processor()
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        vector = model.get_text_features(**inputs).pooler_output[0]
    return vector.tolist()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.embeddings.audio_embedder <query text>")
        raise SystemExit(1)

    embedding = embed_text_for_audio_search(" ".join(sys.argv[1:]))
    print(f"dimension={len(embedding)}, first 5 values={embedding[:5]}")

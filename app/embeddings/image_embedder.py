"""
Multimodal (image) embedding for the RAG pipeline.

Responsibility: convert a downloaded image, or a text query meant to
search for images, into a vector in the SAME embedding space - so a
text query like "authentication flow diagram" can be compared against
stored image embeddings by plain cosine similarity. No chunking,
retrieval, or storage happens here.

Design:
- Provider used: sentence-transformers `clip-ViT-B-32` - a CLIP model
  wrapped for the sentence-transformers API, so it slots into the same
  "load a model once, call .encode()" pattern as
  app.embeddings.embedder, but happens to accept BOTH PIL images and
  text strings and map them into ONE shared vector space. That shared
  space is what makes it "multimodal" - unlike all-MiniLM-L6-v2, which
  only ever understands text.
- This MUST stay a separate model (and, in app.vectorstore.store, a
  separate collection) from app.embeddings.embedder: CLIP's vector
  space has nothing in common with MiniLM's, so an image embedded here
  can only ever be meaningfully compared against OTHER vectors from
  THIS model (either other images, or a query embedded with
  embed_text_for_image_search() below) - never against a MiniLM
  text-chunk embedding.
- Images are fetched over HTTP and decoded with Pillow here, rather
  than requiring the caller to already have image bytes, so
  app.ingestion.ingest can just pass image URLs straight through (the
  same URLs app.ingestion.loader.extract_image_records() already
  parsed out of a crawled page). A broken/unreachable image URL logs a
  warning and is skipped rather than raising, so one dead image link
  can't break ingestion of every other image/document - the same
  "one bad source can't break everything else" pattern used
  throughout this project (see loader.py's crawl_website()).
"""

import logging
from io import BytesIO

import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_CLIP_MODEL_NAME = "clip-ViT-B-32"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazily load the local CLIP model, so importing this module is
    cheap and the (one-time, several-hundred-MB) model download only
    happens when an image embedding call is actually attempted.
    """
    global _model
    if _model is None:
        logger.info("Loading local multimodal (CLIP) model '%s'...", _CLIP_MODEL_NAME)
        _model = SentenceTransformer(_CLIP_MODEL_NAME)
        logger.info("Multimodal model loaded.")
    return _model


def _download_image(image_url: str, timeout: int = 10) -> Image.Image | None:
    """
    Download and decode an image from a URL.

    Returns None (after logging a warning) instead of raising if the
    image can't be fetched or decoded, so one broken image link
    doesn't break embedding of every other image.
    """
    try:
        response = requests.get(
            image_url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAGIngestBot/1.0)"},
        )
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as error:
        logger.warning("Skipping image '%s' (could not download/decode): %s", image_url, error)
        return None


def download_image(image_url: str, timeout: int = 10) -> Image.Image | None:
    """
    Public wrapper around _download_image, for callers outside this
    module that need the raw decoded image itself rather than an
    embedding vector - e.g. app.ingestion.ingest downloads each image
    ONCE and passes the same decoded Image to both embed_images() below
    (for CLIP similarity search) and app.ocr.image_ocr.extract_text_from_image()
    (to pull out any code/text rendered inside the image), instead of
    downloading the same URL twice.
    """
    return _download_image(image_url, timeout=timeout)


def embed_images(images: list[Image.Image]) -> list[list[float]]:
    """
    Embed a batch of already-downloaded Pillow images with the local
    CLIP model.

    Args:
        images: Decoded images (e.g. from download_image()).

    Returns:
        A list of embedding vectors, in the same order as `images`.
    """
    if not images:
        return []

    model = _get_model()
    vectors = model.encode(images, convert_to_numpy=True, show_progress_bar=False)
    logger.info("Embedded %d image(s) (dimension=%d)", len(images), vectors.shape[1])
    return [vector.tolist() for vector in vectors]


def embed_image_urls(image_urls: list[str]) -> dict[str, list[float]]:
    """
    Download and embed a batch of image URLs with the local CLIP model.

    Args:
        image_urls: Image URLs to fetch and embed.

    Returns:
        A dict mapping each successfully-embedded URL to its embedding
        vector. A URL that failed to download/decode is simply absent
        from the result rather than raising, so callers only need to
        handle "was this URL embedded or not" rather than catching
        errors themselves.
    """
    if not image_urls:
        return {}

    images: list[Image.Image] = []
    valid_urls: list[str] = []
    for image_url in image_urls:
        image = _download_image(image_url)
        if image is not None:
            images.append(image)
            valid_urls.append(image_url)

    if not images:
        return {}

    vectors = embed_images(images)
    return dict(zip(valid_urls, vectors))


def embed_text_for_image_search(query_text: str) -> list[float]:
    """
    Embed a natural-language query with the SAME CLIP model used for
    stored images, so it can be compared against them by cosine
    similarity (see module docstring - this only works because both
    sides use this one model; never mix this with
    app.embeddings.embedder's all-MiniLM-L6-v2 vectors).
    """
    model = _get_model()
    vector = model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)[0]
    return vector.tolist()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.embeddings.image_embedder <image_url> [<image_url> ...]")
        raise SystemExit(1)

    embeddings_by_url = embed_image_urls(sys.argv[1:])
    for url, vector in embeddings_by_url.items():
        print(f"{url}: dimension={len(vector)}, first 5 values={vector[:5]}")

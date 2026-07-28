"""
Embedding for the RAG pipeline.

Responsibility: convert each Chunk's text into a fixed-length numeric
vector (an embedding) that captures its meaning. No retrieval or LLM
generation happens here.

Design:
- embed_texts() wraps the actual embedding call in ONE place, so
  swapping embedding providers later (an API-based model, a different
  local model, etc.) means changing this one function, not every caller.
- Chunks are embedded in a single batch call where possible, since
  the model accepts a list of texts at once — more efficient than
  embedding one chunk at a time.
- No API key, no network call, no billing: the model runs locally.

Provider used: sentence-transformers `all-MiniLM-L6-v2` (384 dimensions).
This is a free, open-source model that runs entirely on your machine —
no API key or internet connection needed after the first download.
See data/raw/README.md section 8 for a comparison of embedding models.
"""

import logging
from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer

from app.chunking.chunker import Chunk

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


@dataclass
class EmbeddedChunk:
    """A Chunk paired with its embedding vector."""

    content: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


def _get_model() -> SentenceTransformer:
    """
    Lazily load the local embedding model, so importing this module is
    cheap and the (one-time, ~90MB) model download only happens when an
    embedding call is actually attempted.
    """
    global _model
    if _model is None:
        logger.info("Loading local embedding model '%s'...", _EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of raw text strings using the local model.

    Args:
        texts: List of text strings to embed.

    Returns:
        A list of embedding vectors, in the same order as `texts`.
    """
    if not texts:
        return []

    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    logger.info("Embedded %d text(s) (dimension=%d)", len(texts), vectors.shape[1])
    return [vector.tolist() for vector in vectors]


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """
    Embed a list of Chunks, returning EmbeddedChunk objects that carry
    both the original chunk content/metadata and its embedding vector.

    Args:
        chunks: Chunks produced by app.chunking.chunker.chunk_document.

    Returns:
        A list of EmbeddedChunk, one per input chunk, same order.
    """
    if not chunks:
        return []

    texts = [chunk.content for chunk in chunks]
    embeddings = embed_texts(texts)

    return [
        EmbeddedChunk(
            content=chunk.content,
            embedding=embedding,
            metadata=dict(chunk.metadata),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]


if __name__ == "__main__":
    from app.chunking.chunker import chunk_document
    from app.ingestion.loader import load_markdown_document

    doc = load_markdown_document("data/raw/README.md")
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=30)

    print(f"Embedding {len(chunks)} chunks using local model "
          f"'{_EMBEDDING_MODEL_NAME}'...")
    embedded_chunks = embed_chunks(chunks)

    print(f"Done. Produced {len(embedded_chunks)} embeddings.")
    first = embedded_chunks[0]
    print(f"\nVector dimension: {len(first.embedding)}")
    print(f"First 5 values of chunk 0's embedding: {first.embedding[:5]}")
    print(f"Chunk 0 section: {first.metadata.get('section')}")

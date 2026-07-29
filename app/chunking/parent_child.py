"""
Parent-child chunk expansion.

Turns a flat list of "parent-size" chunks (the ~150-token output of
app.chunking.chunker.chunk_document) into a flat list of BOTH parents
and much smaller "child" sentence windows, tagged with:

    chunk_role:          "parent" | "child"
    parent_chunk_index:  (on children) the chunk_index of the parent
                         this child was carved out of

Why:
- Small children (~40 words each) let dense/BM25 retrievers hit a
  precise sentence that answers a specific question - way more
  precise than always searching over ~150-token blobs, which get
  diluted by all their off-topic sentences.
- The retriever then swaps each child hit for its full parent so the
  LLM still gets the surrounding paragraph as context, not just the
  matched sentence. This is the "small-to-big" / "sentence-window"
  retrieval pattern.

Storage tradeoff:
- Each document takes roughly 2-4x more Chroma rows than
  parents-only, since one parent ~= 3-6 children. Latency to store is
  proportional; retrieval latency is unaffected (only children are
  searched by the parent-child retriever; dense/BM25 filter to
  chunk_role="parent"). Worth the cost for KBs where precision on
  specific facts matters (github source code, web page prose) and
  not for KBs where per-page retrieval already works well (PDF,
  DOCX, Markdown - see hybrid_retriever._HYBRID_KBS).
"""

import re
from copy import deepcopy

from app.chunking.chunker import Chunk, chunk_document
from app.ingestion.loader import Document

# Which source types get parent-child expansion at ingest time (see
# module docstring). Kept as a set so extending to another KB later is
# a one-line change. "code" is the source_type set by
# app.ingestion.loader.load_github_repository for GitHub-ingested
# files; "web" is set for both single-page load_website_document and
# crawl_website results.
_PARENT_CHILD_SOURCE_TYPES = {"code", "web"}

# Which KBs the retriever should treat as parent-child stores (see
# app.retrieval.hybrid_retriever). The KB name is what
# app.vectorstore.store maps each source_type to via _SOURCE_TYPE_TO_KB;
# these two sets are conceptually linked, so both live here.
_PARENT_CHILD_KBS = {"github", "web"}

# Chunks with these content_types are already fine-grained by
# structure (AST classes/methods/functions, whole tables, individual
# table rows) - splitting them into 40-word sentence windows would
# just fragment coherent structural units into arbitrary pieces
# without adding retrieval precision. They still get chunk_role=parent
# so dense + BM25 see them, but no children are generated for them.
_SKIP_CHILDREN_CONTENT_TYPES = {"class", "method", "function", "table", "table_row"}

# Sentence splitter: crude but reliable across markdown/html/plain
# prose. Splits after a sentence-ending punctuation followed by
# whitespace. Doesn't handle code (no punctuation in the CS sense) or
# lists specifically, but the greedy window aggregation below cleans
# up short/orphan fragments.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Target size for each child window, in whitespace-split words. 40 is
# ~50-60 tokens - small enough that a single query sentence matches
# strongly on cosine similarity/BM25 term overlap, big enough that a
# child chunk contains a self-contained thought (not a 5-word
# fragment). Sentence boundaries take precedence over the target -
# we'll overshoot rather than cut mid-sentence.
_CHILD_TARGET_WORDS = 40

# Drop trailing child fragments below this many words - they're
# usually orphan punctuation ("Yes.") or repeated headers left over
# from the parent's whitespace, adding storage cost without giving
# retrieval anything meaningful to match on.
_MIN_CHILD_WORDS = 5


def _sentence_windows(text: str) -> list[str]:
    """
    Split text into sentence windows of ~_CHILD_TARGET_WORDS words each.

    Aggregates greedily: adds one sentence at a time until the running
    window would exceed the target, then emits it and starts a new
    window. Never splits within a sentence.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    windows: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > _CHILD_TARGET_WORDS:
            windows.append(" ".join(current))
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words

    if current and current_words >= _MIN_CHILD_WORDS:
        windows.append(" ".join(current))
    # If the ONLY window we built is below _MIN_CHILD_WORDS, still
    # emit it - we'd rather have a small child chunk than lose the
    # parent entirely to the retriever.
    elif current and not windows:
        windows.append(" ".join(current))

    return windows


def expand_to_parent_child(chunks: list[Chunk]) -> list[Chunk]:
    """
    Given chunks produced by chunk_document (which we treat as
    parents), return a new flat list containing:

        [parent_0, parent_1, ..., parent_N,
         child_0_of_parent_0, child_1_of_parent_0, ...,
         child_0_of_parent_1, ...]

    Every returned chunk is annotated with `chunk_role`; children
    additionally carry `parent_chunk_index` pointing at their source
    parent's `chunk_index`. Parents keep their original chunk_index
    unchanged; children are re-numbered starting from len(parents) so
    the (kb, document_id, chunk_index) tuple stays globally unique.

    Deep-copies parent metadata so downstream mutations on children
    don't leak back into parents.
    """
    parents: list[Chunk] = []
    for parent in chunks:
        annotated = Chunk(
            content=parent.content,
            metadata={**deepcopy(parent.metadata), "chunk_role": "parent"},
        )
        parents.append(annotated)

    children: list[Chunk] = []
    child_index = len(parents)
    for parent in chunks:
        parent_index = parent.metadata.get("chunk_index")
        if parent.metadata.get("content_type") in _SKIP_CHILDREN_CONTENT_TYPES:
            # Already-structured chunks (code AST units, tables) are
            # fine-grained by their own semantics; sentence-window
            # splitting would just fragment them uselessly.
            continue
        for child_text in _sentence_windows(parent.content):
            child = Chunk(
                content=child_text,
                metadata={
                    **deepcopy(parent.metadata),
                    "chunk_role": "child",
                    "chunk_index": child_index,
                    "parent_chunk_index": parent_index,
                },
            )
            children.append(child)
            child_index += 1

    return parents + children


def chunk_with_parent_child(document: Document, **chunk_kwargs) -> list[Chunk]:
    """
    Drop-in replacement for chunk_document() that ALSO runs parent-
    child expansion when the document's source_type is in
    _PARENT_CHILD_SOURCE_TYPES.

    Every ingest path in the codebase should call this instead of
    chunk_document() directly, so a document destined for a
    parent-child-aware KB gets both parent chunks (for the ensemble's
    dense + BM25 retrievers, unchanged behavior) AND child chunks
    (for the new sentence-window retriever) in one pass. Documents
    for other KBs (PDF/DOCX/Markdown) pass through untouched.

    **chunk_kwargs are forwarded to chunk_document (chunk_size,
    chunk_overlap) so callers don't have to know which underlying
    function to configure.
    """
    parents = chunk_document(document, **chunk_kwargs)
    if document.metadata.get("source_type") in _PARENT_CHILD_SOURCE_TYPES:
        return expand_to_parent_child(parents)
    return parents

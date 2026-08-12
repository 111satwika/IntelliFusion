"""
Semantic near-duplicate answer cache.

The exact-match cache in app.generation.llm_generator only helps a
byte-for-byte repeat question (same prompt string in, same answer
out). A REPHRASED repeat ("what embedding model handles text here" vs
"which model does this platform use for text embeddings") still pays
the full ~1-3 minute CPU generation cost, since different phrasing
embeds differently, retrieves at least slightly different chunks, and
so produces a different prompt. This module closes that gap - but a
similarity-based cache carries a real risk a plain hash-match cache
doesn't: a false-positive match returns a WRONG answer, not just a
slow one. Every design choice below exists to contain that risk.

Core mechanism: retrieval + CRAG still run normally (cheap, ~2-10s) to
get the REAL chunk set a fresh answer would use right now. Only
GENERATION (the expensive part) is skipped, and only when a past entry
passes TWO independent checks at once:
  1. The resolved question's embedding is highly similar to the
     entry's question embedding (cosine).
  2. The freshly-retrieved chunk set substantially OVERLAPS the
     entry's chunk set (overlap coefficient |A∩B|/min(|A|,|B|), not
     Jaccard - Jaccard penalizes benign size asymmetry, e.g. CRAG
     dropping one chunk on one run but not the other, even when the
     smaller set sits entirely inside the larger one).
Requiring BOTH is the actual safety mechanism: a question phrased
similarly but genuinely about something else will retrieve different
chunks, so the overlap check catches what similarity alone would miss.

Chunk identity is (document_id, chunk_index, sha256(content)[:16]) -
NOT bare (document_id, chunk_index). Re-ingesting a document upserts
by that same id (see app.vectorstore.store), so the identical id pair
can point at different text after a re-ingest/edit; folding a content
hash into identity makes a cache entry self-invalidate the moment its
underlying evidence actually changes, entirely locally - no wiring
into store.py's ingest/delete paths needed.

Callers (cli.py) are additionally responsible for two further gates
this module has no visibility into on its own:
  - Only call with history-free (first-turn-of-thread) requests. A
    thread's prior turns get rendered into the generation prompt
    regardless of any cache decision (see app.prompting.prompt_builder),
    so two requests could pass both checks above while still differing
    by an entire conversation-history block baked into the actual
    prompt.
  - Never call for a request that will attach images to a vision
    model - vision_image_urls comes from a completely separate CLIP
    embedding space with no representation in the text-chunk-overlap
    check.

Rollout: this is a module-global flag (the cache itself is a shared
resource - a hit populated by one session should benefit every
session, unlike the per-session RAG-quality toggles in session_store.py),
seeded once from the SEMANTIC_ANSWER_CACHE_ENABLED environment
variable, default OFF. No per-session UI control - see this module's
lack of a session_store hook; the substitute for user-level control is
disclosure (see insight.py's "served_from_cache" field), not a
per-session escape hatch.

Thresholds were empirically tuned against the real all-MiniLM-L6-v2
model and real ingested content (5 genuine paraphrase pairs about this
platform vs. 4 topically-similar-but-genuinely-different pairs, run
through the real embed_texts() + retrieve() at top_k=3) - same method
the KB-routing threshold in app.routing.router went through (a guessed
0.4 needed corpus-verified retuning to 0.2). Findings that shaped the
numbers below:
  - Question-embedding similarity ALONE is not reliably discriminative
    at this scale: a genuine paraphrase pair scored as low as 0.554,
    while a genuinely-different pair ("what is CRAG" vs "what is
    Self-RAG") scored 0.733 - higher than several true paraphrases.
  - The chunk-overlap coefficient is the actually load-bearing safety
    signal: every distinct-topic pair scored overlap=0.000 (top_k=3
    genuinely different questions retrieve genuinely different
    evidence), while genuine paraphrases that happened to retrieve
    shared evidence scored 0.667-1.000. No distinct-topic pair in this
    sample ever produced a false-positive overlap score.
  - Consequence: some genuine paraphrases legitimately MISS the cache
    when the two phrasings happen to pull disjoint top-3 chunks (seen
    for 2/5 pairs here) - an accepted conservative miss, not a bug:
    if the retrieved evidence actually differs, reusing the old answer
    would not be safe regardless of question-similarity.
Both thresholds are set just below the weakest genuine-paraphrase
score observed (similarity 0.609, overlap 0.667) while sitting above
every distinct-topic overlap score (always 0.000) - re-verify if the
corpus or embedding model changes meaningfully.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass

_MAX_CHUNK_PREVIEW_CHARS = 500  # bound how much content feeds the identity hash, mirrors crag.py's own preview bound

_SEMANTIC_CACHE_MAXSIZE = 128  # matches llm_generator._ANSWER_CACHE_MAXSIZE

# Empirically tuned against the real embedding model + real ingested
# content - see module docstring for the exact measurements.
_SIMILARITY_THRESHOLD = 0.6
_OVERLAP_THRESHOLD = 0.6

_enabled = os.environ.get("SEMANTIC_ANSWER_CACHE_ENABLED", "") == "1"

_cache: list["SemanticCacheEntry"] = []


@dataclass
class SemanticCacheEntry:
    question: str  # resolved/contextualized form, kept for observability/debugging only
    question_embedding: list[float]
    chunk_ids: frozenset[tuple[str, int, str]]
    kb: str | None
    repository: str | None
    answer: str
    owner: str  # closes a narrow cross-user gap: without this, two owners
    # who happen to have ingested byte-identical content under the same
    # document_id (e.g. both scrape the same public URL) could match
    # each other's cache entries via the (kb, repository) + chunk-overlap
    # checks alone - explicit owner scoping removes that edge case
    # entirely rather than relying on it being emergently rare.


def set_enabled(enabled: bool) -> None:
    """Turn the semantic cache on/off process-wide."""
    global _enabled
    _enabled = bool(enabled)


def is_enabled() -> bool:
    return _enabled


def clear_cache() -> None:
    """Wipe every cached entry."""
    _cache.clear()


def find_cached_answer(
    question_embedding: list[float],
    chunks: list[dict],
    kb: str | None,
    repository: str | None,
    owner: str,
    *,
    enabled: bool | None = None,
) -> str | None:
    """
    Return a previously-cached answer if some entry passes both the
    question-similarity and chunk-overlap checks, else None.

    ``enabled`` overrides the process-global flag for this call only
    (matches every other enable/disable toggle in this codebase);
    omit it to fall back to ``is_enabled()``.

    Scoped to entries with an EXACT (kb, repository, owner) match
    before any fuzzy comparison runs - a question about one repo must
    never serve an answer scoped to a different one, and (as of the
    owner field) one account must never serve an answer generated for
    a different account, even if both happen to have ingested
    identical content under the same kb/repository. Among the
    remaining candidates, returns the answer of whichever entry has
    the highest question-similarity among those that clear BOTH
    thresholds.
    """
    effective_enabled = _enabled if enabled is None else enabled
    if not effective_enabled or not chunks or not _cache:
        return None

    candidate_ids = _chunk_identity(chunks)
    best_score = -1.0
    best_answer: str | None = None
    for entry in _cache:
        if entry.kb != kb or entry.repository != repository or entry.owner != owner:
            continue
        similarity = _cosine_similarity(question_embedding, entry.question_embedding)
        if similarity < _SIMILARITY_THRESHOLD:
            continue
        overlap = _overlap_coefficient(candidate_ids, entry.chunk_ids)
        if overlap < _OVERLAP_THRESHOLD:
            continue
        if similarity > best_score:
            best_score = similarity
            best_answer = entry.answer
    return best_answer


def store_answer(
    question: str,
    question_embedding: list[float],
    chunks: list[dict],
    kb: str | None,
    repository: str | None,
    answer: str,
    owner: str,
) -> None:
    """
    Record a freshly-generated answer for future near-duplicate hits.
    No-op when the module is disabled, there are no chunks to key on,
    or the answer is empty - only a complete, successful answer is
    worth caching (same principle llm_generator's exact-match cache
    already follows).
    """
    if not _enabled or not chunks or not answer:
        return
    entry = SemanticCacheEntry(
        question=question,
        question_embedding=list(question_embedding),
        chunk_ids=_chunk_identity(chunks),
        kb=kb,
        repository=repository,
        answer=answer,
        owner=owner,
    )
    _cache.append(entry)
    while len(_cache) > _SEMANTIC_CACHE_MAXSIZE:
        _cache.pop(0)  # FIFO eviction - matching is fuzzy, not a single exact key, so LRU-on-hit doesn't apply cleanly here


# ---------- Internals ----------


def _chunk_identity(chunks: list[dict]) -> frozenset[tuple[str, int, str]]:
    """
    A chunk set's identity for overlap comparison: (document_id,
    chunk_index, content_hash) per chunk, not bare (document_id,
    chunk_index) - see module docstring for why the content hash
    matters (re-ingestion upserts by the same id, so a bare-id match
    could count stale evidence as "the same").
    """
    identities = set()
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        document_id = metadata.get("document_id") or "?"
        chunk_index = metadata.get("chunk_index")
        content_preview = (chunk.get("content") or "")[:_MAX_CHUNK_PREVIEW_CHARS]
        content_hash = hashlib.sha256(content_preview.encode("utf-8")).hexdigest()[:16]
        identities.add((str(document_id), chunk_index, content_hash))
    return frozenset(identities)


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Same formula as app.routing.router._cosine_similarity -
    duplicated rather than imported, matching this codebase's existing
    precedent of small, self-contained math helpers per module (see
    also app.routing.github_intent)."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _overlap_coefficient(a: frozenset, b: frozenset) -> float:
    """|A ∩ B| / min(|A|, |B|) - see module docstring for why this
    instead of Jaccard."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))

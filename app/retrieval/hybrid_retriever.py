"""
Hybrid retrieval (dense + BM25 fused with RRF, then cross-encoder
reranked) for text KBs whose corpora reward BOTH lexical and semantic
matching. Currently applied to PDF, DOCX, and Markdown; see
_HYBRID_KBS below.

Why hybrid for these KBs:
- Dense (bi-encoder cosine) retrieval is strong on paraphrase and
  semantic overlap but weak on exact terms - a report that mentions
  "Q3 2024 EBITDA" verbatim can rank below a paraphrased chunk that
  says roughly the same thing about a different quarter.
- BM25 (sparse, lexical) is the opposite: strong on exact terms and
  rare tokens (numbers, acronyms, table headers, product names) that
  dense embeddings often smear together, weak on paraphrase.
- Cross-encoder reranking then re-scores the fused top candidates by
  actually attending over (query, chunk) TOGETHER, not just comparing
  independent embeddings - much more accurate than bi-encoder cosine
  on a small pool, at the cost of being too slow to run over the full
  KB (why it comes last, on a shortlist).

Pipeline (per query):
  1. Dense retrieval  → top-N candidates with cosine similarity.
  2. BM25 retrieval   → top-N candidates over the same KB corpus,
                        scored by token overlap (rank_bm25.BM25Okapi).
  3. Reciprocal Rank Fusion combines the two ranked lists into one
     candidate pool (see _reciprocal_rank_fusion).
  4. Cross-encoder ('cross-encoder/ms-marco-MiniLM-L-6-v2') re-scores
     the top RRF candidates by attending over (query, chunk_text).
  5. Return the top_k final hits.

BM25 index lifecycle:
- Built lazily on first query per KB: fetch every chunk in that KB via
  collection.get(include=["documents", "metadatas"]) and index the
  documents. Cached in-process (module-level, keyed by KB name).
- Invalidated whenever a KB's chunks are added/deleted (see
  invalidate_bm25_cache, called from app.vectorstore.store's ingest
  and delete paths). Any next query rebuilds only that KB's index.
- Rebuild cost scales linearly with |KB corpus|; on typical
  thousands-of-chunks KBs this is ~50-300ms, well under the LLM
  cost that follows.

Cross-encoder lifecycle:
- Loaded lazily on first use, cached module-globally. First load
  downloads ~90MB from HuggingFace (one-time), subsequent loads
  from local cache take ~1-2s.
- Scoring 20 (query, chunk) pairs on CPU takes ~100-300ms.
"""

import logging
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.embeddings.embedder import embed_texts
from app.vectorstore.store import _get_collection, query_embedding

logger = logging.getLogger(__name__)


# Which KBs get the hybrid dense+BM25+cross-encoder strategy. Kept as
# a module-level set instead of hard-coded string checks so extending
# to another prose-heavy KB is a one-line change. GitHub and Web
# additionally layer a parent-child sentence-window retriever on top
# (see _PARENT_CHILD_KBS).
_HYBRID_KBS = {"pdf", "docx", "markdown", "github", "web", "audio", "video"}

# Which KBs store parent+child chunks (see app.chunking.parent_child).
# Retrieval for these KBs adds a third ranked list to the RRF fusion:
# dense retrieval over CHILDREN, resolved back to their parents. Also
# forces dense + BM25 to filter to chunk_role="parent" so they don't
# accidentally return sentence fragments.
_PARENT_CHILD_KBS = {"github", "web"}

# How many candidates each of the two retrievers contributes before
# RRF. Deliberately wider than the final top_k: RRF's whole point is
# that a document ranked #40 by dense but #3 by BM25 (or vice versa)
# should still make it into the fused shortlist, so both pools need
# to be wide enough for that overlap to happen.
_DENSE_POOL_SIZE = 40
_BM25_POOL_SIZE = 40

# How many child hits the parent-child retriever pulls before
# resolving them to parents and deduplicating. Roughly aligned with
# dense/BM25 pool sizes so the third ranked list contributes
# comparable weight to RRF, but child hits collapse ~3-6x to parents
# during resolution so the actual parent count entering fusion is
# closer to _DENSE_POOL_SIZE / 4.
_PARENT_CHILD_POOL_SIZE = 60

# How many RRF-fused candidates get sent to the cross-encoder. Small
# enough to keep cross-encoder latency bounded (~200ms on CPU), large
# enough that a correct-but-not-top-ranked chunk still gets a chance
# to be reranked into the final top_k.
_RERANK_POOL_SIZE = 20

# RRF constant (see _reciprocal_rank_fusion). 60 is the value from the
# original RRF paper (Cormack et al. 2009); it damps the top-rank
# advantage enough that a document appearing in BOTH ranked lists at
# rank ~10 beats one appearing at #1 in only one list.
_RRF_K = 60

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---- BM25 index (module-cached) --------------------------------------


class _BM25Index:
    """A BM25Okapi index over one KB, plus the parallel arrays needed
    to map BM25 scores back to (content, metadata) chunk dicts, and
    the tokenized corpus so token-overlap filtering can distinguish
    "matches but got IDF=0" from "doesn't match at all"."""

    def __init__(
        self,
        contents: list[str],
        metadatas: list[dict],
        bm25: BM25Okapi,
        tokenized_corpus: list[list[str]],
    ):
        self.contents = contents
        self.metadatas = metadatas
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus


_bm25_index_by_kb: dict[str, _BM25Index | None] = {}


def _bm25_tokenize(text: str) -> list[str]:
    """Simple lowercase alnum tokenization. Same signal BM25 needs
    (word-level term frequencies) without pulling in a heavy
    tokenizer. Deliberately keeps stopwords: BM25's IDF weighting
    already down-weights common terms, and dropping them here would
    also drop numbers/acronyms whose length passes the >1 char guard."""
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 1]


def _build_bm25_index(kb: str) -> _BM25Index | None:
    """Fetch every chunk in the KB and build a fresh BM25 index over
    them. Returns None if the KB is empty (no chunks to index yet).

    For parent-child KBs (see _PARENT_CHILD_KBS), only PARENT chunks
    are indexed - children are sentence fragments meant only for the
    parent-child retriever's precision-first path, and adding them to
    BM25 would double-count every parent's text (once in the parent,
    once in each child slice).
    """
    collection = _get_collection(kb)
    if kb in _PARENT_CHILD_KBS:
        result = collection.get(
            where={"chunk_role": "parent"}, include=["documents", "metadatas"]
        )
    else:
        result = collection.get(include=["documents", "metadatas"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    if not documents:
        return None

    tokenized_corpus = [_bm25_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("Built BM25 index for KB '%s' over %d chunk(s)", kb, len(documents))
    return _BM25Index(documents, metadatas, bm25, tokenized_corpus)


def _get_bm25_index(kb: str) -> _BM25Index | None:
    """Return the cached BM25 index for kb, building it if absent.
    Returns None if the KB has no chunks."""
    if kb not in _bm25_index_by_kb:
        _bm25_index_by_kb[kb] = _build_bm25_index(kb)
    return _bm25_index_by_kb[kb]


def invalidate_bm25_cache(kb: str | None = None) -> None:
    """Drop the cached BM25 index so the next query rebuilds it.

    Called from the vectorstore's ingest / delete paths whenever the
    underlying chunks change. Passing None invalidates every cached
    KB (used by the delete-all-KBs paths).
    """
    global _bm25_index_by_kb
    if kb is None:
        _bm25_index_by_kb = {}
    else:
        _bm25_index_by_kb.pop(kb, None)


# ---- Cross-encoder (module-cached) -----------------------------------


_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    """Lazily load the cross-encoder once per process."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder '%s'...", _CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


# ---- Retrieval components --------------------------------------------


def _dense_search(
    query_vector: list[float], kb: str, top_k: int, where: dict | None
) -> list[dict]:
    """Thin wrapper: same as retriever.py's dense retrieval, but scoped
    to a single KB and returned as a ranked list (already ordered).

    For parent-child KBs, an extra where={"chunk_role": "parent"}
    clause is folded in so dense doesn't accidentally rank sentence
    fragments alongside their parents - children are the
    parent-child retriever's job, not this one's.
    """
    effective_where = _add_parent_role_filter(where, kb)
    return query_embedding(query_vector, top_k=top_k, where=effective_where, kb=kb)


def _add_parent_role_filter(where: dict | None, kb: str) -> dict | None:
    """Fold a chunk_role="parent" clause into an existing where filter
    when kb stores parent+child chunks. Preserves any other clauses
    the caller provided (e.g. repository scope).
    """
    if kb not in _PARENT_CHILD_KBS:
        return where
    role_clause = {"chunk_role": "parent"}
    if where is None:
        return role_clause
    if "$and" in where:
        return {"$and": [*where["$and"], role_clause]}
    # Wrap a flat where + the role clause under $and so Chroma parses
    # them as a conjunction rather than a single ambiguous filter.
    return {"$and": [where, role_clause]}


def _bm25_search(query_text: str, kb: str, top_k: int, where: dict | None) -> list[dict]:
    """Rank chunks by BM25 score against the query, optionally filtered
    by metadata (repository, content_type, etc.) to keep BM25's results
    consistent with the dense retriever's where-filter behavior."""
    index = _get_bm25_index(kb)
    if index is None:
        return []

    tokenized_query = _bm25_tokenize(query_text)
    if not tokenized_query:
        return []

    scores = index.bm25.get_scores(tokenized_query)
    query_tokens = set(tokenized_query)

    # Pair every scored doc with its metadata, apply the where filter
    # BEFORE sorting/truncating so a filter-matching but slightly
    # lower-scoring chunk isn't accidentally dropped by an unfiltered
    # top-N truncation upstream. Docs that share NO query tokens are
    # excluded (BM25's raw score alone can't tell them apart from
    # "matches but got IDF=0 because the term is corpus-frequent" -
    # explicit token-overlap check makes the intent unambiguous).
    scored = []
    for content, metadata, score, doc_tokens in zip(
        index.contents, index.metadatas, scores, index.tokenized_corpus
    ):
        if not (query_tokens & set(doc_tokens)):
            continue
        if not _matches_where(metadata, where):
            continue
        scored.append(
            {
                "content": content,
                "metadata": metadata,
                "bm25_score": float(score),
                # Distance/similarity are dense-retriever concepts;
                # BM25 doesn't produce them. Kept as None so downstream
                # code that reads hit.get("similarity") still works.
                "distance": None,
                "similarity": None,
            }
        )

    scored.sort(key=lambda hit: hit["bm25_score"], reverse=True)
    return scored[:top_k]


def _parent_child_search(
    query_vector: list[float], kb: str, top_k: int, where: dict | None
) -> list[dict]:
    """
    Sentence-window / "small-to-big" retrieval.

    Dense-searches only the KB's CHILD chunks (small sentence-level
    slices produced at ingest time by app.chunking.parent_child), then
    for each child hit resolves back to its full parent chunk. Multiple
    children pointing at the same parent collapse into a single hit -
    the best-ranked child's rank position is what feeds RRF later, so a
    parent with multiple strong child matches still ranks appropriately
    even after dedup.

    Returned hits carry the PARENT's content and metadata (the LLM
    always sees the full paragraph, not the matched fragment), plus a
    "matched_child_content" field for observability so you can inspect
    which specific sentence triggered the hit.

    Returns [] if the KB isn't parent-child-aware or hasn't been
    ingested with parent-child expansion yet (no chunks tagged
    chunk_role=child).
    """
    if kb not in _PARENT_CHILD_KBS:
        return []

    child_where = _combine_where(where, {"chunk_role": "child"})
    child_hits = query_embedding(query_vector, top_k=top_k, where=child_where, kb=kb)
    if not child_hits:
        return []

    # Group by (document_id, parent_chunk_index) so multiple child hits
    # from the same parent don't produce duplicate parent hits. Keep
    # the child with the highest similarity (i.e. first, since Chroma
    # already returns hits ranked by descending similarity).
    seen: set[tuple] = set()
    parent_keys: list[tuple] = []
    best_child_by_parent: dict[tuple, dict] = {}
    for child in child_hits:
        document_id = child["metadata"].get("document_id")
        parent_chunk_index = child["metadata"].get("parent_chunk_index")
        if document_id is None or parent_chunk_index is None:
            continue
        key = (document_id, parent_chunk_index)
        if key in seen:
            continue
        seen.add(key)
        parent_keys.append(key)
        best_child_by_parent[key] = child

    if not parent_keys:
        return []

    parents = _fetch_parents(kb, parent_keys)
    if not parents:
        return []

    # Materialize parent hits, preserving the child-based ranking order
    # so RRF sees the parents in the same order Chroma returned their
    # children.
    parent_hits: list[dict] = []
    for key in parent_keys:
        parent = parents.get(key)
        if parent is None:
            continue
        best_child = best_child_by_parent[key]
        parent_hits.append(
            {
                "content": parent["content"],
                "metadata": parent["metadata"],
                # No cosine similarity for the parent itself (it wasn't
                # the chunk we scored against); the child's similarity
                # stays available in matched_child_similarity so
                # downstream can inspect it if useful.
                "distance": None,
                "similarity": None,
                "matched_child_content": best_child["content"],
                "matched_child_similarity": best_child.get("similarity"),
            }
        )
    return parent_hits


def _combine_where(base: dict | None, extra: dict) -> dict:
    """Combine an existing where filter with a new clause via $and,
    unwrapping any existing $and so we don't nest them unnecessarily.
    """
    if base is None:
        return extra
    if "$and" in base:
        return {"$and": [*base["$and"], extra]}
    return {"$and": [base, extra]}


def _fetch_parents(kb: str, parent_keys: list[tuple]) -> dict[tuple, dict]:
    """
    Look up parent chunks by their (document_id, chunk_index) keys.

    Batches into a single collection.get() call per unique
    document_id when possible, then indexes results into a dict keyed
    on (document_id, chunk_index) so the caller can retrieve each
    parent in O(1) regardless of which child triggered the lookup.

    Silently skips (returns nothing for) any parent key that no
    matching chunk exists for - possible if a child's parent was
    deleted separately (rare) or the child was ingested before this
    version of the schema.
    """
    if not parent_keys:
        return {}

    collection = _get_collection(kb)
    # One get() per distinct document_id keeps the where filter simple:
    # document_id + chunk_role="parent" filters down to a per-document
    # parent list, then we index by chunk_index in Python.
    document_ids = {doc_id for doc_id, _ in parent_keys}
    parents_by_key: dict[tuple, dict] = {}
    for doc_id in document_ids:
        result = collection.get(
            where={
                "$and": [
                    {"document_id": doc_id},
                    {"chunk_role": "parent"},
                ]
            },
            include=["documents", "metadatas"],
        )
        for content, metadata in zip(
            result.get("documents") or [], result.get("metadatas") or []
        ):
            key = (metadata.get("document_id"), metadata.get("chunk_index"))
            parents_by_key[key] = {"content": content, "metadata": metadata}
    return parents_by_key


def _matches_where(metadata: dict, where: dict | None) -> bool:
    """Evaluate a subset of Chroma's `where` filter grammar against a
    single chunk's metadata, so BM25 can honor the same filters the
    dense retriever gets natively from Chroma.

    Supports the two shapes the retriever actually builds today
    (see retriever._build_where): flat equality ({"repository": "x"})
    and the "$in" operator ({"content_type": {"$in": [...]}}), plus
    Chroma's "$and" wrapper. Any unrecognized operator falls through
    as a non-match (safe default: better to miss a hit than to return
    the wrong one).
    """
    if not where:
        return True

    if "$and" in where:
        return all(_matches_where(metadata, clause) for clause in where["$and"])

    for key, condition in where.items():
        actual = metadata.get(key)
        if isinstance(condition, dict):
            if "$in" in condition:
                if actual not in condition["$in"]:
                    return False
            else:
                return False  # unrecognized operator
        else:
            if actual != condition:
                return False
    return True


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], k: int = _RRF_K
) -> list[dict]:
    """Reciprocal Rank Fusion (Cormack et al. 2009).

    For each document d appearing in one or more ranked lists:
        rrf_score(d) = sum over lists of 1 / (k + rank_in_list(d))

    A document only needs to appear in ONE list to be in the fused
    output; documents appearing in multiple lists get their scores
    summed and rise to the top. k damps the top-rank advantage so a
    document ranked #5 in both lists beats one ranked #1 in a single
    list - which is exactly the behavior we want from a fusion of
    dense + BM25 (a chunk that's a strong match on BOTH signals is a
    better candidate than one that's a very strong match on only one).

    Identity for dedup: each hit's (document_id, chunk_index) - the
    same key retriever.py uses for its own cross-KB dedup. Chunks
    missing document_id (rare, synthetic-metadata edge case) are
    keyed by their content, so they still participate in fusion.
    """
    rrf_scores: dict[tuple, float] = {}
    hits_by_key: dict[tuple, dict] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            key = _fusion_key(hit)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            # Keep the first hit dict we see for each key; content/
            # metadata are identical across sources anyway (same chunk,
            # different scores).
            hits_by_key.setdefault(key, hit)

    fused = []
    for key, score in rrf_scores.items():
        hit = dict(hits_by_key[key])  # shallow copy so we don't mutate cached BM25 hits
        hit["rrf_score"] = score
        fused.append(hit)

    fused.sort(key=lambda h: h["rrf_score"], reverse=True)
    return fused


def _fusion_key(hit: dict) -> tuple:
    """Same dedup identity as retriever.py's cross-KB loop."""
    document_id = hit["metadata"].get("document_id")
    if document_id is None:
        return ("__no_doc_id__", hit["content"][:200])
    return (document_id, hit["metadata"].get("chunk_index"))


def _cross_encoder_rerank(
    query_text: str, hits: list[dict], top_k: int
) -> list[dict]:
    """Re-score a shortlist of hits with the cross-encoder and return
    the top_k. Adds a "cross_encoder_score" field to each hit for
    observability."""
    if not hits:
        return []

    pairs = [(query_text, hit["content"]) for hit in hits]
    model = _get_cross_encoder()
    scores = model.predict(pairs, show_progress_bar=False)

    for hit, score in zip(hits, scores):
        hit["cross_encoder_score"] = float(score)

    hits.sort(key=lambda h: h["cross_encoder_score"], reverse=True)
    return hits[:top_k]


# ---- Public entry point ----------------------------------------------


def retrieve_hybrid(
    query_text: str,
    query_vector: list[float],
    kb: str,
    top_k: int,
    where: dict | None = None,
    variants: list[str] | None = None,
    hyde_answer: str | None = None,
    keywords: list[str] | None = None,
) -> list[dict]:
    """
    Full pipeline: dense + BM25 (+ parent-child, for KBs that support
    it) → RRF → cross-encoder rerank → top_k.

    Args:
        query_text: The raw query string (needed for BM25 tokenization
            and the cross-encoder, which both consume text).
        query_vector: The query's embedding (needed for dense search).
            Pre-computed by the caller so it's not embedded twice.
        kb: Which KB to search (must be in _HYBRID_KBS; the caller is
            responsible for dispatching non-hybrid KBs elsewhere).
            KBs also in _PARENT_CHILD_KBS gain the parent-child
            retriever as a third ranked list.
        top_k: How many final results to return.
        where: Chroma-style metadata filter (e.g. repository scope).
            Passed through to every retriever so the fused list stays
            consistent with the caller's scoping.
        variants: Optional list of additional text queries produced by
            app.retrieval.query_transform (rewrite / paraphrases /
            sub-questions). Each variant is embedded and dense-searched
            + BM25-searched independently; every ranked list joins the
            RRF fusion. The original ``query_text`` is always included
            regardless. Cross-encoder rerank still scores against the
            ORIGINAL ``query_text`` so the final ordering never drifts
            away from the user's actual intent.
        hyde_answer: Optional HyDE-style hypothetical answer paragraph.
            When provided, it is embedded and added as an extra
            dense-only ranked list (no BM25 - a synthesized paragraph
            has little exact-token value beyond what the paraphrases
            already give BM25).
        keywords: Optional topical keyword list produced by
            query_transform. When provided, the space-joined keywords
            are added as an extra BM25-only ranked list (no dense -
            dense of a 3-word bag-of-terms is a poor query).

    Returns:
        A list of hit dicts (same shape as query_embedding's output,
        with additional "bm25_score", "rrf_score", and
        "cross_encoder_score" fields populated where applicable).
    """
    logger.info(
        "Hybrid retrieval (kb=%s): query=%r top_k=%d where=%r variants=%d hyde=%s keywords=%d",
        kb, query_text, top_k, where,
        len(variants) if variants else 0,
        bool(hyde_answer),
        len(keywords) if keywords else 0,
    )

    ranked_lists: list[list[dict]] = []

    # Primary pass: original query text with the pre-embedded vector.
    ranked_lists.append(_dense_search(query_vector, kb, _DENSE_POOL_SIZE, where))
    ranked_lists.append(_bm25_search(query_text, kb, _BM25_POOL_SIZE, where))

    # Extra passes for each transformation variant (rewrite,
    # paraphrases, sub-queries). Each contributes its own dense + BM25
    # ranked lists to the RRF fusion. We deliberately re-embed here
    # rather than requiring the caller to precompute vectors, because
    # the caller is often the CLI/UI which shouldn't need to know
    # about variant vector shapes. Extra embed cost is small (~10ms
    # per variant on CPU) compared to the LLM answer step.
    extra_variants = [
        v for v in (variants or [])
        if v and v.strip() and v.strip().lower() != query_text.strip().lower()
    ]
    if extra_variants:
        variant_vectors = embed_texts(extra_variants)
        for variant_text, variant_vector in zip(extra_variants, variant_vectors):
            ranked_lists.append(_dense_search(variant_vector, kb, _DENSE_POOL_SIZE, where))
            ranked_lists.append(_bm25_search(variant_text, kb, _BM25_POOL_SIZE, where))

    # HyDE: dense-only extra pass using the embedding of a synthesized
    # hypothetical answer paragraph. Skipped when empty/trivial - a
    # one-word "hyde" would just noise up the fusion.
    if hyde_answer and len(hyde_answer.strip()) > 10:
        hyde_vector = embed_texts([hyde_answer])[0]
        ranked_lists.append(_dense_search(hyde_vector, kb, _DENSE_POOL_SIZE, where))

    # Keywords: BM25-only extra pass using the space-joined keyword
    # bag. Sparse retrieval loves exact rare terms and doesn't care
    # about grammar - this is where "PR number 12345" style lookups
    # actually get boosted into the shortlist.
    if keywords:
        kw_query = " ".join(k for k in keywords if k and k.strip())
        if kw_query.strip():
            ranked_lists.append(_bm25_search(kw_query, kb, _BM25_POOL_SIZE, where))

    # Parent-child retrieval always uses the primary query vector: the
    # child index is a sentence-window index whose value is a precise
    # semantic match against the actual user question, not a bag of
    # paraphrases which would just muddy the child ranking.
    parent_child_hits: list[dict] = []
    if kb in _PARENT_CHILD_KBS:
        parent_child_hits = _parent_child_search(
            query_vector, kb, _PARENT_CHILD_POOL_SIZE, where
        )
        ranked_lists.append(parent_child_hits)

    logger.info(
        "Hybrid (kb=%s): %d ranked lists entering RRF fusion",
        kb, len(ranked_lists),
    )

    fused = _reciprocal_rank_fusion(ranked_lists)
    shortlist = fused[:_RERANK_POOL_SIZE]

    reranked = _cross_encoder_rerank(query_text, shortlist, top_k)
    logger.info(
        "Hybrid (kb=%s): %d fused → %d after cross-encoder rerank",
        kb, len(fused), len(reranked),
    )
    return reranked


# ---- Backward-compatibility shim -------------------------------------


def retrieve_pdf_hybrid(
    query_text: str,
    query_vector: list[float],
    top_k: int,
    where: dict | None = None,
) -> list[dict]:
    """Kept for callers that still hard-code "pdf"; new callers should
    use retrieve_hybrid(kb=...) directly."""
    return retrieve_hybrid(query_text, query_vector, kb="pdf", top_k=top_k, where=where)

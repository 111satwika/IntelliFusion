"""
Adaptive retrieval for the GitHub KB.

Applies app.routing.github_intent's per-query strategy decision to the
GitHub KB only:

- explanation:  dense-only over the github KB + cross-encoder rerank.
                Skips BM25 fusion so semantic paraphrase isn't diluted
                by literal-token noise (a "how does X work" query
                rarely shares tokens with the implementation code).
- exact_code:   BM25-only + cross-encoder rerank. Identifier lookups
                are lexical; dense embeddings smear identifier names
                together while BM25 keeps them distinct.
- navigational: metadata-filtered dense retrieval scoped to a specific
                file_name parsed from the query. Falls back to hybrid
                if the filter matches nothing.
- history:      app.retrieval.github_history.retrieve_git_history
                (GitHub REST commits API - see that module). Falls
                back to hybrid on no repo / API failure / empty
                result, so a stale token or an offline moment never
                loses the ability to answer.
- activity:     dense retrieval scoped to content_type in {issue,
                pull_request, discussion} (see
                app.ingestion.ingest.ingest_github_activity) + cross-
                encoder rerank. Unlike every other intent, issues/PRs
                are typically a handful of chunks swimming in a KB of
                hundreds/thousands of code chunks - plain similarity
                ranking routinely loses that contest even when an
                issue/PR chunk is the ONLY thing that actually answers
                the question, so this intent sidesteps ranking with a
                metadata filter instead of trying to out-embed the
                rest of the repo. Falls back to hybrid if the repo has
                no ingested activity at all.
- structural:   app.graph.graph_retrieval.retrieve_by_graph - walks
                the per-repo call/import/inherit graph built at
                ingest time, materializes touched nodes into chunks,
                then cross-encoder-reranks them. Falls back to BM25
                (if a symbol was extracted) or hybrid otherwise, so
                pre-GraphRAG repos still answer.
- visual:       standard hybrid pipeline (the caller in cli.py already
                layers CLIP image retrieval + marker-based image
                surfacing on top of ANY answer, so this branch's job
                is just to produce the strongest text context).
- general:      standard hybrid pipeline (dense + BM25 + parent-child,
                RRF, cross-encoder rerank). Identical to what
                retrieve() already does when scoped to kb="github" -
                the fallback keeps existing behavior unchanged.

Only called from cli.py's _prepare_context_and_images when the user
scoped the chat to the GitHub KB; every other retrieval path (cross-KB
chat, per-KB tabs for pdf/docx/markdown/web) is untouched, so this
module is purely additive.

Depends on private helpers (`_dense_search`, `_bm25_search`,
`_cross_encoder_rerank`) from app.retrieval.hybrid_retriever. Those
are underscore-prefixed but stable within this repo - re-exporting
them would duplicate implementation logic that the hybrid pipeline
already owns.
"""

import logging

from app.embeddings.embedder import embed_texts
from app.graph.graph_retrieval import retrieve_by_graph
from app.retrieval.github_history import retrieve_git_history
from app.retrieval.hybrid_retriever import (
    _bm25_search,
    _cross_encoder_rerank,
    _dense_search,
    retrieve_hybrid,
)
from app.retrieval.query_transform import transform_query
from app.routing.github_intent import GitHubIntent, classify_github_intent

logger = logging.getLogger(__name__)

_KB = "github"

# content_type values app.ingestion.ingest.ingest_github_activity writes
# for issues/PRs/discussions (see app.ingestion.loader.load_github_issues/
# load_github_pull_requests/load_github_discussions) - what the
# "activity" intent filters retrieval down to.
_ACTIVITY_CONTENT_TYPES = ["issue", "pull_request", "discussion"]

# How many candidates each intent's first-stage retriever pulls before
# cross-encoder rerank. Aligned with the values hybrid_retriever uses
# for its own pools (_DENSE_POOL_SIZE / _BM25_POOL_SIZE = 40) so the
# rerank step sees a comparable-sized shortlist and shortlist quality
# doesn't silently regress vs. the general hybrid path.
_INTENT_POOL_SIZE = 40

# Which metadata_hint fields (see app.routing.github_intent
# ._extract_metadata_hints) each intent narrows retrieval by. Kept as
# a table (rather than hard-coded per-branch) so adding a new hint
# field only requires listing it once, and so an intent that
# shouldn't narrow by an extracted hint (e.g. explanation shouldn't
# be scoped to a symbol_name) can drop it explicitly.
_INTENT_METADATA_FIELDS = {
    "navigational": ("file_name", "file_dir", "language"),
    "explanation":  ("file_name", "file_dir", "language"),
    "exact_code":   ("language", "symbol_name"),
    "structural":   ("symbol_name",),  # only used by the exact_code fallback -
                                        # graph traversal handles narrowing itself.
    # visual/general/history don't narrow by metadata hints here -
    # visual/general use the standard hybrid pipeline (broader recall
    # is the point), and history dispatches to a completely different
    # retriever (GitHub REST commits API) that has its own path filter.
}


def _expand_hint_to_clause(hint_key: str, hint_value: str) -> dict:
    """Turn one extracted hint into a Chroma where clause.

    Most hints map 1:1 to a real metadata field ($eq). The one
    exception is symbol_name: an identifier the user names could be
    a class, method, or function, so it expands to an $or across the
    three real fields the AST chunker writes.
    """
    if hint_key == "symbol_name":
        return {
            "$or": [
                {"class_name": hint_value},
                {"method_name": hint_value},
                {"function_name": hint_value},
            ]
        }
    return {hint_key: hint_value}


def _build_where(repository: str | None, extras: list[dict] | None = None) -> dict | None:
    """Combine repository scope + any number of extra clauses into a
    Chroma-style where filter under $and.

    Every extra clause is applied as a conjunction: if the user names
    both a directory AND a language, retrieval is narrowed to chunks
    matching BOTH. A single clause (or zero clauses, with no
    repository) short-circuits the $and wrapper to keep filters
    compact and readable in logs.
    """
    clauses: list[dict] = []
    if repository:
        clauses.append({"repository": repository})
    if extras:
        clauses.extend(extras)
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _hints_to_clauses(
    metadata_hints: dict[str, str], allowed_fields: tuple[str, ...]
) -> list[dict]:
    """Turn extracted hints into Chroma where clauses, keeping only
    the fields this intent is allowed to narrow by (per
    _INTENT_METADATA_FIELDS)."""
    return [
        _expand_hint_to_clause(key, value)
        for key, value in metadata_hints.items()
        if key in allowed_fields
    ]


def _dense_then_rerank(
    query_text: str, query_vector: list[float], top_k: int, where: dict | None
) -> list[dict]:
    """Dense-only retrieval (over the github KB) + cross-encoder rerank.
    Used by the explanation branch and, with a tighter filter, by the
    navigational branch."""
    dense_hits = _dense_search(query_vector, _KB, _INTENT_POOL_SIZE, where)
    if not dense_hits:
        return []
    return _cross_encoder_rerank(query_text, dense_hits, top_k)


def _bm25_then_rerank(
    query_text: str, top_k: int, where: dict | None
) -> list[dict]:
    """BM25-only retrieval (over the github KB) + cross-encoder rerank.
    Used by the exact_code branch."""
    bm25_hits = _bm25_search(query_text, _KB, _INTENT_POOL_SIZE, where)
    if not bm25_hits:
        return []
    return _cross_encoder_rerank(query_text, bm25_hits, top_k)


def _run_hybrid(
    query_text: str,
    query_vector: list[float],
    top_k: int,
    where: dict | None,
    *,
    query_transform_enabled: bool | None = None,
) -> list[dict]:
    """The existing fused pipeline for the github KB - dense + BM25 +
    parent-child, RRF, cross-encoder rerank. Used both as the intent
    strategy for visual/general and as a fallback when the more
    targeted strategies (history, navigational filter) return
    nothing.

    Query transformation (rewrite / paraphrases / sub-questions /
    HyDE / keywords - see app.retrieval.query_transform) is applied
    here so the general/visual paths (and every intent that falls
    back to hybrid) benefit from wider recall. Structural / exact_code
    / navigational-with-hint / history do NOT go through this
    function, so their targeted retrieval stays precise and cheap.
    """
    transform = transform_query(query_text, kb=_KB, enabled=query_transform_enabled)
    return retrieve_hybrid(
        query_text, query_vector, kb=_KB, top_k=top_k, where=where,
        variants=transform.variants,
        hyde_answer=transform.hyde_answer,
        keywords=transform.keywords,
    )


def retrieve_github_adaptive(
    query_text: str,
    top_k: int = 5,
    repository: str | None = None,
    *,
    query_transform_enabled: bool | None = None,
) -> tuple[list[dict], GitHubIntent]:
    """
    Route a GitHub-KB query to the retrieval strategy that best fits
    it, and return the retrieved chunks along with the intent decision
    (so the UI can show which strategy was chosen and which metadata
    filters were applied - a small transparency win for users trying
    to understand why an answer is or isn't grounded).

    Metadata narrowing per intent (see _INTENT_METADATA_FIELDS):
      - navigational: file_name, file_dir, language.
      - explanation:  file_name, file_dir, language - so an
                      explanation query that names a specific file
                      still stays scoped to that file.
      - exact_code:   language, symbol_name - identifier hints
                      (expanded to $or over class_name / method_name /
                      function_name) turn BM25 into a much more
                      precise lookup when the user names a specific
                      symbol.
      - activity:     content_type in {issue, pull_request,
                      discussion} - a fixed filter, not extracted from
                      the query text (see _ACTIVITY_CONTENT_TYPES).
      - visual, general, history: no metadata narrowing here
                      (history has its own path filter inside the
                      commits API).

    Args:
        query_text: The user's natural-language question.
        top_k:      How many chunks to return.
        repository: Optional "owner/repo" scope. Required for the
                    "history" intent (with no repository we can't hit
                    the commits API); optional but recommended for
                    every other intent to avoid answering a
                    just-ingested repo's question from a different
                    previously-ingested repo (same rationale as
                    app.retrieval.retriever.retrieve).

    Returns:
        (chunks, decision) - the retrieved chunks in the standard
        retrieve() shape, plus the GitHubIntent that decided the
        strategy (including every metadata hint that was extracted,
        even the ones this intent chose not to apply).
    """
    decision = classify_github_intent(query_text)
    intent = decision.intent
    hint_fields = _INTENT_METADATA_FIELDS.get(intent, ())
    hint_clauses = _hints_to_clauses(decision.metadata_hints, hint_fields)
    scoped_where = _build_where(repository, extras=hint_clauses)
    unscoped_where = _build_where(repository)

    # History is the one path that doesn't use the vector store at all:
    # commits live in git, not Chroma. If we can't or shouldn't call
    # the API (no repository scope, API error, empty result), fall
    # back to hybrid so the user still gets a useful text answer.
    if intent == "history":
        if repository is None:
            logger.info(
                "GitHub intent=history but no repository scope - falling back to hybrid."
            )
        else:
            hits = retrieve_git_history(
                query_text, repository=repository, top_k=top_k
            )
            if hits:
                return hits, decision
            logger.info("Git history returned no commits - falling back to hybrid.")
        query_vector = embed_texts([query_text])[0]
        return _run_hybrid(
            query_text, query_vector, top_k, unscoped_where,
            query_transform_enabled=query_transform_enabled,
        ), decision

    if intent == "activity":
        # Metadata-filtered, not extracted from the query (unlike
        # file_name/symbol_name hints) - "activity" itself already
        # means "restrict to issue/PR/discussion chunks", so the
        # filter is unconditional whenever this intent wins.
        activity_where = _build_where(
            repository, extras=[{"content_type": {"$in": _ACTIVITY_CONTENT_TYPES}}]
        )
        query_vector = embed_texts([query_text])[0]
        hits = _dense_then_rerank(query_text, query_vector, top_k, activity_where)
        if hits:
            return hits, decision
        logger.info(
            "Activity intent found no issue/PR/discussion chunks for repository=%r - "
            "falling back to hybrid.",
            repository,
        )
        return _run_hybrid(
            query_text, query_vector, top_k, unscoped_where,
            query_transform_enabled=query_transform_enabled,
        ), decision

    if intent == "explanation":
        query_vector = embed_texts([query_text])[0]
        hits = _dense_then_rerank(query_text, query_vector, top_k, scoped_where)
        # If the hint filter narrowed too aggressively (e.g. wrong
        # file_name, or file_dir the user misremembered), retry
        # without the metadata narrowing so we don't silently return
        # zero context to the LLM.
        if not hits and hint_clauses:
            logger.info(
                "Explanation with metadata_hints=%r matched nothing - retrying without hints.",
                decision.metadata_hints,
            )
            hits = _dense_then_rerank(query_text, query_vector, top_k, unscoped_where)
        return hits, decision

    if intent == "structural":
        # GraphRAG path: only meaningful when we know both the repo
        # AND the symbol the user is asking about. Missing either
        # signal, we fall through to exact_code semantics (BM25 on
        # the symbol name if we have it, hybrid otherwise) - a
        # graceful degradation that keeps this intent from ever
        # returning nothing.
        symbol = decision.metadata_hints.get("symbol_name")
        if repository and symbol:
            graph_hits = retrieve_by_graph(query_text, symbol, repository)
            if graph_hits:
                # Rerank the graph-materialized shortlist with the
                # SAME cross-encoder every other branch uses, so the
                # final ranking is comparable across intents.
                return _cross_encoder_rerank(query_text, graph_hits, top_k), decision
            logger.info(
                "Graph retrieval empty for repo=%r symbol=%r - falling back to BM25.",
                repository, symbol,
            )
        else:
            logger.info(
                "Structural intent without repo+symbol (repo=%r, hints=%r) - "
                "falling back to BM25/hybrid.",
                repository, decision.metadata_hints,
            )
        # Graph miss: try the exact_code strategy next (BM25 scoped
        # by symbol_name if present) - it's the closest thing to
        # \"find the definition\" without a graph. If THAT is empty,
        # unfiltered hybrid is the last resort.
        hits = _bm25_then_rerank(query_text, top_k, scoped_where) if hint_clauses else []
        if hits:
            return hits, decision
        query_vector = embed_texts([query_text])[0]
        return _run_hybrid(
            query_text, query_vector, top_k, unscoped_where,
            query_transform_enabled=query_transform_enabled,
        ), decision

    if intent == "exact_code":
        hits = _bm25_then_rerank(query_text, top_k, scoped_where)
        if not hits and hint_clauses:
            logger.info(
                "exact_code with metadata_hints=%r matched nothing - retrying without hints.",
                decision.metadata_hints,
            )
            hits = _bm25_then_rerank(query_text, top_k, unscoped_where)
        return hits, decision

    if intent == "navigational" and hint_clauses:
        # navigational fires either from a keyword ("in file X") or
        # from an unconditional file_name/file_dir extraction (see
        # classify_github_intent). Either way, hint_clauses are what
        # define "which slice of the repo"; without them, navigational
        # is indistinguishable from general, so fall through to hybrid.
        query_vector = embed_texts([query_text])[0]
        hits = _dense_then_rerank(query_text, query_vector, top_k, scoped_where)
        if hits:
            return hits, decision
        logger.info(
            "Navigational filter %r matched nothing - falling back to hybrid.",
            decision.metadata_hints,
        )
        return _run_hybrid(
            query_text, query_vector, top_k, unscoped_where,
            query_transform_enabled=query_transform_enabled,
        ), decision

    # visual, general, and navigational-without-hints all fall through
    # to the standard hybrid pipeline. visual's image handling is
    # layered on separately by cli.py's _prepare_context_and_images
    # (marker-based extraction + CLIP retrieval), so this branch's job
    # is just to produce the strongest text context possible.
    query_vector = embed_texts([query_text])[0]
    return _run_hybrid(
        query_text, query_vector, top_k, unscoped_where,
        query_transform_enabled=query_transform_enabled,
    ), decision

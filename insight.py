"""
Framework-agnostic "how was this answer checked" logic - the same
plain-language explanations the Streamlit build's ui/panels.py used to
render directly with st.markdown/st.expander, now returning a plain
JSON-able dict instead, for api/chat.py to send to the frontend.

Purely observability - every decision described here was already made
by the retrieval/generation pipeline (see cli.py) before this runs.
"""

import logging

from app.retrieval.query_transform import transform_query
from session_store import KB_LABELS

logger = logging.getLogger(__name__)

GITHUB_INTENT_UI = {
    "explanation":  ("Explanation",  "dense semantic search"),
    "exact_code":   ("Exact code",   "BM25 keyword search"),
    "navigational": ("Navigational", "file-scoped search"),
    "history":      ("History",      "GitHub commits API"),
    "activity":     ("Activity",     "issue/PR/discussion search"),
    "visual":       ("Visual",       "hybrid + image retrieval"),
    "structural":   ("Structural",   "code graph traversal"),
    "general":      ("General",      "hybrid retrieval (fallback)"),
}
GITHUB_INTENT_PLAIN = {
    "explanation": "this question asked how or why something works, so it searched by meaning rather than exact wording.",
    "exact_code": "this question named a specific function or symbol, so it searched for that exact text.",
    "navigational": "this question pointed at a specific file or folder, so the search was narrowed to it.",
    "history": "this question asked about a past change, so it went straight to the commit history rather than a text search.",
    "activity": "this question asked about issues, pull requests, or discussions, so the search was narrowed to that activity instead of the code itself.",
    "visual": "this question referenced an image or diagram, so image search ran alongside the text search.",
    "structural": "this question asked how code pieces relate, so it walked the code's call/inheritance graph.",
    "general": "no specific signal in the question, so it ran the standard hybrid search.",
}


def _crag_line(crag_result) -> str | None:
    # No "is CRAG enabled" guard here - a None/non-LLM-graded
    # crag_result already fully signals "CRAG didn't run for this
    # request", regardless of whether that's because the module is
    # globally off or because this session's setting is off (see
    # session_store's per-session crag_enabled).
    if crag_result is None or not getattr(crag_result, "used_llm", False):
        return None
    if crag_result.fell_back:
        return "Before answering: the relevance check didn't run cleanly, so every retrieved passage was kept as a precaution."
    if crag_result.dropped_indices:
        n = len(crag_result.dropped_indices)
        return (
            f"Before answering: the system checked whether the retrieved material actually addressed "
            f"your question, and dropped {n} irrelevant passage{'s' if n != 1 else ''} before answering."
        )
    return "Before answering: the system checked whether the retrieved material actually addressed your question — all of it did."


def _self_rag_line(critique_result) -> str | None:
    # Same reasoning as _crag_line: critique_result being None already
    # means Self-RAG didn't run for this request, session-scoped
    # setting or not.
    if critique_result is None:
        return None
    if critique_result.fell_back:
        return "After answering: the self-check didn't run cleanly, so treat this answer as unverified."
    if critique_result.hallucinations:
        n = len(critique_result.hallucinations)
        return f"After answering: {n} sentence{'s' if n != 1 else ''} in the answer couldn't be verified against the sources — see below."
    return "After answering: every sentence was checked against the sources — nothing was flagged as unsupported."


def _routing_details(decision, target_kbs: list[str], forced_kb: str | None) -> dict:
    return {
        "forced_kb": forced_kb,
        "explicit_kb_signal": list(decision.kbs) if decision.kbs else None,
        "method": decision.method,
        "kb_scores": [
            {
                "kb": kb_name,
                "score": round(score, 3),
                "meets_threshold": score >= 0.40,
                "searched": (forced_kb is not None and kb_name == forced_kb)
                or (forced_kb is None and kb_name in target_kbs),
            }
            for kb_name, score in sorted(decision.kb_scores.items(), key=lambda kv: kv[1], reverse=True)
        ] if decision.kb_scores else [],
        "route_scores": [
            {"route": route, "score": round(score, 3), "used": route in decision.routes}
            for route, score in sorted(decision.scores.items(), key=lambda kv: kv[1], reverse=True)
        ] if decision.scores else [],
    }


def _github_intent_details(github_intent, repository: str | None) -> dict:
    label, strategy = GITHUB_INTENT_UI.get(github_intent.intent, (github_intent.intent, "unknown strategy"))
    return {
        "intent": github_intent.intent,
        "label": label,
        "strategy": strategy,
        "method": github_intent.method,
        "repository": repository,
        "metadata_hints": dict(getattr(github_intent, "metadata_hints", {}) or {}),
        "scores": [
            {"intent": intent, "score": round(score, 3), "chosen": intent == github_intent.intent}
            for intent, score in sorted(github_intent.scores.items(), key=lambda kv: kv[1], reverse=True)
        ] if github_intent.scores else [],
    }


def _query_transform_details(question: str, kb: str | None, *, enabled: bool) -> dict | None:
    if not enabled:
        return None
    try:
        result = transform_query(question, kb=kb, enabled=True)
    except Exception:  # noqa: BLE001 - observability must not break the chat response
        logger.exception("Query-transform detail failed to fetch cached result.")
        return None
    non_trivial = (
        result.used_llm
        and not result.fell_back
        and (
            result.rewritten.strip().lower() != question.strip().lower()
            or result.sub_queries
            or result.paraphrases
            or result.keywords
            or bool(result.hyde_answer)
        )
    )
    if not non_trivial:
        return None
    return {
        "rewritten": result.rewritten if result.rewritten.strip().lower() != question.strip().lower() else None,
        "sub_queries": list(result.sub_queries),
        "paraphrases": list(result.paraphrases),
        "hyde_answer": result.hyde_answer or None,
        "keywords": list(result.keywords),
    }


def _contextualization_line(contextualization) -> str | None:
    if contextualization is None:
        return None
    if not contextualization.used_llm or contextualization.fell_back:
        return None
    if contextualization.resolved.strip().lower() == contextualization.original.strip().lower():
        return None
    return f'Because this looked like a follow-up, your question was understood as: "{contextualization.resolved}"'


def _contextualization_details(contextualization) -> dict | None:
    # Takes the ALREADY-COMPUTED ContextualizeResult directly rather
    # than re-invoking contextualize_query() the way
    # _query_transform_details() re-calls transform_query() - that
    # module has an LRU cache making a repeat call free, but
    # contextualize_query() deliberately has none (see its module
    # docstring: history changes every turn, so caching would be a
    # near-permanent miss) - re-calling it here would cost a real
    # duplicate Ollama call on every single turn.
    if contextualization is None or not contextualization.used_llm:
        return None
    return {
        "original": contextualization.original,
        "resolved": contextualization.resolved,
        "fell_back": contextualization.fell_back,
        "changed": contextualization.resolved.strip().lower() != contextualization.original.strip().lower(),
    }


def _semantic_cache_line(used_semantic_cache: bool) -> str | None:
    # Unlike every other pipeline decision surfaced on this panel,
    # there's no per-session toggle a user could check to control this
    # (see app.generation.semantic_cache's module docstring - it's a
    # process-global, env-var-controlled flag) - this line IS the
    # substitute for that control: it makes a cache hit visible even
    # though the user can't turn it off themselves.
    if not used_semantic_cache:
        return None
    return "This answer was served from a cached response to a very similar earlier question, not freshly generated."


def _crag_details(crag_result) -> dict | None:
    if crag_result is None or not getattr(crag_result, "used_llm", False):
        return None
    return {
        "overall": crag_result.overall,
        "fell_back": crag_result.fell_back,
        "counts": dict(crag_result.counts),
        "verdicts": [
            {
                "chunk": v.index + 1,
                "relevance": v.relevance,
                "kept": v.is_keeper,
                "reason": v.reason,
            }
            for v in crag_result.verdicts
        ],
    }


def _self_rag_details(critique_result) -> dict | None:
    if critique_result is None:
        return None
    return {
        "fell_back": critique_result.fell_back,
        "dimensions": critique_result.as_dimension_rows(),
        "hallucinations": list(critique_result.hallucinations),
    }


def build_insight(
    *,
    question: str,
    kb: str | None,
    decision,
    target_kbs: list[str],
    forced_kb: str | None = None,
    github_intent=None,
    repository: str | None = None,
    crag_result=None,
    critique_result=None,
    query_transform_enabled: bool = False,
    contextualization=None,
    used_semantic_cache: bool = False,
) -> dict:
    """
    Build the full "how was this answer checked" payload as a plain
    dict, ready to JSON-serialize into the chat-stream's final SSE
    event. Mirrors ui/panels.py's render_insight_panel exactly, minus
    the Streamlit rendering calls.

    ``contextualization`` is the ContextualizeResult cli.py's
    ask_with_vision_stream already computed (see
    app.retrieval.contextualize) - passed through directly, not
    re-derived, since that module has no cache to make a repeat call free.

    ``used_semantic_cache`` is also already computed by cli.py's
    ask_with_vision_stream (see app.generation.semantic_cache) -
    surfaced here purely for disclosure, see _semantic_cache_line.
    """
    crag_line = _crag_line(crag_result)
    self_rag_line = _self_rag_line(critique_result)
    contextualization_line = _contextualization_line(contextualization)
    semantic_cache_line = _semantic_cache_line(used_semantic_cache)

    flagged = bool(getattr(critique_result, "hallucinations", None)) or (
        crag_result is not None and getattr(crag_result, "overall", None) == "incorrect"
    )
    if flagged:
        pill_label, pill_kind = "Needs review", "warn"
    elif crag_line or self_rag_line:
        pill_label, pill_kind = "Verified", "good"
    else:
        pill_label, pill_kind = "Unverified", "muted"

    if forced_kb == "github" and github_intent is not None:
        plain = GITHUB_INTENT_PLAIN.get(github_intent.intent, "ran the standard hybrid search.")
        scope_bit = f"repo {repository}" if repository else "all ingested GitHub repos"
        searched_line = f"Searched: GitHub ({scope_bit}) — {plain}"
    elif forced_kb is not None:
        searched_line = f"Searched: {KB_LABELS.get(forced_kb, forced_kb)} only (this view is scoped to one source)."
    else:
        kbs_str = ", ".join(target_kbs) if target_kbs else "every populated source"
        searched_line = f"Searched: {kbs_str}."
        # Cross-KB search: "github" can be one of several target_kbs,
        # not the whole story - append its adaptive-intent strategy
        # rather than replacing the line (see
        # app.retrieval.retriever.retrieve's github dispatch).
        if github_intent is not None:
            plain = GITHUB_INTENT_PLAIN.get(github_intent.intent, "ran the standard hybrid search.")
            searched_line += f" For GitHub specifically: {plain}"

    return {
        "pill_label": pill_label,
        "pill_kind": pill_kind,
        "searched": searched_line,
        "contextualized_as": contextualization_line,
        "served_from_cache": semantic_cache_line,
        "before_answering": crag_line,
        "after_answering": self_rag_line,
        "hallucinations": list(critique_result.hallucinations) if critique_result is not None else [],
        "technical_details": {
            "routing": _routing_details(decision, target_kbs, forced_kb),
            "github_intent": _github_intent_details(github_intent, repository) if github_intent is not None else None,
            "query_transform": _query_transform_details(
                question, kb, enabled=query_transform_enabled
            ),
            "contextualization": _contextualization_details(contextualization),
            "semantic_cache": used_semantic_cache,
            "crag": _crag_details(crag_result),
            "self_rag": _self_rag_details(critique_result),
        },
    }

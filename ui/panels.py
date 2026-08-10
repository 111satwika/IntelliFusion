"""
Observability panels: routing decisions, GitHub-intent badges,
query-transform variants, CRAG verdicts, Self-RAG critiques, and the
per-KB pending-document list. Pure render functions - the retrieval/
generation pipeline has already made every decision shown here by the
time these run; nothing in this module changes pipeline behavior.
"""

import logging

import streamlit as st

from app.retrieval.crag import is_enabled as crag_enabled
from app.retrieval.query_transform import is_enabled as query_transform_enabled, transform_query
from app.generation.self_reflection import is_enabled as self_rag_enabled
from app.vectorstore.store import delete_document, delete_repository, list_documents

from ui.icons import icon_tile
from ui.state import KB_LABELS, get_pending, remove_pending

logger = logging.getLogger(__name__)

# (icon name from ui.icons.PATHS, background tint token, icon color token) per KB.
_ROW_ICON_STYLE = {
    "markdown": ("markdown", "var(--surface-sunken)", "var(--ink-soft)"),
    "pdf": ("pdf", "var(--bad-wash)", "var(--bad)"),
    "docx": ("docx", "var(--accent-wash)", "var(--accent-strong)"),
    "web": ("web", "var(--accent-wash)", "var(--accent-strong)"),
    "github": ("github", "var(--surface-sunken)", "var(--ink)"),
}


def render_row_icon(kb: str) -> str:
    """Return an inline-HTML rounded icon tile for a source-list row (see render_source_rows)."""
    icon_name, bg, fg = _ROW_ICON_STYLE[kb]
    return icon_tile(icon_name, tile=44, radius=10, bg=bg, fg=fg, icon_size=22)


def render_images(image_hits: list[dict]) -> None:
    """Render any relevant retrieved images below an answer, with a citation."""
    for image_hit in image_hits:
        metadata = image_hit["metadata"]
        st.image(metadata["image_url"], caption=metadata.get("alt_text"))
        page_url = metadata.get("page_url")
        if page_url:
            st.caption(f"From: {page_url}")


def render_status_pill(label: str, kind: str) -> str:
    """Return an inline-HTML span for a colored status badge. `kind` is
    one of "green" (Indexed), "amber" (Pending), "slate" (neutral) -
    see ui/theme.py's .pill-* classes."""
    return f'<span class="pill pill-{kind}">{label}</span>'


def render_source_rows(kb: str) -> None:
    """
    Render every document currently stored in a KB - both permanently
    "Indexed" ones and this-session "Pending" ones (see ui.state's
    pending-document tracker) - as one unified list of bordered rows,
    each with a status pill and a remove ("Discard chunks from the
    KB") action. Pending rows additionally get a "Save" action that
    just removes the session's abandonment-cleanup claim on the
    document, since its chunks are already stored.

    Rows are keyed by document_id (repository, for kb="github") - see
    app.vectorstore.store.list_documents().
    """
    label = KB_LABELS[kb]
    indexed = {doc["id"]: doc["chunk_count"] for doc in list_documents(kb)}
    pending_entries = {
        (entry.get("repository") or entry.get("document_id")): entry
        for entry in get_pending(kb)
    }

    if not indexed:
        st.caption(f"No {label} sources indexed yet.")
        return

    for doc_id in sorted(indexed):
        chunk_count = indexed[doc_id]
        pending_entry = pending_entries.get(doc_id)
        with st.container(border=True):
            col_icon, col_label, col_pill, col_save, col_remove = st.columns([1, 6, 2, 1, 1])
            col_icon.markdown(render_row_icon(kb), unsafe_allow_html=True)
            col_label.markdown(
                f"**{doc_id}**  \n"
                f"<span style='color:#64748B; font-size:0.85rem;'>{label} · {chunk_count} chunks</span>",
                unsafe_allow_html=True,
            )
            if pending_entry is not None:
                col_pill.markdown(render_status_pill("Pending", "amber"), unsafe_allow_html=True)
                entry_key = "repository" if pending_entry.get("repository") else "document_id"
                if col_save.button("Save", key=f"save_{kb}_{doc_id}"):
                    remove_pending(kb, entry_key, doc_id)
                    st.rerun()
            else:
                col_pill.markdown(render_status_pill("Indexed", "green"), unsafe_allow_html=True)
            if col_remove.button("✕", key=f"remove_{kb}_{doc_id}", help="Remove this source"):
                if kb == "github":
                    delete_repository(doc_id)
                else:
                    delete_document(doc_id, kb=kb)
                if pending_entry is not None:
                    entry_key = "repository" if pending_entry.get("repository") else "document_id"
                    remove_pending(kb, entry_key, doc_id)
                st.rerun()


# Human-readable label + retrieval-strategy caption + a plain-language
# sentence per GitHub intent (see app.routing.github_intent and
# app.retrieval.github_adaptive). The plain sentence is what
# render_insight_panel's "Searched" line uses; label/strategy back the
# technical-details table.
GITHUB_INTENT_UI = {
    "explanation":  ("Explanation",  "dense semantic search"),
    "exact_code":   ("Exact code",   "BM25 keyword search"),
    "navigational": ("Navigational", "file-scoped search"),
    "history":      ("History",      "GitHub commits API"),
    "visual":       ("Visual",       "hybrid + image retrieval"),
    "structural":   ("Structural",   "code graph traversal"),
    "general":      ("General",      "hybrid retrieval (fallback)"),
}
GITHUB_INTENT_PLAIN = {
    "explanation": "this question asked how or why something works, so it searched by meaning rather than exact wording.",
    "exact_code": "this question named a specific function or symbol, so it searched for that exact text.",
    "navigational": "this question pointed at a specific file or folder, so the search was narrowed to it.",
    "history": "this question asked about a past change, so it went straight to the commit history rather than a text search.",
    "visual": "this question referenced an image or diagram, so image search ran alongside the text search.",
    "structural": "this question asked how code pieces relate, so it walked the code's call/inheritance graph.",
    "general": "no specific signal in the question, so it ran the standard hybrid search.",
}


def _crag_line(crag_result) -> str | None:
    """Plain-language 'before answering' sentence for the insight panel, or None if CRAG didn't run."""
    if not crag_enabled() or crag_result is None or not getattr(crag_result, "used_llm", False):
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
    """Plain-language 'after answering' sentence for the insight panel, or None if Self-RAG didn't run."""
    if not self_rag_enabled() or critique_result is None:
        return None
    if critique_result.fell_back:
        return "After answering: the self-check didn't run cleanly, so treat this answer as unverified."
    if critique_result.hallucinations:
        n = len(critique_result.hallucinations)
        return f"After answering: {n} sentence{'s' if n != 1 else ''} in the answer couldn't be verified against the sources — see below."
    return "After answering: every sentence was checked against the sources — nothing was flagged as unsupported."


def render_insight_panel(
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
) -> None:
    """
    One consolidated "how was this answer checked" panel, replacing
    what used to be five separate expanders (routing, GitHub-intent
    badge, query-transform variants, CRAG verdict, Self-RAG critique).
    Plain-language summary by default; every number the old panels
    showed is still available under "Technical details" for anyone
    who wants it - nothing is silently dropped, just de-prioritized.

    Purely observability - every decision described here was already
    made by the retrieval/generation pipeline before this renders.
    """
    crag_line = _crag_line(crag_result)
    self_rag_line = _self_rag_line(critique_result)

    # Overall status pill: "Needs review" beats "Verified" beats
    # "Unverified" (nothing was actually checked, because both
    # verification toggles are off) - never claim more confidence than
    # was actually earned.
    flagged = bool(getattr(critique_result, "hallucinations", None)) or (
        crag_result is not None and getattr(crag_result, "overall", None) == "incorrect"
    )
    if flagged:
        pill_label, pill_kind = "Needs review", "warn"
    elif crag_line or self_rag_line:
        pill_label, pill_kind = "Verified", "good"
    else:
        pill_label, pill_kind = "Unverified", "muted"

    # "Searched" line.
    if kb == "github" and github_intent is not None:
        plain = GITHUB_INTENT_PLAIN.get(github_intent.intent, "ran the standard hybrid search.")
        scope_bit = f"repo `{repository}`" if repository else "all ingested GitHub repos"
        searched_line = f"**Searched:** GitHub ({scope_bit}) — {plain}"
    elif forced_kb is not None:
        searched_line = f"**Searched:** {KB_LABELS.get(forced_kb, forced_kb)} only (this tab is scoped to one source)."
    else:
        kbs_str = ", ".join(target_kbs) if target_kbs else "every populated source"
        searched_line = f"**Searched:** {kbs_str}."

    with st.expander("How this answer was checked", expanded=False):
        top_label, top_pill = st.columns([5, 1])
        top_label.markdown("###### Summary")
        top_pill.markdown(render_status_pill(pill_label, pill_kind), unsafe_allow_html=True)

        st.markdown(
            f'<div class="insight-desc"><p>{searched_line}</p>'
            + (f"<p>{crag_line}</p>" if crag_line else "")
            + (f"<p>{self_rag_line}</p>" if self_rag_line else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        if critique_result is not None and critique_result.hallucinations:
            for sentence in critique_result.hallucinations:
                st.markdown(f"- *{sentence}*")

        with st.expander("Technical details", expanded=False):
            _render_routing_details(decision, target_kbs, forced_kb)
            if kb == "github" and github_intent is not None:
                _render_github_intent_details(github_intent, repository)
            _render_query_transform_details(question, kb)
            if crag_result is not None and getattr(crag_result, "used_llm", False):
                _render_crag_details(crag_result)
            if critique_result is not None:
                _render_self_rag_details(critique_result)


def _render_routing_details(decision, target_kbs: list[str], forced_kb: str | None) -> None:
    if forced_kb is not None:
        st.markdown(f"**Routing bypassed** — forced to KB `{forced_kb}`.")
    else:
        st.markdown(
            f"**Explicit KB signal:** {', '.join(decision.kbs) if decision.kbs else '_none — fell back to every populated KB_'} "
            f"· method: `{decision.method}`"
        )
    st.markdown("KB similarity scores (threshold ≥ 0.40 to count as an explicit match):")
    if decision.kb_scores:
        kb_rows = [
            {
                "KB": kb_name,
                "score": round(score, 3),
                "≥ threshold": "✓" if score >= 0.40 else "",
                "searched": "✓" if (forced_kb is not None and kb_name == forced_kb) or (forced_kb is None and kb_name in target_kbs) else "",
            }
            for kb_name, score in sorted(decision.kb_scores.items(), key=lambda kv: kv[1], reverse=True)
        ]
        st.table(kb_rows)
    if decision.scores:
        st.markdown("Content-type route scores (code / table / image / general, threshold ≥ 0.55):")
        route_rows = [
            {"route": route, "score": round(score, 3), "used": "✓" if route in decision.routes else ""}
            for route, score in sorted(decision.scores.items(), key=lambda kv: kv[1], reverse=True)
        ]
        st.table(route_rows)


def _render_github_intent_details(github_intent, repository: str | None) -> None:
    label, strategy = GITHUB_INTENT_UI.get(github_intent.intent, (github_intent.intent, "(unknown strategy)"))
    st.markdown(f"**GitHub intent:** {label} ({strategy}) · method: `{github_intent.method}`")
    hints = getattr(github_intent, "metadata_hints", {}) or {}
    if hints:
        st.table([{"field": k, "value": v} for k, v in hints.items()])
    if github_intent.scores:
        intent_rows = [
            {"intent": intent, "semantic_score": round(score, 3), "chosen": "✓" if intent == github_intent.intent else ""}
            for intent, score in sorted(github_intent.scores.items(), key=lambda kv: kv[1], reverse=True)
        ]
        st.table(intent_rows)


def _render_query_transform_details(question: str, kb: str | None) -> None:
    if not query_transform_enabled():
        return
    try:
        result = transform_query(question, kb=kb)
    except Exception:  # noqa: BLE001 - observability panel must not crash chat
        logger.exception("Query transform detail failed to fetch cached result.")
        return
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
        return
    st.markdown("**Query variants used:**")
    if result.rewritten and result.rewritten.strip().lower() != question.strip().lower():
        st.markdown(f"- Rewritten: {result.rewritten}")
    for sub in result.sub_queries:
        st.markdown(f"- Sub-question: {sub}")
    for para in result.paraphrases:
        st.markdown(f"- Paraphrase: {para}")
    if result.hyde_answer:
        st.markdown(f"- HyDE: {result.hyde_answer}")
    if result.keywords:
        chips = " ".join(render_status_pill(k, "muted") for k in result.keywords)
        st.markdown(f"- Keywords: {chips}", unsafe_allow_html=True)


def _render_crag_details(crag_result) -> None:
    counts = crag_result.counts
    st.markdown(
        f"**Corrective RAG** — correct: {counts['correct']} · partial: {counts['partial']} · incorrect: {counts['incorrect']}"
    )
    rows = [
        {
            "chunk": f"#{v.index + 1}",
            "relevance": v.relevance,
            "kept": "✓" if v.is_keeper else "✗",
            "reason": v.reason,
        }
        for v in crag_result.verdicts
    ]
    st.table(rows)


def _render_self_rag_details(critique_result) -> None:
    st.markdown("**Self-Reflective RAG scores:**")
    st.table(critique_result.as_dimension_rows())

"""
Streamlit chat UI for IntelliFusion.

Run with:
    streamlit run app_ui.py

Responsibility: provide a simple browser-based chat interface. No
retrieval, prompting, or generation logic lives here - this file only
renders the chat and calls cli.ask_with_vision(), the same pipeline
function the plain CLI (cli.py) uses. That way the CLI and the UI can
never drift apart or duplicate pipeline logic.

Pending-document tracking: documents ingested through the KB tabs
land in Chroma immediately (so the user can chat against them right
away), but are also recorded in data/pending_documents.json keyed by
Streamlit session id. A "Save" button per document removes it from
that tracker, promoting it to a normal permanent KB entry. A
"Discard" button both removes it from the tracker AND deletes its
chunks from the KB. On every app startup, any session in the tracker
whose last_seen_at is older than _ABANDONED_SESSION_MINUTES is
considered abandoned and its pending chunks are auto-deleted - this
is our stand-in for a real "session ended" hook (Streamlit doesn't
provide one).
"""

import json
import logging
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from app.chunking.parent_child import chunk_with_parent_child
from app.embeddings.embedder import embed_chunks
from app.ingestion.ingest import _ingest_images, ingest_github_repo
from app.ingestion.loader import (
    crawl_website,
    load_docx_document,
    load_markdown_document,
    load_pdf_document,
    load_website_document,
)
from app.logging_config import configure_logging
from app.routing.router import classify_route
from app.vectorstore.store import (
    add_embedded_chunks,
    count,
    delete_document,
    delete_repository,
    list_populated_kbs,
    list_repositories,
)
from cli import ask_with_vision, ask_with_vision_stream

configure_logging()
logger = logging.getLogger(__name__)

_PENDING_TRACKER_PATH = Path(__file__).resolve().parent / "data" / "pending_documents.json"
_ABANDONED_SESSION_MINUTES = 30

_KB_LABELS = {
    "markdown": "Markdown",
    "pdf": "PDF",
    "docx": "DOCX",
    "web": "Web",
    "github": "GitHub",
}


# ---------- Pending-document tracker (JSON on disk) ----------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_tracker() -> dict:
    """Load the pending-doc tracker, returning an empty structure if absent."""
    if not _PENDING_TRACKER_PATH.exists():
        return {"sessions": {}}
    try:
        return json.loads(_PENDING_TRACKER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Pending-doc tracker unreadable; starting fresh.")
        return {"sessions": {}}


def _save_tracker(tracker: dict) -> None:
    _PENDING_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PENDING_TRACKER_PATH.write_text(json.dumps(tracker, indent=2), encoding="utf-8")


def _empty_pending() -> dict:
    return {kb: [] for kb in _KB_LABELS}


def _cleanup_abandoned_sessions(tracker: dict, current_session_id: str) -> dict:
    """
    Delete pending chunks from any session whose last_seen_at is older
    than _ABANDONED_SESSION_MINUTES (and isn't the current session).
    Returns the updated tracker. The threshold protects against
    accidentally nuking a still-open session in another browser tab.
    """
    threshold = datetime.now(timezone.utc) - timedelta(minutes=_ABANDONED_SESSION_MINUTES)
    surviving: dict[str, dict] = {}
    for session_id, session_data in tracker.get("sessions", {}).items():
        if session_id == current_session_id:
            surviving[session_id] = session_data
            continue
        try:
            last_seen = datetime.fromisoformat(session_data.get("last_seen_at", ""))
        except ValueError:
            last_seen = datetime.min.replace(tzinfo=timezone.utc)
        if last_seen >= threshold:
            surviving[session_id] = session_data
            continue
        # Abandoned - purge its pending chunks from the KBs.
        pending = session_data.get("pending", {})
        for kb, entries in pending.items():
            for entry in entries:
                if entry.get("repository"):
                    delete_repository(entry["repository"])
                elif entry.get("document_id"):
                    delete_document(entry["document_id"], kb=kb)
        logger.info("Cleaned up abandoned session %s (%d KB(s))", session_id, len(pending))
    tracker["sessions"] = surviving
    return tracker


def _ensure_session(tracker: dict, session_id: str) -> dict:
    """Make sure the current session has an entry in the tracker."""
    sessions = tracker.setdefault("sessions", {})
    if session_id not in sessions:
        sessions[session_id] = {
            "started_at": _now_iso(),
            "last_seen_at": _now_iso(),
            "pending": _empty_pending(),
        }
    else:
        sessions[session_id]["last_seen_at"] = _now_iso()
        # Defensively make sure every KB key exists (in case _KB_LABELS
        # gained a new KB after the tracker file was first written).
        for kb in _KB_LABELS:
            sessions[session_id]["pending"].setdefault(kb, [])
    return tracker


def _get_pending(kb: str) -> list[dict]:
    tracker = _load_tracker()
    session = tracker.get("sessions", {}).get(st.session_state["session_id"], {})
    return session.get("pending", {}).get(kb, [])


def _add_pending(kb: str, entry: dict) -> None:
    tracker = _load_tracker()
    tracker = _ensure_session(tracker, st.session_state["session_id"])
    tracker["sessions"][st.session_state["session_id"]]["pending"][kb].append(entry)
    _save_tracker(tracker)


def _remove_pending(kb: str, entry_key: str, entry_value: str) -> None:
    tracker = _load_tracker()
    session = tracker.get("sessions", {}).get(st.session_state["session_id"])
    if not session:
        return
    session["pending"][kb] = [
        entry for entry in session["pending"][kb] if entry.get(entry_key) != entry_value
    ]
    _save_tracker(tracker)


# ---------- Ingestion helpers (reuse the existing pipeline modules) ----------


def _ingest_documents(documents, document_id_override: str | None = None) -> tuple[int, list[str]]:
    """
    Chunk, embed, and store a list of Documents. If document_id_override
    is given, force every chunk's document_id metadata to that string
    BEFORE storing (so file uploads land under their original filename
    rather than the temp path the loader saw). Returns (total_chunks,
    unique_document_ids_stored).
    """
    total_chunks = 0
    document_ids: list[str] = []
    for document in documents:
        if document_id_override is not None:
            document.metadata["document_id"] = document_id_override
        chunks = chunk_with_parent_child(document)
        for chunk in chunks:
            if document_id_override is not None:
                chunk.metadata["document_id"] = document_id_override
        embedded = embed_chunks(chunks)
        add_embedded_chunks(embedded)
        total_chunks += len(chunks)
        doc_id = document.metadata.get("document_id")
        if doc_id and doc_id not in document_ids:
            document_ids.append(doc_id)
    return total_chunks, document_ids


def _ingest_uploaded_files(uploaded_files, loader, kb: str) -> None:
    """
    Ingest each uploaded file and record it as pending in the tracker.
    document_id is forced to the ORIGINAL uploaded filename (not the
    temp path) so the same file re-uploaded upserts instead of piling
    up duplicates, and so the Pending list shows the real filename.
    """
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            result = loader(tmp_path)
            documents = result if isinstance(result, list) else [result]
            _ingest_documents(documents, document_id_override=uploaded_file.name)
            _add_pending(kb, {"document_id": uploaded_file.name, "label": uploaded_file.name})
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------- Streamlit boilerplate ----------


st.set_page_config(page_title="IntelliFusion", page_icon="🧠")
st.title("🧠 IntelliFusion")
st.caption(
    "Multi-Source Knowledge Intelligence Platform — ask questions across "
    "Markdown, PDF, DOCX, websites and GitHub repos. Answers are grounded "
    "in retrieved context only (local embeddings + Chroma + a local Ollama "
    "model), with query routing to the right knowledge base(s)."
)

# Give this browser session a stable id, then run the "clean up sessions
# that were abandoned > _ABANDONED_SESSION_MINUTES ago" pass exactly
# once per new session (that's what the "session_id not yet in state"
# check gates).
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex
    _tracker = _load_tracker()
    _tracker = _cleanup_abandoned_sessions(_tracker, st.session_state["session_id"])
    _tracker = _ensure_session(_tracker, st.session_state["session_id"])
    _save_tracker(_tracker)
else:
    # Keep the current session's last_seen_at fresh, so a long chat
    # session (with no ingest activity) isn't mistaken for abandoned
    # by the next browser tab that opens the app.
    _tracker = _ensure_session(_load_tracker(), st.session_state["session_id"])
    _save_tracker(_tracker)

# Per-KB chat histories, plus a separate global one for the cross-KB
# chat area below the tabs. Each is a list of {"role", "content",
# "images"} dicts.
if "messages_by_kb" not in st.session_state:
    st.session_state["messages_by_kb"] = {kb: [] for kb in _KB_LABELS}
if "messages_global" not in st.session_state:
    st.session_state["messages_global"] = []

top_k = st.sidebar.slider("Top-K chunks to retrieve", min_value=1, max_value=8, value=3)
use_vision = st.sidebar.checkbox(
    "Always use vision model on images (slow, may time out on CPU)",
    value=False,
    help=(
        "Code screenshots are already auto-detected and sent to the vision "
        "model regardless of this setting (see app.ocr.image_classifier). "
        "Turning this on ALSO sends every other loosely-relevant image to "
        "the vision model, which can take several minutes per question on "
        "CPU-only machines and may time out."
    ),
)

st.sidebar.divider()
st.sidebar.subheader("Cross-KB chat: scope to a repo")
_ingested_repos = list_repositories()
repo_filter = st.sidebar.selectbox(
    "Repository (optional)",
    options=["(search everything)"] + _ingested_repos,
    help=(
        "Only affects the cross-KB chat area below the tabs. Each KB "
        "tab's own chat searches only that KB regardless of this setting."
    ),
)
active_repository = repo_filter if repo_filter != "(search everything)" else None


# ---------- Rendering helpers ----------


def _render_images(image_hits: list[dict]) -> None:
    """Render any relevant retrieved images below an answer, with a citation."""
    for image_hit in image_hits:
        metadata = image_hit["metadata"]
        st.image(metadata["image_url"], caption=metadata.get("alt_text"))
        page_url = metadata.get("page_url")
        if page_url:
            st.caption(f"From: {page_url}")


def _run_streaming_query(
    question: str,
    *,
    top_k: int,
    use_vision: bool,
    kb: str | None = None,
    repository: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Retrieve + call ask_with_vision_stream + render tokens live inside
    the current st.chat_message container.

    Wraps two invisible-but-slow phases in a visible st.spinner so the
    assistant bubble doesn't sit blank for ~15-30s while the user
    thinks the app is frozen:
      1. retrieval (~2-3s: query embed + Chroma query + cross-encoder rerank)
      2. LLM prompt-processing on CPU (~15-30s for a 7B model chewing
         through a ~3000-token prompt before it emits the first token).

    Once the first token arrives the spinner exits and st.write_stream
    takes over to render the remaining tokens live. Returns the fully-
    accumulated answer string (for persisting to session state) plus
    the image hits (for _render_images to display below).
    """
    with st.spinner("Thinking… (retrieval + LLM warm-up, first token can take ~15-30 s on CPU)"):
        token_iter, image_hits = ask_with_vision_stream(
            question,
            top_k=top_k,
            use_vision=use_vision,
            kb=kb,
            repository=repository,
        )
        # Force the FIRST token before exiting the spinner so the
        # spinner covers all initial latency; every subsequent token
        # streams live via st.write_stream below.
        try:
            first_token = next(token_iter)
        except StopIteration:
            first_token = None

    if first_token is None:
        # Model produced nothing (unusual - typically a config/timeout
        # issue). Surface an explicit message rather than an empty bubble
        # so the user can tell "no answer" apart from "still thinking".
        st.warning("The model returned an empty response. Check the terminal log for errors.")
        return "", image_hits

    def _stream_with_first():
        yield first_token
        yield from token_iter

    answer = st.write_stream(_stream_with_first())
    return answer, image_hits


def _render_pending_docs(kb: str) -> None:
    """
    Render the pending-in-this-session list for a KB, with per-doc
    Save (keep in KB, remove from pending) and Discard (delete chunks
    from KB, remove from pending) buttons.
    """
    pending = _get_pending(kb)
    if not pending:
        st.caption("No pending documents in this session.")
        return
    st.markdown("**Pending in this session** (save to keep, or they'll be auto-removed):")
    for entry in pending:
        col_label, col_save, col_discard = st.columns([6, 1, 1])
        col_label.markdown(f"• `{entry['label']}`")
        entry_key = "repository" if entry.get("repository") else "document_id"
        entry_value = entry[entry_key]
        # Streamlit widget keys need to be unique per (kb, entry) pair.
        widget_suffix = f"{kb}_{entry_key}_{entry_value}"
        if col_save.button("💾 Save", key=f"save_{widget_suffix}"):
            _remove_pending(kb, entry_key, entry_value)
            st.rerun()
        if col_discard.button("🗑 Discard", key=f"discard_{widget_suffix}"):
            if entry_key == "repository":
                delete_repository(entry_value)
            else:
                delete_document(entry_value, kb=kb)
            _remove_pending(kb, entry_key, entry_value)
            st.rerun()


def _render_routing_panel(decision, target_kbs: list[str], forced_kb: str | None = None) -> None:
    """
    Show which KB(s) and content-type route(s) the router picked for a
    query, with per-KB and per-route similarity scores. Purely
    observability - the retriever has already made the decision by the
    time this renders.

    When forced_kb is set (per-tab chat), the panel highlights that
    routing was BYPASSED for this query and shows what routing *would*
    have picked, so the user can spot when they've asked a question in
    the "wrong" tab (e.g. a code question inside the PDF tab).
    """
    if forced_kb is not None:
        would_route_here = forced_kb in (decision.kbs or [])
        top_kb = max(decision.kb_scores.items(), key=lambda kv: kv[1])[0] if decision.kb_scores else None
        top_score = decision.kb_scores.get(top_kb, 0.0) if top_kb else 0.0
        this_score = decision.kb_scores.get(forced_kb, 0.0)
        header = (
            f"🔀 Routing (bypassed) → forced KB: **{forced_kb}** "
            f"| this KB score: {this_score:.3f} "
            f"| top KB by score: **{top_kb}** ({top_score:.3f})"
        )
    else:
        header = (
            f"🔀 Routing → searched KBs: {', '.join(target_kbs) if target_kbs else '(none)'} "
            f"| routes: {', '.join(decision.routes)} | method: {decision.method}"
        )

    with st.expander(header, expanded=False):
        if forced_kb is not None:
            if would_route_here:
                st.info(f"✅ Routing would have also picked `{forced_kb}` for this query.")
            elif decision.kbs:
                st.warning(
                    f"⚠️ Routing would have picked {decision.kbs} for this query, "
                    f"not `{forced_kb}`. You may want to ask this in a different tab."
                )
            else:
                st.caption(
                    "No explicit KB signal in this query — routing would have searched "
                    "every populated KB. The current tab restricts to just this one."
                )
        else:
            st.markdown(
                f"**Explicit KB signal:** {', '.join(decision.kbs) if decision.kbs else '_none — fell back to every populated KB_'}"
            )

        st.markdown("**KB similarity scores** (threshold ≥ 0.40 to count as an explicit match):")
        if decision.kb_scores:
            kb_rows = [
                {
                    "KB": kb,
                    "score": round(score, 3),
                    "≥ threshold": "✅" if score >= 0.40 else "",
                    "searched": "✅" if (forced_kb is not None and kb == forced_kb) or (forced_kb is None and kb in target_kbs) else "",
                }
                for kb, score in sorted(decision.kb_scores.items(), key=lambda kv: kv[1], reverse=True)
            ]
            st.table(kb_rows)
        else:
            st.caption("(no KB similarity scores available)")

        # Every text KB now goes through the hybrid retrieval
        # pipeline (dense + BM25 → RRF → cross-encoder). GitHub and
        # Web add a third RRF input on top of that: a parent-child
        # sentence-window retriever that searches small "child" slices
        # and resolves each hit back to its full parent paragraph, so
        # the LLM still gets the surrounding context and not just the
        # matched sentence. Flag both tiers in the routing panel so
        # it's obvious which KBs use which extra layers.
        hybrid_kbs = {"pdf", "docx", "markdown", "github", "web"}
        parent_child_kbs = {"github", "web"}
        in_scope = {forced_kb} if forced_kb is not None else set(target_kbs)
        hybrid_kbs_in_scope = hybrid_kbs & in_scope
        parent_child_in_scope = parent_child_kbs & in_scope
        if hybrid_kbs_in_scope:
            kbs_str = ", ".join(sorted(hybrid_kbs_in_scope)).upper()
            st.info(
                f"📄 **{kbs_str} KB(s) use hybrid retrieval:** "
                "Dense (MiniLM) + BM25 → Reciprocal Rank Fusion → Cross-Encoder rerank "
                "(`ms-marco-MiniLM-L-6-v2`)."
            )
        if parent_child_in_scope:
            pc_str = ", ".join(sorted(parent_child_in_scope)).upper()
            st.info(
                f"🧩 **{pc_str} KB(s) also use parent-child retrieval:** "
                "children (sentence windows, ~40 words) are dense-searched for precision, "
                "then resolved to their full parent paragraph before the cross-encoder rerank."
            )

        st.markdown("**Content-type route scores** (code / table / image / general, threshold ≥ 0.55):")
        if decision.scores:
            route_rows = [
                {
                    "route": route,
                    "score": round(score, 3),
                    "used": "✅" if route in decision.routes else "",
                }
                for route, score in sorted(decision.scores.items(), key=lambda kv: kv[1], reverse=True)
            ]
            st.table(route_rows)
        else:
            st.caption("(no route similarity scores available)")


def _render_kb_chat(kb: str) -> None:
    """
    Render a chat scoped to a single KB: history + text_input form.
    Uses a form (not st.chat_input) so it renders reliably inside a
    tab; st.chat_input is reserved for the top-level cross-KB chat.
    """
    label = _KB_LABELS[kb]
    st.markdown(f"**💬 Ask about your {label} documents**")
    for message in st.session_state["messages_by_kb"][kb]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_images(message.get("images", []))
            routing = message.get("routing")
            if routing is not None:
                _render_routing_panel(routing["decision"], routing["target_kbs"], forced_kb=kb)
    with st.form(key=f"chat_form_{kb}", clear_on_submit=True):
        question = st.text_input(
            f"Ask about {label}",
            key=f"chat_input_{kb}",
            label_visibility="collapsed",
            placeholder=f"Ask a question about your {label} content...",
        )
        submitted = st.form_submit_button("Send")
    if submitted and question:
        logger.info("UI (kb=%s) question: %r", kb, question)
        st.session_state["messages_by_kb"][kb].append({"role": "user", "content": question})
        # Classify for observability (retriever bypasses this because
        # we force kb=kb) so the user can see what routing WOULD have
        # picked and spot mis-tabbed questions.
        decision = classify_route(question)
        # Show the just-submitted user turn immediately (before rerun)
        # so it sits above the streaming assistant response, matching
        # the message order the user will see after rerun renders
        # everything from session state.
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            # _run_streaming_query wraps retrieval + LLM warm-up in a
            # visible spinner (so the bubble doesn't look frozen for
            # ~15-30s while the 7B model on CPU processes the prompt),
            # then streams tokens live via st.write_stream once the
            # first token arrives. Returns the fully-accumulated answer
            # for persisting to session state.
            answer, image_hits = _run_streaming_query(
                question, top_k=top_k, use_vision=use_vision, kb=kb
            )
            _render_images(image_hits)
        st.session_state["messages_by_kb"][kb].append(
            {
                "role": "assistant",
                "content": answer,
                "images": image_hits,
                "routing": {"decision": decision, "target_kbs": [kb]},
            }
        )
        st.rerun()


def _render_upload_kb_tab(kb: str, uploader_types: list[str], loader) -> None:
    """Render a file-upload-driven KB tab (markdown/pdf/docx)."""
    label = _KB_LABELS[kb]
    st.metric(f"Chunks in {label} KB", count(kb))
    uploaded = st.file_uploader(
        f"Add {label} files",
        type=uploader_types,
        accept_multiple_files=True,
        key=f"upload_{kb}",
    )
    if uploaded and st.button(f"Ingest {label} files", key=f"btn_ingest_{kb}"):
        with st.spinner(f"Ingesting {len(uploaded)} {label} file(s)..."):
            try:
                _ingest_uploaded_files(uploaded, loader, kb=kb)
                st.success(f"Ingested {len(uploaded)} file(s). Marked as pending — save to keep.")
                st.rerun()
            except Exception as error:  # noqa: BLE001 - surface any loader failure to the UI
                logger.exception("%s ingestion failed", label)
                st.error(f"Failed to ingest: {error}")
    _render_pending_docs(kb)
    st.divider()
    _render_kb_chat(kb)


# ---------- Main layout: 5 KB tabs ----------


st.subheader("📚 Knowledge Bases")
_md_tab, _pdf_tab, _docx_tab, _web_tab, _gh_tab = st.tabs(
    ["📄 Markdown", "📕 PDF", "📘 DOCX", "🌐 Web", "🐙 GitHub"]
)

with _md_tab:
    _render_upload_kb_tab("markdown", ["md"], load_markdown_document)

with _pdf_tab:
    _render_upload_kb_tab("pdf", ["pdf"], load_pdf_document)

with _docx_tab:
    _render_upload_kb_tab("docx", ["docx"], load_docx_document)

with _web_tab:
    st.metric("Chunks in Web KB", count("web"))
    web_url = st.text_input(
        "Ingest a URL",
        placeholder="https://example.com/docs/page",
        key="input_web_url",
    )
    # Crawl mode fetches the whole in-scope doc site starting from
    # the URL (via app.ingestion.loader.crawl_website). Required for
    # JS-rendered SPA docs like IBM Docs (ibm.com/docs/en/...), whose
    # single-page fetch returns a near-empty HTML shell and produces
    # 0 chunks - crawl_website special-cases IBM Docs and uses their
    # public TOC+content API to pull every page reliably. Leave
    # unchecked for a plain static single-page fetch.
    crawl_mode = st.checkbox(
        "Crawl the whole site from this URL (needed for IBM Docs / SPA docs)",
        key="input_web_crawl",
    )
    # When crawling, default to the WHOLE in-scope site. Only cap
    # page count if the user explicitly opts in - useful for smoke
    # tests against big doc sites, or to keep an ingest short while
    # tuning chunking/retrieval.
    limit_pages = st.checkbox(
        "Limit number of pages to crawl",
        key="input_web_limit_pages",
        disabled=not crawl_mode,
    )
    max_pages = st.number_input(
        "Max pages to crawl",
        min_value=1,
        max_value=5000,
        value=30,
        step=1,
        key="input_web_max_pages",
        disabled=not (crawl_mode and limit_pages),
    )
    if web_url and st.button("Ingest URL", key="btn_web"):
        with st.spinner(f"{'Crawling' if crawl_mode else 'Fetching'} '{web_url}'..."):
            try:
                if crawl_mode:
                    # Effectively "unlimited" when the user didn't opt
                    # into a cap - crawl_website still respects its
                    # in-scope filter (same host + nested-under-seed
                    # path) so this can't run away across the whole
                    # internet, only exhausts the seed's sub-site.
                    effective_max_pages = int(max_pages) if limit_pages else 10_000
                    documents = crawl_website(web_url, max_pages=effective_max_pages)
                    if not documents:
                        st.warning(
                            f"Crawl of '{web_url}' returned 0 pages - the URL may be "
                            "unreachable, out of scope, or the site returned no crawlable links."
                        )
                    else:
                        _ingest_documents(documents)
                        for doc in documents:
                            _ingest_images(doc)
                        # NOTE: crawled pages are auto-saved (NOT marked
                        # pending). A whole-site crawl is a "keep it all"
                        # workflow - the user has already opted in by
                        # ticking Crawl mode and letting it run for
                        # hours - so requiring a per-page Save click and
                        # relying on the 30-minute abandonment timer is
                        # the wrong shape here (an idle Streamlit tab
                        # during a long crawl would otherwise wipe the
                        # whole ingest via _cleanup_abandoned_sessions).
                        # Single-page ingest below keeps the pending
                        # flow, where a "test then save" workflow does
                        # make sense.
                        st.success(
                            f"Crawled, ingested, and saved {len(documents)} page(s) from '{web_url}'."
                        )
                        st.rerun()
                else:
                    document = load_website_document(web_url)
                    if not document.content.strip():
                        st.warning(
                            f"Fetched '{web_url}' but got 0 characters of extractable text. "
                            "This usually means the site is JavaScript-rendered (e.g. IBM Docs) - "
                            "try enabling 'Crawl the whole site from this URL' above."
                        )
                    else:
                        _ingest_documents([document])
                        _ingest_images(document)
                        _add_pending("web", {"document_id": web_url, "label": web_url})
                        st.success(f"Ingested '{web_url}'. Marked as pending — save to keep.")
                        st.rerun()
            except Exception as error:  # noqa: BLE001
                logger.exception("Website ingestion failed for '%s'", web_url)
                st.error(f"Failed to ingest '{web_url}': {error}")
    _render_pending_docs("web")
    st.divider()
    _render_kb_chat("web")

with _gh_tab:
    st.metric("Chunks in GitHub KB", count("github"))
    if _ingested_repos:
        st.markdown("**All ingested repos:**")
        for repo in _ingested_repos:
            st.markdown(f"- `{repo}`")
    github_repo_url = st.text_input(
        "Repo (owner/repo or full URL)",
        placeholder="e.g. https://github.com/owner/repo",
        key="input_gh_repo",
    )
    github_branch = st.text_input(
        "Branch (optional)",
        placeholder="default branch if left blank",
        key="input_gh_branch",
    )
    if github_repo_url and st.button("Ingest repo", key="btn_gh"):
        with st.spinner(f"Ingesting '{github_repo_url}'..."):
            try:
                ingest_github_repo(github_repo_url, branch=github_branch or None)
                # ingest_github_repo doesn't return the "owner/repo"
                # label back, so derive it the same way loader.py does:
                # take the last two path segments, strip a trailing .git.
                parsed = github_repo_url.rstrip("/").removesuffix(".git")
                parts = parsed.split("/")
                owner_repo = "/".join(parts[-2:]) if len(parts) >= 2 else parsed
                _add_pending(
                    "github", {"repository": owner_repo, "label": owner_repo}
                )
                st.success(
                    f"Ingested '{owner_repo}'. Marked as pending — save to keep, "
                    f"or Discard to remove every chunk from this repo."
                )
                st.rerun()
            except Exception as error:  # noqa: BLE001
                logger.exception("GitHub repo ingestion failed for '%s'", github_repo_url)
                st.error(f"Failed to ingest '{github_repo_url}': {error}")
    _render_pending_docs("github")
    st.divider()
    _render_kb_chat("github")

st.divider()


# ---------- Cross-KB chat (uses routing to pick KBs) ----------


st.subheader("🔀 Cross-KB chat")
st.caption(
    "Uses query routing to pick which KB(s) to search - handy for questions "
    "that span multiple sources or don't obviously belong to one KB."
)

for message in st.session_state["messages_global"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_images(message.get("images", []))
        routing = message.get("routing")
        if routing is not None:
            _render_routing_panel(routing["decision"], routing["target_kbs"])

question = st.chat_input("Ask a question about your ingested sources...")

if question:
    logger.info("UI (cross-KB) question received: %r (top_k=%d)", question, top_k)
    st.session_state["messages_global"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Classify BEFORE calling _run_streaming_query so we can
        # show the same decision the retriever will make internally.
        # One extra embed call per question - negligible vs. LLM.
        decision = classify_route(question)
        target_kbs = decision.kbs or list_populated_kbs()
        # _run_streaming_query wraps retrieval + LLM warm-up in a
        # visible spinner (so the bubble doesn't look frozen for
        # ~15-30s while the 7B model on CPU processes the prompt),
        # then streams tokens live via st.write_stream once the
        # first token arrives.
        answer, image_hits = _run_streaming_query(
            question, top_k=top_k, use_vision=use_vision, repository=active_repository
        )
        _render_images(image_hits)
        _render_routing_panel(decision, target_kbs)

    st.session_state["messages_global"].append(
        {
            "role": "assistant",
            "content": answer,
            "images": image_hits,
            "routing": {"decision": decision, "target_kbs": target_kbs},
        }
    )




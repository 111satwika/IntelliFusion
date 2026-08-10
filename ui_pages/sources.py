"""
Sources page: ingest documents into each of the 5 knowledge bases
(Markdown / PDF / DOCX / Web / GitHub), see everything already
indexed (or still pending-in-this-session), and chat scoped to just
that KB for testing what you just ingested.

*** DEPRECATED (see app_ui.py's module docstring): no controls here
for ingesting GitHub Issues/Pull Requests/Discussions - that's only
available via the FastAPI webapp (POST /api/sources/github/activity,
see app.ingestion.ingest.ingest_github_activity).
"""

import logging

import streamlit as st

from app.ingestion.ingest import _ingest_images, ingest_github_repo
from app.ingestion.loader import (
    crawl_website,
    load_docx_document,
    load_markdown_document,
    load_pdf_document,
    load_website_document,
)
from app.vectorstore.store import count, list_documents, list_repositories

from ui.chat import render_kb_chat
from ui.icons import svg
from ui.ingestion import ingest_documents, ingest_uploaded_files
from ui.panels import render_source_rows
from ui.state import KB_LABELS, add_pending

logger = logging.getLogger(__name__)

_top_k = st.session_state["top_k"]
_use_vision = st.session_state["use_vision"]

_total_chunks = sum(count(kb) for kb in KB_LABELS)
_total_sources = sum(len(list_documents(kb)) for kb in KB_LABELS)

header_left, header_right = st.columns([5, 2], vertical_alignment="bottom")
with header_left:
    st.markdown(f'<div class="page-title-row">{svg("folder", size=24, stroke="var(--accent)")}<h1>Sources</h1></div>', unsafe_allow_html=True)
with header_right:
    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
        st.markdown(
            f'<span class="chip">{_total_sources} source{"s" if _total_sources != 1 else ""} · {_total_chunks} chunks</span>',
            unsafe_allow_html=True,
        )
        with st.popover("⋯"):
            if st.button("Refresh counts"):
                st.rerun()

st.caption("Add sources to each knowledge base, then chat with just that source below.")
st.divider()

if "active_source_type" not in st.session_state:
    st.session_state["active_source_type"] = "markdown"

_card_cols = st.columns(len(KB_LABELS))
for _col, _kb in zip(_card_cols, KB_LABELS):
    _is_active = st.session_state["active_source_type"] == _kb
    # Key PREFIX (not just value) encodes active state for ui/theme.py's
    # CSS, same trick as the sidebar nav (see ui/sidebar.py).
    _card_key = f"source_card_active_{_kb}" if _is_active else f"source_card_{_kb}"
    with _col:
        with st.container(border=True, key=_card_key, horizontal_alignment="center"):
            icon_fg = "var(--accent-strong)" if _is_active else "var(--ink-soft)"
            st.markdown(
                f'<div style="width:40px; height:40px; border-radius:10px; margin:0 auto 8px; '
                f'background:{"var(--surface)" if _is_active else "var(--surface-sunken)"}; '
                f'display:flex; align-items:center; justify-content:center;">'
                f'{svg(_kb, size=21, stroke=icon_fg)}</div>',
                unsafe_allow_html=True,
            )
            _picked = st.button(
                KB_LABELS[_kb],
                key=f"pick_{_kb}",
                type="tertiary",
                use_container_width=True,
            )
        if _picked:
            st.session_state["active_source_type"] = _kb
            st.rerun()

st.divider()
active_kb = st.session_state["active_source_type"]


def _render_upload_kb_tab(kb: str, uploader_types: list[str], loader) -> None:
    """Render a file-upload-driven KB tab (markdown/pdf/docx)."""
    label = KB_LABELS[kb]
    uploaded = st.file_uploader(
        f"Add {label} files",
        type=uploader_types,
        accept_multiple_files=True,
        key=f"upload_{kb}",
    )
    if uploaded and st.button(f"Ingest {label} files", key=f"btn_ingest_{kb}"):
        with st.spinner(f"Ingesting {len(uploaded)} {label} file(s)..."):
            try:
                ingest_uploaded_files(uploaded, loader, kb=kb)
                st.success(f"Ingested {len(uploaded)} file(s). Marked as pending — save to keep.")
                st.rerun()
            except Exception as error:  # noqa: BLE001 - surface any loader failure to the UI
                logger.exception("%s ingestion failed", label)
                st.error(f"Failed to ingest: {error}")
    render_source_rows(kb)
    st.divider()
    render_kb_chat(kb, top_k=_top_k, use_vision=_use_vision)


if active_kb == "markdown":
    _render_upload_kb_tab("markdown", ["md"], load_markdown_document)

elif active_kb == "pdf":
    _render_upload_kb_tab("pdf", ["pdf"], load_pdf_document)

elif active_kb == "docx":
    _render_upload_kb_tab("docx", ["docx"], load_docx_document)

elif active_kb == "web":
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
                        ingest_documents(documents)
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
                        ingest_documents([document])
                        _ingest_images(document)
                        add_pending("web", {"document_id": web_url, "label": web_url})
                        st.success(f"Ingested '{web_url}'. Marked as pending — save to keep.")
                        st.rerun()
            except Exception as error:  # noqa: BLE001
                logger.exception("Website ingestion failed for '%s'", web_url)
                st.error(f"Failed to ingest '{web_url}': {error}")
    render_source_rows("web")
    st.divider()
    render_kb_chat("web", top_k=_top_k, use_vision=_use_vision)

elif active_kb == "github":
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
                add_pending(
                    "github", {"repository": owner_repo, "label": owner_repo}
                )
                st.success(
                    f"Ingested '{owner_repo}'. Marked as pending — save to keep, "
                    f"or remove it below to delete every chunk from this repo."
                )
                st.rerun()
            except Exception as error:  # noqa: BLE001
                logger.exception("GitHub repo ingestion failed for '%s'", github_repo_url)
                st.error(f"Failed to ingest '{github_repo_url}': {error}")
    render_source_rows("github")
    st.divider()
    # Per-tab repo scope. Distinct from the Chat page's cross-KB repo
    # picker (which only affects that page) - the GitHub KB can hold
    # multiple ingested repos and mixing chunks from two different
    # codebases in one answer leads to wrong-repo citations (a
    # `Router` class from repoA quoted next to a method from repoB).
    # Also mandatory for structural (GraphRAG loads
    # {owner}__{repo}.gpickle) and history (needs a repo to hit the
    # commits API) intents - both silently downgrade to hybrid without
    # a repo scope.
    _github_scope = st.selectbox(
        "Scope questions to a repo",
        options=["(all ingested GitHub repos)"] + list_repositories(),
        key="github_tab_repo_scope",
        help=(
            "When multiple repos are ingested, pick one here so "
            "citations and graph traversal stay inside that repo. "
            "Leaving this on '(all)' can mix chunks from different "
            "repos in a single answer."
        ),
    )
    _github_scope_repo = (
        _github_scope if _github_scope != "(all ingested GitHub repos)" else None
    )
    render_kb_chat("github", top_k=_top_k, use_vision=_use_vision, repository=_github_scope_repo)

"""
Chat page (default/landing page): the cross-KB chat, which uses query
routing to pick which knowledge base(s) to search - the primary way
most users will interact with the app, so it's the first thing they
see rather than being buried below the ingestion tabs.
"""

import streamlit as st

from app.vectorstore.store import count, list_populated_kbs, list_repositories
from ui.chat import render_cross_kb_chat
from ui.icons import svg
from ui.state import KB_LABELS

st.markdown(
    f'<div class="page-title-row">{svg("brand", size=24, stroke="var(--accent)")}<h1>IntelliFusion</h1></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Ask questions across Markdown, PDF, DOCX, websites and GitHub repos. "
    "Answers are grounded in retrieved context only, with query routing to "
    "the right knowledge base(s)."
)

_populated = list_populated_kbs()
if _populated:
    cols = st.columns(len(KB_LABELS))
    for col, kb in zip(cols, KB_LABELS):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:8px; color:var(--muted); '
                    f'font-size:0.8rem; font-weight:600; margin-bottom:8px;">'
                    f'{svg(kb, size=15, stroke="var(--muted)")}{KB_LABELS[kb]}</div>'
                    f'<div class="num" style="font-size:1.4rem; font-weight:600;">{count(kb):,}</div>',
                    unsafe_allow_html=True,
                )
else:
    st.info("No sources indexed yet — add some on the **Sources** page to get started.")

st.divider()

_ingested_repos = list_repositories()
repo_filter = st.selectbox(
    "Scope this chat to a repository (optional)",
    options=["(search everything)"] + _ingested_repos,
    key="cross_kb_repo_scope",
    help="Only affects this chat. Each Sources tab's own chat searches only that KB regardless of this setting.",
)
active_repository = repo_filter if repo_filter != "(search everything)" else None

render_cross_kb_chat(
    top_k=st.session_state["top_k"],
    use_vision=st.session_state["use_vision"],
    repository=active_repository,
)

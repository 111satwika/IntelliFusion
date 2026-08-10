"""
Settings page: retrieval and advanced-RAG toggles, reached via the nav
rail instead of always sitting in the sidebar. Widget keys ("top_k",
"use_vision") are read directly from st.session_state by the Chat and
Sources pages - defaults are pre-seeded in ui.state.init_session() so
those pages work correctly even before this page has ever been
visited in a session.

*** DEPRECATED (see app_ui.py's module docstring): the toggles below
flip the process-global module flags directly (fine for this
single-session Streamlit process), NOT the per-session settings
mechanism api/settings.py + session_store.py use for the current
FastAPI webapp - the two are not interoperable.
"""

import streamlit as st

from app.generation.self_reflection import (
    is_enabled as self_rag_enabled,
    set_enabled as set_self_rag_enabled,
)
from app.retrieval.crag import is_enabled as crag_enabled, set_enabled as set_crag_enabled
from app.retrieval.query_transform import (
    is_enabled as query_transform_enabled,
    set_enabled as set_query_transform_enabled,
)
from ui.icons import svg

st.markdown(
    f'<div class="page-title-row">{svg("settings", size=24, stroke="var(--accent)")}<h1>Settings</h1></div>',
    unsafe_allow_html=True,
)
st.caption("These apply to every chat in the app.")

with st.container(border=True):
    st.subheader("Retrieval")
    st.slider(
        "Chunks per answer",
        min_value=1, max_value=8,
        key="top_k",
        help="How many passages are retrieved and handed to the model as context.",
    )
    st.checkbox(
        "Always use vision on images",
        key="use_vision",
        help=(
            "Code screenshots are read automatically. This also sends every "
            "other loosely-relevant image to the vision model, which can take "
            "several minutes per question on CPU-only machines and may time out."
        ),
    )

with st.container(border=True):
    st.subheader("Answer quality")
    st.caption("Extra verification passes. Each one adds a few seconds per question.")

    query_transform_toggle = st.checkbox(
        "Rewrite unclear questions",
        value=query_transform_enabled(),
        key="query_transform_toggle",
        help=(
            "Expands short or compound questions into clearer search terms "
            "(rewrite, paraphrases, sub-questions, keyword/HyDE expansions) "
            "before retrieving. Adds ~2-4 s to non-trivial questions on CPU; "
            "trivial short questions skip this entirely."
        ),
    )
    set_query_transform_enabled(query_transform_toggle)

    crag_toggle = st.checkbox(
        "Fact-check sources before answering",
        value=crag_enabled(),
        key="crag_toggle",
        help=(
            "Grades each retrieved passage for relevance to your question, drops "
            "the irrelevant ones, and searches again with a reworded question if "
            "too few useful passages remain. (Corrective RAG.) Adds ~15-25 s on CPU."
        ),
    )
    set_crag_enabled(crag_toggle)

    self_rag_toggle = st.checkbox(
        "Fact-check the answer after writing it",
        value=self_rag_enabled(),
        key="self_rag_toggle",
        help=(
            "Re-reads the finished answer against its sources and flags any "
            "sentence that isn't actually backed by them. (Self-Reflective RAG.) "
            "Adds ~20-40 s on CPU but doesn't affect first-token latency."
        ),
    )
    set_self_rag_enabled(self_rag_toggle)

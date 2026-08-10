"""
Streamlit multi-page UI for IntelliFusion.

*** DEPRECATED - superseded by server.py + webapp/ (FastAPI + plain
HTML/JS), which is now the primary UI (see the README's "How to run
it" section). This file is kept runnable for reference, but is NOT
being kept in sync with newer platform features:

  - The RAG-quality toggles (CRAG / query-transform / Self-RAG) on
    ui_pages/settings.py still flip the process-global module flags
    in app.retrieval.crag / app.retrieval.query_transform /
    app.generation.self_reflection directly, not the per-session
    settings mechanism the FastAPI app uses (see session_store.py,
    api/settings.py) - fine for this single-session Streamlit process,
    but a fully separate, non-interoperable toggle path from the one
    the webapp uses.
  - ui_pages/sources.py has no ingestion controls for GitHub Issues /
    Pull Requests / Discussions (see app.ingestion.ingest's
    ingest_github_activity - webapp/api/sources.py only).
  - ui_pages/evaluation.py has no display for the Precision@k/Recall@k
    metrics added to app.evaluation.metrics (webapp only).

If you're picking this codebase up fresh, start with `server.py`
instead. This file may be removed in a future cleanup once nothing
still depends on it.

Run with:
    streamlit run app_ui.py

Responsibility: this file is a thin entry point only - page config,
theme, shared session-state init, the global settings sidebar, and
navigation wiring. No retrieval, prompting, or generation logic lives
here; the actual pages (ui_pages/) call into the ui/ package, which in
turn calls cli.ask_with_vision_stream(), the same pipeline function
the plain CLI (cli.py) uses. That way the CLI and the UI can never
drift apart or duplicate pipeline logic.

Pages are referenced below by PATH STRING, not Python import, so the
dependency graph stays one-directional: ui_pages/* -> ui/* -> cli.py,
app.* (see ui/__init__.py).
"""

import logging

import streamlit as st

from app.logging_config import configure_logging
from ui.sidebar import render_nav
from ui.state import init_session
from ui.theme import inject_theme

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="IntelliFusion", page_icon="🧠", layout="wide")
inject_theme()
init_session()

pages = [
    st.Page("ui_pages/sources.py", title="Sources", icon=":material/folder:"),
    st.Page("ui_pages/chat.py", title="Chat", icon=":material/chat_bubble:", default=True),
    st.Page("ui_pages/evaluation.py", title="Evaluation", icon=":material/bar_chart:"),
    st.Page("ui_pages/settings.py", title="Settings", icon=":material/settings:"),
]
current_page = st.navigation(pages, position="hidden")
render_nav(pages, current_page.title)
current_page.run()

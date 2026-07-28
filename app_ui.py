"""
Streamlit chat UI for the Version 1 RAG pipeline.

Run with:
    streamlit run app_ui.py

Responsibility: provide a simple browser-based chat interface. No
retrieval, prompting, or generation logic lives here - this file only
renders the chat and calls cli.ask_with_vision(), the same pipeline
function the plain CLI (cli.py) uses. That way the CLI and the UI can
never drift apart or duplicate pipeline logic.
"""

import logging

import streamlit as st

from app.ingestion.ingest import ingest_github_repo
from app.logging_config import configure_logging
from app.vectorstore.store import list_repositories
from cli import ask_with_vision

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="GitHub Codebase Intelligence Platform", page_icon="🔎")
st.title("🔎 GitHub Codebase Intelligence Platform")
st.caption(
    "Version 1 — Basic RAG over README.md. Answers are grounded in retrieved "
    "context only (local embeddings + Chroma + a local Ollama model)."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

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
st.sidebar.subheader("Ingest a GitHub repo")
github_repo_url = st.sidebar.text_input(
    "Repo (owner/repo or full URL)",
    placeholder="e.g. https://github.com/owner/repo",
)
github_branch = st.sidebar.text_input(
    "Branch (optional)", placeholder="default branch if left blank"
)
if st.sidebar.button("Ingest repo", disabled=not github_repo_url):
    with st.sidebar:
        with st.spinner(f"Ingesting '{github_repo_url}'..."):
            try:
                stored = ingest_github_repo(github_repo_url, branch=github_branch or None)
                st.success(f"Stored {stored} chunk(s) from '{github_repo_url}'.")
            except Exception as error:
                logger.exception("GitHub repo ingestion failed for '%s'", github_repo_url)
                st.error(f"Failed to ingest '{github_repo_url}': {error}")

st.sidebar.divider()
st.sidebar.subheader("Scope questions to a repo")
_ingested_repos = list_repositories()
repo_filter = st.sidebar.selectbox(
    "Repository (optional)",
    options=["(search everything)"] + _ingested_repos,
    help=(
        "The vector store holds chunks from EVERY source ever ingested "
        "(README, PDFs, websites, and every GitHub repo). Without scoping "
        "to a specific repo here, a question can be answered using another "
        "previously-ingested repo's chunks instead of the one you just "
        "ingested."
    ),
)
active_repository = repo_filter if repo_filter != "(search everything)" else None


def _render_images(image_hits: list[dict]) -> None:
    """Render any relevant retrieved images below an answer, with a citation."""
    for image_hit in image_hits:
        metadata = image_hit["metadata"]
        st.image(metadata["image_url"], caption=metadata.get("alt_text"))
        page_url = metadata.get("page_url")
        if page_url:
            st.caption(f"From: {page_url}")


# Replay past messages so they stay visible across reruns.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_images(message.get("images", []))

question = st.chat_input("Ask a question about this repository...")

if question:
    logger.info("UI question received: %r (top_k=%d)", question, top_k)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, image_hits = ask_with_vision(
                question, top_k=top_k, use_vision=use_vision, repository=active_repository
            )
        st.markdown(answer)
        _render_images(image_hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "images": image_hits}
    )


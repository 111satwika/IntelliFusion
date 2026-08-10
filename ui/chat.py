"""
Chat rendering: the streaming-query runner shared by every chat
surface, the per-KB chat (scoped to one knowledge base), and the
cross-KB chat (routed across every populated KB).

*** DEPRECATED (see app_ui.py's module docstring): ask_with_vision_stream
is called below without crag_enabled/query_transform_enabled/
self_rag_enabled, so it falls back to whatever ui_pages/settings.py's
process-global flags say - not the per-session settings the FastAPI
webapp (api/chat.py) reads instead.
"""

import logging

import streamlit as st

from app.routing.github_intent import classify_github_intent
from app.routing.router import classify_route
from app.vectorstore.store import list_populated_kbs
from cli import ask_with_vision_stream

from app.generation.self_reflection import (
    critique_answer,
    is_enabled as self_rag_enabled,
)

from ui.icons import avatar_data_uri
from ui.panels import render_images, render_insight_panel
from ui.state import KB_LABELS

logger = logging.getLogger(__name__)

_ASSISTANT_AVATAR = avatar_data_uri("brand", bg_hex="#0E7C86")
_USER_AVATAR = avatar_data_uri("user", bg_hex="#9A9C95")


def _avatar_for(role: str) -> str:
    return _ASSISTANT_AVATAR if role == "assistant" else _USER_AVATAR


def run_streaming_query(
    question: str,
    *,
    top_k: int,
    use_vision: bool,
    kb: str | None = None,
    repository: str | None = None,
) -> tuple[str, list[dict], list[dict], object, object]:
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
    takes over to render the remaining tokens live.

    Returns:
        (answer, image_hits, chunks, crag_result, critique_result)
        - answer: the fully-accumulated answer string (for persisting
          to session state).
        - image_hits: for render_images to display below the answer.
        - chunks: the (post-CRAG-filter) chunks that fed the LLM,
          for rendering the CRAG panel and for calling critique_answer.
        - crag_result: CragResult from the retrieval-time evaluator,
          or None when CRAG is disabled / not applicable. Used to
          render the "Corrective RAG" panel.
        - critique_result: CritiqueResult from Self-RAG, or None when
          Self-RAG is disabled. Used to render the "Self-Reflective
          RAG" panel.
    """
    with st.spinner("Thinking… (retrieval + LLM warm-up, first token can take ~15-30 s on CPU)"):
        # 5th/6th return values (contextualization, from conversation
        # memory - see app.retrieval.contextualize; and
        # used_semantic_cache, from app.generation.semantic_cache) are
        # intentionally discarded here: this Streamlit build has no
        # per-session history to pass in (see this module's own
        # deprecation notice), so contextualization is always a no-op
        # trivial result, and this build has no insight panel to
        # disclose a cache hit through.
        token_iter, image_hits, chunks, crag_result, _contextualization, _used_semantic_cache, _audio_clip_hits = ask_with_vision_stream(
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
        return "", image_hits, chunks, crag_result, None

    def _stream_with_first():
        yield first_token
        yield from token_iter

    answer = st.write_stream(_stream_with_first())

    # Self-RAG runs AFTER streaming so first-token latency stays
    # untouched. The cli.py streaming tail already warmed the LRU
    # cache during generation (when enabled), so this call is a
    # cache hit and doesn't add another Ollama round-trip.
    critique_result = None
    if self_rag_enabled():
        with st.spinner("Critiquing answer (Self-RAG)…"):
            try:
                critique_result = critique_answer(question, chunks, answer or "")
            except Exception:  # noqa: BLE001 - critique failures never break the chat
                logger.exception("Self-RAG critique failed in UI; skipping panel.")
                critique_result = None

    return answer, image_hits, chunks, crag_result, critique_result


def render_kb_chat(kb: str, top_k: int, use_vision: bool, repository: str | None = None) -> None:
    """
    Render a chat scoped to a single KB: history + text_input form.
    Uses a form (not st.chat_input) so it renders reliably inside a
    tab; st.chat_input is reserved for the top-level cross-KB chat.

    ``repository`` scopes retrieval to a single "owner/repo" within
    the KB (currently only meaningful for kb="github", which is the
    only KB whose vector store holds mixed-repo content - see
    app.retrieval.github_adaptive._build_where). None means "search
    every repo in this KB", matching the pre-picker behavior for
    backwards compatibility with the web/pdf/docx/markdown tabs that
    don't have a repo concept.
    """
    label = KB_LABELS[kb]
    st.markdown(f"**Ask about your {label} documents**")
    for message in st.session_state["messages_by_kb"][kb]:
        with st.chat_message(message["role"], avatar=_avatar_for(message["role"])):
            st.markdown(message["content"])
            render_images(message.get("images", []))
            # The insight panel uses the RESULT OBJECTS snapshotted on
            # the message dict at write time (routing decision,
            # github_intent, crag/critique results) rather than
            # re-running anything - their inputs (chunk set / answer
            # text) aren't preserved beyond the snapshot, and
            # re-running would either be a cache miss (wasting a full
            # LLM round-trip on scroll) or wrong (if caches were
            # evicted). No routing snapshot = no panel, matching "this
            # message predates the insight-panel feature or its toggles
            # were off when it was written".
            routing = message.get("routing")
            if message["role"] == "assistant" and routing is not None:
                render_insight_panel(
                    question=message.get("user_question", ""),
                    kb=kb,
                    decision=routing["decision"],
                    target_kbs=routing["target_kbs"],
                    forced_kb=kb,
                    github_intent=message.get("github_intent"),
                    # Use the scope that was active when the message was
                    # written, not the currently-selected picker value -
                    # otherwise switching the picker would rewrite
                    # history's answers to look like they came from a
                    # different scope than they actually did.
                    repository=message.get("github_repository"),
                    crag_result=message.get("crag_result"),
                    critique_result=message.get("critique_result"),
                )
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
        # For the GitHub KB, additionally classify the adaptive-RAG
        # intent (explanation / exact_code / navigational / history /
        # visual / general) so the assistant bubble can show a badge
        # explaining which retrieval strategy was chosen. Cheap enough
        # (~1 query embedding) to run in the UI even though the
        # retriever calls it again internally - the alternative would
        # be plumbing the decision back through the streaming call
        # chain, which is more code for a purely observability signal.
        github_intent = classify_github_intent(question) if kb == "github" else None
        # Show the just-submitted user turn immediately (before rerun)
        # so it sits above the streaming assistant response, matching
        # the message order the user will see after rerun renders
        # everything from session state.
        with st.chat_message("user", avatar=_USER_AVATAR):
            st.markdown(question)
        with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
            # run_streaming_query wraps retrieval + LLM warm-up in a
            # visible spinner (so the bubble doesn't look frozen for
            # ~15-30s while the 7B model on CPU processes the prompt),
            # then streams tokens live via st.write_stream once the
            # first token arrives. Returns the fully-accumulated answer
            # for persisting to session state.
            answer, image_hits, chunks, crag_result, critique_result = run_streaming_query(
                question, top_k=top_k, use_vision=use_vision, kb=kb,
                repository=repository,
            )
            render_images(image_hits)
            # Live-message insight panel: same panel as the history
            # replay, but rendered here for the just-produced answer so
            # the user doesn't have to wait for a rerun to see it.
            render_insight_panel(
                question=question,
                kb=kb,
                decision=decision,
                target_kbs=[kb],
                forced_kb=kb,
                github_intent=github_intent,
                repository=repository,
                crag_result=crag_result,
                critique_result=critique_result,
            )
        st.session_state["messages_by_kb"][kb].append(
            {
                "role": "assistant",
                "content": answer,
                "images": image_hits,
                "routing": {"decision": decision, "target_kbs": [kb]},
                "github_intent": github_intent,
                # Snapshot the scope so history replay stays truthful
                # even after the user changes the picker.
                "github_repository": repository if kb == "github" else None,
                # Snapshot the user's question so history replay can
                # render the query-transform panel for that specific
                # question (the message list is a flat sequence, so
                # without this snapshot the assistant message wouldn't
                # know which user turn it corresponds to).
                "user_question": question,
                # Snapshot CRAG / Self-RAG result objects for history
                # replay. We do NOT snapshot the chunk list itself:
                # it can be large and the panels only need the
                # already-computed result objects.
                "crag_result": crag_result,
                "critique_result": critique_result,
            }
        )
        st.rerun()


def render_cross_kb_chat(top_k: int, use_vision: bool, repository: str | None = None) -> None:
    """
    Chat that uses query routing to pick which KB(s) to search - handy
    for questions that span multiple sources or don't obviously belong
    to one KB. ``repository`` optionally narrows every search to one
    already-ingested "owner/repo".
    """
    for message in st.session_state["messages_global"]:
        with st.chat_message(message["role"], avatar=_avatar_for(message["role"])):
            st.markdown(message["content"])
            render_images(message.get("images", []))
            routing = message.get("routing")
            if message["role"] == "assistant" and routing is not None:
                # Cross-KB chat doesn't force a kb, so kb=None here
                # (matches the transform_query call the retriever
                # actually made).
                render_insight_panel(
                    question=message.get("user_question", ""),
                    kb=None,
                    decision=routing["decision"],
                    target_kbs=routing["target_kbs"],
                    crag_result=message.get("crag_result"),
                    critique_result=message.get("critique_result"),
                )

    question = st.chat_input("Ask a question about your ingested sources...")

    if question:
        logger.info("UI (cross-KB) question received: %r (top_k=%d)", question, top_k)
        st.session_state["messages_global"].append({"role": "user", "content": question})
        with st.chat_message("user", avatar=_USER_AVATAR):
            st.markdown(question)

        with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
            # Classify BEFORE calling run_streaming_query so we can
            # show the same decision the retriever will make internally.
            # One extra embed call per question - negligible vs. LLM.
            decision = classify_route(question)
            target_kbs = decision.kbs or list_populated_kbs()
            answer, image_hits, chunks, crag_result, critique_result = run_streaming_query(
                question, top_k=top_k, use_vision=use_vision, repository=repository
            )
            render_images(image_hits)
            # Cross-KB path passes kb=None into transform_query (via
            # retrieve() -> retrieve_hybrid), so the insight panel here
            # does the same.
            render_insight_panel(
                question=question,
                kb=None,
                decision=decision,
                target_kbs=target_kbs,
                crag_result=crag_result,
                critique_result=critique_result,
            )

        st.session_state["messages_global"].append(
            {
                "role": "assistant",
                "content": answer,
                "images": image_hits,
                "routing": {"decision": decision, "target_kbs": target_kbs},
                "user_question": question,
                "crag_result": crag_result,
                "critique_result": critique_result,
            }
        )

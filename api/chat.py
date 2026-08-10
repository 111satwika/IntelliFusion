"""
Chat streaming API: wraps cli.ask_with_vision_stream (the same
pipeline entry point the old CLI/Streamlit builds used) and streams
tokens to the browser over SSE, followed by one final event carrying
the image hits and the "how was this answer checked" insight payload
(see insight.py).

Chat history is kept server-side, per session, per conversation thread
(see session_store.get_history/append_history/clear_history) - one
thread per KB tab on the Sources page ("markdown"/"pdf"/"docx"/"web"/
"github") plus one for the cross-KB Chat page (thread_key "__cross__"),
matching the exact 6 independent conversation threads webapp/app.js
already renders. Used to resolve follow-up questions ("what about
tests for that?") via app.retrieval.contextualize before retrieval,
and to give the LLM the recent turns at prompt-build time (see
cli._prepare_context_and_images) - gated by the per-session
conversation_memory_enabled setting, off by default like the other
advanced toggles. The visible chat UI itself is still purely
client-rendered HTML with no fetch-on-load (a page reload clears what
you SEE, but the server keeps remembering underneath - see webapp's
"Clear conversation" button for the deliberate way to reset it).
"""

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.generation.self_reflection import critique_answer
from app.routing.github_intent import classify_github_intent
from app.routing.router import classify_route
from app.vectorstore.store import list_populated_kbs
from cli import ask_with_vision_stream

from api.deps import get_session_id
from insight import build_insight
from session_store import append_history, clear_history, get_history, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    kb: str | None = None
    repository: str | None = None


class ClearHistoryRequest(BaseModel):
    kb: str | None = None


def _thread_key(kb: str | None) -> str:
    """Same thread identity webapp/app.js already organizes its 6
    independent visible chat threads by - the cross-KB Chat page uses
    "__cross__", each Sources-page per-KB thread uses the KB name."""
    return kb or "__cross__"


def _sse(payload: dict) -> str:
    # JSON-encoding (not raw text) so embedded newlines/quotes in
    # tokens or answer text can't break SSE's line-based "data: ..."
    # framing.
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/stream")
def chat_stream(body: ChatRequest, session_id: str = Depends(get_session_id)):
    settings = get_settings(session_id)
    top_k = settings["top_k"]
    use_vision = settings["use_vision"]
    crag_enabled = settings["crag_enabled"]
    query_transform_enabled = settings["query_transform_enabled"]
    self_rag_enabled = settings["self_rag_enabled"]
    conversation_memory_enabled = settings["conversation_memory_enabled"]
    question = body.question
    kb = body.kb
    repository = body.repository
    thread_key = _thread_key(kb)
    history = get_history(session_id, thread_key)

    logger.info("chat question kb=%s: %r", kb, question)

    # Classified BEFORE streaming so the insight panel describes the
    # exact same decision the retriever made internally - mirrors the
    # Streamlit build's render_kb_chat/render_cross_kb_chat.
    decision = classify_route(question)
    target_kbs = [kb] if kb else (decision.kbs or list_populated_kbs())
    # Not just "kb == 'github'" (explicitly-scoped chat) - a cross-KB
    # question (kb=None) can still route to "github" as one of several
    # target_kbs, and app.retrieval.retriever.retrieve() now dispatches
    # that KB's slice through the same adaptive-intent retrieval either
    # way (see retriever.py). This call is cheap (~1 embedding) and
    # purely for the insight panel - the retriever already made this
    # same classification internally to decide its own strategy.
    github_intent = classify_github_intent(question) if "github" in target_kbs else None

    def event_stream():
        token_iter, image_hits, chunks, crag_result, contextualization, used_semantic_cache, audio_clip_hits = ask_with_vision_stream(
            question, top_k=top_k, use_vision=use_vision, kb=kb, repository=repository,
            crag_enabled=crag_enabled, query_transform_enabled=query_transform_enabled,
            self_rag_enabled=self_rag_enabled,
            history=history, conversation_memory_enabled=conversation_memory_enabled,
        )
        answer_parts: list[str] = []
        for token in token_iter:
            answer_parts.append(token)
            yield _sse({"token": token})
        answer = "".join(answer_parts)

        # Recorded only now that the answer is fully assembled - a
        # client that disconnects mid-stream never leaves a partial/
        # empty answer in this thread's remembered history.
        if answer:
            append_history(session_id, thread_key, question, answer)

        # Self-RAG runs AFTER streaming so first-token latency is
        # unaffected. cli.py's streaming tail already warmed the
        # LRU cache during generation (when enabled), so this call
        # is a cache hit, not a second Ollama round-trip.
        critique_result = None
        if self_rag_enabled:
            try:
                critique_result = critique_answer(question, chunks, answer or "", enabled=True)
            except Exception:  # noqa: BLE001 - critique failure must not break the chat
                logger.exception("Self-RAG critique failed; continuing without it.")

        insight = build_insight(
            question=question,
            kb=kb,
            decision=decision,
            target_kbs=target_kbs,
            forced_kb=kb,
            github_intent=github_intent,
            repository=repository,
            crag_result=crag_result,
            critique_result=critique_result,
            query_transform_enabled=query_transform_enabled,
            contextualization=contextualization,
            used_semantic_cache=used_semantic_cache,
        )
        yield _sse({
            "event": "done",
            "images": [
                {
                    "image_url": hit["metadata"]["image_url"],
                    "alt_text": hit["metadata"].get("alt_text"),
                    "page_url": hit["metadata"].get("page_url"),
                    # Present only for video-frame hits (see
                    # app.ingestion.ingest._ingest_video_frames) - None
                    # for ordinary web-image hits, same as page_url
                    # above being None for frames.
                    "origin": hit["metadata"].get("origin"),
                    "video_document_id": hit["metadata"].get("video_document_id"),
                    "timestamp_seconds": hit["metadata"].get("timestamp_seconds"),
                }
                for hit in image_hits
            ],
            # Native acoustic-similarity matches (see
            # app.embeddings.audio_embedder) - [] unless the question
            # routed to "sound" AND audio clips were ever ingested
            # (ENABLE_AUDIO_SIMILARITY_SEARCH=1 at ingest time).
            "audio_clips": [
                {
                    "audio_url": hit["metadata"]["source_audio_url"],
                    "start_seconds": hit["metadata"]["start_seconds"],
                    "end_seconds": hit["metadata"].get("end_seconds"),
                    "document_id": hit["metadata"]["document_id"],
                }
                for hit in audio_clip_hits
            ],
            "insight": insight,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/clear")
def clear_chat_history(body: ClearHistoryRequest, session_id: str = Depends(get_session_id)):
    """Empty one thread's remembered conversation (see _thread_key) -
    used by the webapp's "Clear conversation" button. Does not touch
    any other thread or session."""
    clear_history(session_id, _thread_key(body.kb))
    return {"cleared": True}

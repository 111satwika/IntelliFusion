"""
Simple command-line interface for the Version 1 RAG pipeline.

Usage:
    python cli.py "What is this repository about?"
    python cli.py "How does chunking work in this project?" --top-k 5
    python cli.py --ingest-github owner/repo
    python cli.py --ingest-github https://github.com/owner/repo --branch main "What does this repo do?"
    python cli.py "How does auth work?" --repo owner/repo

Ties together every module built so far, in order:
    retrieve()       -> app.retrieval.retriever
    build_prompt()   -> app.prompting.prompt_builder
    generate_answer()-> app.generation.llm_generator

This file intentionally contains no pipeline logic of its own - it
only parses CLI args and calls the three functions above, so the
pipeline stays testable/usable from other code (e.g. a future web
API) without depending on this CLI.
"""

import argparse
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field  # noqa: F401 - kept for compatibility with existing imports; no direct use in this module.

from app.embeddings.embedder import embed_texts
from app.generation import semantic_cache
from app.generation.llm_generator import (
    generate_answer,
    generate_answer_stream,
    generate_answer_with_images,
)
from app.generation.self_reflection import (
    CritiqueResult,
    critique_answer,
    is_enabled as _self_rag_enabled,
)
from app.logging_config import configure_logging
from app.prompting.prompt_builder import build_prompt
from app.retrieval.contextualize import ContextualizeResult, contextualize_query
from app.retrieval.crag import CragResult, evaluate_chunks, is_enabled as _crag_enabled
from app.retrieval.github_adaptive import retrieve_github_adaptive
from app.retrieval.query_transform import transform_query
from app.retrieval.retriever import retrieve, retrieve_audio_clips, retrieve_images
from app.routing.router import classify_route
from app.vectorstore.store import get_chunk_by_document_id

logger = logging.getLogger(__name__)

# Images are matched via a separate multimodal (CLIP) embedding space
# (see app.embeddings.image_embedder), whose absolute cosine-similarity
# scores run lower than all-MiniLM-L6-v2's for genuinely relevant text
# - so this threshold is deliberately loose, just enough to filter out
# clearly-unrelated images rather than to mean "highly confident match".
IMAGE_SIMILARITY_THRESHOLD = 0.2
MAX_IMAGES_FOR_VISION = 2

# Matches the "[Image: alt] (url)" markers that _build_website_document
# inlines into each web-KB chunk right where the <img> sits in the HTML
# (see app.ingestion.loader._describe_image_tag). If a retrieved text
# chunk contains one of these, the doc page's author has already told
# us that image belongs next to that prose - a much stronger relevance
# signal than CLIP cosine similarity between the query and the image's
# pixels, especially for corpora (like IBM docs) where every setup
# screenshot looks visually similar to every other setup screenshot.
_IMAGE_MARKER_RE = re.compile(r"\[Image:\s*([^\]]*)\]\s*\(([^)\s]+)\)")


def _images_referenced_in_chunks(chunks: list[dict]) -> list[dict]:
    """
    Extract images explicitly named in the retrieved chunks (via the
    "[Image: alt] (url)" markers, see _IMAGE_MARKER_RE) and return
    them shaped like retrieve_images() hits so they can be displayed
    alongside CLIP-retrieved ones.

    The chunk's OWN text is telling us these images belong next to it,
    so similarity is set to 1.0 - by definition they pass any
    IMAGE_SIMILARITY_THRESHOLD without needing a separate CLIP-based
    relevance check.
    """
    seen_urls: set[str] = set()
    hits: list[dict] = []
    for chunk in chunks:
        content = chunk.get("content") or ""
        chunk_metadata = chunk.get("metadata") or {}
        page_url = chunk_metadata.get("url")
        for match in _IMAGE_MARKER_RE.finditer(content):
            alt_text = match.group(1).strip() or None
            image_url = match.group(2).strip()
            if not image_url or image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            hits.append(
                {
                    "metadata": {
                        "image_url": image_url,
                        "alt_text": alt_text,
                        "page_url": page_url,
                    },
                    "similarity": 1.0,
                }
            )
    return hits


# Default owner for plain `python cli.py "question"` usage, which has
# no login/account concept of its own - matches
# app.vectorstore.store._DEFAULT_OWNER's value (duplicated as a plain
# string rather than importing across layers; cli.py is meant to stay
# usable standalone, without depending on api/deps.py's LOCAL_OWNER).
_DEFAULT_OWNER = "local"


def ask(query_text: str, top_k: int = 3, repository: str | None = None, owner: str = _DEFAULT_OWNER) -> str:
    """Run the full retrieve -> build_prompt -> generate_answer pipeline."""
    chunks = retrieve(query_text, top_k=top_k, repository=repository, owner=owner)
    prompt = build_prompt(query_text, chunks)
    return generate_answer(prompt)


def _is_code_screenshot(image_url: str, owner: str) -> bool:
    """
    True if the given image's own OCR/vision text chunk was classified
    as a source-code screenshot (content_type == "image_code") at
    ingestion time (see app.ocr.image_classifier, app.ingestion.ingest),
    as opposed to an ordinary image (content_type == "image_ocr").

    Used to auto-enable the vision model for exactly the screenshots
    it's actually useful for (code rendered as pixels, where OCR alone
    can be garbled/inaccurate), without paying the vision-model cost
    for every loosely CLIP-relevant image.
    """
    chunk = get_chunk_by_document_id(image_url, owner)
    return chunk is not None and chunk["metadata"].get("content_type") == "image_code"


# CRAG re-retrieval settings. If CRAG's filter leaves fewer than this
# many "keeper" chunks, we do ONE extra retrieval pass using the
# query-transform rewrite so the LLM has something usable to work
# with. Kept small (2) so ambiguous-but-mostly-useful chunk sets
# don't get an unnecessary second round-trip.
_CRAG_MIN_KEEPERS = 2


def _apply_crag(
    query_text: str,
    chunks: list[dict],
    *,
    top_k: int,
    repository: str | None,
    kb: str | None,
    owner: str,
    crag_enabled: bool | None = None,
    query_transform_enabled: bool | None = None,
) -> tuple[list[dict], CragResult | None]:
    """Grade the retrieved chunks with the CRAG evaluator, drop the
    "incorrect" ones, and if too few keepers survive, do ONE
    supplementary retrieval pass using the rewritten query variant.

    Returns:
        (chunks_after_crag, crag_result) - crag_result is None when
        CRAG is disabled or produced no filtering opinion. Callers use
        the CragResult to render the retrieval-evaluation panel
        without having to re-run the evaluator.

    Behavior when CRAG is disabled: evaluate_chunks() returns a
    trivial "correct" result with every chunk kept, so this
    function is effectively an identity pass. That's what makes it
    safe to call unconditionally from the retrieval path.
    """
    if not chunks:
        return chunks, None
    result = evaluate_chunks(query_text, chunks, enabled=crag_enabled)
    logger.info(
        "CRAG verdict=%s kept=%d dropped=%d used_llm=%s",
        result.overall, len(result.kept_indices), len(result.dropped_indices),
        result.used_llm,
    )
    # No filtering when the evaluator wasn't consulted or all chunks
    # passed. This is by far the common case when CRAG is off.
    if not result.used_llm:
        return chunks, None
    if not result.dropped_indices:
        return chunks, result

    kept = [chunks[i] for i in result.kept_indices if 0 <= i < len(chunks)]

    # If filtering left us with enough context, use it directly.
    if len(kept) >= _CRAG_MIN_KEEPERS:
        return kept, result

    # Not enough surviving chunks - try ONE supplementary retrieval
    # using the query-transform rewrite. This deliberately does NOT
    # loop: a second CRAG pass on the supplementary results would
    # risk unbounded retries in ambiguous cases. We just augment the
    # keeper set with whatever the rewrite surfaces and hand THAT to
    # the LLM (a wider, possibly-noisier context is still better than
    # 0-1 chunks).
    transform = transform_query(query_text, kb=kb, enabled=query_transform_enabled)
    rewritten = transform.rewritten
    if not rewritten or rewritten.strip().lower() == query_text.strip().lower():
        logger.info("CRAG re-retrieve: no rewrite available; returning kept chunks only.")
        return kept, result

    logger.info("CRAG re-retrieve using rewrite=%r", rewritten)
    try:
        if kb == "github":
            extra_chunks, _ = retrieve_github_adaptive(
                rewritten, top_k=top_k, repository=repository, owner=owner,
                query_transform_enabled=query_transform_enabled,
            )
        else:
            extra_chunks = retrieve(
                rewritten, top_k=top_k, repository=repository, kb=kb, owner=owner,
                query_transform_enabled=query_transform_enabled,
            )
    except Exception:  # noqa: BLE001 - re-retrieval failure just leaves us with the keepers
        logger.exception("CRAG re-retrieve failed; returning kept chunks only.")
        return kept, result

    # Dedupe by (document_id, chunk_index); prefer originally-kept
    # chunks first so their position in the shortlist reflects the
    # cross-encoder ranking that got them there.
    seen: set[tuple] = set()
    merged: list[dict] = []
    for hit in [*kept, *extra_chunks]:
        md = hit.get("metadata") or {}
        key = (md.get("document_id"), md.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged[:top_k], result


def ask_with_vision(
    query_text: str,
    top_k: int = 3,
    use_vision: bool = False,
    repository: str | None = None,
    kb: str | None = None,
    *,
    owner: str = _DEFAULT_OWNER,
    crag_enabled: bool | None = None,
    query_transform_enabled: bool | None = None,
    self_rag_enabled: bool | None = None,
    history: list[dict] | None = None,
    conversation_memory_enabled: bool | None = None,
) -> tuple[str, list[dict]]:
    """
    Same pipeline as ask(), but additionally retrieves relevant images
    for DISPLAY, and (only when explicitly opted into via
    `use_vision=True`) sends them to a vision-capable Ollama model (see
    app.generation.llm_generator.generate_answer_with_images) alongside
    the text context - so a question whose real answer only exists
    inside a screenshot (e.g. a code example rendered as an image
    rather than real HTML text) can still be answered by literally
    showing the model the picture, not just its alt text.

    `use_vision` defaults to False: on CPU-only setups the vision model
    (llava) reliably takes 2.5-10+ minutes per call and frequently
    times out (see /memories/repo/rag-platform-notes.md). Worse, a
    slow/timed-out vision call keeps the CPU busy, which then starves
    the text-only fallback call right after it, causing IT to time out
    too - i.e. attempting vision by default made even normally-fast
    text-only answers unreliable. OCR'd text from these same images is
    already part of the normal retrieved text context by the time this
    function runs (stored as an ordinary extra chunk at ingestion time,
    see app.ocr.image_ocr), so most code-screenshot questions are
    already answerable from `chunks` alone without ever calling the
    vision model. Pass `use_vision=True` explicitly to opt back in for
    ALL relevant images.

    Even with `use_vision=False`, vision is still AUTO-enabled for any
    relevant image already classified as a source-code screenshot
    (content_type == "image_code", see _is_code_screenshot) - this is
    exactly the case where plain OCR is most likely to be garbled/
    inaccurate (see app.ocr.image_classifier), so it's worth the extra
    cost even by default, while still not paying that cost for every
    other loosely-relevant image.

    `repository` optionally scopes retrieval to one already-ingested
    "owner/repo" (see app.retrieval.retriever.retrieve's `repository`
    argument) - without it, a question is answered from EVERY source
    ever ingested into the shared vector store, not just the repo the
    user most recently ingested/cares about.

    Returns:
        (answer_text, image_hits_used) - image_hits_used is the list
        of retrieved image dicts (see
        app.retrieval.retriever.retrieve_images) that cleared
        IMAGE_SIMILARITY_THRESHOLD, for callers that want to DISPLAY
        them (e.g. app_ui.py), regardless of whether they were also
        sent to the vision model.
    """
    prompt, display_images, vision_image_urls, chunks, _crag, _contextualization, cached_answer, _audio_clip_hits = (
        _prepare_context_and_images(
            query_text, top_k=top_k, use_vision=use_vision, repository=repository, kb=kb,
            owner=owner, crag_enabled=crag_enabled, query_transform_enabled=query_transform_enabled,
            history=history, conversation_memory_enabled=conversation_memory_enabled,
        )
    )
    if cached_answer is not None:
        # Semantic near-duplicate hit (see app.generation.semantic_cache)
        # - _prepare_context_and_images already guarantees this is only
        # ever non-None for a history-free, non-vision request, so
        # skipping straight to the cached answer here is safe.
        answer = cached_answer
    elif vision_image_urls:
        try:
            answer = generate_answer_with_images(prompt, vision_image_urls)
        except Exception:
            # Vision models are slow on CPU and can time out; the text
            # context already includes any OCR'd text from these same
            # images (see app.ocr.image_ocr), so falling back to the
            # text-only pipeline still gives a useful answer instead
            # of a hard failure.
            logger.exception("Vision generation failed; falling back to text-only answer.")
            answer = generate_answer(prompt)
    else:
        answer = generate_answer(prompt)
        if not history and not vision_image_urls:
            try:
                question_embedding = embed_texts([_contextualization.resolved])[0]
                semantic_cache.store_answer(
                    _contextualization.resolved, question_embedding, chunks, kb, repository, answer, owner
                )
            except Exception:  # noqa: BLE001 - cache store failure must not break the answer
                logger.exception("Semantic cache store failed; continuing without caching this answer.")

    # Self-RAG (see app.generation.self_reflection): critique the
    # finished answer against the chunks. Returns None when Self-RAG
    # is disabled; the caller can ignore it in that case. Cached, so
    # the UI can retrieve the same critique later without a second
    # LLM round-trip.
    effective_self_rag = _self_rag_enabled() if self_rag_enabled is None else self_rag_enabled
    if effective_self_rag:
        try:
            critique_answer(query_text, chunks, answer, enabled=True)
        except Exception:  # noqa: BLE001 - critique failure must not break the answer
            logger.exception("Self-RAG critique failed; continuing without critique.")

    return answer, display_images


def ask_with_vision_stream(
    query_text: str,
    top_k: int = 3,
    use_vision: bool = False,
    repository: str | None = None,
    kb: str | None = None,
    *,
    owner: str = _DEFAULT_OWNER,
    crag_enabled: bool | None = None,
    query_transform_enabled: bool | None = None,
    self_rag_enabled: bool | None = None,
    history: list[dict] | None = None,
    conversation_memory_enabled: bool | None = None,
) -> tuple[Iterator[str], list[dict], list[dict], CragResult | None, ContextualizeResult, bool, list[dict]]:
    """
    Streaming counterpart to ask_with_vision(): identical retrieval,
    prompt-assembly, and image-hit logic, but returns a token iterator
    for the answer instead of the finished string, so callers (e.g.
    app_ui.py's st.write_stream) can render tokens live as Ollama
    produces them. Especially useful on CPU-only setups where a full
    7B-model answer takes 2-3 minutes - the user sees the answer
    taking shape instead of staring at a spinner.

    Semantic differences vs ask_with_vision():
      * When the vision model would fire (auto-detected code
        screenshots, or use_vision=True), we FALL BACK to the
        non-streaming vision path and yield the finished answer as a
        single chunk. Streaming a multi-image vision request requires
        piecing tokens together from a different Ollama endpoint
        response shape, and vision runs are the outlier - the vast
        majority of RAG queries are text-only and benefit fully from
        live streaming.
      * On vision failure we fall back to text-only STREAMING (not the
        non-streaming path), so the timeout-free behavior is preserved
        even in the fallback case.

    Returns:
        (token_iterator, image_hits_for_display, chunks, crag_result,
        contextualization, used_semantic_cache, audio_clip_hits)
        - image_hits_for_display matches ask_with_vision's second
          return value exactly.
        - chunks are the (post-CRAG-filter) chunks that fed the LLM;
          the UI passes them to critique_answer() after streaming
          completes to render the Self-RAG panel.
        - crag_result is the retrieval-time evaluator's verdict, or
          None when CRAG is disabled / not applicable. Used by the UI
          to render the CRAG panel without re-running the evaluator.
        - contextualization is the ContextualizeResult from resolving
          `query_text` against `history` (see
          app.retrieval.contextualize) - always present (never None),
          with `resolved == original` when conversation memory is
          disabled/there's no history/the question already looked
          standalone. Used by the UI to show "resolved your question
          as: ..." without a second Ollama call.
        - used_semantic_cache is True when the answer was served from
          app.generation.semantic_cache instead of a fresh Ollama call
          (see that module for the exact conditions) - used by the UI
          to disclose this, since (unlike the other RAG-quality
          toggles) there's no per-session control to show/hide instead.
        - audio_clip_hits matches _prepare_context_and_images's own
          audio_clip_hits exactly (see that function's docstring) -
          acoustic-similarity matches for the UI to surface alongside
          the answer, always [] unless the query routed to "sound"
          AND audio clips were ever ingested.
    """
    prompt, display_images, vision_image_urls, chunks, crag_result, contextualization, cached_answer, audio_clip_hits = (
        _prepare_context_and_images(
            query_text, top_k=top_k, use_vision=use_vision, repository=repository, kb=kb,
            owner=owner, crag_enabled=crag_enabled, query_transform_enabled=query_transform_enabled,
            history=history, conversation_memory_enabled=conversation_memory_enabled,
        )
    )

    def _critique_after(answer: str) -> None:
        """Hook to run Self-RAG on the finished answer. No-op when the
        critic is disabled. Failures are swallowed so a broken critic
        can never affect the user-visible answer."""
        effective_self_rag = _self_rag_enabled() if self_rag_enabled is None else self_rag_enabled
        if not effective_self_rag:
            return
        try:
            critique_answer(query_text, chunks, answer, enabled=True)
        except Exception:  # noqa: BLE001
            logger.exception("Self-RAG critique failed; continuing without critique.")

    def _maybe_cache_after(answer: str) -> None:
        """Hook to store a freshly-generated answer in the semantic
        cache. Never called for a vision-augmented answer (see
        _vision_then_maybe_stream, which does not call this) or when
        `history` is non-empty - see app.generation.semantic_cache's
        module docstring for why both gates matter. Failures are
        swallowed so a broken cache can never affect the user-visible
        answer."""
        if history or vision_image_urls or not answer:
            return
        try:
            question_embedding = embed_texts([contextualization.resolved])[0]
            semantic_cache.store_answer(
                contextualization.resolved, question_embedding, chunks, kb, repository, answer, owner
            )
        except Exception:  # noqa: BLE001
            logger.exception("Semantic cache store failed; continuing without caching this answer.")

    if cached_answer is not None:
        # Semantic near-duplicate hit (see app.generation.semantic_cache)
        # - _prepare_context_and_images already guarantees this is only
        # ever non-None for a history-free, non-vision request, so this
        # branch never conflicts with the vision path below. Still runs
        # Self-RAG on the cached text, same as llm_generator's own
        # exact-match cache already does - a cache hit isn't exempt from
        # the same post-hoc verification a freshly-generated answer gets.
        def _cached_then_critique() -> Iterator[str]:
            yield cached_answer
            _critique_after(cached_answer)

        return _cached_then_critique(), display_images, chunks, crag_result, contextualization, True, audio_clip_hits

    if vision_image_urls:
        # Non-streaming vision path with a text-only STREAMING fallback
        # if vision fails/times out. Wrapped in an inner generator so
        # ask_with_vision_stream can uniformly return an Iterator[str]
        # to its caller regardless of which path actually runs.
        def _vision_then_maybe_stream() -> Iterator[str]:
            answer_buffer: list[str] = []
            try:
                token = generate_answer_with_images(prompt, vision_image_urls)
                answer_buffer.append(token)
                yield token
            except Exception:
                logger.exception("Vision generation failed; falling back to streaming text-only answer.")
                for token in generate_answer_stream(prompt):
                    answer_buffer.append(token)
                    yield token
            _critique_after("".join(answer_buffer))

        return _vision_then_maybe_stream(), display_images, chunks, crag_result, contextualization, False, audio_clip_hits

    def _stream_and_critique() -> Iterator[str]:
        """Wrapper around generate_answer_stream that accumulates the
        finished answer so Self-RAG (if enabled) can critique it AFTER
        the last token is yielded - keeping the user's first-token
        latency unchanged."""
        answer_buffer: list[str] = []
        for token in generate_answer_stream(prompt):
            answer_buffer.append(token)
            yield token
        answer = "".join(answer_buffer)
        _critique_after(answer)
        _maybe_cache_after(answer)

    return _stream_and_critique(), display_images, chunks, crag_result, contextualization, False, audio_clip_hits


def _prepare_context_and_images(
    query_text: str,
    top_k: int,
    use_vision: bool,
    repository: str | None,
    kb: str | None,
    *,
    owner: str = _DEFAULT_OWNER,
    crag_enabled: bool | None = None,
    query_transform_enabled: bool | None = None,
    history: list[dict] | None = None,
    conversation_memory_enabled: bool | None = None,
) -> tuple[str, list[dict], list[str], list[dict], CragResult | None, ContextualizeResult, str | None, list[dict]]:
    """
    Shared retrieval + prompt-assembly + image-hit computation for
    ask_with_vision() and ask_with_vision_stream(), extracted so the
    two entry points can't drift in what gets retrieved, what shows
    up in the UI, or what gets sent to the vision model.

    Returns:
        prompt              - finished prompt string ready for the LLM.
        display_images      - image hits to render below the answer in
                              the UI (CLIP-retrieved that cleared
                              IMAGE_SIMILARITY_THRESHOLD, PLUS every
                              image explicitly named by an [Image: ...]
                              (url) marker in the retrieved chunks -
                              see _images_referenced_in_chunks).
        vision_image_urls   - image_urls to actually send to the
                              vision model (subset of display_images:
                              code screenshots when use_vision=False,
                              all CLIP hits when use_vision=True, and
                              always empty when CLIP retrieved nothing;
                              chunk-referenced images are never sent
                              to vision, only displayed).
        chunks              - the retrieved chunks (post-CRAG-filter,
                              if CRAG is enabled). Returned so the
                              streaming caller can hand them to the
                              Self-RAG critic AFTER generation and
                              render the retrieval-evaluation panel.
        crag_result         - the CragResult from the retrieval-time
                              evaluator, or None when CRAG is
                              disabled / not applicable. Callers use
                              it to render the CRAG panel without
                              re-running the evaluator.
        contextualization   - the ContextualizeResult from resolving
                              query_text against history (see
                              app.retrieval.contextualize) - always
                              present, `resolved == original` when
                              conversation memory is off/there's no
                              history/the question already looked
                              standalone. `resolved` is what actually
                              got searched for; `original` (and
                              `history`) is what the LLM sees on the
                              "Question:" line.
        cached_answer       - a previously-generated answer for a
                              near-duplicate earlier question (see
                              app.generation.semantic_cache), or None
                              if there's no such hit / the cache is
                              disabled / this request doesn't qualify
                              for it at all. Only ever non-None when
                              `history` is empty (first turn of a
                              thread) AND no image will be sent to the
                              vision model - see semantic_cache's
                              module docstring for why both gates
                              matter. Callers use this to skip calling
                              generate_answer/generate_answer_stream
                              entirely.
        audio_clip_hits      - acoustic-similarity hits from
                              app.retrieval.retriever.retrieve_audio_clips
                              (see app.embeddings.audio_embedder), for
                              callers that want to surface "sounds
                              like" matches alongside the answer.
                              Always [] unless the query routed to the
                              "sound" content-type AND
                              ENABLE_AUDIO_SIMILARITY_SEARCH=1 was set
                              at ingestion time (otherwise nothing was
                              ever embedded/stored to match against).
    """
    # Conversation memory (see app.retrieval.contextualize): resolve a
    # context-dependent follow-up ("what about tests for that?") into
    # a standalone form BEFORE retrieval, using this thread's recent
    # turns. Every retrieval-related call below (adaptive/standard
    # retrieve, and CRAG's own supplementary re-retrieval) searches for
    # `contextualization.resolved`, not the raw query_text, so a
    # follow-up's embedding/BM25 search isn't chasing an unresolved
    # pronoun. The ORIGINAL query_text (plus history) still goes to
    # build_prompt() below - the LLM resolves references itself from
    # the history block shown right above the question, so "Question:"
    # always matches what the user actually typed.
    contextualization = contextualize_query(
        query_text, history or [], kb=kb, enabled=conversation_memory_enabled
    )
    resolved_query = contextualization.resolved

    # GitHub-KB queries go through the adaptive retriever, which picks
    # a per-query strategy (dense-only for explanation, BM25-only for
    # exact_code, metadata-filtered dense for navigational, GitHub REST
    # commits API for history, hybrid for visual/general) - see
    # app.retrieval.github_adaptive. Every other KB, and any cross-KB
    # query (kb is None), goes through the standard retrieve() path
    # unchanged, so this branch is purely additive.
    if kb == "github":
        chunks, github_intent = retrieve_github_adaptive(
            resolved_query, top_k=top_k, repository=repository, owner=owner,
            query_transform_enabled=query_transform_enabled,
        )
        logger.info(
            "GitHub adaptive: intent=%s method=%s file_hint=%r chunks=%d",
            github_intent.intent, github_intent.method,
            github_intent.file_hint, len(chunks),
        )
    else:
        chunks = retrieve(
            resolved_query, top_k=top_k, repository=repository, kb=kb, owner=owner,
            query_transform_enabled=query_transform_enabled,
        )

    # Corrective RAG (see app.retrieval.crag): grade each retrieved
    # chunk's relevance, drop the "incorrect" ones, and if the
    # filtered set is too thin, do ONE re-retrieval with the rewritten
    # query variant. When CRAG is disabled the evaluator returns a
    # trivial "correct" result with every chunk kept, so this branch
    # is a no-op for users who haven't opted in.
    chunks, crag_result = _apply_crag(
        resolved_query, chunks, top_k=top_k, repository=repository, kb=kb, owner=owner,
        crag_enabled=crag_enabled, query_transform_enabled=query_transform_enabled,
    )

    prompt = build_prompt(query_text, chunks, history=history)

    # Query routing (see app.routing.router): the image retriever uses
    # a completely separate model/vector space (CLIP) from text
    # retrieval, so - unlike retrieve()'s content_type routes, which
    # only ever ADD extra candidates - it's worth skipping entirely
    # when the query doesn't look at all visual, to avoid paying the
    # CLIP embedding + image-search cost on every single question.
    # Computed once and reused for the "sound" check below too, so a
    # question that fires both content-type routes doesn't pay for a
    # second classify_route() embedding call.
    content_type_routes = classify_route(query_text).routes

    if "image" in content_type_routes:
        try:
            # kb scopes to just this KB's images (web screenshots vs.
            # video frames vs. none for markdown/pdf/docx/github/audio)
            # - without it, a per-KB chat tab could surface an image
            # from a completely different source (see
            # retrieve_images's own docstring). None (cross-KB chat)
            # searches every image regardless of origin, unchanged.
            image_hits = retrieve_images(query_text, top_k=MAX_IMAGES_FOR_VISION, kb=kb, owner=owner)
        except Exception:
            logger.exception("Image retrieval failed; continuing text-only.")
            image_hits = []
    else:
        image_hits = []

    # Same reasoning as image_hits above, but for native acoustic
    # similarity (CLAP - see app.embeddings.audio_embedder). A no-op
    # (empty result, not an error) if ENABLE_AUDIO_SIMILARITY_SEARCH
    # was never turned on at ingestion time - nothing was ever stored
    # to match against, so query_audio_clip_embedding just returns [].
    if "sound" in content_type_routes:
        try:
            audio_clip_hits = retrieve_audio_clips(query_text, top_k=MAX_IMAGES_FOR_VISION, kb=kb, owner=owner)
        except Exception:
            logger.exception("Audio clip retrieval failed; continuing without it.")
            audio_clip_hits = []
    else:
        audio_clip_hits = []

    relevant_images = [hit for hit in image_hits if hit["similarity"] >= IMAGE_SIMILARITY_THRESHOLD]

    if use_vision:
        images_to_send = relevant_images
    else:
        images_to_send = [
            hit for hit in relevant_images if _is_code_screenshot(hit["metadata"]["image_url"], owner)
        ]
    vision_image_urls = [hit["metadata"]["image_url"] for hit in images_to_send]

    # Additionally SURFACE (for UI display only, not for the vision
    # model) every image that a retrieved chunk explicitly names via
    # its "[Image: alt] (url)" marker - the chunk's own text tells us
    # these images belong next to the answer being generated, whether
    # or not CLIP's separate embedding space happens to rank them
    # highly for this query. This is what makes the "setup screenshot"
    # for e.g. a Targetprocess automation rule appear alongside the
    # rule's JSON code, even when the query ("give me the code for X")
    # contains no visual keywords for the router (see
    # app.routing.router._IMAGE_KEYWORDS) and image retrieval is
    # therefore skipped entirely. Merged after the vision decision so
    # the (small) LLM-call cost stays bounded by CLIP + MAX_IMAGES_FOR_VISION.
    seen_urls = {hit["metadata"].get("image_url") for hit in relevant_images}
    for referenced_hit in _images_referenced_in_chunks(chunks):
        if referenced_hit["metadata"]["image_url"] not in seen_urls:
            relevant_images.append(referenced_hit)
            seen_urls.add(referenced_hit["metadata"]["image_url"])

    # Semantic near-duplicate answer cache (see app.generation.semantic_cache):
    # only attempted for a history-free first turn with no vision image
    # attached - see that module's docstring for exactly why each gate
    # exists (a thread's prior turns and any attached image both affect
    # the real answer in ways this cache has no visibility into). A
    # lookup failure must never break the answer, so it's caught and
    # simply treated as a miss.
    cached_answer: str | None = None
    if not history and not vision_image_urls and semantic_cache.is_enabled():
        try:
            question_embedding = embed_texts([contextualization.resolved])[0]
            cached_answer = semantic_cache.find_cached_answer(question_embedding, chunks, kb, repository, owner)
        except Exception:  # noqa: BLE001 - cache lookup failure must not break the answer
            logger.exception("Semantic cache lookup failed; continuing without it.")

    return prompt, relevant_images, vision_image_urls, chunks, crag_result, contextualization, cached_answer, audio_clip_hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question about this repository (Version 1 RAG pipeline)."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Your natural-language question. Optional when using --ingest-github alone.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many chunks to retrieve as context (default: 3).",
    )
    parser.add_argument(
        "--ingest-github",
        metavar="REPO_URL",
        help=(
            "Ingest a public GitHub repo before answering (e.g. 'owner/repo' or "
            "a full https://github.com/... URL). Can be combined with a question, "
            "or used on its own to just ingest."
        ),
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Branch to ingest with --ingest-github (default: the repo's own default branch).",
    )
    parser.add_argument(
        "--repo",
        metavar="OWNER/REPO",
        default=None,
        help=(
            "Scope the question to one already-ingested GitHub repo (e.g. 'owner/repo'). "
            "Without this, questions are answered from the ENTIRE shared vector store "
            "(every source ever ingested), which can surface the wrong repo's chunks once "
            "more than one repo has been ingested."
        ),
    )
    args = parser.parse_args()

    configure_logging()

    if args.ingest_github:
        from app.ingestion.ingest import ingest_github_repo

        print(f"Ingesting GitHub repo '{args.ingest_github}'...")
        stored = ingest_github_repo(args.ingest_github, branch=args.branch)
        print(f"Stored {stored} chunk(s) from '{args.ingest_github}'.\n")

    if not args.question:
        return

    logger.info("CLI question received: %r (top_k=%d)", args.question, args.top_k)

    print(f"Question: {args.question}\n")
    print("Thinking...\n")
    answer, image_hits = ask_with_vision(args.question, top_k=args.top_k, repository=args.repo)

    print("=" * 70)
    print("Answer:")
    print(answer)
    if image_hits:
        print("\n" + "-" * 70)
        print(f"(Answer used {len(image_hits)} relevant image(s) as visual context:)")
        for hit in image_hits:
            print(f" - {hit['metadata'].get('image_url')}")


if __name__ == "__main__":
    main()

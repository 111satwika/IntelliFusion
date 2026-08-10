"""
Corrective RAG (CRAG) - retrieval-time relevance evaluation.

After the base retriever returns its top-k chunks, this module asks
the local LLM to grade each chunk's relevance to the query. The
per-chunk verdicts collapse into an overall "correct / ambiguous /
incorrect" verdict on the retrieved evidence, which the caller then
uses to decide whether to:

  * "correct":   use the chunks as-is (no correction needed).
  * "ambiguous": drop the incorrect-scored chunks; if too few remain,
                 do ONE re-retrieval pass using a rewritten query
                 variant and merge the results (deduped) before
                 handing the shortlist off to the LLM.
  * "incorrect": surface the failure - the caller can choose to
                 refuse cleanly ("I don't have enough grounded
                 information to answer") rather than have the LLM
                 hallucinate over irrelevant snippets.

Design constraints:
  * Single Ollama call per evaluation, JSON-schema output, so N chunks
    cost the same LLM time as 1. The prompt lists every chunk with its
    index and asks for one verdict per index.
  * lru_cache keyed by (sha256(query), sha256(chunk_ids)) so repeat
    queries during a session don't re-evaluate identical chunk sets.
  * Silent fallback to "correct" when the evaluator fails - retrieval
    already returned chunks, so falling back to "use them" is strictly
    less harmful than dropping them.
  * Runtime enable flag via set_enabled() - callers don't have to
    thread a kwarg through 4 layers of the retrieval / generation
    stack (same pattern as app.retrieval.query_transform).

Why LLM-as-judge instead of the paper's fine-tuned T5 retrieval
evaluator: keeps the deployment story to one model (qwen2.5:7b) and
avoids downloading + hosting a second model just for a boolean-ish
signal. Structured JSON output plus temperature=0 keeps the judge's
verdicts stable enough to be useful.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

import requests

logger = logging.getLogger(__name__)


_OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "qwen2.5:7b-instruct"

# Read timeout for the CRAG evaluator's Ollama call. Aligned with the
# query transformer's timeout - same model, similar prompt size, same
# CPU-bound first-token warm-up cost. Set high enough that a cold
# Ollama doesn't cause a spurious fallback.
_EVAL_TIMEOUT_S = 180

# How many chunks are excerpted into the evaluator prompt. Truncated
# per-chunk so a very long chunk doesn't blow the prompt window; the
# evaluator only needs a taste of the content to judge relevance.
_MAX_CHUNK_PREVIEW_CHARS = 500

# Overall-verdict aggregation thresholds. Kept as module constants so
# tuning is a one-line change rather than a doc-hunt through this
# function. A chunk counts toward "correct" if the judge marked it
# "correct"; "partial" chunks are salvageable (kept on ambiguous
# verdict) but not by themselves enough to declare the retrieval
# correct overall.
_VERDICT_CORRECT_MIN_RATIO = 0.6      # >=60% correct  -> overall correct
_VERDICT_AMBIGUOUS_MIN_KEEPERS = 1    # any correct OR partial -> ambiguous
                                       # (else -> incorrect)

# Runtime enable flag - off by default so imports don't accidentally
# require Ollama; the UI/CLI turns it on explicitly via set_enabled().
_enabled = False


@dataclass
class ChunkVerdict:
    """Per-chunk relevance verdict from the LLM evaluator."""

    index: int
    relevance: str  # "correct" | "partial" | "incorrect"
    reason: str

    @property
    def is_keeper(self) -> bool:
        """Whether this chunk survives CRAG filtering. Partial chunks
        are kept - they usually contribute background context even when
        they don't directly answer the question, and the cross-encoder
        already ranked them above the discarded majority."""
        return self.relevance in ("correct", "partial")


@dataclass
class CragResult:
    """The full CRAG verdict + filtered chunk indices for one query."""

    query: str
    overall: str  # "correct" | "ambiguous" | "incorrect"
    verdicts: list[ChunkVerdict] = field(default_factory=list)
    kept_indices: list[int] = field(default_factory=list)
    dropped_indices: list[int] = field(default_factory=list)
    used_llm: bool = False
    fell_back: bool = False

    @property
    def counts(self) -> dict[str, int]:
        """Small helper for the UI - number of chunks in each verdict
        bucket, for the retrieval-evaluation expander."""
        out = {"correct": 0, "partial": 0, "incorrect": 0}
        for v in self.verdicts:
            if v.relevance in out:
                out[v.relevance] += 1
        return out


def set_enabled(enabled: bool) -> None:
    """Turn CRAG evaluation on/off process-wide.

    Called by the UI when the sidebar toggle changes; safe to call
    every render because it's just a module-flag write.
    """
    global _enabled
    _enabled = bool(enabled)


def is_enabled() -> bool:
    return _enabled


def evaluate_chunks(
    query_text: str, chunks: list[dict], *, enabled: bool | None = None
) -> CragResult:
    """Evaluate whether the retrieved chunks are relevant to the query.

    When the module is disabled or the input is empty, returns a
    trivial "correct" result with every chunk kept - the caller can
    proceed as if CRAG weren't in the pipeline at all.

    ``enabled`` overrides the process-global toggle for this call
    only (used by callers that need a per-session decision, e.g. the
    chat API resolving a session's own CRAG setting); omit it to fall
    back to ``is_enabled()`` as before.

    Cached by (query_hash, chunk_ids_hash) so a chat that re-asks the
    same question over the same chunk set pays zero extra LLM cost.
    """
    query_text = (query_text or "").strip()
    if not query_text or not chunks:
        return CragResult(
            query=query_text,
            overall="correct",
            verdicts=[],
            kept_indices=list(range(len(chunks))),
        )

    effective_enabled = _enabled if enabled is None else enabled
    if not effective_enabled:
        return CragResult(
            query=query_text,
            overall="correct",
            verdicts=[],
            kept_indices=list(range(len(chunks))),
        )

    key = _cache_key(query_text, chunks)
    return _cached_evaluate(key, query_text, tuple(_hashable_chunk(c) for c in chunks))


def clear_cache() -> None:
    """Wipe the CRAG evaluation cache."""
    _cached_evaluate.cache_clear()


# ---------- Internals ----------


def _cache_key(query_text: str, chunks: list[dict]) -> str:
    """Hash-based cache key combining query text and the ordered
    document_id/chunk_index tuple of the retrieved chunks. Different
    chunk sets = different cache entries; same chunks in a different
    order = also different (order matters because the LLM indexes by
    position in the prompt)."""
    ids = "|".join(_chunk_id(c) for c in chunks)
    material = f"{query_text}\n---\n{ids}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _chunk_id(chunk: dict) -> str:
    md = chunk.get("metadata") or {}
    doc = md.get("document_id") or "?"
    idx = md.get("chunk_index")
    return f"{doc}::{idx}"


def _hashable_chunk(chunk: dict) -> tuple[str, str]:
    """Reduce a chunk to (id, content_preview) so it can go into an
    lru_cache key argument list (dicts aren't hashable). Content
    preview truncated so cache key size stays bounded even for long
    parent chunks."""
    return (_chunk_id(chunk), (chunk.get("content") or "")[:_MAX_CHUNK_PREVIEW_CHARS])


@lru_cache(maxsize=128)
def _cached_evaluate(
    _key: str, query_text: str, _chunk_signature: tuple
) -> CragResult:
    """LRU-cached evaluation. _key uniquifies the cache entry; the
    chunk signature is passed through so the payload isn't
    re-derived from the hash. We rebuild the "real" chunks from the
    signature (id + content preview) - that's all the judge needs to
    grade relevance."""
    # Rebuild the minimal chunk shape the prompt needs.
    chunks_for_prompt = [
        {"id": cid, "content": content}
        for cid, content in _chunk_signature
    ]
    try:
        raw = _call_ollama_json(query_text, chunks_for_prompt)
    except Exception:  # noqa: BLE001 - any evaluator failure falls back
        logger.exception("CRAG evaluator LLM call failed; falling back to 'correct'.")
        return CragResult(
            query=query_text,
            overall="correct",
            verdicts=[],
            kept_indices=list(range(len(chunks_for_prompt))),
            used_llm=True,
            fell_back=True,
        )
    return _parse_eval_json(query_text, len(chunks_for_prompt), raw)


def _build_prompt(query_text: str, chunks: list[dict]) -> str:
    """Assemble the JSON-schema prompt for the CRAG evaluator.

    Each chunk gets a numbered header + a truncated preview. The judge
    is asked to grade each chunk on the same three-value scale as the
    output schema, and to give an OVERALL verdict.
    """
    chunk_blocks = []
    for i, chunk in enumerate(chunks):
        preview = (chunk.get("content") or "")[:_MAX_CHUNK_PREVIEW_CHARS]
        chunk_blocks.append(f"[{i}] {preview}")
    chunks_text = "\n\n".join(chunk_blocks)
    return f"""You are a retrieval-evaluator for a RAG system. Given a user
question and the chunks the retriever returned, judge how relevant
each chunk actually is to answering the question.

Return ONLY a single JSON object matching this schema exactly:

{{
  "chunks": [
    {{"index": <0-based chunk index>, "relevance": "correct" | "partial" | "incorrect", "reason": "<one short sentence>"}},
    ...
  ],
  "overall": "correct" | "ambiguous" | "incorrect"
}}

Rules for each per-chunk verdict:
- "correct":   directly answers or contains material clearly needed to
               answer the user's question.
- "partial":   related to the topic and useful as background, but does
               not by itself answer the question.
- "incorrect": off-topic or contradicts the question - a purely
               distracting chunk.

Rules for the overall verdict:
- "correct":   most chunks are relevant; the retrieved set is enough.
- "ambiguous": some relevant, some not - filtering will help but the
               remainder may still be thin.
- "incorrect": none of the chunks meaningfully address the question.

Include exactly one entry per chunk index (0..{len(chunks) - 1}).

User question:
{query_text}

Retrieved chunks:
{chunks_text}
"""


def _call_ollama_json(query_text: str, chunks: list[dict]) -> str:
    """POST to Ollama's non-streaming endpoint and return the raw
    response body. Streaming JSON doesn't help - we can't act on the
    verdicts until the whole schema is present."""
    prompt = _build_prompt(query_text, chunks)
    logger.info(
        "CRAG evaluator: calling Ollama for query=%r chunks=%d",
        query_text, len(chunks),
    )
    response = requests.post(
        _OLLAMA_URL,
        json={
            "model": _DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False,
            # Structured output - Ollama constrains generation to
            # valid JSON, so we don't have to strip fences or trailing
            # prose in the parser.
            "format": "json",
            "options": {
                # Deterministic - same input, same verdict on cache
                # miss.
                "temperature": 0.0,
                # Enough tokens for N per-chunk verdicts + overall,
                # capped so a runaway model can't spin forever.
                "num_predict": 1024,
            },
        },
        timeout=_EVAL_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "")


def _parse_eval_json(query_text: str, num_chunks: int, raw: str) -> CragResult:
    """Parse the evaluator's JSON. Any parse or schema failure -> fall
    back to "correct" so chunks aren't silently dropped on evaluator
    misbehavior. Malformed verdicts are treated as "correct" (safer
    default than "incorrect")."""
    fallback = CragResult(
        query=query_text,
        overall="correct",
        verdicts=[],
        kept_indices=list(range(num_chunks)),
        used_llm=True,
        fell_back=True,
    )
    if not raw or not raw.strip():
        logger.warning("CRAG evaluator returned empty response; falling back to 'correct'.")
        return fallback
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        stripped = _strip_code_fence(raw)
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("CRAG evaluator returned non-JSON; falling back. raw=%r", raw[:200])
            return fallback

    if not isinstance(obj, dict):
        return fallback

    verdicts: list[ChunkVerdict] = []
    seen_indices: set[int] = set()
    for entry in _coerce_list(obj.get("chunks")):
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= num_chunks or idx in seen_indices:
            continue
        seen_indices.add(idx)
        relevance = _coerce_str(entry.get("relevance")).lower()
        if relevance not in ("correct", "partial", "incorrect"):
            # Unknown verdict from the judge - treat as correct so
            # we don't silently drop a chunk the retriever ranked
            # highly.
            relevance = "correct"
        reason = _coerce_str(entry.get("reason")) or "(no reason given)"
        verdicts.append(ChunkVerdict(index=idx, relevance=relevance, reason=reason))

    # Chunks the judge didn't grade - default to "correct" for the
    # same reason: don't drop what the retriever surfaced.
    for missing in range(num_chunks):
        if missing not in seen_indices:
            verdicts.append(
                ChunkVerdict(index=missing, relevance="correct", reason="(not graded)")
            )
    verdicts.sort(key=lambda v: v.index)

    kept = [v.index for v in verdicts if v.is_keeper]
    dropped = [v.index for v in verdicts if not v.is_keeper]

    overall = _coerce_str(obj.get("overall")).lower()
    if overall not in ("correct", "ambiguous", "incorrect"):
        overall = _derive_overall(verdicts)

    # Overall consistency guard: if the judge said "correct" but its
    # own per-chunk grades imply the opposite (majority incorrect),
    # trust the per-chunk grades - they're what the caller will act
    # on downstream.
    derived = _derive_overall(verdicts)
    if overall == "correct" and derived == "incorrect":
        overall = derived

    return CragResult(
        query=query_text,
        overall=overall,
        verdicts=verdicts,
        kept_indices=kept,
        dropped_indices=dropped,
        used_llm=True,
        fell_back=False,
    )


def _derive_overall(verdicts: list[ChunkVerdict]) -> str:
    """Fallback overall-verdict aggregation from per-chunk grades,
    used when the judge's "overall" field is missing/malformed OR when
    it contradicts its own per-chunk grades."""
    if not verdicts:
        return "correct"
    n = len(verdicts)
    correct = sum(1 for v in verdicts if v.relevance == "correct")
    keepers = sum(1 for v in verdicts if v.is_keeper)
    if correct / n >= _VERDICT_CORRECT_MIN_RATIO:
        return "correct"
    if keepers >= _VERDICT_AMBIGUOUS_MIN_KEEPERS:
        return "ambiguous"
    return "incorrect"


_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_code_fence(raw: str) -> str:
    match = _CODE_FENCE.match(raw)
    if match:
        return match.group(1)
    return raw


def _coerce_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_list(value) -> list:
    if isinstance(value, list):
        return value
    return []

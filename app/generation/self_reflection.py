"""
Self-Reflective RAG (Self-RAG) - generation-time answer critique.

After the LLM produces an answer, this module asks the same local LLM
(acting as a judge) to score the answer on three dimensions matching
the reflection tokens from the Self-RAG paper:

  * grounded  ([IsSup]):  does every claim in the answer trace back
                          to the retrieved chunks? (0-1)
  * relevant  ([IsRel]):  are the retrieved chunks the right ones for
                          this question? (0-1)
  * useful    ([IsUse]):  does the answer actually address what was
                          asked, at appropriate depth? (0-1)

  * hallucinations:       list of specific claims from the answer that
                          the judge could not find support for in the
                          chunks. Rendered under the critique expander
                          so the user can spot the exact sentences to
                          be skeptical of.

Design constraints:
  * Post-hoc (runs AFTER streaming completes) so it never delays the
    user's first token - the answer is already fully rendered by the
    time the critic starts.
  * Single Ollama call, JSON-schema output.
  * lru_cache keyed by (query_hash, answer_hash) - re-asking the same
    question won't produce a bit-identical answer (temperature 0 keeps
    it stable enough for cache hits during a session).
  * Silent fallback on evaluator failure - the UI just skips the
    critique panel; the answer itself is unaffected.
  * Runtime enable flag via set_enabled(), same pattern as
    app.retrieval.query_transform and app.retrieval.crag.

Why LLM-as-judge instead of the paper's fine-tuned reflection tokens:
same reason as CRAG's design - training + hosting a second model
(especially one that emits special tokens embedded in generation
itself) is a step change in deployment complexity, and the LLM-as-
judge signal is strong enough at temperature=0 to be useful for the
"is this answer trustworthy?" UI badge.
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

# Read timeout for the critic's Ollama call. Same reasoning as the
# CRAG evaluator - on CPU the first-token warm-up cost dominates, and
# a truly stuck Ollama shouldn't add minutes to every question.
_CRITIQUE_TIMEOUT_S = 180

# How much of each chunk the critic sees when checking grounding.
# Truncated per-chunk so a long parent chunk doesn't blow the context
# window - the critic only needs enough of each chunk to check whether
# specific claims appear in it.
_MAX_CHUNK_PREVIEW_CHARS = 800

# How much of the answer the critic sees. Long-form answers get
# truncated so the critic's own prompt stays bounded; the truncation
# happens sentence-wise where possible so we don't cut a claim in
# half.
_MAX_ANSWER_CHARS = 4000

# Confidence threshold: any score under this triggers a "low
# confidence" badge in the UI. Deliberately liberal at 0.5 rather
# than 0.7 - the judge tends toward the middle of the 0-1 range and
# a stricter threshold would light up the warning on almost every
# answer, teaching users to ignore it.
_LOW_CONFIDENCE_THRESHOLD = 0.5

# Runtime enable flag.
_enabled = False


@dataclass
class DimensionScore:
    """Score for one Self-RAG reflection dimension."""

    score: float  # 0.0 - 1.0
    reason: str


@dataclass
class CritiqueResult:
    """The full Self-RAG critique for one (query, chunks, answer)
    triple, plus derived UI helpers."""

    query: str
    grounded: DimensionScore
    relevant: DimensionScore
    useful: DimensionScore
    hallucinations: list[str] = field(default_factory=list)
    used_llm: bool = False
    fell_back: bool = False

    @property
    def min_score(self) -> float:
        """The weakest dimension - what the UI's low-confidence badge
        actually gates on. Using min() rather than a mean keeps a
        single-dimension collapse (e.g. answer clearly hallucinated
        but otherwise on-topic) from getting averaged away."""
        return min(self.grounded.score, self.relevant.score, self.useful.score)

    @property
    def is_low_confidence(self) -> bool:
        return self.min_score < _LOW_CONFIDENCE_THRESHOLD

    def as_dimension_rows(self) -> list[dict]:
        """Small helper for the UI - rows for a st.table showing
        one line per reflection dimension with score + reason."""
        return [
            {
                "dimension": "grounded",
                "score": round(self.grounded.score, 2),
                "reason": self.grounded.reason,
            },
            {
                "dimension": "relevant",
                "score": round(self.relevant.score, 2),
                "reason": self.relevant.reason,
            },
            {
                "dimension": "useful",
                "score": round(self.useful.score, 2),
                "reason": self.useful.reason,
            },
        ]


def set_enabled(enabled: bool) -> None:
    """Turn Self-RAG critique on/off process-wide."""
    global _enabled
    _enabled = bool(enabled)


def is_enabled() -> bool:
    return _enabled


def critique_answer(
    query_text: str, chunks: list[dict], answer: str, *, enabled: bool | None = None
) -> CritiqueResult | None:
    """Judge the answer's grounding/relevance/utility against the
    retrieved chunks.

    Returns ``None`` when the module is disabled or the answer is
    empty - the caller (UI) then knows to skip the critique panel
    entirely rather than rendering a dashboard of zeros.

    ``enabled`` overrides the process-global toggle for this call
    only (used by callers resolving a per-session setting instead of
    the shared default); omit it to fall back to ``is_enabled()``.

    Cached by (query_hash, answer_hash) so the history-replay path
    doesn't re-critique on every rerun.
    """
    query_text = (query_text or "").strip()
    answer = (answer or "").strip()
    effective_enabled = _enabled if enabled is None else enabled
    if not effective_enabled:
        return None
    if not query_text or not answer:
        return None
    if not chunks:
        # No chunks to ground against - critiquing "grounded" would be
        # meaningless. UI still gets None so it can suppress the panel.
        return None

    key = _cache_key(query_text, answer)
    return _cached_critique(
        key, query_text, tuple(_hashable_chunk(c) for c in chunks), answer[:_MAX_ANSWER_CHARS]
    )


def clear_cache() -> None:
    """Wipe the critique cache."""
    _cached_critique.cache_clear()


# ---------- Internals ----------


def _cache_key(query_text: str, answer: str) -> str:
    material = f"{query_text}\n---\n{answer}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _hashable_chunk(chunk: dict) -> tuple[str, str]:
    md = chunk.get("metadata") or {}
    doc = md.get("document_id") or "?"
    idx = md.get("chunk_index")
    cid = f"{doc}::{idx}"
    return (cid, (chunk.get("content") or "")[:_MAX_CHUNK_PREVIEW_CHARS])


@lru_cache(maxsize=128)
def _cached_critique(
    _key: str, query_text: str, chunk_signature: tuple, answer: str
) -> CritiqueResult:
    chunks_for_prompt = [
        {"id": cid, "content": content} for cid, content in chunk_signature
    ]
    try:
        raw = _call_ollama_json(query_text, chunks_for_prompt, answer)
    except Exception:  # noqa: BLE001
        logger.exception("Self-RAG critic LLM call failed; returning neutral fallback.")
        return _fallback_critique(query_text)
    return _parse_critique_json(query_text, raw)


def _build_prompt(query_text: str, chunks: list[dict], answer: str) -> str:
    """Assemble the JSON-schema prompt for the Self-RAG critic."""
    chunk_blocks = []
    for i, chunk in enumerate(chunks):
        preview = (chunk.get("content") or "")[:_MAX_CHUNK_PREVIEW_CHARS]
        chunk_blocks.append(f"[Chunk {i}] {preview}")
    chunks_text = "\n\n".join(chunk_blocks)

    return f"""You are a strict answer-quality evaluator for a retrieval-
augmented question answering system. Given a user question, the
retrieved chunks the answer was built from, and the answer itself,
score the answer on three dimensions.

Return ONLY a single JSON object matching this schema exactly:

{{
  "grounded":       {{"score": <0.0-1.0>, "reason": "<one short sentence>"}},
  "relevant":       {{"score": <0.0-1.0>, "reason": "<one short sentence>"}},
  "useful":         {{"score": <0.0-1.0>, "reason": "<one short sentence>"}},
  "hallucinations": [<up to 5 short sentences quoting or paraphrasing
                     specific claims from the answer that are NOT
                     supported by the retrieved chunks. Empty list if
                     the answer is fully grounded.>]
}}

Scoring rubric (be strict; low scores are OK):
- grounded:  1.0 if every factual claim in the answer is directly
             supported by the retrieved chunks; 0.5 if some claims
             are supported and some appear to be generic knowledge
             not from the chunks; 0.0 if the answer is largely
             independent of the chunks.
- relevant:  1.0 if the retrieved chunks are the correct source for
             this question; 0.5 if they touch the topic but miss the
             specific ask; 0.0 if the chunks are off-topic.
- useful:    1.0 if the answer directly and completely addresses the
             user's question; 0.5 if it addresses it partially or
             vaguely; 0.0 if it evades or dodges.

The "hallucinations" list should be empty when grounded >= 0.9. Only
call out specific claims - do NOT restate the whole answer.

User question:
{query_text}

Retrieved chunks:
{chunks_text}

Answer being critiqued:
{answer}
"""


def _call_ollama_json(query_text: str, chunks: list[dict], answer: str) -> str:
    prompt = _build_prompt(query_text, chunks, answer)
    logger.info(
        "Self-RAG critic: calling Ollama for query=%r answer_chars=%d chunks=%d",
        query_text, len(answer), len(chunks),
    )
    response = requests.post(
        _OLLAMA_URL,
        json={
            "model": _DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                # Room for 3 scored dimensions + up to 5 hallucination
                # sentences; capped for the same runaway-model
                # protection as everywhere else.
                "num_predict": 1024,
            },
        },
        timeout=_CRITIQUE_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "")


def _parse_critique_json(query_text: str, raw: str) -> CritiqueResult:
    fallback = _fallback_critique(query_text)
    if not raw or not raw.strip():
        logger.warning("Self-RAG critic returned empty response; using neutral fallback.")
        return fallback
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        stripped = _strip_code_fence(raw)
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("Self-RAG critic returned non-JSON; using neutral fallback. raw=%r", raw[:200])
            return fallback

    if not isinstance(obj, dict):
        return fallback

    return CritiqueResult(
        query=query_text,
        grounded=_parse_dimension(obj.get("grounded")),
        relevant=_parse_dimension(obj.get("relevant")),
        useful=_parse_dimension(obj.get("useful")),
        hallucinations=[
            h for h in (_coerce_str(x) for x in _coerce_list(obj.get("hallucinations")))
            if h
        ][:5],
        used_llm=True,
        fell_back=False,
    )


def _parse_dimension(value) -> DimensionScore:
    """Coerce one dimension entry to a DimensionScore. Missing /
    malformed dimensions get a neutral 0.5 with a "(unrated)" reason
    so downstream code can still render the table without special-
    casing None."""
    if not isinstance(value, dict):
        return DimensionScore(score=0.5, reason="(unrated)")
    try:
        score = float(value.get("score", 0.5))
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(1.0, score))
    reason = _coerce_str(value.get("reason")) or "(no reason given)"
    return DimensionScore(score=score, reason=reason)


def _fallback_critique(query_text: str) -> CritiqueResult:
    """Neutral 0.5 across the board - used on evaluator failure so the
    UI doesn't render either a green "high confidence" (misleading if
    we couldn't check) or a red "low confidence" (misleading if the
    answer might be fine) badge."""
    neutral = DimensionScore(score=0.5, reason="(critic unavailable)")
    return CritiqueResult(
        query=query_text,
        grounded=neutral,
        relevant=neutral,
        useful=neutral,
        hallucinations=[],
        used_llm=True,
        fell_back=True,
    )


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

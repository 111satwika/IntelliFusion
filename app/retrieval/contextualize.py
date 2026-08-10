"""
Conversation-memory query contextualization.

A follow-up question inside a multi-turn chat often only makes sense
given what was just discussed ("what about tests for that?"). Embedded
on its own, "that" carries no retrievable meaning - dense/BM25 search
has no idea what it refers to, so a naive re-embed-and-search of the
raw follow-up returns unrelated chunks. This module rewrites such a
follow-up into a standalone question BEFORE retrieval runs, using the
session's recent conversation turns (see session_store.py's per-thread
history) as context for one Ollama call.

This is deliberately a SEPARATE module from app.retrieval.query_transform,
not an added mode of it:
  - query_transform's LRU cache is keyed on (query text, kb) alone,
    which assumes a query's meaning is stable across calls. A
    follow-up's meaning depends on turn-varying history, so caching by
    text alone would silently return a stale rewrite from a different
    conversation. This module does not cache at all - history changes
    on every single turn, so a cache would be a near-permanent miss.
  - query_transform's job is "given a well-formed query, produce
    variants that widen retrieval recall" (paraphrases, HyDE,
    keywords). This module's job is a different, prior step: "figure
    out what the user is even asking, given what was said before."
    Conflating the two would blur both responsibilities.

Fast path: a question with no history, or one that already reads as
fully self-contained (no pronoun/demonstrative referring to something
unstated, no continuation opener like "what about", not suspiciously
short), skips the LLM entirely - a normal standalone question pays
zero extra latency.

Enable/disable: same runtime-flag pattern as
app.retrieval.crag/query_transform/app.generation.self_reflection - a
process-global default (off, so imports never require Ollama), plus a
per-call ``enabled`` override so a caller can resolve a per-session
setting instead (see session_store.py's conversation_memory_enabled).

Failure handling: any Ollama/parse failure falls back to the original,
unmodified question - retrieval still runs, just without the benefit
of contextualization, exactly the "purely additive" failure posture
every other LLM-assist module in this pipeline uses.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


_OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "qwen2.5:7b-instruct"

# Same reasoning as query_transform.py's _TRANSFORM_TIMEOUT_S - on CPU
# the model spends most of the budget on prompt processing before the
# first token, so this is looser than it looks.
_CONTEXTUALIZE_TIMEOUT_S = 180

# How many of the thread's most recent turns are shown to the LLM when
# resolving a follow-up. Independent of session_store._MAX_HISTORY_TURNS
# (which bounds how much is STORED) - only the last few turns are ever
# actually relevant to resolving "what about that", and a shorter
# prompt means a faster call.
_MAX_TURNS_IN_PROMPT = 3

# Each history answer is truncated to this many characters when built
# into the prompt, so one long prior answer can't blow the prompt size
# - only enough is needed for the model to know what "that" was.
_MAX_ANSWER_PREVIEW_CHARS = 300

# Fast-path heuristic: a question is treated as context-dependent (and
# sent to the LLM) if it contains one of these pronouns/demonstratives
# with no clear antecedent of its own...
_REFERENCE_WORDS = {
    "it", "its", "that", "this", "they", "them", "their", "those", "these",
    "he", "she", "him", "her",
}
# ...or opens with a continuation phrase that only makes sense attached
# to a prior turn...
_CONTINUATION_OPENERS = (
    "what about", "and ", "also", "what else", "how about", "same for",
    "same with", "what if", "why not", "or ",
)
# ...or is short enough that it's plausibly an elliptical follow-up
# rather than a genuinely complete question ("compare A and B and C
# and D and E" is long AND self-contained; "and that one?" is short
# and almost certainly isn't).
_FAST_PATH_MIN_WORDS_FOR_STANDALONE = 7

# Runtime enable flag - off by default, same as crag/query_transform/
# self_reflection, so importing this module never implies Ollama must
# be up. The chat API resolves the per-session setting and passes it
# in via `enabled=`; this global is only the CLI/eval-runner fallback.
_enabled = False


@dataclass
class ContextualizeResult:
    """Result of attempting to resolve a follow-up question against
    recent conversation turns.

    ``used_llm``/``fell_back`` are for observability, same convention
    as query_transform.TransformResult/crag.CragResult - the insight
    panel uses them to honestly report "no history to resolve against",
    "already looked standalone", "resolved via LLM", or "LLM was tried
    but we fell back".
    """

    original: str
    resolved: str
    used_llm: bool = False
    fell_back: bool = False


def set_enabled(enabled: bool) -> None:
    """Turn conversation-memory contextualization on/off process-wide."""
    global _enabled
    _enabled = bool(enabled)


def is_enabled() -> bool:
    return _enabled


def contextualize_query(
    question: str,
    history: list[dict],
    kb: str | None = None,
    *,
    enabled: bool | None = None,
) -> ContextualizeResult:
    """Resolve `question` into a standalone form using recent turns.

    Returns a ContextualizeResult with `resolved == original` (no LLM
    call made) when: the module is disabled, there's no history for
    this thread, or the question already reads as self-contained (see
    `_looks_context_dependent`).

    ``enabled`` overrides the process-global toggle for this call only
    (used by callers resolving a per-session setting instead of the
    shared default); omit it to fall back to ``is_enabled()``.

    No caching - see module docstring for why.
    """
    question = (question or "").strip()
    if not question:
        return ContextualizeResult(original="", resolved="")

    effective_enabled = _enabled if enabled is None else enabled
    if not effective_enabled:
        return ContextualizeResult(original=question, resolved=question)

    if not _looks_context_dependent(question, history):
        logger.debug("Contextualization fast-path (already standalone): %r", question)
        return ContextualizeResult(original=question, resolved=question)

    try:
        raw = _call_ollama_json(question, history, kb)
    except Exception:  # noqa: BLE001 - any contextualization failure falls back
        logger.exception("Contextualization LLM call failed; falling back to original question.")
        return ContextualizeResult(original=question, resolved=question, used_llm=True, fell_back=True)
    return _parse_response(question, raw)


# ---------- Internals ----------


def _looks_context_dependent(question: str, history: list[dict]) -> bool:
    """Cheap heuristic: is this question worth spending an LLM call on
    to resolve against history? See module docstring's three signals.
    """
    if not history:
        return False
    text = question.lower().strip()
    words = re.findall(r"[a-z0-9']+", text)
    if any(word in _REFERENCE_WORDS for word in words):
        return True
    if text.startswith(_CONTINUATION_OPENERS):
        return True
    if len(words) <= _FAST_PATH_MIN_WORDS_FOR_STANDALONE:
        return True
    return False


def _format_history_for_prompt(history: list[dict]) -> str:
    recent = history[-_MAX_TURNS_IN_PROMPT:]
    blocks = []
    for turn in recent:
        answer = (turn.get("answer") or "").strip()
        if len(answer) > _MAX_ANSWER_PREVIEW_CHARS:
            answer = answer[:_MAX_ANSWER_PREVIEW_CHARS] + "…"
        blocks.append(f"Q: {turn.get('question', '')}\nA: {answer}")
    return "\n\n".join(blocks)


def _build_prompt(question: str, history: list[dict], kb: str | None) -> str:
    kb_hint = f" (about the '{kb}' knowledge base)" if kb else ""
    history_text = _format_history_for_prompt(history)
    return f"""You are resolving a follow-up question in an ongoing chat{kb_hint} into a
standalone question that makes sense with NO other context.

Conversation so far:
{history_text}

New message from the user:
{question}

Return ONLY a single JSON object matching this schema exactly, with no
prose commentary before or after:

{{"standalone_question": "<the new message, rewritten to replace any pronoun or implicit reference to the conversation above with what it actually refers to>"}}

Rules:
- If the new message is already a complete, standalone question that
  doesn't depend on the conversation above, return it unchanged.
- Preserve the user's actual intent and wording as much as possible -
  only resolve references, do not add new claims or change the scope
  of what's being asked.
- Preserve entity names, code symbols, and numeric identifiers verbatim.
- Keep it under 40 words.
"""


def _call_ollama_json(question: str, history: list[dict], kb: str | None) -> str:
    prompt = _build_prompt(question, history, kb)
    logger.info("Contextualizing follow-up question=%r against %d prior turn(s)", question, len(history))
    response = requests.post(
        _OLLAMA_URL,
        json={
            "model": _DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_predict": 128,
            },
        },
        timeout=_CONTEXTUALIZE_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "")


def _parse_response(question: str, raw: str) -> ContextualizeResult:
    fallback = ContextualizeResult(original=question, resolved=question, used_llm=True, fell_back=True)
    if not raw or not raw.strip():
        logger.warning("Contextualization returned empty response; falling back.")
        return fallback
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        stripped = _strip_code_fence(raw)
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("Contextualization returned non-JSON; falling back. raw=%r", raw[:200])
            return fallback

    if not isinstance(obj, dict):
        return fallback

    resolved = _coerce_str(obj.get("standalone_question")) or question
    return ContextualizeResult(original=question, resolved=resolved, used_llm=True, fell_back=False)


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

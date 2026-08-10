"""
Query transformation for the RAG pipeline.

Given a raw user query, this module produces a small family of derived
queries that all get run through the existing hybrid retriever, so
that lexical mismatches, compound questions, and vocabulary drift
between the user and the corpus don't silently drop good chunks out
of the shortlist.

Four transformation techniques share a single LLM call (Ollama,
`qwen2.5:7b-instruct` by default), because paying the ~2-4 s round
trip four times over would be prohibitive:

  1. Rewrite         - a clean single-sentence version of the query,
                       stripping filler ("please", "could you...")
                       and canonicalizing pronouns/references.
  2. Decomposition   - if the query is compound ("compare A and B",
                       "explain X and then Y"), split into 2-4 focused
                       sub-questions.
  3. Multi-query     - two paraphrases with different word choices,
                       so a corpus phrased differently from the user
                       still has BM25 overlap.
  4. Expansion       - a HyDE-style hypothetical answer paragraph
                       (used as an extra dense query) plus 3-5
                       keywords (used as an extra BM25 query).

Fast path: trivial queries (short, single-question, no conjunctions)
skip the LLM entirely and just return the original query unchanged,
so short lookups like "what is BM25" don't pay a 3 s transformer tax.

Enable/disable: a module-level toggle (`set_enabled`) lets the UI turn
transformation on/off without threading a flag through every layer of
the retrieval call chain. Off by default so imports don't accidentally
depend on Ollama being up; the UI/CLI flip it on at startup.

Failure handling: if Ollama is down or the JSON output is malformed,
the transformer silently falls back to the original query only. The
existing hybrid pipeline already works without transformation, so this
module is purely additive.

Caching: an LRU cache keyed by (sha256(query), kb) means the same
question re-asked in the same process pays zero LLM cost the second
time (and any subsequent time). Cache size is bounded so a
long-running Streamlit process doesn't leak memory.
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


# Ollama endpoint. Kept as a module constant (not imported from
# llm_generator) so this module can be unit-tested without pulling in
# the streaming/vision helpers.
_OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "qwen2.5:7b-instruct"

# How many candidate paraphrases the LLM is asked for. 2 is enough to
# cover word-choice drift ("delete" vs "remove", "PR" vs "pull
# request") without inflating the retrieval fan-out beyond what RRF
# can meaningfully fuse.
_NUM_PARAPHRASES = 2

# Maximum sub-queries kept from a decomposed query. Prevents a very
# chatty LLM from producing 8 sub-questions and blowing up the fan-out.
_MAX_SUB_QUERIES = 4

# Maximum keywords kept from expansion. Keeps the derived BM25 query
# short and topical rather than a bag of everything the LLM thought
# might be related.
_MAX_KEYWORDS = 5

# Fast-path heuristic thresholds - a query passing ALL of these is
# considered trivial and skips the LLM call entirely.
_FAST_PATH_MAX_WORDS = 5
_FAST_PATH_MAX_CHARS = 40

# Words that strongly suggest a compound / multi-part question worth
# transforming even if it's under the fast-path length threshold.
_COMPLEX_TRIGGERS = {
    "and", "also", "then", "vs", "versus", "compare", "both",
    "difference", "why", "how", "explain", "list",
}

# Read timeout for the transformer's Ollama call. Deliberately looser
# than a "chat" timeout because on CPU the model can spend most of the
# budget on prompt processing (the JSON-schema prompt is ~250 tokens)
# before it starts emitting tokens - the answer generator uses 300s
# for the same reason. Set high enough that a cold Ollama warm-up
# doesn't cause an unnecessary fallback, low enough that a truly
# stuck Ollama can't add minutes to every retrieval.
_TRANSFORM_TIMEOUT_S = 180

# Runtime enable flag. Off by default so `from ... import ...` at
# import time doesn't accidentally require Ollama; the UI/CLI turns it
# on explicitly via set_enabled(True).
_enabled = False


@dataclass
class TransformResult:
    """A transformed query, ready to feed into the hybrid retriever.

    ``variants`` is the deduplicated list of TEXT queries to embed +
    BM25-search (dense retrieval runs one query_embed per variant).
    ``hyde_answer`` is a separate dense-only extra query (embedded but
    NOT put through BM25). ``keywords`` is a separate BM25-only extra
    query (tokenized but NOT embedded).

    ``used_llm`` and ``fell_back`` are set for observability so the UI
    expander can honestly show what happened: "fast path", "LLM ran
    and succeeded", or "LLM was tried but we fell back".
    """

    original: str
    rewritten: str
    sub_queries: list[str] = field(default_factory=list)
    paraphrases: list[str] = field(default_factory=list)
    hyde_answer: str = ""
    keywords: list[str] = field(default_factory=list)
    used_llm: bool = False
    fell_back: bool = False

    @property
    def variants(self) -> list[str]:
        """Deduplicated ordered list of TEXT queries to run through
        the full hybrid pipeline (dense + BM25).

        Order matters for observability only (RRF is order-independent
        within each ranked list). Original query is always first so
        that if downstream code truncates variants, the user's actual
        question is never dropped.
        """
        seen: set[str] = set()
        out: list[str] = []
        for candidate in [self.original, self.rewritten, *self.sub_queries, *self.paraphrases]:
            candidate = (candidate or "").strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        return out


def set_enabled(enabled: bool) -> None:
    """Turn query transformation on/off process-wide.

    Called by the UI when the sidebar toggle changes; safe to call
    every render because it's just a module-flag write.
    """
    global _enabled
    _enabled = bool(enabled)


def is_enabled() -> bool:
    return _enabled


def transform_query(
    query_text: str, kb: str | None = None, *, enabled: bool | None = None
) -> TransformResult:
    """Return a TransformResult for the given query.

    When the module is disabled, or the query trips the fast-path
    heuristic, no LLM call is made and the result is a trivial
    TransformResult wrapping just the original query.

    ``enabled`` overrides the process-global toggle for this call
    only (used by callers resolving a per-session setting instead of
    the shared default); omit it to fall back to ``is_enabled()``.

    Cached by (sha256(query), kb) - re-asking the same question in the
    same process pays zero extra LLM cost.
    """
    query_text = (query_text or "").strip()
    if not query_text:
        return TransformResult(original="", rewritten="")

    effective_enabled = _enabled if enabled is None else enabled
    if not effective_enabled:
        return TransformResult(original=query_text, rewritten=query_text)

    if not _looks_complex(query_text):
        logger.debug("Query transform fast-path (trivial): %r", query_text)
        return TransformResult(original=query_text, rewritten=query_text)

    key = _cache_key(query_text, kb)
    return _cached_transform(key, query_text, kb)


def clear_cache() -> None:
    """Wipe the transformation cache (e.g. after switching model)."""
    _cached_transform.cache_clear()


# ---------- Internals ----------


def _cache_key(query_text: str, kb: str | None) -> str:
    """Hash-based cache key. Full query string is not used as the key
    directly because lru_cache stores every distinct argument as a
    separate entry, and long queries would be needlessly held in
    memory - the sha256 digest is 64 chars regardless.
    """
    material = f"{kb or '-'}::{query_text}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@lru_cache(maxsize=256)
def _cached_transform(_key: str, query_text: str, kb: str | None) -> TransformResult:
    """LRU-cached transform - _key is the hash from _cache_key and is
    what actually makes lru_cache unique per query; query_text is
    passed through as the second arg so the payload isn't re-derived
    from the hash."""
    try:
        raw = _call_ollama_json(query_text, kb)
    except Exception:  # noqa: BLE001 - any transformer failure falls back
        logger.exception("Query transformer LLM call failed; falling back to original query.")
        return TransformResult(
            original=query_text, rewritten=query_text, used_llm=True, fell_back=True
        )
    return _parse_transform_json(query_text, raw)


def _looks_complex(query_text: str) -> bool:
    """Cheap heuristic: is this query worth spending an LLM call on?

    A query is transformed if it's long enough to plausibly contain
    multiple sub-questions, OR it's short but uses conjunctions /
    "how"/"why"/"explain" that reward paraphrasing, OR it has a
    question mark late in the sentence (suggests compound).
    """
    text = query_text.lower()
    words = re.findall(r"[a-z0-9']+", text)
    if len(words) > _FAST_PATH_MAX_WORDS or len(query_text) > _FAST_PATH_MAX_CHARS:
        return True
    if any(trigger in words for trigger in _COMPLEX_TRIGGERS):
        return True
    if text.count("?") > 1:
        return True
    return False


def _build_prompt(query_text: str, kb: str | None) -> str:
    """Assemble the JSON-schema prompt for the transformer.

    Kept as a plain string (not a jinja/template dependency) because
    it's a single f-string interpolation and the schema is stable.
    """
    kb_hint = ""
    if kb:
        kb_hint = (
            f"The query will be run against the '{kb}' knowledge base "
            "(a mix of prose and code)."
        )
    return f"""You are a query transformation assistant for a retrieval-augmented
question answering system. Given a user's raw question, produce a JSON
object with the following transformations to improve retrieval recall.

{kb_hint}

Return ONLY a single JSON object matching this schema exactly, with no
prose commentary before or after:

{{
  "rewrite": "<a clean, single-sentence rewording of the question, keeping the same intent>",
  "is_complex": <true if the question genuinely contains multiple sub-questions, else false>,
  "sub_queries": [<0 to {_MAX_SUB_QUERIES} focused sub-questions if is_complex is true, else empty list>],
  "paraphrases": [<exactly {_NUM_PARAPHRASES} distinct paraphrases using different word choices>],
  "hyde_answer": "<a 1-2 sentence hypothetical answer paragraph as if you already knew the answer, used only for embedding similarity - do not add caveats or 'I don't know'>",
  "keywords": [<3 to {_MAX_KEYWORDS} topical keywords or technical terms likely to appear in relevant documents>]
}}

Rules:
- Every field is required. Empty lists are fine but the keys must exist.
- Do not include markdown fences, comments, or trailing text.
- Preserve entity names, code symbols, and numeric identifiers verbatim
  in every string field.
- Keep every string under 30 words.

User question:
{query_text}
"""


def _call_ollama_json(query_text: str, kb: str | None) -> str:
    """POST to Ollama's non-streaming /api/generate endpoint and
    return the raw response body's `response` field (which is what
    the model actually produced).

    Non-streaming here (unlike the answer generator's streaming path)
    because we need the FULL JSON before we can parse it - streaming
    a JSON blob doesn't help us start work any earlier.
    """
    prompt = _build_prompt(query_text, kb)
    logger.info("Query transformer: calling Ollama for query=%r kb=%r", query_text, kb)
    response = requests.post(
        _OLLAMA_URL,
        json={
            "model": _DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False,
            # Structured-output mode: forces the model to emit valid
            # JSON, so we don't have to strip markdown fences or
            # trailing prose in the parser. Requires Ollama >=0.5;
            # older Ollama silently ignores this key and we still get
            # the parser's fallback path.
            "format": "json",
            "options": {
                # Deterministic; we want the same query to produce the
                # same transformation across cache misses.
                "temperature": 0.0,
                # 512 tokens is enough headroom for the full JSON
                # schema (rewrite + 4 sub-queries + 2 paraphrases +
                # hyde + 5 keywords) plus a bit of slack; capping it
                # prevents a runaway model from producing pages of
                # extra text.
                "num_predict": 512,
            },
        },
        timeout=_TRANSFORM_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "")


def _parse_transform_json(query_text: str, raw: str) -> TransformResult:
    """Parse the LLM's JSON response into a TransformResult.

    Any parse or schema failure -> fall back to the original query.
    The fallback is intentional: the retrieval pipeline works fine
    without transformation, so a malformed LLM response is just a
    quality miss, not a user-visible error.
    """
    if not raw or not raw.strip():
        logger.warning("Query transformer returned empty response; falling back.")
        return TransformResult(
            original=query_text, rewritten=query_text, used_llm=True, fell_back=True
        )
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Occasionally the model wraps JSON in code fences despite the
        # prompt instructions; try one salvage attempt before giving up.
        stripped = _strip_code_fence(raw)
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("Query transformer returned non-JSON; falling back. raw=%r", raw[:200])
            return TransformResult(
                original=query_text, rewritten=query_text, used_llm=True, fell_back=True
            )

    if not isinstance(obj, dict):
        return TransformResult(
            original=query_text, rewritten=query_text, used_llm=True, fell_back=True
        )

    rewritten = _coerce_str(obj.get("rewrite")) or query_text
    is_complex = bool(obj.get("is_complex"))
    sub_queries = _coerce_str_list(obj.get("sub_queries"))[:_MAX_SUB_QUERIES] if is_complex else []
    paraphrases = _coerce_str_list(obj.get("paraphrases"))[:_NUM_PARAPHRASES]
    hyde_answer = _coerce_str(obj.get("hyde_answer"))
    keywords = _coerce_str_list(obj.get("keywords"))[:_MAX_KEYWORDS]

    return TransformResult(
        original=query_text,
        rewritten=rewritten,
        sub_queries=sub_queries,
        paraphrases=paraphrases,
        hyde_answer=hyde_answer,
        keywords=keywords,
        used_llm=True,
        fell_back=False,
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


def _coerce_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _coerce_str(item)
        if text:
            out.append(text)
    return out

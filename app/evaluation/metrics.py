"""
Evaluation metrics for the RAG pipeline.

Two families:

1. RETRIEVAL metrics (deterministic, need ground-truth chunk IDs):

   * hit_rate_at_k     - is at least one expected chunk/doc in top-k?
   * mrr               - reciprocal rank of the first expected hit
                         (0 if none of the expected hits appear).
   * ndcg_at_k         - discounted cumulative gain normalized to
                         [0, 1]. Binary relevance (in expected set or
                         not) - we don't have graded judgments.
   * precision_at_k    - fraction of the top-k retrieved chunks that
                         are actually expected.
   * recall_at_k       - fraction of the expected chunks that show up
                         somewhere in the top-k.

   All five come in a ``document`` variant (matches by document_id)
   and a ``chunk`` variant (matches by "document_id::chunk_index").
   Document-level is more forgiving - it credits getting the right
   PAGE even if the wrong section of the page won ranking.

2. LLM-AS-JUDGE metrics (RAGAS-style, one Ollama call each):

   * faithfulness      - how well every claim in the answer is
                         supported by the retrieved chunks. Roughly
                         "no hallucinations". [0-1]
   * answer_relevance  - does the answer actually address the
                         question that was asked? [0-1]
   * context_relevance - are the retrieved chunks the right ones for
                         this question? Retrieval-quality signal
                         that works even without ground-truth IDs.
                         [0-1]

Design constraints:
  * All LLM-judge calls use ``format:"json"`` with a small schema,
    temperature=0, num_predict bounded. Same Ollama endpoint /
    default model as the rest of the pipeline (see llm_generator).
  * Failure of any single metric is caught and logged; the item
    just gets a ``None`` for that metric so the aggregate can
    ignore it rather than counting it as zero. This matters when
    Ollama times out on one question in a 50-item run.
  * Prompts are deliberately short and structured. LLM-as-judge is
    unreliable when the judge has too much to read - we bound each
    chunk preview and truncate the answer just like the Self-RAG
    critic does.

Not implemented (deliberately):
  * ROUGE / BLEU / BERTScore against ``expected_answer`` - these
    reward surface overlap and punish valid paraphrases. Faithfulness
    + answer-relevance are better signals for RAG-style Q&A.
  * Ground-truth answer correctness via LLM-judge - could add later,
    but faithfulness+relevance already captures most of what
    "correctness" means when the retrieval is trusted.
"""

from __future__ import annotations

import json
import logging
import math
import re

import requests

logger = logging.getLogger(__name__)


_OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "qwen2.5:7b-instruct"

# Same generous timeout as CRAG / Self-RAG - CPU cold-starts can push
# a single judge call past 60s, and we'd rather have a slow eval than
# a failed one.
_JUDGE_TIMEOUT_S = 180

_MAX_CHUNK_PREVIEW_CHARS = 600
_MAX_ANSWER_CHARS = 3000


# ---------- Retrieval metrics ----------


def _chunk_key(chunk: dict) -> str:
    """Chunk-level identity key: document_id::chunk_index. Matches the
    ``expected_chunk_ids`` field in dataset items."""
    md = chunk.get("metadata") or {}
    doc = md.get("document_id", "?")
    idx = md.get("chunk_index", "?")
    return f"{doc}::{idx}"


def _document_key(chunk: dict) -> str:
    md = chunk.get("metadata") or {}
    return str(md.get("document_id", "?"))


def hit_rate_at_k(
    retrieved: list[dict], expected_ids: list[str], *, mode: str, k: int | None = None
) -> float:
    """1.0 if any of the top-k retrieved chunks match an expected ID,
    else 0.0.

    ``mode`` is "chunk" or "document" - selects which identity key
    to compare on.
    """
    if not expected_ids:
        return 0.0
    key_fn = _chunk_key if mode == "chunk" else _document_key
    expected_set = set(expected_ids)
    top = retrieved if k is None else retrieved[:k]
    for hit in top:
        if key_fn(hit) in expected_set:
            return 1.0
    return 0.0


def mrr(retrieved: list[dict], expected_ids: list[str], *, mode: str) -> float:
    """Mean reciprocal rank of the FIRST retrieved chunk that matches
    any expected ID. 0.0 if none of the expected IDs are retrieved.

    Single-query MRR = 1/rank_of_first_hit. Aggregated across items
    the runner takes the arithmetic mean, matching the classical
    definition.
    """
    if not expected_ids:
        return 0.0
    key_fn = _chunk_key if mode == "chunk" else _document_key
    expected_set = set(expected_ids)
    for rank, hit in enumerate(retrieved, start=1):
        if key_fn(hit) in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: list[dict], expected_ids: list[str], *, mode: str, k: int | None = None
) -> float:
    """Normalized Discounted Cumulative Gain at k, binary relevance.

    We don't have graded relevance judgments (no "chunk A is more
    relevant than chunk B" labels), so gain is 1.0 for chunks in the
    expected set and 0.0 otherwise. The discount is log2(rank+1)
    starting at rank 1 (so the top hit gets full credit).

    Returned in [0, 1]. NaN-safe: returns 0.0 when no expected IDs are
    provided or when the ideal DCG is 0.
    """
    if not expected_ids:
        return 0.0
    key_fn = _chunk_key if mode == "chunk" else _document_key
    expected_set = set(expected_ids)
    top = retrieved if k is None else retrieved[:k]
    # Actual DCG
    dcg = 0.0
    for rank, hit in enumerate(top, start=1):
        if key_fn(hit) in expected_set:
            dcg += 1.0 / math.log2(rank + 1)
    # Ideal DCG: all expected hits packed at the top (up to k or the
    # count of expected IDs, whichever is smaller).
    ideal_hits = min(len(expected_set), len(top))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(
    retrieved: list[dict], expected_ids: list[str], *, mode: str, k: int | None = None
) -> float:
    """Fraction of the top-k retrieved chunks that match an expected
    ID. 0.0 if no expected IDs are provided or the top-k slice is
    empty.
    """
    if not expected_ids:
        return 0.0
    key_fn = _chunk_key if mode == "chunk" else _document_key
    expected_set = set(expected_ids)
    top = retrieved if k is None else retrieved[:k]
    if not top:
        return 0.0
    matches = sum(1 for hit in top if key_fn(hit) in expected_set)
    return matches / len(top)


def recall_at_k(
    retrieved: list[dict], expected_ids: list[str], *, mode: str, k: int | None = None
) -> float:
    """Fraction of the expected IDs that appear somewhere in the
    top-k retrieved chunks. 0.0 if no expected IDs are provided.
    """
    if not expected_ids:
        return 0.0
    key_fn = _chunk_key if mode == "chunk" else _document_key
    expected_set = set(expected_ids)
    top = retrieved if k is None else retrieved[:k]
    found = {key_fn(hit) for hit in top} & expected_set
    return len(found) / len(expected_set)


# ---------- LLM-as-judge metrics ----------


def faithfulness(
    question: str,
    chunks: list[dict],
    answer: str,
    *,
    model: str = _DEFAULT_MODEL,
) -> float | None:
    """Score in [0, 1]: how well every claim in the answer is
    supported by the retrieved chunks. High = grounded, low =
    hallucinating.

    Returns ``None`` if the judge call fails - the aggregate then
    skips this item for this metric rather than pulling the mean
    down with a fake zero.
    """
    if not chunks or not (answer or "").strip():
        return None
    prompt = _build_judge_prompt(
        role=(
            "You are grading how well an AI assistant's answer is "
            "supported by the retrieved reference passages. Return a "
            "single JSON object with fields \"score\" (a float in "
            "[0, 1]) and \"reason\" (one short sentence). Score 1.0 "
            "means every claim in the answer is directly supported by "
            "the passages; 0.0 means the answer is entirely "
            "unsupported or contradicts the passages. Ignore "
            "grammatical style; grade only factual support."
        ),
        question=question,
        chunks=chunks,
        answer=answer,
    )
    return _call_judge(prompt, model=model, label="faithfulness")


def answer_relevance(
    question: str,
    answer: str,
    *,
    model: str = _DEFAULT_MODEL,
) -> float | None:
    """Score in [0, 1]: does the answer actually address the question
    that was asked? Doesn't look at retrieved chunks - purely a
    question<->answer alignment score.
    """
    if not (answer or "").strip():
        return None
    prompt = _build_judge_prompt(
        role=(
            "You are grading whether an AI assistant's answer "
            "addresses the user's question. Return a single JSON "
            "object with fields \"score\" (a float in [0, 1]) and "
            "\"reason\" (one short sentence). Score 1.0 means the "
            "answer directly addresses the question and stays on "
            "topic; 0.0 means the answer is off-topic, evasive, or "
            "answers a different question. Do NOT judge factual "
            "correctness - only topical alignment."
        ),
        question=question,
        chunks=None,
        answer=answer,
    )
    return _call_judge(prompt, model=model, label="answer_relevance")


def context_relevance(
    question: str,
    chunks: list[dict],
    *,
    model: str = _DEFAULT_MODEL,
) -> float | None:
    """Score in [0, 1]: are the retrieved chunks the right ones for
    this question? Retrieval-quality signal that doesn't need
    ground-truth IDs - the judge just reads the passages and says
    whether they look relevant.

    This is the LLM-judge counterpart to hit_rate/MRR: same
    intent (did retrieval succeed?), different assumption (no labels
    needed).
    """
    if not chunks:
        return None
    prompt = _build_judge_prompt(
        role=(
            "You are grading whether the retrieved reference "
            "passages are relevant to the user's question. Return a "
            "single JSON object with fields \"score\" (a float in "
            "[0, 1]) and \"reason\" (one short sentence). Score 1.0 "
            "means most passages clearly address the question; 0.0 "
            "means the passages are unrelated. Ignore whether they "
            "fully answer the question - only judge topical relevance."
        ),
        question=question,
        chunks=chunks,
        answer=None,
    )
    return _call_judge(prompt, model=model, label="context_relevance")


# ---------- Internals ----------


def _build_judge_prompt(
    role: str,
    question: str,
    chunks: list[dict] | None,
    answer: str | None,
) -> str:
    """Assemble a small structured prompt for a judge call. All three
    LLM-judge metrics share this shape so the parsing / retry logic
    can be identical."""
    parts: list[str] = [role, "", f"QUESTION:\n{question.strip()}"]
    if chunks:
        parts.append("")
        parts.append("REFERENCE PASSAGES:")
        for i, chunk in enumerate(chunks, start=1):
            content = (chunk.get("content") or "").strip()
            if len(content) > _MAX_CHUNK_PREVIEW_CHARS:
                content = content[:_MAX_CHUNK_PREVIEW_CHARS] + "…"
            parts.append(f"[{i}] {content}")
    if answer:
        answer_stripped = answer.strip()
        if len(answer_stripped) > _MAX_ANSWER_CHARS:
            answer_stripped = answer_stripped[:_MAX_ANSWER_CHARS] + "…"
        parts.append("")
        parts.append(f"ANSWER:\n{answer_stripped}")
    parts.append("")
    parts.append(
        'Reply with ONLY a JSON object of the form '
        '{"score": <float in [0,1]>, "reason": "<one short sentence>"}. '
        "No prose, no markdown fences."
    )
    return "\n".join(parts)


def _call_judge(prompt: str, *, model: str, label: str) -> float | None:
    """One Ollama JSON call, parse out ``score``. Returns None on any
    failure (network, timeout, unparseable JSON, out-of-range score)
    so the caller can distinguish "we don't know" from a literal 0.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 128},
    }
    try:
        resp = requests.post(_OLLAMA_URL, json=payload, timeout=_JUDGE_TIMEOUT_S)
        resp.raise_for_status()
        body = resp.json()
    except Exception:  # noqa: BLE001
        logger.exception("Judge call failed for metric=%s", label)
        return None

    raw = (body.get("response") or "").strip()
    raw = _strip_code_fence(raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Rare fallback - the model sometimes emits extra prose despite
        # format:"json". Grab the first {...} span and try again.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning(
                "Judge for metric=%s returned unparseable output: %r", label, raw[:200]
            )
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning(
                "Judge for metric=%s returned unparseable output: %r", label, raw[:200]
            )
            return None

    score = obj.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        logger.warning("Judge for metric=%s returned non-numeric score: %r", label, score)
        return None
    if not math.isfinite(score):
        return None
    # Clamp defensively - some judge runs return 1.2 or -0.1 despite
    # the [0,1] instruction, and it's better to clip than to skip.
    return max(0.0, min(1.0, score))


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` fences a judge sometimes emits despite
    format:"json". Same helper shape as query_transform / crag."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()

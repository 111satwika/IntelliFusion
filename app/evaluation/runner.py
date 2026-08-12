"""
Evaluation runner - executes a dataset through the RAG pipeline and
computes metrics.

Runs the EXACT same code paths the UI and CLI use (via
``cli._prepare_context_and_images`` for retrieval+CRAG+prompt, then
``generate_answer`` for the answer) so a metric change here always
reflects what real users experience - no parallel implementation
drift.

Public API:
  * ``EvalConfig`` - what to run and how.
  * ``ItemResult`` - per-item outcome + metrics.
  * ``EvalReport`` - the whole run: config + items + aggregates.
  * ``run_evaluation(items, config)`` - synchronous batch runner.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.evaluation.dataset import EvalItem
from app.evaluation.metrics import (
    answer_relevance,
    context_relevance,
    faithfulness,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.generation.llm_generator import generate_answer
from app.retrieval.crag import set_enabled as _set_crag_enabled
from app.retrieval.query_transform import set_enabled as _set_query_transform_enabled

# _prepare_context_and_images bundles retrieval + CRAG + prompt-build
# so we pick up any future changes (e.g. new re-ranking stage) for
# free. Importing an underscore-prefixed function from cli.py is a
# small deliberate coupling - the alternative is duplicating that
# 60-line function and letting it drift.
from cli import _prepare_context_and_images

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Knobs for one evaluation run."""

    default_kb: str | None = None
    top_k: int = 5
    # Feature-flag switches - applied process-wide for the duration
    # of the run, restored to the previous value on exit so an eval
    # run doesn't silently change what the UI does next.
    use_crag: bool = False
    use_query_transform: bool = True
    # Which LLM-judge metrics to compute. Each one is an Ollama round
    # trip - on CPU this is the dominant cost, so allowing users to
    # opt out of the slower ones lets them iterate faster.
    judge_faithfulness: bool = True
    judge_answer_relevance: bool = True
    judge_context_relevance: bool = True


@dataclass
class ItemResult:
    """Per-question outcome and metrics."""

    item: EvalItem
    kb_used: str | None
    chunks: list[dict]
    answer: str
    # Latency (seconds). context_prep covers retrieve + CRAG +
    # prompt-build; generation is the LLM call that produced the
    # answer. total is wall-clock for the whole item so overhead
    # (metric calls) shows up as (total - context_prep - generation).
    latency_context_prep_s: float
    latency_generation_s: float
    latency_total_s: float
    # Retrieval metrics - None when no ground truth was provided.
    doc_hit_rate: float | None = None
    doc_mrr: float | None = None
    doc_ndcg: float | None = None
    doc_precision: float | None = None
    doc_recall: float | None = None
    chunk_hit_rate: float | None = None
    chunk_mrr: float | None = None
    chunk_ndcg: float | None = None
    chunk_precision: float | None = None
    chunk_recall: float | None = None
    # LLM-judge metrics - None when disabled or the judge failed.
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_relevance: float | None = None
    # Populated only when the item errored out end-to-end. Preserved
    # in the report so the reader can see what failed instead of
    # having a mysteriously-missing question.
    error: str | None = None

    def metrics_row(self) -> dict:
        """Dict of just the numeric metrics + kb - suitable for a
        Streamlit dataframe or CSV row."""
        return {
            "question": self.item.question[:80],
            "kb": self.kb_used or "-",
            "chunks": len(self.chunks),
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_relevance": self.context_relevance,
            "doc_hit_rate": self.doc_hit_rate,
            "doc_mrr": self.doc_mrr,
            "doc_ndcg": self.doc_ndcg,
            "doc_precision": self.doc_precision,
            "doc_recall": self.doc_recall,
            "chunk_hit_rate": self.chunk_hit_rate,
            "chunk_mrr": self.chunk_mrr,
            "chunk_ndcg": self.chunk_ndcg,
            "chunk_precision": self.chunk_precision,
            "chunk_recall": self.chunk_recall,
            "latency_total_s": round(self.latency_total_s, 2),
        }


@dataclass
class EvalReport:
    """Full evaluation run report."""

    config: EvalConfig
    started_at: str
    ended_at: str
    items: list[ItemResult] = field(default_factory=list)

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def failed_items(self) -> int:
        return sum(1 for r in self.items if r.error is not None)

    def aggregate(self) -> dict:
        """Compute mean of each metric across items where the metric
        was applicable. Missing (None) values are IGNORED, not
        treated as zero, so a partial run doesn't misleadingly drag
        the average down.
        """
        agg: dict[str, float | None] = {}
        metric_names = (
            "faithfulness", "answer_relevance", "context_relevance",
            "doc_hit_rate", "doc_mrr", "doc_ndcg", "doc_precision", "doc_recall",
            "chunk_hit_rate", "chunk_mrr", "chunk_ndcg", "chunk_precision", "chunk_recall",
            "latency_total_s", "latency_context_prep_s", "latency_generation_s",
        )
        for name in metric_names:
            values = [getattr(r, name) for r in self.items if getattr(r, name) is not None]
            agg[name] = round(statistics.fmean(values), 3) if values else None
        agg["items_with_faithfulness"] = sum(1 for r in self.items if r.faithfulness is not None)
        agg["items_with_retrieval_gt"] = sum(1 for r in self.items if r.item.has_retrieval_ground_truth)
        return agg


# ---------- Runner ----------


def run_evaluation(
    items: list[EvalItem],
    config: EvalConfig,
    *,
    owner: str = "local",
    progress_callback=None,
) -> EvalReport:
    """Execute the dataset. Returns a fully-populated ``EvalReport``.

    ``owner``: which account's ingested data to evaluate against
    (defaults to "local", i.e. auth disabled) - evaluation stays a
    single-operator workflow (see module docstring), it just needs to
    know WHICH operator's data to search now that documents are
    owner-scoped, so results don't come back empty for an account
    other than "local".

    ``progress_callback(index, total, item)`` is invoked before each
    item runs - the Streamlit UI uses this to update a progress bar.
    Safe to omit for CLI runs.

    Feature-flag state is snapshotted before the run and restored
    after, so an eval doesn't leak toggle state into the running UI
    process.

    app.generation.semantic_cache is force-DISABLED for the whole run
    (not configurable via EvalConfig, unlike CRAG/query-transform) -
    a cache hit would silently skip real generation for whatever
    question happened to look like a near-duplicate of one already
    asked through the live chat UI, defeating the entire point of
    measuring actual generation quality. Also snapshotted/restored so
    a run doesn't leave the cache disabled for the live UI afterward.
    """
    started_at = datetime.now(timezone.utc).isoformat()

    # Snapshot + set global feature flags for the duration of the run.
    # We reach into module state directly (via is_enabled/set_enabled)
    # so the toggle behavior matches exactly what the UI checkboxes
    # do.
    from app.generation import semantic_cache as _semantic_cache
    from app.retrieval.crag import is_enabled as _crag_is_enabled
    from app.retrieval.query_transform import is_enabled as _qt_is_enabled
    prev_crag = _crag_is_enabled()
    prev_qt = _qt_is_enabled()
    prev_semantic_cache = _semantic_cache.is_enabled()
    _set_crag_enabled(config.use_crag)
    _set_query_transform_enabled(config.use_query_transform)
    _semantic_cache.set_enabled(False)

    results: list[ItemResult] = []
    try:
        for idx, item in enumerate(items):
            if progress_callback is not None:
                try:
                    progress_callback(idx, len(items), item)
                except Exception:  # noqa: BLE001 - UI callback never breaks the run
                    logger.exception("progress_callback failed; continuing run.")
            results.append(_run_one(item, config, owner))
    finally:
        _set_crag_enabled(prev_crag)
        _set_query_transform_enabled(prev_qt)
        _semantic_cache.set_enabled(prev_semantic_cache)

    ended_at = datetime.now(timezone.utc).isoformat()
    return EvalReport(
        config=config,
        started_at=started_at,
        ended_at=ended_at,
        items=results,
    )


def _run_one(item: EvalItem, config: EvalConfig, owner: str) -> ItemResult:
    """Execute one item end-to-end. Errors are captured on the
    result rather than raised, so one broken question can't abort a
    50-item batch."""
    kb_used = item.kb or config.default_kb
    t_start = time.perf_counter()

    try:
        t_ctx0 = time.perf_counter()
        # 6th/7th return values (contextualization, from conversation
        # memory - see app.retrieval.contextualize; and cached_answer,
        # from app.generation.semantic_cache) are intentionally
        # discarded: eval items are single questions with no
        # conversation thread, so history is never passed and
        # contextualization is always a no-op trivial result, and a
        # cached answer would defeat the whole point of evaluating
        # actual generation quality.
        prompt, _display_images, _vision_urls, chunks, _crag_result, _contextualization, _cached_answer, _audio_clip_hits = (
            _prepare_context_and_images(
                item.question,
                top_k=config.top_k,
                use_vision=False,
                repository=None,
                kb=kb_used,
                owner=owner,
            )
        )
        latency_context_prep = time.perf_counter() - t_ctx0

        t_gen0 = time.perf_counter()
        answer = generate_answer(prompt) or ""
        latency_generation = time.perf_counter() - t_gen0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Evaluation item failed: %r", item.question)
        return ItemResult(
            item=item,
            kb_used=kb_used,
            chunks=[],
            answer="",
            latency_context_prep_s=0.0,
            latency_generation_s=0.0,
            latency_total_s=time.perf_counter() - t_start,
            error=f"{type(exc).__name__}: {exc}",
        )

    result = ItemResult(
        item=item,
        kb_used=kb_used,
        chunks=chunks,
        answer=answer,
        latency_context_prep_s=latency_context_prep,
        latency_generation_s=latency_generation,
        latency_total_s=time.perf_counter() - t_start,
    )

    # Retrieval metrics - only when the item provided ground truth.
    # Chunk-level and document-level are independent (an item can
    # have one, the other, or both).
    if item.expected_chunk_ids:
        result.chunk_hit_rate = hit_rate_at_k(
            chunks, item.expected_chunk_ids, mode="chunk", k=config.top_k
        )
        result.chunk_mrr = mrr(chunks, item.expected_chunk_ids, mode="chunk")
        result.chunk_ndcg = ndcg_at_k(
            chunks, item.expected_chunk_ids, mode="chunk", k=config.top_k
        )
        result.chunk_precision = precision_at_k(
            chunks, item.expected_chunk_ids, mode="chunk", k=config.top_k
        )
        result.chunk_recall = recall_at_k(
            chunks, item.expected_chunk_ids, mode="chunk", k=config.top_k
        )
    if item.expected_document_ids:
        result.doc_hit_rate = hit_rate_at_k(
            chunks, item.expected_document_ids, mode="document", k=config.top_k
        )
        result.doc_mrr = mrr(chunks, item.expected_document_ids, mode="document")
        result.doc_ndcg = ndcg_at_k(
            chunks, item.expected_document_ids, mode="document", k=config.top_k
        )
        result.doc_precision = precision_at_k(
            chunks, item.expected_document_ids, mode="document", k=config.top_k
        )
        result.doc_recall = recall_at_k(
            chunks, item.expected_document_ids, mode="document", k=config.top_k
        )

    # LLM-judge metrics - one Ollama call each. Order matters only for
    # perceived latency in UI progress; results are independent.
    if config.judge_faithfulness:
        result.faithfulness = faithfulness(item.question, chunks, answer)
    if config.judge_answer_relevance:
        result.answer_relevance = answer_relevance(item.question, answer)
    if config.judge_context_relevance:
        result.context_relevance = context_relevance(item.question, chunks)

    # Refresh total-latency so it accounts for the judge round-trips
    # too - otherwise "total_s" would look artificially fast.
    result.latency_total_s = time.perf_counter() - t_start
    return result

"""
Evaluation report writer.

Writes each run to two files under a caller-chosen directory:

  * ``<timestamp>_<name>.json`` - full raw report with per-item
    results (chunks, answer, all metrics, latency). Machine-readable;
    consumed by any downstream analysis / A/B comparison.
  * ``<timestamp>_<name>.md`` - human-friendly Markdown summary with
    the aggregate table + per-item drill-down. Meant to be pasted
    into a PR or opened in the VS Code Markdown preview.

Both files are keyed by the SAME timestamp+name so it's trivial to
find the pair for one run.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.runner import EvalReport, ItemResult

logger = logging.getLogger(__name__)


def write_report(
    report: EvalReport,
    out_dir: str | Path,
    *,
    name: str = "eval",
) -> tuple[Path, Path]:
    """Persist an EvalReport as (JSON, Markdown). Returns both paths.

    ``name`` gets slugified into the filename so users can identify
    runs at a glance (``2026-08-03_141507_web_crag_on.md``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    slug = _slugify(name)
    stem = f"{stamp}_{slug}"

    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    json_path.write_text(_report_to_json(report), encoding="utf-8")
    md_path.write_text(_report_to_markdown(report), encoding="utf-8")

    logger.info("Wrote evaluation report: %s / %s", md_path.name, json_path.name)
    return md_path, json_path


# ---------- Internals ----------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Lowercase / dash-separated. Kept dumb-simple - filesystem
    names, not URLs."""
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug or "eval"


def _report_to_json(report: EvalReport) -> str:
    """Full raw dump. Chunks get flattened to (id, content_preview)
    tuples so the file stays reviewable in a text editor - a report
    of 20 items with full 3000-char chunks would be megabytes."""
    payload = {
        "config": asdict(report.config),
        "started_at": report.started_at,
        "ended_at": report.ended_at,
        "total_items": report.total_items,
        "failed_items": report.failed_items,
        "aggregate": report.aggregate(),
        "items": [_item_to_dict(r) for r in report.items],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _item_to_dict(result: ItemResult) -> dict:
    return {
        "question": result.item.question,
        "kb_used": result.kb_used,
        "notes": result.item.notes,
        "expected_document_ids": result.item.expected_document_ids,
        "expected_chunk_ids": result.item.expected_chunk_ids,
        "expected_answer": result.item.expected_answer,
        "answer": result.answer,
        "chunk_count": len(result.chunks),
        "retrieved_chunk_keys": [
            _chunk_signature(c) for c in result.chunks
        ],
        "metrics": {
            "faithfulness": result.faithfulness,
            "answer_relevance": result.answer_relevance,
            "context_relevance": result.context_relevance,
            "doc_hit_rate": result.doc_hit_rate,
            "doc_mrr": result.doc_mrr,
            "doc_ndcg": result.doc_ndcg,
            "doc_precision": result.doc_precision,
            "doc_recall": result.doc_recall,
            "chunk_hit_rate": result.chunk_hit_rate,
            "chunk_mrr": result.chunk_mrr,
            "chunk_ndcg": result.chunk_ndcg,
            "chunk_precision": result.chunk_precision,
            "chunk_recall": result.chunk_recall,
        },
        "latency_s": {
            "context_prep": round(result.latency_context_prep_s, 3),
            "generation": round(result.latency_generation_s, 3),
            "total": round(result.latency_total_s, 3),
        },
        "error": result.error,
    }


def _chunk_signature(chunk: dict) -> str:
    md = chunk.get("metadata") or {}
    doc = md.get("document_id", "?")
    idx = md.get("chunk_index", "?")
    return f"{doc}::{idx}"


def _report_to_markdown(report: EvalReport) -> str:
    """Human-readable summary. Sections:
      1. Run metadata (config, timing).
      2. Aggregate table (one row per metric with mean).
      3. Per-item results (question, metrics, timing, error if any).
    """
    lines: list[str] = []
    lines.append(f"# Evaluation report")
    lines.append("")
    lines.append(f"- **Started**: {report.started_at}")
    lines.append(f"- **Ended**:   {report.ended_at}")
    lines.append(f"- **Items**:   {report.total_items} total, {report.failed_items} failed")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for k, v in asdict(report.config).items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Mean |")
    lines.append("| --- | --- |")
    for name, value in report.aggregate().items():
        lines.append(f"| `{name}` | {_fmt(value)} |")
    lines.append("")

    lines.append("## Per-item results")
    lines.append("")
    for i, r in enumerate(report.items, start=1):
        lines.append(f"### {i}. {r.item.question}")
        lines.append("")
        if r.error:
            lines.append(f"**ERROR**: `{r.error}`")
            lines.append("")
            continue
        lines.append(f"- **KB**: `{r.kb_used or '-'}`  ·  **Chunks retrieved**: {len(r.chunks)}")
        lines.append(
            f"- **Latency**: total {r.latency_total_s:.2f}s "
            f"(context {r.latency_context_prep_s:.2f}s, gen {r.latency_generation_s:.2f}s)"
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| faithfulness | {_fmt(r.faithfulness)} |")
        lines.append(f"| answer_relevance | {_fmt(r.answer_relevance)} |")
        lines.append(f"| context_relevance | {_fmt(r.context_relevance)} |")
        lines.append(f"| doc_hit_rate | {_fmt(r.doc_hit_rate)} |")
        lines.append(f"| doc_mrr | {_fmt(r.doc_mrr)} |")
        lines.append(f"| doc_ndcg | {_fmt(r.doc_ndcg)} |")
        lines.append(f"| doc_precision | {_fmt(r.doc_precision)} |")
        lines.append(f"| doc_recall | {_fmt(r.doc_recall)} |")
        lines.append(f"| chunk_hit_rate | {_fmt(r.chunk_hit_rate)} |")
        lines.append(f"| chunk_mrr | {_fmt(r.chunk_mrr)} |")
        lines.append(f"| chunk_ndcg | {_fmt(r.chunk_ndcg)} |")
        lines.append(f"| chunk_precision | {_fmt(r.chunk_precision)} |")
        lines.append(f"| chunk_recall | {_fmt(r.chunk_recall)} |")
        lines.append("")
        # Truncated answer preview - full text is in the JSON companion.
        preview = (r.answer or "").strip().replace("\n", " ")
        if len(preview) > 400:
            preview = preview[:400] + "…"
        lines.append(f"> {preview or '_(empty answer)_'}")
        lines.append("")

    return "\n".join(lines)


def _fmt(value) -> str:
    """Render a metric cell: numbers to 3 decimals, None -> em-dash."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)

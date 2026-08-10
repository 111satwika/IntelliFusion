"""
Evaluation API: list JSONL datasets, run a batch through the same
pipeline as Chat (as a background job - runs can take many minutes),
poll progress, and fetch the last finished report.

The "last report" is kept in a process-global variable (matching how
the CRAG/query-transform/Self-RAG toggles are already process-global -
see app/retrieval/crag.py etc.) rather than per-session: evaluation is
inherently a single-operator workflow, and reports are already written
to disk (data/eval/reports/) regardless.
"""

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.evaluation.dataset import load_dataset
from app.evaluation.report import write_report
from app.evaluation.runner import EvalConfig, run_evaluation

from jobs import start_job

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

_EVAL_DIR = Path("data/eval")
_EVAL_REPORTS_DIR = _EVAL_DIR / "reports"

_last_report_serialized: dict | None = None


@router.get("/datasets")
def list_datasets():
    if not _EVAL_DIR.exists():
        return []
    return [str(p.relative_to(Path("."))) for p in sorted(_EVAL_DIR.glob("*.jsonl"))]


class RunRequest(BaseModel):
    dataset_path: str
    default_kb: str | None = "web"
    top_k: int = 5
    use_crag: bool = False
    use_query_transform: bool = True
    judge_faithfulness: bool = True
    judge_answer_relevance: bool = True
    judge_context_relevance: bool = True
    report_name: str = "eval"


def _serialize_report(report) -> dict:
    return {
        "aggregate": report.aggregate(),
        "items": [
            {
                **r.metrics_row(),
                "question": r.item.question,
                "kb_used": r.kb_used,
                "chunk_count": len(r.chunks),
                "latency_total_s": r.latency_total_s,
                "latency_context_prep_s": r.latency_context_prep_s,
                "latency_generation_s": r.latency_generation_s,
                "answer": r.answer,
                "expected_answer": r.item.expected_answer,
                "error": r.error,
            }
            for r in report.items
        ],
    }


def _run_evaluation_job(body: RunRequest, *, progress):
    global _last_report_serialized

    items = load_dataset(Path(body.dataset_path))
    config = EvalConfig(
        default_kb=body.default_kb if body.default_kb != "(auto)" else None,
        top_k=body.top_k,
        use_crag=body.use_crag,
        use_query_transform=body.use_query_transform,
        judge_faithfulness=body.judge_faithfulness,
        judge_answer_relevance=body.judge_answer_relevance,
        judge_context_relevance=body.judge_context_relevance,
    )

    def _progress_cb(idx, total, item):
        progress({"index": idx, "total": total, "question": item.question[:80]})

    report = run_evaluation(items, config, progress_callback=_progress_cb)
    md_path, json_path = write_report(report, _EVAL_REPORTS_DIR, name=body.report_name or "eval")

    serialized = _serialize_report(report)
    serialized["md_path"] = str(md_path)
    serialized["json_path"] = str(json_path)
    _last_report_serialized = serialized
    return serialized


@router.post("/run")
def run(body: RunRequest):
    job_id = start_job(_run_evaluation_job, body)
    return {"job_id": job_id}


@router.get("/last")
def last_report():
    return _last_report_serialized

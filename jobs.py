"""
Minimal in-memory background-job registry.

Ingestion (GitHub repo clone + embed, website crawl, PDF/DOCX/Markdown
parse) and evaluation runs are confirmed multi-minute, fully
synchronous, blocking calls (see app/ingestion/ingest.py,
app/evaluation/runner.py) - running them inline in an HTTP request
handler would hold the connection open for the whole duration. Instead
they run in a plain background thread; the client gets a job_id back
immediately and polls GET /api/jobs/{job_id} for status.

In-memory only (no persistence across server restarts) - acceptable
for a single-process local tool exactly like the pending-document
tracker and Chroma client already are.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Job:
    status: str = "running"  # "running" | "done" | "error"
    progress: dict = field(default_factory=dict)
    result: Any = None
    error: str | None = None


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def start_job(fn: Callable[..., Any], *args, **kwargs) -> str:
    """
    Run fn(*args, progress=progress_setter, **kwargs) in a background
    thread and return a job_id immediately. fn receives an extra
    keyword-only `progress` callable it can call with any JSON-able
    dict to update this job's reported progress.
    """
    job_id = uuid.uuid4().hex
    job = Job()
    with _lock:
        _jobs[job_id] = job

    def _set_progress(payload: dict) -> None:
        with _lock:
            job.progress = payload

    def _run() -> None:
        try:
            result = fn(*args, progress=_set_progress, **kwargs)
            with _lock:
                job.result = result
                job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surface every failure to the polling client
            logger.exception("Background job %s failed.", job_id)
            with _lock:
                job.error = str(exc)
                job.status = "error"

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)

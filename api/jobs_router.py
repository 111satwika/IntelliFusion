"""Shared background-job polling endpoint, used by both ingestion (api/sources.py) and evaluation (api/evaluation.py) jobs."""

from fastapi import APIRouter

from jobs import get_job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        return {"status": "not_found"}
    return {"status": job.status, "progress": job.progress, "result": job.result, "error": job.error}

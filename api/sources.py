"""
Sources API: list what's ingested per knowledge base, ingest new
documents/URLs/repos (as background jobs - see jobs.py, these can take
minutes), and save/delete pending or indexed documents.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile
from pydantic import BaseModel

from app.chunking.parent_child import chunk_with_parent_child
from app.embeddings.embedder import embed_chunks
from app.ingestion.ingest import (
    _ingest_audio_clips,
    _ingest_images,
    _ingest_video_frames,
    ingest_github_activity,
    ingest_github_repo,
)
from app.ingestion.loader import (
    crawl_website,
    load_audio_document,
    load_docx_document,
    load_markdown_document,
    load_pdf_document,
    load_video_document,
    load_website_document,
)
from app.vectorstore.store import (
    add_embedded_chunks,
    count,
    delete_document,
    delete_repository,
    list_documents,
    list_repositories,
)

from api.deps import get_session_id
from jobs import start_job
from session_store import KB_LABELS, add_pending, get_pending, remove_pending

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sources", tags=["sources"])

_LOADERS = {
    "markdown": load_markdown_document,
    "pdf": load_pdf_document,
    "docx": load_docx_document,
    "audio": load_audio_document,
    "video": load_video_document,
}


def _ingest_documents(documents, document_id_override: str | None = None) -> tuple[int, list[str]]:
    """Chunk, embed, and store a list of Documents. See ui/ingestion.py (Streamlit build) for the original of this logic."""
    total_chunks = 0
    document_ids: list[str] = []
    for document in documents:
        if document_id_override is not None:
            document.metadata["document_id"] = document_id_override
        chunks = chunk_with_parent_child(document)
        for chunk in chunks:
            if document_id_override is not None:
                chunk.metadata["document_id"] = document_id_override
        embedded = embed_chunks(chunks)
        add_embedded_chunks(embedded)
        total_chunks += len(chunks)
        doc_id = document.metadata.get("document_id")
        if doc_id and doc_id not in document_ids:
            document_ids.append(doc_id)
    return total_chunks, document_ids


@router.get("")
def get_summary():
    return {
        "total_sources": sum(len(list_documents(kb)) for kb in KB_LABELS),
        "total_chunks": sum(count(kb) for kb in KB_LABELS),
        "kbs": {kb: count(kb) for kb in KB_LABELS},
    }


@router.get("/{kb}")
def list_sources(kb: str, session_id: str = Depends(get_session_id)):
    indexed = {doc["id"]: doc["chunk_count"] for doc in list_documents(kb)}
    pending_ids = {
        (entry.get("repository") or entry.get("document_id"))
        for entry in get_pending(session_id, kb)
    }
    return [
        {"id": doc_id, "chunk_count": chunk_count, "status": "pending" if doc_id in pending_ids else "indexed"}
        for doc_id, chunk_count in sorted(indexed.items())
    ]


@router.get("/github/repositories")
def github_repositories():
    return list_repositories()


def _run_file_ingest(kb: str, filename: str, suffix: str, content: bytes, *, session_id: str, progress):
    progress({"message": f"Ingesting {filename}..."})
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        documents = _LOADERS[kb](tmp_path)
        documents = documents if isinstance(documents, list) else [documents]
        _ingest_documents(documents, document_id_override=filename)
        # Must happen before the temp file is deleted below - this is
        # the only place an uploaded audio/video file's bytes exist at
        # all (see app.ingestion.ingest._ingest_video_frames/
        # _ingest_audio_clips). document_id_override above already
        # updated each document's metadata in place, so document_id
        # here is the uploaded filename, not the temp path.
        if kb == "video":
            for document in documents:
                _ingest_video_frames(document, tmp_path)
        if kb in ("audio", "video"):
            for document in documents:
                _ingest_audio_clips(document, tmp_path)
        add_pending(session_id, kb, {"document_id": filename, "label": filename})
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"document_id": filename}


@router.post("/{kb}/upload")
async def upload_file(kb: str, file: UploadFile, session_id: str = Depends(get_session_id)):
    content = await file.read()
    suffix = Path(file.filename).suffix
    job_id = start_job(_run_file_ingest, kb, file.filename, suffix, content, session_id=session_id)
    return {"job_id": job_id}


class WebIngestRequest(BaseModel):
    url: str
    crawl: bool = False
    max_pages: int | None = None


def _run_web_ingest(url: str, crawl: bool, max_pages: int | None, *, session_id: str, progress):
    if crawl:
        progress({"message": f"Crawling {url}..."})
        effective_max_pages = max_pages if max_pages else 10_000
        documents = crawl_website(url, max_pages=effective_max_pages)
        if not documents:
            raise ValueError(
                f"Crawl of '{url}' returned 0 pages - the URL may be unreachable, out of scope, "
                "or the site returned no crawlable links."
            )
        _ingest_documents(documents)
        for doc in documents:
            _ingest_images(doc)
        # Crawled pages are auto-saved (not pending) - a whole-site
        # crawl is a "keep it all" workflow the user already opted
        # into by ticking crawl mode (see the Streamlit build's
        # ui_pages/sources.py for the same reasoning).
        return {"pages_ingested": len(documents), "saved": True}
    progress({"message": f"Fetching {url}..."})
    document = load_website_document(url)
    if not document.content.strip():
        raise ValueError(
            f"Fetched '{url}' but got 0 characters of extractable text. This usually means the "
            "site is JavaScript-rendered - try enabling crawl mode."
        )
    _ingest_documents([document])
    _ingest_images(document)
    add_pending(session_id, "web", {"document_id": url, "label": url})
    return {"pages_ingested": 1, "saved": False, "document_id": url}


@router.post("/web/ingest")
def ingest_web(body: WebIngestRequest, session_id: str = Depends(get_session_id)):
    job_id = start_job(_run_web_ingest, body.url, body.crawl, body.max_pages, session_id=session_id)
    return {"job_id": job_id}


class GitHubIngestRequest(BaseModel):
    repo_url: str
    branch: str | None = None


def _run_github_ingest(repo_url: str, branch: str | None, *, session_id: str, progress):
    progress({"message": f"Ingesting {repo_url}..."})
    ingest_github_repo(repo_url, branch=branch)
    parsed = repo_url.rstrip("/").removesuffix(".git")
    parts = parsed.split("/")
    owner_repo = "/".join(parts[-2:]) if len(parts) >= 2 else parsed
    add_pending(session_id, "github", {"repository": owner_repo, "label": owner_repo})
    return {"repository": owner_repo}


@router.post("/github/ingest")
def ingest_github(body: GitHubIngestRequest, session_id: str = Depends(get_session_id)):
    job_id = start_job(_run_github_ingest, body.repo_url, body.branch, session_id=session_id)
    return {"job_id": job_id}


class GitHubActivityIngestRequest(BaseModel):
    repo_url: str
    include_issues: bool = True
    include_prs: bool = True
    include_discussions: bool = True
    max_items: int = 200


def _run_github_activity_ingest(
    repo_url: str,
    include_issues: bool,
    include_prs: bool,
    include_discussions: bool,
    max_items: int,
    *,
    session_id: str,
    progress,
):
    progress({"message": f"Ingesting activity for {repo_url}..."})
    result = ingest_github_activity(
        repo_url,
        include_issues=include_issues,
        include_prs=include_prs,
        include_discussions=include_discussions,
        max_items=max_items,
    )
    add_pending(session_id, "github", {"repository": result["repository"], "label": result["repository"]})
    return result


@router.post("/github/activity")
def ingest_github_activity_route(
    body: GitHubActivityIngestRequest, session_id: str = Depends(get_session_id)
):
    job_id = start_job(
        _run_github_activity_ingest,
        body.repo_url, body.include_issues, body.include_prs, body.include_discussions, body.max_items,
        session_id=session_id,
    )
    return {"job_id": job_id}


class SaveRequest(BaseModel):
    doc_id: str


@router.post("/{kb}/save")
def save_source(kb: str, body: SaveRequest, session_id: str = Depends(get_session_id)):
    entry_key = "repository" if kb == "github" else "document_id"
    remove_pending(session_id, kb, entry_key, body.doc_id)
    return {"ok": True}


@router.delete("/{kb}")
def delete_source(kb: str, doc_id: str, session_id: str = Depends(get_session_id)):
    if kb == "github":
        delete_repository(doc_id)
        remove_pending(session_id, kb, "repository", doc_id)
    else:
        delete_document(doc_id, kb=kb)
        remove_pending(session_id, kb, "document_id", doc_id)
    return {"ok": True}

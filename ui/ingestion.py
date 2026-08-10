"""Ingestion helpers (reuse the existing pipeline modules)."""

import tempfile
from pathlib import Path

from app.chunking.parent_child import chunk_with_parent_child
from app.embeddings.embedder import embed_chunks
from app.vectorstore.store import add_embedded_chunks

from ui.state import add_pending


def ingest_documents(documents, document_id_override: str | None = None) -> tuple[int, list[str]]:
    """
    Chunk, embed, and store a list of Documents. If document_id_override
    is given, force every chunk's document_id metadata to that string
    BEFORE storing (so file uploads land under their original filename
    rather than the temp path the loader saw). Returns (total_chunks,
    unique_document_ids_stored).
    """
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


def ingest_uploaded_files(uploaded_files, loader, kb: str) -> None:
    """
    Ingest each uploaded file and record it as pending in the tracker.
    document_id is forced to the ORIGINAL uploaded filename (not the
    temp path) so the same file re-uploaded upserts instead of piling
    up duplicates, and so the Pending list shows the real filename.
    """
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            result = loader(tmp_path)
            documents = result if isinstance(result, list) else [result]
            ingest_documents(documents, document_id_override=uploaded_file.name)
            add_pending(kb, {"document_id": uploaded_file.name, "label": uploaded_file.name})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

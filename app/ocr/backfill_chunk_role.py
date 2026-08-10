"""
One-off maintenance script: add the missing chunk_role="parent" field
to already-stored OCR/vision-extracted image-text chunks.

Why this exists: app.ingestion.ingest._ingest_images and
app.ocr.finish_image_ingest previously stored these chunks without a
chunk_role field at all (they bypass app.chunking.parent_child, which
is the only code that sets chunk_role). app.retrieval.hybrid_retriever
hard-filters both its dense search and its BM25 index to
chunk_role="parent" for the "web"/"github" KBs - so every chunk stored
before this fix is silently invisible to retrieval, even though it's
sitting in the vector store. Both storage sites now set chunk_role
correctly for new ingests; this script repairs chunks stored before
that fix, in place, with no re-embedding (metadata-only patch, so it's
fast even over a large corpus).

Usage: python -m app.ocr.backfill_chunk_role
"""

import logging

from app.vectorstore.store import _get_collection

logger = logging.getLogger(__name__)

# Only these two content_types are ever produced by the OCR/vision
# extraction path this script repairs - scoping to them (rather than
# "any chunk missing chunk_role") means this can never touch a chunk
# for an unrelated reason.
_IMAGE_CONTENT_TYPES = {"image_ocr", "image_code"}

# The only KBs app.retrieval.hybrid_retriever applies the
# chunk_role="parent" filter to (app.retrieval.hybrid_retriever._PARENT_CHILD_KBS).
_AFFECTED_KBS = ["web", "github"]


def backfill_chunk_role(kb: str) -> int:
    """Patch every image_ocr/image_code chunk in `kb` that's missing
    chunk_role, setting it to "parent". Returns how many were patched."""
    collection = _get_collection(kb)
    result = collection.get(include=["metadatas"])
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []

    patch_ids: list[str] = []
    patch_metadatas: list[dict] = []
    for chunk_id, metadata in zip(ids, metadatas):
        if metadata.get("content_type") not in _IMAGE_CONTENT_TYPES:
            continue
        if metadata.get("chunk_role"):
            continue  # already correct (stored after the fix)
        updated = dict(metadata)
        updated["chunk_role"] = "parent"
        patch_ids.append(chunk_id)
        patch_metadatas.append(updated)

    if patch_ids:
        collection.update(ids=patch_ids, metadatas=patch_metadatas)
    logger.info("KB '%s': patched %d chunk(s) with missing chunk_role", kb, len(patch_ids))
    return len(patch_ids)


if __name__ == "__main__":
    from app.logging_config import configure_logging

    configure_logging()
    total = 0
    for kb_name in _AFFECTED_KBS:
        total += backfill_chunk_role(kb_name)
    print(f"Done. Patched {total} chunk(s) total across {_AFFECTED_KBS}.")

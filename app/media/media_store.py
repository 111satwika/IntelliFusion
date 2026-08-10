"""
Servable local media files (video frame thumbnails, full audio/video
source copies) for the native acoustic/visual similarity search
feature (see app.embeddings.image_embedder / app.embeddings.audio_embedder).

Responsibility: save a file to a predictable, safe location under
data/media/ and return the URL path server.py's StaticFiles mount will
serve it from. No embedding, chunking, or retrieval happens here - this
module only persists bytes and returns a URL.

Why this exists: website images have a real external URL a browser can
fetch directly (webapp/app.js renders `<img src="{image_url}">` as-is,
and api/chat.py passes that string straight through - see
app.embeddings.image_embedder's module docstring). A sampled video
frame, decoded via OpenCV from a local file, has no such URL - nothing
serves it. Same problem for audio clip playback: a CLAP-embedded 10s
window has no standalone file to point an <audio> element at. This
module is the minimal fix: persist what's needed to disk once, at
ingest time, and hand back a URL server.py's new `/media` StaticFiles
mount can serve.

Design:
- Filenames are NEVER derived from user-controlled strings (original
  upload filename, video title, etc.) - always a hash of document_id
  plus a numeric suffix. This closes path traversal at the
  filename-generation layer, independent of StaticFiles' own (already
  audited) path-resolution protection - defense in depth, not
  redundant.
- save_media_copy() is idempotent (skips the copy if the target
  already exists) so re-ingesting an unchanged file doesn't re-copy a
  potentially large video every time - matches this codebase's
  existing "re-ingestion is safe/idempotent" convention (see
  app.vectorstore.store's upsert-by-stable-id pattern).
- `ext` is checked against an explicit whitelist before any copy is
  made, so an unexpected extension is rejected rather than silently
  landing in the publicly-served tree.
"""

import hashlib
import logging
import shutil
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

MEDIA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "media"
FRAMES_DIR = MEDIA_DIR / "frames"
SOURCES_DIR = MEDIA_DIR / "sources"

# Same extensions app.ingestion.ingest._LOADERS_BY_EXTENSION recognizes
# for audio/video - a source file save is only ever attempted for a
# file that already made it through ingestion, so this whitelist is a
# defensive check, not the primary gate.
_ALLOWED_SOURCE_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
}


def _document_hash(document_id: str) -> str:
    """Stable, filesystem-safe, non-user-controlled filename component."""
    return hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:16]


def save_frame_thumbnail(document_id: str, timestamp_seconds: float, image: Image.Image) -> str:
    """
    Save one sampled video frame as a small JPEG thumbnail, return the
    URL path it will be servable at (via server.py's `/media` mount).

    Filename is deterministic (same document_id + timestamp always
    produces the same path), so re-ingesting the same video overwrites
    the same thumbnails instead of accumulating stale duplicates.
    """
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_ms = int(timestamp_seconds * 1000)
    filename = f"{_document_hash(document_id)}_{timestamp_ms}.jpg"
    path = FRAMES_DIR / filename
    image.convert("RGB").save(path, format="JPEG", quality=85)
    return f"/media/frames/{filename}"


def save_media_copy(document_id: str, file_path: str) -> str | None:
    """
    Copy an ingested audio/video source file into the served media
    tree ONCE, so audio-clip playback has a real file to seek into
    (see module docstring - one full copy per document, not one file
    per embedded clip).

    Idempotent: if the target already exists, the copy is skipped -
    re-ingesting an unchanged file is cheap, not a repeated large-file
    copy.

    Returns None (and logs a warning) instead of raising if the
    extension isn't in the allowed set - a defensive check that should
    never actually trigger for a file that already passed through
    ingestion's own extension-based loader dispatch.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in _ALLOWED_SOURCE_EXTENSIONS:
        logger.warning("Refusing to save media copy for unrecognized extension %r (document_id=%r)", ext, document_id)
        return None

    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_document_hash(document_id)}{ext}"
    dest = SOURCES_DIR / filename
    if not dest.exists():
        shutil.copyfile(file_path, dest)
    return f"/media/sources/{filename}"


def delete_media_for_document(document_id: str) -> int:
    """
    Remove every saved frame thumbnail and source-file copy for a
    document, keyed off the same hash save_frame_thumbnail()/
    save_media_copy() derive their filenames from.

    Called from app.vectorstore.store.delete_document() so deleting a
    document from the UI doesn't leave orphaned files under data/media/
    behind forever - before this, a deleted video's frames/audio copy
    had no cleanup path at all, since they're keyed by content hash,
    not by any Chroma id delete_document() already knew to remove.

    A no-op (returns 0) for a document that was never a video/audio
    source - safe to call unconditionally for every deleted document,
    same as app.vectorstore.store already does for the image/audio-clip
    collections themselves.

    Returns:
        How many files were actually deleted.
    """
    doc_hash = _document_hash(document_id)
    deleted = 0

    for path in FRAMES_DIR.glob(f"{doc_hash}_*.jpg"):
        path.unlink(missing_ok=True)
        deleted += 1

    for path in SOURCES_DIR.glob(f"{doc_hash}.*"):
        path.unlink(missing_ok=True)
        deleted += 1

    if deleted:
        logger.info("Deleted %d media file(s) for document_id=%r", deleted, document_id)
    return deleted

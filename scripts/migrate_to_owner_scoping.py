"""
One-off migration: assign every pre-existing chunk/file to a single
real account, so the per-user data isolation added across
app.vectorstore.store / app.media.media_store / app.graph.code_graph
actually scopes today's data instead of it being invisible to every
account (nothing pre-Phase-2 ever wrote an "owner" metadata field).

NOT part of the app's normal code path - run this by hand, once,
against a stopped app container's data/ directory.

--------------------------------------------------------------------
IMPORTANT: stop the app (and Ollama) container before running this
with --apply. This script opens the same Chroma SQLite-backed store
the running app uses; a concurrent write from the live server while
this script is mid-migration risks corrupting the database. `docker
compose stop app` (Ollama can keep running) is enough - no need to
tear down the whole stack.
--------------------------------------------------------------------

What this migrates, and why each piece needs its own treatment:

1. Every text-KB chunk (markdown/pdf/docx/web/github/audio/video
   collections - "audio"/"video" here means transcript-text chunks,
   not the binary audio-clip/image collections below). The storage id
   scheme changed from f"{document_id}::chunk_{n}" (etc.) to
   f"{owner}::{document_id}::chunk_{n}" - since Chroma has no
   "rename an id in place" operation, each chunk is re-inserted under
   its new id (delete old id + upsert new id, same embedding/content,
   metadata gains "owner") rather than metadata-patched in place. This
   also sidesteps a real latent bug a metadata-only patch would leave
   behind: if a still-bare-id chunk were ever re-ingested after
   migration, the new code would compute a NEW owner-prefixed id and
   upsert a second, duplicate chunk alongside the untouched old one
   instead of cleanly overwriting it.

2. The image collection (website images + video frame thumbnails).
   Same id-rewrite treatment, PLUS: a video frame's `image_url` is
   itself a filename hash of `document_id` alone (pre-migration - see
   app.media.media_store's git history), and the id-doubles-as-url
   collection scheme means the id, the url, AND the actual file on
   disk all have to move together. Website images (image_url is an
   external http(s) URL) only need the id rewrite - there's no local
   file to rename.

3. The audio-clip collection. Same id-rewrite treatment, PLUS the
   `source_audio_url` field (and the underlying data/media/sources/
   file it points at) needs the same owner-mixed-hash rename video
   frames get, for the same file-identity reason.

4. data/graphs/{repo}.gpickle -> data/graphs/{username}/{repo}.gpickle
   (app.graph.code_graph._graph_path now namespaces graphs per account).

Idempotent and safe to re-run: any chunk that ALREADY has a real
(non-empty) "owner" metadata value is left untouched and counted
separately, rather than being silently reassigned - protects against
double-migrating, or against clobbering data a differently-scoped run
already touched.

Usage:
    python scripts/migrate_to_owner_scoping.py --username alice            # dry run (default)
    python scripts/migrate_to_owner_scoping.py --username alice --apply    # actually migrate
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import user_store  # noqa: E402
from app.vectorstore import store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("migrate_to_owner_scoping")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_GRAPHS_DIR = _DATA_DIR / "graphs"
_FRAMES_DIR = _DATA_DIR / "media" / "frames"
_SOURCES_DIR = _DATA_DIR / "media" / "sources"

_UPSERT_BATCH_SIZE = 500


def _old_document_hash(document_id: str) -> str:
    """Pre-migration file-hash scheme (bare document_id, no owner) -
    see app.media.media_store._document_hash's current (owner-mixed)
    version for what this is being migrated TO."""
    return hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:16]


def _new_document_hash(document_id: str, owner: str) -> str:
    return hashlib.sha1(f"{owner}::{document_id}".encode("utf-8")).hexdigest()[:16]


def _has_real_owner(metadata: dict) -> bool:
    return bool(metadata.get("owner"))


def backup_data_dir() -> Path:
    """Copy the whole data/ directory aside before touching anything.
    Cheap insurance for a one-shot, hand-run, hard-to-reverse script -
    a bad run can be undone by restoring this copy."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = _REPO_ROOT / f"data_backup_{timestamp}"
    logger.info("Backing up %s -> %s ...", _DATA_DIR, backup_dir)
    shutil.copytree(_DATA_DIR, backup_dir)
    logger.info("Backup complete.")
    return backup_dir


def migrate_text_kbs(username: str, apply: bool) -> None:
    for kb in store.KB_NAMES:
        collection = store._get_collection(kb)
        result = collection.get(include=["documents", "embeddings", "metadatas"])
        ids = result["ids"]
        documents = result["documents"]
        embeddings = result["embeddings"]
        metadatas = result["metadatas"]

        old_ids: list[str] = []
        new_ids: list[str] = []
        new_documents: list[str] = []
        new_embeddings: list = []
        new_metadatas: list[dict] = []
        already_scoped = 0

        for old_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            if _has_real_owner(metadata):
                already_scoped += 1
                continue
            old_ids.append(old_id)
            new_ids.append(f"{username}::{old_id}")
            new_documents.append(document)
            new_embeddings.append(embedding)
            new_metadatas.append({**metadata, "owner": username})

        logger.info(
            "[%s] %d chunk(s) to migrate, %d already scoped (skipped), %d total.",
            kb, len(old_ids), already_scoped, len(ids),
        )

        if not apply or not old_ids:
            continue

        collection.delete(ids=old_ids)
        for start in range(0, len(new_ids), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE
            collection.upsert(
                ids=new_ids[start:end],
                embeddings=new_embeddings[start:end],
                documents=new_documents[start:end],
                metadatas=new_metadatas[start:end],
            )
        logger.info("[%s] migrated %d chunk(s).", kb, len(old_ids))


def migrate_image_collection(username: str, apply: bool) -> None:
    collection = store._get_image_collection()
    result = collection.get(include=["documents", "embeddings", "metadatas"])
    ids = result["ids"]
    documents = result["documents"]
    embeddings = result["embeddings"]
    metadatas = result["metadatas"]

    old_ids: list[str] = []
    new_ids: list[str] = []
    new_documents: list[str] = []
    new_embeddings: list = []
    new_metadatas: list[dict] = []
    frame_renames: list[tuple[Path, Path]] = []
    already_scoped = 0

    for old_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
        if _has_real_owner(metadata):
            already_scoped += 1
            continue

        new_metadata = {**metadata, "owner": username}
        image_url = metadata.get("image_url", old_id)

        if metadata.get("origin") == "video_frame" and image_url.startswith("/media/frames/"):
            video_document_id = metadata.get("video_document_id")
            old_filename = image_url.rsplit("/", 1)[-1]
            # filename is "{hash}_{timestamp_ms}.jpg" - keep the
            # timestamp suffix, only the hash prefix changes.
            _old_hash_prefix, _, suffix = old_filename.partition("_")
            new_hash = _new_document_hash(video_document_id, username)
            new_filename = f"{new_hash}_{suffix}"
            new_image_url = f"/media/frames/{new_filename}"
            frame_renames.append((_FRAMES_DIR / old_filename, _FRAMES_DIR / new_filename))
            new_metadata["image_url"] = new_image_url
            new_id = f"{username}::{new_image_url}"
        else:
            new_id = f"{username}::{image_url}"

        old_ids.append(old_id)
        new_ids.append(new_id)
        new_documents.append(document)
        new_embeddings.append(embedding)
        new_metadatas.append(new_metadata)

    logger.info(
        "[image] %d chunk(s) to migrate (%d frame file rename(s)), %d already scoped (skipped), %d total.",
        len(old_ids), len(frame_renames), already_scoped, len(ids),
    )

    if not apply:
        return

    for old_path, new_path in frame_renames:
        if old_path.exists():
            old_path.rename(new_path)
        else:
            logger.warning("Frame file missing on disk, skipping rename: %s", old_path)

    if old_ids:
        collection.delete(ids=old_ids)
        for start in range(0, len(new_ids), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE
            collection.upsert(
                ids=new_ids[start:end],
                embeddings=new_embeddings[start:end],
                documents=new_documents[start:end],
                metadatas=new_metadatas[start:end],
            )
        logger.info("[image] migrated %d chunk(s), renamed %d frame file(s).", len(old_ids), len(frame_renames))


def migrate_audio_clip_collection(username: str, apply: bool) -> None:
    collection = store._get_audio_clip_collection()
    result = collection.get(include=["documents", "embeddings", "metadatas"])
    ids = result["ids"]
    documents = result["documents"]
    embeddings = result["embeddings"]
    metadatas = result["metadatas"]

    old_ids: list[str] = []
    new_ids: list[str] = []
    new_documents: list[str] = []
    new_embeddings: list = []
    new_metadatas: list[dict] = []
    source_renames: dict[Path, Path] = {}  # dedupe: one source file backs many clips
    already_scoped = 0

    for old_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
        if _has_real_owner(metadata):
            already_scoped += 1
            continue

        document_id = metadata["document_id"]
        new_metadata = {**metadata, "owner": username}
        source_url = metadata.get("source_audio_url")

        if source_url and source_url.startswith("/media/sources/"):
            old_filename = source_url.rsplit("/", 1)[-1]
            ext = Path(old_filename).suffix
            new_filename = f"{_new_document_hash(document_id, username)}{ext}"
            new_source_url = f"/media/sources/{new_filename}"
            source_renames[_SOURCES_DIR / old_filename] = _SOURCES_DIR / new_filename
            new_metadata["source_audio_url"] = new_source_url

        new_id = f"{username}::{document_id}::clip_{metadata['start_seconds']}"
        old_ids.append(old_id)
        new_ids.append(new_id)
        new_documents.append(document)
        new_embeddings.append(embedding)
        new_metadatas.append(new_metadata)

    logger.info(
        "[audio-clip] %d chunk(s) to migrate (%d source file rename(s)), %d already scoped (skipped), %d total.",
        len(old_ids), len(source_renames), already_scoped, len(ids),
    )

    if not apply:
        return

    for old_path, new_path in source_renames.items():
        if old_path.exists():
            old_path.rename(new_path)
        else:
            logger.warning("Source file missing on disk, skipping rename: %s", old_path)

    if old_ids:
        collection.delete(ids=old_ids)
        for start in range(0, len(new_ids), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE
            collection.upsert(
                ids=new_ids[start:end],
                embeddings=new_embeddings[start:end],
                documents=new_documents[start:end],
                metadatas=new_metadatas[start:end],
            )
        logger.info(
            "[audio-clip] migrated %d chunk(s), renamed %d source file(s).", len(old_ids), len(source_renames)
        )


def migrate_graphs(username: str, apply: bool) -> None:
    if not _GRAPHS_DIR.exists():
        logger.info("[graphs] no data/graphs/ directory, nothing to do.")
        return

    # Only direct-child .gpickle files are pre-migration - anything
    # already under a per-owner subdirectory was migrated by a prior run.
    to_move = sorted(p for p in _GRAPHS_DIR.iterdir() if p.is_file() and p.suffix == ".gpickle")
    logger.info("[graphs] %d graph file(s) to move under data/graphs/%s/.", len(to_move), username)

    if not apply or not to_move:
        return

    target_dir = _GRAPHS_DIR / username
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in to_move:
        path.rename(target_dir / path.name)
    logger.info("[graphs] moved %d graph file(s).", len(to_move))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--username", required=True, help="Account every pre-existing document is assigned to.")
    parser.add_argument("--apply", action="store_true", help="Actually migrate. Without this, only reports what would happen.")
    parser.add_argument("--skip-backup", action="store_true", help="Skip the automatic data/ backup (not recommended).")
    args = parser.parse_args()

    if not user_store.user_exists(args.username):
        logger.error(
            "No account named %r exists yet (checked data/users.json). "
            "Sign up for this account through the running app first, then re-run this script.",
            args.username,
        )
        raise SystemExit(1)

    if args.apply:
        logger.warning(
            "Running with --apply: this will rewrite chunk ids/metadata and rename files on disk. "
            "Make sure the app container is STOPPED before continuing (see this script's module docstring)."
        )
        if not args.skip_backup:
            backup_data_dir()
    else:
        logger.info("Dry run (pass --apply to actually migrate). No data will be changed.")

    migrate_text_kbs(args.username, args.apply)
    migrate_image_collection(args.username, args.apply)
    migrate_audio_clip_collection(args.username, args.apply)
    migrate_graphs(args.username, args.apply)

    logger.info("Done.")


if __name__ == "__main__":
    main()

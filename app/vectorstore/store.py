"""
Vector storage for the RAG pipeline.

Responsibility: persist each EmbeddedChunk's vector (+ text + metadata)
so it can later be searched by similarity. No embedding computation and
no LLM generation happens here — this module only stores and retrieves.

Provider used: Chroma (chromadb), running in embedded/local mode. It
writes its index to a folder on disk (CHROMA_DB_DIR below), so data
persists across runs without needing a separate database server.

Design:
- add_embedded_chunks() and query_embedding() are the main entry
  points other modules should use, so the rest of the pipeline never
  talks to Chroma's API directly (same "one place per concern" pattern
  used in embedder.py). get_table_chunk() is a second, narrower entry
  point: a direct metadata lookup (not a similarity search) used only
  by the retriever's table-completion step.
- Multiple text "knowledge bases" (KBs), not one shared collection:
  text chunks are split across FIVE separate Chroma collections, one
  per source (KB_NAMES: "markdown", "pdf", "docx", "web", "github"),
  instead of all sharing a single collection. This exists so query
  routing (app.routing.router) can search only the KB(s) a query is
  actually about, and so a query with no specific KB signal can still
  skip any KB that has never been ingested into at all
  (list_populated_kbs()) - both are real latency wins once there are
  several sources ingested, not just a rerank/filter tweak. Each
  chunk's KB is decided purely from its `source_type` metadata (set at
  load time - see app.ingestion.loader) via _kb_for_metadata(); chunks
  with no recognized source_type fall back to _DEFAULT_KB. GitHub repo
  files always get source_type="code" regardless of their own file
  type (a repo's README.md included - see load_github_repository), so
  they land in the "github" KB, not "markdown".
- Chroma requires metadata values to be str/int/float/bool — it cannot
  store a list directly. Our chunk metadata has a `block_types` field
  that is a list (e.g. ["heading", "table"]), so _sanitize_metadata()
  joins list values into a comma-separated string before storing, and
  drops any None values (Chroma also rejects None).
- Each chunk needs a unique, stable id. We build it from the source
  file name + chunk_index, so re-ingesting the same document with the
  same chunking parameters produces the same ids (upsert-friendly).
- Distance metric: every collection is created with hnsw:space=
  "cosine", so Chroma indexes and compares vectors using cosine
  distance instead of its default (squared L2/Euclidean). Cosine is
  the standard choice for sentence-transformer embeddings, since these
  models are trained to optimize cosine similarity rather than raw
  vector magnitude. This setting only takes effect when a collection
  is first created, so changing it later requires deleting
  CHROMA_DB_DIR and re-ingesting.
- A further, separate collection (IMAGE_COLLECTION_NAME) stores
  multimodal (CLIP) image embeddings from
  app.embeddings.image_embedder, via add_image_chunks()/
  query_image_embedding()/image_count(). It's distinct from every text
  KB above because CLIP vectors live in a completely different vector
  space than the all-MiniLM-L6-v2 text embeddings stored there -
  cosine similarity is only meaningful between vectors produced by the
  same model.
- Functions that look up one chunk/set of chunks by exact metadata
  (not a similarity search) - get_table_chunk(), get_class_chunk(),
  get_chunk_by_document_id(), get_chunks_by_content_type(),
  list_repositories() - don't take a `kb` argument: they're rare,
  cheap point lookups (not the hot, latency-sensitive similarity-search
  path query_embedding() is), so they simply check every text KB via
  _iter_collections() rather than requiring every caller to already
  know which KB a given id/content_type/repository lives in.
"""

import logging
import random
from pathlib import Path

import chromadb

from app.embeddings.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)

CHROMA_DB_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db")
IMAGE_COLLECTION_NAME = "rag_image_chunks"
AUDIO_CLIP_COLLECTION_NAME = "rag_audio_clip_chunks"


def _invalidate_hybrid_cache(kb: str) -> None:
    """
    Drop any BM25/hybrid-retriever AND KB-routing caches for the given
    KB whenever its stored chunks change. Lazily imported to avoid a
    circular dependency (both hybrid_retriever and router import from
    this module).

    Silently no-ops if a module isn't importable (e.g. in a test
    environment that stubs out retrieval/routing).
    """
    try:
        from app.retrieval.hybrid_retriever import invalidate_bm25_cache
    except ImportError:
        pass
    else:
        invalidate_bm25_cache(kb)

    try:
        from app.routing.router import invalidate_kb_routing_cache
    except ImportError:
        return
    invalidate_kb_routing_cache(kb)


# One Chroma collection per source "knowledge base" (see module
# docstring). Order here is only for readability/logging - it has no
# effect on routing or search results.
KB_NAMES = ["markdown", "pdf", "docx", "web", "github", "audio", "video"]

_COLLECTION_NAME_BY_KB = {kb: f"rag_chunks_{kb}" for kb in KB_NAMES}

# How a chunk's `source_type` metadata (set at load time - see
# app.ingestion.loader) maps to a KB. GitHub-sourced files are always
# tagged source_type="code" regardless of their actual file type (see
# load_github_repository), so "code" is the one source_type that maps
# to a differently-named KB ("github", not "code").
_SOURCE_TYPE_TO_KB = {
    "markdown": "markdown",
    "pdf": "pdf",
    "docx": "docx",
    "web": "web",
    "code": "github",
    "audio": "audio",
    "video": "video",
}

# Fallback KB for a chunk whose metadata has no recognized source_type
# (e.g. hand-built test fixtures, or any future source type added to
# loader.py before this mapping is updated to match) - keeps
# add_embedded_chunks() total rather than raising on unrecognized
# metadata.
_DEFAULT_KB = "markdown"

# Owner used when a chunk's metadata has no "owner" key at all (e.g.
# hand-built test fixtures predating per-user ownership, or a caller
# that hasn't been updated yet) - mirrors api.deps.LOCAL_OWNER's value
# without importing from the api/ layer (this module sits below it).
_DEFAULT_OWNER = "local"

_client = None
_collections: dict[str, object] = {}
_image_collection = None
_audio_clip_collection = None


def _get_client():
    """Lazily create the persistent Chroma client, shared by every collection."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _client


def _get_collection(kb: str = _DEFAULT_KB):
    """
    Lazily create (or fetch, if already created this process) the text
    KB collection for `kb`, so importing this module doesn't touch
    disk until actually needed.
    """
    if kb not in _collections:
        _collections[kb] = _get_client().get_or_create_collection(
            name=_COLLECTION_NAME_BY_KB[kb],
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[kb]


def _iter_collections():
    """Every text KB's collection, for point lookups that must check all of them (see module docstring)."""
    return (_get_collection(kb) for kb in KB_NAMES)


def _kb_for_metadata(metadata: dict) -> str:
    """Which KB a chunk belongs in, based on its `source_type` metadata (see module docstring)."""
    return _SOURCE_TYPE_TO_KB.get(metadata.get("source_type"), _DEFAULT_KB)


def _get_image_collection():
    """
    Lazily create the image collection (see module docstring for why
    this is separate from every text KB collection).
    """
    global _image_collection
    if _image_collection is None:
        _image_collection = _get_client().get_or_create_collection(
            name=IMAGE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _image_collection


def _get_audio_clip_collection():
    """
    Lazily create the audio-clip collection (see
    app.embeddings.audio_embedder's module docstring for why this is
    separate from both the text KBs AND the image collection - CLAP's
    vector space has nothing in common with either).
    """
    global _audio_clip_collection
    if _audio_clip_collection is None:
        _audio_clip_collection = _get_client().get_or_create_collection(
            name=AUDIO_CLIP_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _audio_clip_collection


def _sanitize_metadata(metadata: dict) -> dict:
    """
    Convert a chunk's metadata into a form Chroma accepts: only
    str/int/float/bool values, no None, no lists.
    """
    clean: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, list):
            clean[key] = ", ".join(str(item) for item in value)
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def add_embedded_chunks(chunks: list[EmbeddedChunk]) -> None:
    """
    Store a list of EmbeddedChunks in the vector store.

    Each chunk is routed to its KB collection (see module docstring)
    based on its own `source_type` metadata - a single call can mix
    chunks bound for different KBs (e.g. ingest_all() chunking several
    source types in one pass); they're grouped and upserted into each
    KB's collection separately, transparently to the caller.

    Args:
        chunks: EmbeddedChunks produced by
            app.embeddings.embedder.embed_chunks.
    """
    if not chunks:
        return

    chunks_by_kb: dict[str, list[EmbeddedChunk]] = {}
    for chunk in chunks:
        chunks_by_kb.setdefault(_kb_for_metadata(chunk.metadata), []).append(chunk)

    for kb, kb_chunks in chunks_by_kb.items():
        collection = _get_collection(kb)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in kb_chunks:
            document_id = chunk.metadata.get("document_id") or chunk.metadata.get("file_name", "document")
            chunk_index = chunk.metadata.get("chunk_index", len(ids))
            page_number = chunk.metadata.get("page_number")
            # Storage id is owner-prefixed so two owners' documents that
            # happen to share a document_id (e.g. both upload a file
            # called "report.pdf") never collide/overwrite each other -
            # deliberately only the internal Chroma id, not the
            # document_id metadata VALUE, which stays untouched since
            # it's a user-visible display string (Sources list, chat
            # toasts, evaluation fixtures) and is independently
            # re-derived by app.graph.code_graph at graph-build time.
            owner = chunk.metadata.get("owner", _DEFAULT_OWNER)

            # PDF chunks carry page_number (recovered from page markers -
            # see app.chunking.chunker._pack_section) even though chunk_index
            # is already unique on its own (continuous across the whole
            # file, never reset per page). page_number is still folded into
            # the id so it stays legible/debuggable, and so any future
            # source that DOES restart chunk_index per page/unit remains
            # collision-safe too.
            if page_number is not None:
                ids.append(f"{owner}::{document_id}::page_{page_number}::chunk_{chunk_index}")
            else:
                ids.append(f"{owner}::{document_id}::chunk_{chunk_index}")

            embeddings.append(chunk.embedding)
            documents.append(chunk.content)
            # owner is written back explicitly (not just used for the
            # id above) so it's always a real, queryable metadata
            # field on every stored chunk - including for chunks whose
            # caller never set one, defaulting through to
            # _DEFAULT_OWNER, so a where={"owner": ...} filter can
            # never silently miss a chunk that simply never had the
            # field at all.
            metadatas.append(_sanitize_metadata({**chunk.metadata, "owner": owner}))

        # upsert: re-running ingestion on the same document overwrites the
        # same ids instead of creating duplicates.
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Stored/updated %d chunk(s) in KB '%s' (collection '%s')", len(ids), kb, _COLLECTION_NAME_BY_KB[kb])
        _invalidate_hybrid_cache(kb)


def query_embedding(query_vector: list[float], top_k: int = 5, where: dict | None = None, kb: str = _DEFAULT_KB) -> list[dict]:
    """
    Find the top_k stored chunks most similar to a query embedding,
    searching only the given KB's collection (see module docstring).

    Args:
        query_vector: The embedding of the user's query (same model/
            dimension as the stored chunk embeddings).
        top_k: How many results to return.
        where: Optional Chroma metadata filter (e.g.
            {"repository": "owner/repo"}) applied BEFORE the similarity
            search, so with multiple repos/documents ingested into the
            same KB a query can be scoped to just one of them instead
            of competing against that whole KB. None (the default)
            searches the entire KB, unfiltered.
        kb: Which KB collection to search (one of KB_NAMES) - callers
            (see app.retrieval.retriever) decide this via query
            routing, so only the KB(s) actually relevant to a query
            are ever searched instead of always searching everything.

    Returns:
        A list of dicts, each with "content", "metadata", "distance"
        (cosine distance, lower = more similar), and "similarity"
        (cosine similarity, higher = more similar; similarity = 1 -
        distance), ordered from most to least similar.
    """
    collection = _get_collection(kb)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
    )

    hits: list[dict] = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for content, metadata, distance in zip(documents, metadatas, distances):
        hits.append(
            {
                "content": content,
                "metadata": metadata,
                "distance": distance,
                "similarity": 1 - distance,
            }
        )

    logger.info("Query returned %d result(s) for top_k=%d in KB '%s'", len(hits), top_k, kb)
    return hits


def sample_kb_embeddings(kb: str, limit: int = 40, owner: str | None = None) -> list[list[float]]:
    """
    Return up to `limit` embeddings already stored in `kb`'s
    collection, chosen by random sample (not just the first N Chroma
    happens to return, which could skew toward whichever document was
    ingested first).

    Used by app.routing.router to route a query by comparing it
    against a sample of what's ACTUALLY been ingested into each KB,
    instead of a fixed set of hand-written example questions that
    can't anticipate every deployment's real subject matter (e.g. a
    KB full of Targetprocess automation-rule docs needs a very
    different "what does a typical question here look like" signal
    than a KB of API reference PDFs would). No extra embedding cost:
    Chroma already stores each chunk's vector, so this is a plain
    fetch, not a re-embed.

    owner is deliberately OPTIONAL and defaults to sampling across
    EVERY owner: unlike every other owner-scoped function in this
    module, this one never returns/discloses any chunk content - it
    only nudges which KB a query gets ROUTED to (a soft signal, not a
    content boundary). Fully owner-scoping it would require the same
    (owner, kb)-keyed caching overhaul this module's BM25 index got
    (see app.retrieval.hybrid_retriever) purely to prevent one owner's
    ingested-topic mix from mildly influencing another owner's routing
    scores - a real but low-severity gap, deliberately deferred rather
    than expanding this change's blast radius for a non-disclosure risk.

    Returns [] for an empty/unpopulated KB - the caller treats that the
    same as any other KB with no semantic signal to offer.
    """
    collection = _get_collection(kb)
    where = {"owner": owner} if owner is not None else None
    all_ids = collection.get(where=where, include=[])["ids"]
    if not all_ids:
        return []
    sample_ids = random.sample(all_ids, min(limit, len(all_ids)))
    result = collection.get(ids=sample_ids, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return []
    # Cast every component to a native Python float - Chroma returns
    # embeddings as numpy float32 arrays, and a bare list(vector) keeps
    # numpy.float32 scalars in the list. Those silently poison every
    # downstream score derived from them (app.routing.router's cosine
    # similarities) with numpy scalar types instead of plain floats -
    # harmless for arithmetic, but numpy.bool_ (from e.g. `score >=
    # 0.40` in insight.py) is NOT json-serializable the way a native
    # Python bool is, so an unconverted embedding here eventually broke
    # the entire /api/chat/stream response with a 500 mid-stream.
    return [[float(x) for x in vector] for vector in embeddings]


def get_table_chunk(table_id: str, owner: str) -> dict | None:
    """
    Fetch the whole-table chunk (content_type == "table") matching a
    given table_id, as opposed to any of its individual per-row chunks
    which share the same table_id.

    Used by app.retrieval.retriever to "complete" a table: when a
    retrieved chunk is only one row of a table, the retriever looks up
    the complete table via this function so the LLM sees every row
    instead of just whichever rows happened to rank in the top_k.

    Checks every text KB (the caller doesn't know in advance which KB a
    given table_id lives in - see module docstring) and returns the
    first match. owner is checked independently of table_id (not just
    relied upon via the id scheme) so two owners' tables can never
    cross-resolve even if their table_id metadata happened to coincide.

    Returns:
        A dict with "content" and "metadata" (no "distance"/
        "similarity", since this is a direct metadata lookup, not a
        similarity search), or None if no matching table chunk exists.
    """
    for collection in _iter_collections():
        result = collection.get(
            where={"$and": [{"table_id": table_id}, {"content_type": "table"}, {"owner": owner}]},
            limit=1,
        )
        if result["documents"]:
            return {
                "content": result["documents"][0],
                "metadata": result["metadatas"][0],
                "distance": None,
                "similarity": None,
            }
    return None


def get_class_chunk(class_id: str, owner: str) -> dict | None:
    """
    Fetch the whole-class chunk (content_type == "class") matching a
    given class_id, as opposed to any of its individual per-method
    chunks which share the same class_id.

    The code-chunking counterpart of get_table_chunk() (see that
    function's docstring) - used by
    app.retrieval.retriever._complete_partial_classes() to "complete"
    a class: when a retrieved chunk is only one method, the retriever
    looks up the complete class via this function so the LLM sees the
    whole class (constructor, shared state, sibling methods) instead
    of just that one method in isolation.

    Checks every text KB, same as get_table_chunk() (class/method
    chunks only ever exist in the "github" KB today, but this avoids
    silently breaking if that ever changes). owner is checked
    independently, same reasoning as get_table_chunk().

    Returns:
        A dict with "content" and "metadata" (no "distance"/
        "similarity", since this is a direct metadata lookup, not a
        similarity search), or None if no matching class chunk exists.
    """
    for collection in _iter_collections():
        result = collection.get(
            where={"$and": [{"class_id": class_id}, {"content_type": "class"}, {"owner": owner}]},
            limit=1,
        )
        if result["documents"]:
            return {
                "content": result["documents"][0],
                "metadata": result["metadatas"][0],
                "distance": None,
                "similarity": None,
            }
    return None


def count(kb: str | None = None) -> int:
    """
    Return how many chunks are currently stored.

    Args:
        kb: If given, count only that one KB's collection. None (the
            default) sums every text KB's collection.
    """
    if kb is not None:
        return _get_collection(kb).count()
    return sum(collection.count() for collection in _iter_collections())


def list_populated_kbs(owner: str) -> list[str]:
    """
    Every KB (see KB_NAMES) that has at least one chunk stored FOR
    THIS OWNER.

    Used by app.retrieval.retriever as the default search scope when
    query routing (app.routing.router) finds no explicit KB signal in
    a query: rather than searching every KB unconditionally, it only
    searches the ones that actually have data - a KB nothing has ever
    been ingested into (by this owner) is skipped for free, with no
    risk of missing real results. Owner-scoped rather than using
    count()'s global total, since a KB populated only by a DIFFERENT
    owner shouldn't be treated as a real fallback target for this one -
    Chroma's .count() has no `where` parameter, so this is a
    get(where=...)-and-measure instead of the O(1) count() other
    global stats use.
    """
    populated = []
    for kb in KB_NAMES:
        ids = _get_collection(kb).get(where={"owner": owner}, include=[])["ids"]
        if ids:
            populated.append(kb)
    return populated


def get_chunk_by_document_id(document_id: str, owner: str) -> dict | None:
    """
    Fetch a single stored chunk by its exact document_id, as opposed
    to a similarity search - e.g. looking up the image_ocr/image_code
    text chunk derived from one specific image (whose document_id is
    the image's own URL - see app.ingestion.ingest's _ingest_images(),
    each image is its own single-chunk "document").

    Used by cli.py to check whether a retrieved image was classified
    as a source-code screenshot (content_type == "image_code") at
    ingestion time (see app.ocr.image_classifier), so the vision model
    can be auto-invoked for exactly that case without needing the
    (slow, unreliable) vision path for every loosely-relevant image.

    Checks every text KB (images are ingested from websites, so their
    derived text chunk lives in the "web" KB today, but this avoids
    hard-coding that assumption here). owner is checked independently,
    same reasoning as get_table_chunk().

    Returns:
        A dict with "content" and "metadata" (no "distance"/
        "similarity", since this is a direct metadata lookup), or None
        if no chunk with that document_id exists.
    """
    for collection in _iter_collections():
        result = collection.get(
            where={"$and": [{"document_id": document_id}, {"owner": owner}]}, limit=1
        )
        if result["documents"]:
            return {
                "content": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
    return None


def get_chunks_by_content_type(content_type: str) -> list[dict]:
    """
    Fetch EVERY stored chunk matching a given content_type (e.g.
    "image_ocr"), as opposed to a similarity search.

    Used by one-off maintenance/upgrade scripts (e.g.
    app.ocr.upgrade_code_images) that need to re-process
    already-ingested chunks of one kind in place, without re-running
    the whole (multi-hour) crawl+chunk+embed pipeline.

    Checks every text KB and returns the combined results.

    Returns:
        A list of dicts, each with "id", "content", and "metadata" -
        no "distance"/"similarity", since this isn't a similarity
        search.
    """
    matches: list[dict] = []
    for collection in _iter_collections():
        result = collection.get(where={"content_type": content_type})
        matches.extend(
            {"id": chunk_id, "content": content, "metadata": metadata}
            for chunk_id, content, metadata in zip(result["ids"], result["documents"], result["metadatas"])
        )
    return matches


def list_repositories(owner: str) -> list[str]:
    """
    Every distinct "owner/repo" value among THIS OWNER's stored chunks
    (see app.ingestion.loader.load_github_repository's document_id/
    repository metadata), sorted alphabetically.

    Used by the UI/CLI to let a user pick which already-ingested
    repository to scope a question to (see retrieve()'s `repository`
    argument) - GitHub-sourced chunks all land in the "github" KB (see
    module docstring), but every text KB is checked here for
    robustness rather than hard-coding that assumption. Owner-scoped so
    one account's repo-picker dropdown never reveals what repository
    names another account has privately ingested.

    Returns:
        An empty list if this owner hasn't ingested a GitHub repo yet
        (chunks with no "repository" metadata key, e.g. from
        PDFs/websites, are not included).
    """
    repositories: set[str] = set()
    for collection in _iter_collections():
        result = collection.get(where={"owner": owner}, include=["metadatas"])
        repositories.update(
            metadata.get("repository") for metadata in result["metadatas"] if metadata.get("repository")
        )
    return sorted(repositories)


def list_documents(kb: str, owner: str) -> list[dict]:
    """
    Every distinct document currently stored in one KB FOR THIS OWNER,
    with its stored chunk count.

    For kb="github", documents are grouped by `repository` (one row
    per ingested repo) rather than by individual file `document_id`,
    matching how the GitHub KB is ingested/deleted as a whole repo
    (see delete_repository()). Every other KB is grouped by
    `document_id` (see delete_document()).

    Used by the UI to show what's already indexed in a KB, as opposed
    to what's merely pending-in-this-session (see
    data/pending_documents.json / ui.state). Owner-scoped so the
    Sources page only ever lists documents this account itself
    ingested.

    Returns:
        A list of {"id": str, "chunk_count": int} dicts, sorted by id.
        "id" is a document_id for every KB except "github", where
        it's an "owner/repo" repository name.
    """
    collection = _get_collection(kb)
    result = collection.get(where={"owner": owner}, include=["metadatas"])
    group_field = "repository" if kb == "github" else "document_id"
    counts: dict[str, int] = {}
    for metadata in result["metadatas"]:
        key = metadata.get(group_field)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return [{"id": doc_id, "chunk_count": n} for doc_id, n in sorted(counts.items())]


def delete_document(document_id: str, owner: str, kb: str | None = None) -> int:
    """
    Delete every chunk whose metadata `document_id` matches the given
    value AND belongs to `owner`, PLUS any video frames / audio clips /
    saved media files associated with it. Returns the number of TEXT
    chunks removed (unchanged return contract for existing callers -
    frame/clip/file cleanup counts are only logged, not returned).

    Args:
        document_id: The document_id metadata value written by the
            loader (e.g. the file path for PDF/DOCX/Markdown, the URL
            for a website page).
        owner: Required - closes what was previously a real gap (any
            session could delete any document regardless of who
            ingested it). Checked as an independent `where` clause,
            not inferred from the storage id, so a delete can never
            cross owners even if document_id metadata happens to
            coincide between two owners (e.g. both ingest a file
            literally named "report.pdf").
        kb: Optional - if given, only delete from that specific KB
            (faster). If None, checks every KB (safer when the caller
            doesn't know which KB a document lives in).

    Used by the UI's "discard" and auto-cleanup-of-abandoned-session
    paths to remove ingested-but-not-saved documents (see app_ui.py).

    Image-collection and audio-clip-collection cleanup is checked
    UNCONDITIONALLY of `kb`, since both are separate collections
    outside the per-KB text collections `kb` scopes - a video's frames
    live in the image collection regardless of which text KB argument
    was passed. Before this, deleting a video/audio document left its
    frames, clips, and saved media files under data/media/ orphaned
    forever - there was no cleanup path for them at all (see
    app.media.media_store.delete_media_for_document).
    """
    collections = [_get_collection(kb)] if kb is not None else list(_iter_collections())
    total_deleted = 0
    affected_kbs: set[str] = set()
    for collection in collections:
        existing = collection.get(
            where={"$and": [{"document_id": document_id}, {"owner": owner}]}, include=[]
        )
        ids = existing["ids"]
        if ids:
            collection.delete(ids=ids)
            total_deleted += len(ids)
            # collection.name is e.g. "rag_chunks_pdf" - map back to kb.
            for kb_name, coll_name in _COLLECTION_NAME_BY_KB.items():
                if coll_name == collection.name:
                    affected_kbs.add(kb_name)
    if total_deleted:
        logger.info("Deleted %d chunk(s) for document_id=%r owner=%r", total_deleted, document_id, owner)
    for affected_kb in affected_kbs:
        _invalidate_hybrid_cache(affected_kb)

    # Video frames are tagged video_document_id; website-page images
    # are tagged page_url (== document_id for a web page - see
    # app.ingestion.loader._build_website_document/extract_image_records).
    # A document is never both, so both filters are checked, not just one.
    image_collection = _get_image_collection()
    for field in ("video_document_id", "page_url"):
        existing = image_collection.get(where={"$and": [{field: document_id}, {"owner": owner}]}, include=[])
        ids = existing["ids"]
        if ids:
            image_collection.delete(ids=ids)
            logger.info("Deleted %d image chunk(s) (%s=%r)", len(ids), field, document_id)

    audio_collection = _get_audio_clip_collection()
    existing = audio_collection.get(
        where={"$and": [{"document_id": document_id}, {"owner": owner}]}, include=[]
    )
    ids = existing["ids"]
    if ids:
        audio_collection.delete(ids=ids)
        logger.info("Deleted %d audio clip(s) (document_id=%r)", len(ids), document_id)

    from app.media.media_store import delete_media_for_document

    delete_media_for_document(document_id, owner)

    return total_deleted


def delete_repository(repository: str, owner: str) -> int:
    """
    Delete every chunk whose metadata `repository` matches the given
    "owner/repo" value (see app.ingestion.loader.load_github_repository)
    AND belongs to `owner` (the local account, not to be confused with
    the "owner" half of the GitHub "owner/repo" string). Returns the
    number of chunks removed.

    Only the "github" KB is checked, because that's the only KB
    GitHub-sourced chunks are ever written to (see module docstring's
    _SOURCE_TYPE_TO_KB); other KBs are guaranteed not to have a
    `repository` metadata key.
    """
    collection = _get_collection("github")
    existing = collection.get(where={"$and": [{"repository": repository}, {"owner": owner}]}, include=[])
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
        logger.info("Deleted %d chunk(s) for repository=%r owner=%r", len(ids), repository, owner)
        _invalidate_hybrid_cache("github")
    return len(ids)



def add_image_chunks(image_chunks: list[dict]) -> None:
    """
    Store a list of embedded images in the separate image collection.

    Args:
        image_chunks: list of dicts, each with "embedding" (a CLIP
            image embedding from app.embeddings.image_embedder),
            "content" (the image's alt text, kept as the stored
            document for citation/debugging), and "metadata" (must
            include "image_url" - used as the stable id so re-ingesting
            the same image upserts instead of duplicating - plus
            "alt_text", "page_url", "page_title").
    """
    if not image_chunks:
        return

    collection = _get_image_collection()

    # Chroma's upsert() rejects duplicate ids within a single call (it only
    # upserts-across-calls, not within one) - dedupe by (owner, image_url)
    # first, in case the same image URL appears more than once on the same
    # page (e.g. the same icon reused inline and in a gallery), keeping the
    # last one. Keyed on the PAIR, not bare image_url: the stored id is
    # itself owner-prefixed (see below), so two different owners saving the
    # same public image URL in the same batch call must both survive, not
    # silently collapse into whichever happened to be processed last.
    by_owner_url: dict[tuple[str, str], dict] = {
        (image_chunk["metadata"].get("owner", _DEFAULT_OWNER), image_chunk["metadata"]["image_url"]): image_chunk
        for image_chunk in image_chunks
    }

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for (owner, image_url), image_chunk in by_owner_url.items():
        # Owner-prefixed for the same collision reason as
        # add_embedded_chunks() - two owners saving the same public
        # image URL would otherwise overwrite each other's chunk.
        ids.append(f"{owner}::{image_url}")
        embeddings.append(image_chunk["embedding"])
        documents.append(image_chunk["content"])
        metadatas.append(_sanitize_metadata({**image_chunk["metadata"], "owner": owner}))

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info("Stored/updated %d image chunk(s) in collection '%s'", len(ids), IMAGE_COLLECTION_NAME)


def query_image_embedding(query_vector: list[float], top_k: int = 3, where: dict | None = None) -> list[dict]:
    """
    Find the top_k stored images most similar to a query embedding.

    query_vector MUST come from the same CLIP model used to embed the
    stored images (app.embeddings.image_embedder.embed_text_for_image_search),
    never from the text-only all-MiniLM-L6-l2 model used for
    query_embedding() above - the two live in unrelated vector spaces.

    where: optional Chroma metadata filter (e.g. {"source_kb": "video"})
    - see app.retrieval.retriever.retrieve_images for why this matters:
    without it, a question scoped to one KB tab (e.g. "video") would
    match against EVERY stored image regardless of origin (other
    videos, unrelated web screenshots), since this collection is
    shared across all image sources (see module docstring).

    Returns:
        Same shape as query_embedding(): a list of dicts with
        "content" (alt text), "metadata", "distance", and
        "similarity", ordered from most to least similar.
    """
    collection = _get_image_collection()

    results = collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)

    hits: list[dict] = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for content, metadata, distance in zip(documents, metadatas, distances):
        hits.append(
            {
                "content": content,
                "metadata": metadata,
                "distance": distance,
                "similarity": 1 - distance,
            }
        )

    logger.info("Image query returned %d result(s) for top_k=%d", len(hits), top_k)
    return hits


def image_count() -> int:
    """Return how many images are currently stored."""
    return _get_image_collection().count()


def add_audio_clip_chunks(clip_chunks: list[dict]) -> None:
    """
    Store a list of CLAP-embedded audio clips in the separate
    audio-clip collection.

    Args:
        clip_chunks: list of dicts, each with "embedding" (a CLAP
            audio embedding from app.embeddings.audio_embedder),
            "content" (a short label, kept as the stored document for
            citation/debugging), and "metadata" (must include
            "document_id" and "start_seconds" - used together as the
            stable id, since - unlike a web image's URL - an audio
            clip has no natural external id of its own; mirrors
            add_embedded_chunks()'s own "{document_id}::..." id
            convention rather than add_image_chunks()'s image_url-as-id
            convention).
    """
    if not clip_chunks:
        return

    collection = _get_audio_clip_collection()

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for clip_chunk in clip_chunks:
        metadata = clip_chunk["metadata"]
        owner = metadata.get("owner", _DEFAULT_OWNER)
        clip_id = f"{owner}::{metadata['document_id']}::clip_{metadata['start_seconds']}"
        ids.append(clip_id)
        embeddings.append(clip_chunk["embedding"])
        documents.append(clip_chunk["content"])
        metadatas.append(_sanitize_metadata({**metadata, "owner": owner}))

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info("Stored/updated %d audio clip(s) in collection '%s'", len(ids), AUDIO_CLIP_COLLECTION_NAME)


def query_audio_clip_embedding(query_vector: list[float], top_k: int = 3, where: dict | None = None) -> list[dict]:
    """
    Find the top_k stored audio clips most similar to a query
    embedding.

    query_vector MUST come from the same CLAP model used to embed the
    stored clips (app.embeddings.audio_embedder.embed_text_for_audio_search),
    never from the text-only all-MiniLM-L6-v2 model or the CLIP image
    model - all three live in unrelated vector spaces.

    where: optional Chroma metadata filter (e.g. {"source_kb": "video"}
    - see app.retrieval.retriever.retrieve_audio_clips). Without this,
    a question scoped to one KB tab would match against every clip
    from every ingested audio/video file, since this collection has no
    other notion of "which KB" a clip belongs to.

    Returns:
        Same shape as query_embedding()/query_image_embedding(): a
        list of dicts with "content", "metadata", "distance", and
        "similarity", ordered from most to least similar.
    """
    collection = _get_audio_clip_collection()

    results = collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)

    hits: list[dict] = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for content, metadata, distance in zip(documents, metadatas, distances):
        hits.append(
            {
                "content": content,
                "metadata": metadata,
                "distance": distance,
                "similarity": 1 - distance,
            }
        )

    logger.info("Audio clip query returned %d result(s) for top_k=%d", len(hits), top_k)
    return hits


def audio_clip_count() -> int:
    """Return how many audio clips are currently stored."""
    return _get_audio_clip_collection().count()


if __name__ == "__main__":
    from app.chunking.chunker import chunk_document
    from app.embeddings.embedder import embed_chunks
    from app.ingestion.loader import load_markdown_document

    doc = load_markdown_document("data/raw/README.md")
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=30)
    embedded_chunks = embed_chunks(chunks)

    add_embedded_chunks(embedded_chunks)
    print(f"Stored {len(embedded_chunks)} chunks. Collection now has {count()} total.")

    # Sanity check: embed a query and retrieve the closest chunks.
    from app.embeddings.embedder import embed_texts

    query_text = "What is this repository about?"
    query_vector = embed_texts([query_text])[0]
    results = query_embedding(query_vector, top_k=3)

    print(f"\nTop {len(results)} results for query: '{query_text}'")
    print("=" * 70)
    for i, hit in enumerate(results, start=1):
        section = hit["metadata"].get("section", "")
        print(f"\n#{i} | distance: {hit['distance']:.4f} | section: '{section}'")
        print("-" * 70)
        print(hit["content"][:300])

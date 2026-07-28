"""
Retrieval for the RAG pipeline.

Responsibility: given a user's natural-language question, find the
most relevant stored chunks. No LLM generation happens here — this
module only retrieves context.

Design:
- retrieve() is the single entry point other modules should use, so
  callers never need to know that retrieval is "embed the query, then
  search the vector store" under the hood.
- The query is embedded with the SAME model used for chunks
  (app.embeddings.embedder), because embeddings only make sense
  compared against other embeddings from that same model/vector space.
- Table completeness problem: app.chunking.chunker emits both a
  whole-table chunk AND one chunk per data row (Strategy 4). Individual
  row chunks compete with each other and with the whole-table chunk
  for the same top_k slots, so a table with more rows than top_k can
  easily surface only some of its rows (e.g. 2 of 4) instead of the
  complete table. _complete_partial_tables() fixes this after the
  similarity search: whenever a retrieved chunk is just one row of a
  table, it is swapped for that table's complete whole-table chunk
  (looked up via its shared table_id), so the LLM always sees every
  row of any table it was given even a partial glimpse of.
- Near-duplicate title problem: this corpus has dozens of
  similarly-worded page/section titles (e.g. "Assign X to Y when Z"
  automation rule pages), and the small text embedding model
  (all-MiniLM-L6-v2) can rank a short, keyword-dense but topically
  unrelated chunk above the chunk whose section heading is an almost
  exact lexical match for the query. _rerank_by_title_overlap() fixes
  this after the similarity search: it fetches a wider candidate pool
  than top_k, boosts each candidate's score by how many of the
  query's meaningful words also appear in that chunk's section/
  heading breadcrumb, then re-sorts and truncates to top_k - a cheap
  lexical signal that complements (rather than replaces) the dense
  embedding similarity.
"""

import logging
import math
import re

from app.embeddings.embedder import embed_texts
from app.embeddings.image_embedder import embed_text_for_image_search
from app.routing.router import classify_route
from app.vectorstore.store import (
    KB_NAMES,
    get_class_chunk,
    get_table_chunk,
    list_populated_kbs,
    query_embedding,
    query_image_embedding,
)

logger = logging.getLogger(__name__)

# Which stored content_types each non-"general" text route should be
# narrowed to (see app.routing.router's module docstring for what each
# route means). "general"/"image" are deliberately absent here:
# "general" means no content_type filter at all (today's default,
# unscoped behavior), and "image" isn't a text-collection filter at
# all - it's a signal for callers to ALSO query the separate CLIP
# image retriever (see retrieve_images() below), which retrieve()
# itself never touches.
_ROUTE_CONTENT_TYPES = {
    "code": ["class", "method", "function"],
    "table": ["table", "table_row"],
}

# The "code" content_type filter only ever matches chunks in the
# "github" KB (AST class/method/function chunking only happens for
# Python files loaded via app.ingestion.loader.load_github_repository
# - see app.chunking.code_chunker) - running it against any other KB
# would just waste a query on a filter that can never match there. The
# "table" filter has no such restriction: a table can appear in a
# markdown file, a PDF, a DOCX, or a GitHub repo's own docs, so it's
# applied to every KB actually being searched.
_ROUTE_CONTENT_TYPE_KB_RESTRICTION = {"code": "github"}


# How much weight the title/heading lexical-overlap boost gets relative
# to cosine similarity (which ranges roughly 0-1). Originally 0.25 (a
# small nudge), but real usage showed that natural-language queries
# wrapped around an exact title phrase (e.g. "give me the automation
# rule code for: <title>") dilute the embedding enough that the
# correct chunk's raw similarity drops well below unrelated chunks -
# a 0.25 boost wasn't enough to recover it. Weighting the lexical
# overlap equally with cosine similarity reliably promotes the correct
# chunk back into top_k without needing an exact-phrase-only query.
_TITLE_BOOST_WEIGHT = 1.0

# Generic "what is this repository/project about" style questions have
# no meaningful lexical overlap with any particular section heading (a
# repo's own name isn't literally "repository" or "project"), so
# _rerank_by_title_overlap's boost can't help them - and in practice
# the small text embedding model (all-MiniLM-L6-v2) can rank an
# unrelated file's boilerplate intro well above the README's actual
# project description for exactly this kind of vague query (measured
# directly on a real repo: query-vs-README-description cosine
# similarity was ~0.05, but query-vs-CHANGELOG's "this project adheres
# to Semantic Versioning" boilerplate was ~0.35 - the CHANGELOG text
# just happens to share more surface-level wording with a vague
# "project"/"repository" question despite being the wrong answer).
# Detecting this class of overview question and strongly preferring
# README chunks fixes that: the README is near-universally where a
# project's own description actually lives, so it should win over
# whatever else happens to embed closest to generic phrasing like
# "what is this about".
_OVERVIEW_QUERY_TOKENS = {"repository", "repo", "project", "overview", "purpose", "summary"}
_README_OVERVIEW_BOOST = 1.0

# On top of _README_OVERVIEW_BOOST (which prefers the README file over
# everything else), also prefer the README's own opening/top-level
# chunk over its subsections (e.g. "Getting Started", "Related
# Projects", "Acknowledgements"): a project's actual one-line
# description almost always lives in that opening chunk, right under
# the title, not in a subsection - and subsections can otherwise still
# win on raw similarity purely by chance (e.g. "Related Projects"
# lists other project names, which can embed closer to a vague
# "what is this project about" query than the real description does).
_README_INTRO_BOOST = 1.0

# Sized comfortably larger than the entire image collection (a few
# hundred images) so that query_image_embedding() effectively returns
# every image, letting the lexical-overlap rerank in
# _rerank_images_by_alt_text_overlap() consider the whole corpus
# instead of a similarity-based pool that can miss the correct image
# for longer, natural-language-wrapped queries (see retrieve_images()).
_ALL_IMAGES_POOL_SIZE = 10_000

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "on", "in",
    "for", "and", "or", "its", "it", "this", "that", "when", "if", "then",
    "with", "as", "by", "from", "at", "be", "been", "being", "has",
    "have", "had", "do", "does", "did", "not", "no", "so", "up", "out",
    "into", "than", "who", "what", "how",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase, alnum-only words with stopwords/single-letters dropped."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _STOPWORDS and len(word) > 1
    }


def _is_readme_chunk(hit: dict) -> bool:
    """Whether a hit's source file is a project README, by file name."""
    file_path = (hit["metadata"].get("file_path") or "").lower()
    file_name = file_path.rsplit("/", 1)[-1]
    return file_name in {"readme.md", "readme.rst", "readme.txt", "readme"}


def _is_readme_intro_chunk(hit: dict) -> bool:
    """
    Whether a hit is the README's opening/top-level chunk, identified
    by its section breadcrumb having no " > " (see _README_INTRO_BOOST
    above) - a subsection's breadcrumb looks like "python-dotenv >
    Getting Started", while the opening chunk's is just "python-dotenv".
    """
    return _is_readme_chunk(hit) and " > " not in (hit["metadata"].get("section") or "")


def _rerank_by_title_overlap(query_text: str, hits: list[dict], top_k: int) -> list[dict]:
    """
    Re-sort similarity-search hits by (similarity + a boost for how
    much the chunk's section/heading breadcrumb lexically overlaps the
    query), then keep only the top_k.

    hits is expected to be a wider candidate pool than top_k (see
    retrieve()), since the whole point is to let this promote a
    correct-but-lower-ranked chunk that plain embedding similarity
    pushed just outside the final top_k.

    Also strongly prefers README chunks for generic "what is this
    project/repository about" style questions (see
    _OVERVIEW_QUERY_TOKENS/_README_OVERVIEW_BOOST above) - a separate
    fix from the title-overlap boost, since these vague queries have
    no section heading to lexically match against.
    """
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return hits[:top_k]

    is_overview_query = bool(query_tokens & _OVERVIEW_QUERY_TOKENS)

    def score(hit: dict) -> float:
        similarity = hit.get("similarity") or 0.0
        section_tokens = _tokenize(hit["metadata"].get("section", ""))
        overlap = query_tokens & section_tokens
        boost = len(overlap) / len(query_tokens)
        readme_boost = 0.0
        if is_overview_query and _is_readme_chunk(hit):
            readme_boost = _README_OVERVIEW_BOOST
            if _is_readme_intro_chunk(hit):
                readme_boost += _README_INTRO_BOOST
        return similarity + _TITLE_BOOST_WEIGHT * boost + readme_boost

    return sorted(hits, key=score, reverse=True)[:top_k]


def _complete_partial_tables(hits: list[dict]) -> list[dict]:
    """
    Replace any retrieved single-row table chunk with its complete
    whole-table chunk, so the LLM never sees an arbitrary subset of a
    table's rows just because that's what fit in top_k.

    Rows belonging to a table whose whole-table chunk was already
    retrieved directly (or already substituted in) are dropped rather
    than duplicated, since the complete table already includes them.
    """
    already_complete_table_ids = {
        hit["metadata"].get("table_id")
        for hit in hits
        if hit["metadata"].get("content_type") == "table"
    }
    seen_table_ids = set(already_complete_table_ids)

    completed: list[dict] = []
    for hit in hits:
        metadata = hit["metadata"]
        table_id = metadata.get("table_id")

        if metadata.get("content_type") == "table_row" and table_id:
            if table_id in seen_table_ids:
                continue  # already represented by the complete table
            full_table = get_table_chunk(table_id)
            if full_table is not None:
                seen_table_ids.add(table_id)
                completed.append(full_table)
                continue

        completed.append(hit)

    return completed


def _build_where(repository: str | None, content_types: list[str] | None) -> dict | None:
    """
    Combine the repository scope (see retrieve()'s `repository` arg)
    and a route's content_type restriction (see _ROUTE_CONTENT_TYPES)
    into a single Chroma `where` filter, using `$and` only when both
    are present (Chroma's where DSL wants an explicit boolean operator
    for multi-condition filters, not an implicit AND via a plain
    multi-key dict).
    """
    conditions = []
    if repository:
        conditions.append({"repository": repository})
    if content_types:
        conditions.append({"content_type": {"$in": content_types}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _complete_partial_classes(hits: list[dict]) -> list[dict]:
    """
    Replace any retrieved single-method code chunk with its complete
    whole-class chunk, so the LLM sees the class's constructor, shared
    state, and sibling methods instead of just one method in isolation
    just because that's what fit in top_k.

    The code-chunking counterpart of _complete_partial_tables() (see
    that function's docstring and app.chunking.code_chunker's module
    docstring for the parent/child class-method design this mirrors).
    Methods belonging to a class whose whole-class chunk was already
    retrieved directly (or already substituted in) are dropped rather
    than duplicated, since the complete class already includes them.
    """
    already_complete_class_ids = {
        hit["metadata"].get("class_id")
        for hit in hits
        if hit["metadata"].get("content_type") == "class"
    }
    seen_class_ids = set(already_complete_class_ids)

    completed: list[dict] = []
    for hit in hits:
        metadata = hit["metadata"]
        class_id = metadata.get("class_id")

        if metadata.get("content_type") == "method" and class_id:
            if class_id in seen_class_ids:
                continue  # already represented by the complete class
            full_class = get_class_chunk(class_id)
            if full_class is not None:
                seen_class_ids.add(class_id)
                completed.append(full_class)
                continue

        completed.append(hit)

    return completed


def retrieve(query_text: str, top_k: int = 5, repository: str | None = None) -> list[dict]:
    """
    Find the top_k chunks most relevant to a user's question.

    Args:
        query_text: The user's natural-language question.
        top_k: How many chunks to retrieve.
        repository: Optional "owner/repo" filter (see
            app.ingestion.loader.load_github_repository's `repository`
            metadata). When given, only chunks from that repo are
            searched (repos only ever live in the "github" KB, but
            this filter still applies within it) - otherwise a
            question about a just-ingested repo could be answered from
            an unrelated, previously-ingested repo's chunks instead
            (whichever happen to rank highest). None (the default)
            searches every repo in whichever KB(s) are searched.

    Query routing (see app.routing.router) drives TWO independent
    decisions before any searching happens:
    1. WHICH KB(s) to search (decision.kbs) - an explicit source
       signal in the query (e.g. "in the PDF", "in the codebase") sets
       this; with no such signal, every KB that actually has data is
       searched (list_populated_kbs()) rather than every KB
       unconditionally, so a KB nothing was ever ingested into is
       skipped for free.
    2. WHICH content_type(s) to additionally filter for within each
       searched KB (decision.routes, e.g. "code", "table", plus always
       "general"). Each matched route runs as its OWN separate,
       content_type-scoped query_embedding() call per KB - a real,
       additional retriever, not just a rerank tweak - and every
       route's candidates, across every KB, are merged (deduped by
       document_id/chunk_index) before reranking. Because "general"
       (today's unscoped search) is always one of the routes, a wrong
       or low-confidence routing decision can only ADD extra,
       more-targeted candidates - it never removes anything the
       unscoped search within a searched KB would already have found.

    Returns:
        A list of dicts, each with "content", "metadata", "distance"
        (cosine distance, lower = more similar), and "similarity"
        (cosine similarity, higher = more similar), ordered from most
        to least similar. A retrieved table row may be replaced with
        its complete table (see _complete_partial_tables), in which
        case "distance"/"similarity" are None for that entry.
    """
    logger.info(
        "Retrieving top_k=%d chunks for query: %r (repository=%r)", top_k, query_text, repository
    )
    query_vector = embed_texts([query_text])[0]
    decision = classify_route(query_text)
    target_kbs = decision.kbs or list_populated_kbs()
    # Fetch a wider candidate pool than top_k so _rerank_by_title_overlap
    # has enough rank-adjacent candidates to promote a correct chunk
    # that plain embedding similarity ranked just outside top_k, rather
    # than only ever reranking within an already-truncated list. 25 was
    # too narrow in practice: a query phrased as a natural-language
    # question around an exact title (rather than the title alone) can
    # push the correct chunk's raw similarity rank down near ~30, so
    # the pool needs to be wide enough to still catch it.
    candidate_pool_size = max(top_k * 10, 50)

    seen_chunks: set[tuple] = set()
    candidates: list[dict] = []
    for kb in target_kbs:
        for route in decision.routes:
            content_types = _ROUTE_CONTENT_TYPES.get(route)
            if route != "general" and content_types is None:
                continue  # e.g. "image" - not a text-collection route, retrieve_images() handles it
            restricted_to_kb = _ROUTE_CONTENT_TYPE_KB_RESTRICTION.get(route)
            if restricted_to_kb is not None and restricted_to_kb != kb:
                continue  # e.g. "code" content_type only ever exists in the "github" KB
            where = _build_where(repository, content_types)
            for hit in query_embedding(query_vector, top_k=candidate_pool_size, where=where, kb=kb):
                document_id = hit["metadata"].get("document_id")
                # Only dedup when document_id is actually present - chunks
                # missing it (e.g. some synthetic/legacy metadata) would
                # otherwise all collapse onto the same (None, None) key and
                # wrongly get dropped as "duplicates" of one another.
                if document_id is not None:
                    dedup_key = (document_id, hit["metadata"].get("chunk_index"))
                    if dedup_key in seen_chunks:
                        continue
                    seen_chunks.add(dedup_key)
                candidates.append(hit)

    hits = _rerank_by_title_overlap(query_text, candidates, top_k)
    hits = _complete_partial_tables(hits)
    return _complete_partial_classes(hits)


def _rerank_images_by_alt_text_overlap(query_text: str, hits: list[dict], top_k: int) -> list[dict]:
    """
    Same fix as _rerank_by_title_overlap(), applied to images instead
    of text chunks: re-sort CLIP similarity-search hits by (similarity
    + a boost for how much the image's alt_text lexically overlaps the
    query), then keep only the top_k.

    CLIP maps images and text into the same space by VISUAL content,
    which works well for "show me a picture of X" queries but poorly
    here - most of this corpus's images are visually similar
    documentation screenshots (a light-themed web form with WHEN/AND/
    THEN rows), so CLIP alone can't tell apart which specific
    automation-rule screenshot a question is about; the ground-truth
    image for a query can rank outside the top 50 on visual similarity
    alone. Each image DOES have a descriptive alt_text captured at
    ingestion time that is often an almost exact lexical match for the
    section/question it illustrates (see app.ingestion.ingest), so the
    same lexical-overlap boost that fixed text retrieval applies here
    too.

    Overlap is weighted by IDF (inverse document frequency), not a
    flat fraction: this corpus has many near-duplicate alt_texts like
    "executing javascript to assign the people working on a Feature to
    a newly created User Story" that differ from each other ONLY in
    their entity names (Feature/Epic/Task/Bug/User Story). A flat
    "fraction of query tokens present" score can't tell these apart -
    it counts common filler words (executing, javascript, assign,
    people, working, created, newly) the same as the words that
    actually distinguish one screenshot from another, and ends up
    promoting whichever near-duplicate happens to share the most
    filler words with the query rather than the one whose entity names
    actually match. Weighting each shared token by how rare it is
    across this image collection's alt_texts fixes that: filler words
    that appear in dozens of alt_texts contribute almost nothing,
    while a distinguishing word like "task" or "epic" that appears in
    only a few contributes much more.
    """
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return hits[:top_k]

    alt_token_sets = [_tokenize(hit["metadata"].get("alt_text", "")) for hit in hits]
    doc_count = len(alt_token_sets) or 1
    doc_freq: dict[str, int] = {}
    for tokens in alt_token_sets:
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    def idf(token: str) -> float:
        return math.log((doc_count + 1) / (doc_freq.get(token, 0) + 1)) + 1.0

    query_weight_total = sum(idf(token) for token in query_tokens) or 1.0

    def score(hit: dict, alt_tokens: set[str]) -> float:
        similarity = hit.get("similarity") or 0.0
        overlap = query_tokens & alt_tokens
        boost = sum(idf(token) for token in overlap) / query_weight_total
        return similarity + _TITLE_BOOST_WEIGHT * boost

    scored = [(score(hit, alt_tokens), hit) for hit, alt_tokens in zip(hits, alt_token_sets)]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [hit for _, hit in scored[:top_k]]


def retrieve_images(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Find the top_k stored images most relevant to a user's question,
    using the multimodal CLIP model (app.embeddings.image_embedder)
    rather than the text-only all-MiniLM-L6-v2 model retrieve() above
    uses - CLIP is what makes matching a text query against IMAGE
    embeddings possible at all, since both are mapped into the same
    vector space (see image_embedder module docstring).

    This is always an ADDITIONAL, parallel lookup alongside retrieve()
    - never a replacement for it - since retrieve() searches text
    chunks and this searches images, two separate collections.

    Args:
        query_text: The user's natural-language question.
        top_k: How many images to retrieve.

    Returns:
        A list of dicts, each with "content" (the image's alt text),
        "metadata" (image_url, page_url, alt_text, page_title),
        "distance", and "similarity", ordered from most to least
        similar. Empty list if no images have been ingested yet.
    """
    logger.info("Retrieving top_k=%d image(s) for query: %r", top_k, query_text)
    query_vector = embed_text_for_image_search(query_text)
    # Same wider-candidate-pool + lexical-boost-rerank fix as retrieve()
    # above, applied to images (see _rerank_images_by_alt_text_overlap).
    # Needs a much wider pool than the text version's 50: most of this
    # corpus's images are visually similar documentation screenshots,
    # so CLIP similarity alone is a much weaker/flatter signal here
    # than dense text embedding similarity is for text chunks - in
    # testing, a correct image's raw CLIP rank was as deep as ~130 out
    # of ~480 total images for a bare query, and even ~320 out of ~480
    # for a longer, natural-language-wrapped query (the extra wrapper
    # words dilute the CLIP embedding further). A fixed pool size of a
    # few hundred can still miss it, so - since the whole image
    # collection is small (low hundreds) - just fetch every image and
    # let the lexical-overlap rerank below sort out relevance; this
    # sidesteps having to guess a pool size that works for every
    # possible query phrasing.
    candidates = query_image_embedding(query_vector, top_k=_ALL_IMAGES_POOL_SIZE)
    return _rerank_images_by_alt_text_overlap(query_text, candidates, top_k)


if __name__ == "__main__":
    import sys

    # Usage: python -m app.retrieval.retriever "your question here" [top_k]
    query_text = sys.argv[1] if len(sys.argv) > 1 else "What is this repository about?"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    results = retrieve(query_text, top_k=top_k)

    print(f"Query: '{query_text}'")
    print(f"Top {len(results)} retrieved chunks:")
    print("=" * 70)
    for i, hit in enumerate(results, start=1):
        section = hit["metadata"].get("section", "")
        if hit["similarity"] is None:
            score = "N/A (completed table)"
        else:
            score = f"similarity: {hit['similarity']:.4f} (distance: {hit['distance']:.4f})"
        print(f"\n#{i} | {score} | section: '{section}'")
        print("-" * 70)
        print(hit["content"])

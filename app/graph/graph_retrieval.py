"""
Graph-based retrieval for the "structural" GitHub intent.

Answers questions the vector store can't answer well on its own -
"what calls X", "who imports Y", "subclasses of Z" - by walking the
per-repo code graph built by app.graph.code_graph, then materializing
the touched nodes back into Chroma chunks the cross-encoder reranker
already knows how to score.

Design:
- No new embedding model, no new store. This module reuses the
  existing "github" collection and the same cross-encoder rerank
  path everything else uses, so answer quality above graph-traversal
  is unchanged - GraphRAG only changes which chunks arrive at the
  reranker's shortlist.
- Every graph miss (no graph file, no matching node, no directional
  keyword, no symbol_name in the query) returns an empty list, and
  the caller (app.retrieval.github_adaptive) then falls back to
  hybrid. Structural retrieval is a boost, never a hard requirement.
- Directions are inferred from a small keyword table, not a router.
  The intent classifier already labels the query "structural"; the
  only remaining question is which way to walk (callers vs callees,
  importers vs imported, subclasses vs bases). Keeping that as a
  local keyword table (rather than pushing it into
  github_intent.py's semantic classifier) means one file owns
  "what does structural retrieval mean" end to end.
"""

from __future__ import annotations

import logging

import networkx as nx

from app.graph.code_graph import (
    EDGE_CALLS,
    EDGE_IMPORTS,
    EDGE_INHERITS,
    NODE_CLASS,
    NODE_EXTERNAL,
    NODE_FUNCTION,
    NODE_METHOD,
    load_repository_graph,
)
from app.vectorstore.store import _get_collection

logger = logging.getLogger(__name__)

_KB = "github"


# Keyword -> (edge_kind, direction) mapping. "in" means the query
# asks about EDGES POINTING INTO the seed node (predecessors);
# "out" means edges going OUT (successors). Both directions are
# tried when the query is ambiguous (see _resolve_directions).
_DIRECTION_KEYWORDS: dict[str, tuple[str, str]] = {
    # calls
    "who calls":         (EDGE_CALLS, "in"),
    "what calls":        (EDGE_CALLS, "in"),
    "callers of":        (EDGE_CALLS, "in"),
    "callers":           (EDGE_CALLS, "in"),
    "used by":           (EDGE_CALLS, "in"),
    "who uses":          (EDGE_CALLS, "in"),
    "what does":         (EDGE_CALLS, "out"),   # "what does X call" - refined below
    "calls to":          (EDGE_CALLS, "out"),
    "calls from":        (EDGE_CALLS, "out"),
    "callees of":        (EDGE_CALLS, "out"),
    "dependencies of":   (EDGE_CALLS, "out"),
    "depends on":        (EDGE_CALLS, "out"),
    # imports
    "who imports":       (EDGE_IMPORTS, "in"),
    "importers of":      (EDGE_IMPORTS, "in"),
    "imports of":        (EDGE_IMPORTS, "out"),
    "what does it import":  (EDGE_IMPORTS, "out"),
    "what imports":      (EDGE_IMPORTS, "in"),
    # inheritance
    "subclasses of":     (EDGE_INHERITS, "in"),
    "who extends":       (EDGE_INHERITS, "in"),
    "who inherits":      (EDGE_INHERITS, "in"),
    "extends":           (EDGE_INHERITS, "out"),
    "inherits from":     (EDGE_INHERITS, "out"),
    "base classes of":   (EDGE_INHERITS, "out"),
    "parent class of":   (EDGE_INHERITS, "out"),
}


def _resolve_directions(query_text: str) -> list[tuple[str, str]]:
    """
    Every (edge_kind, direction) the query's phrasing suggests,
    longest-match-first so "what does X call" beats a bare "what
    does".

    Returning a list rather than a single winner lets an ambiguous
    query ("show me the relationships around X") pull chunks from
    both sides - the reranker then sorts them by textual relevance,
    so nothing is lost by being liberal here.

    If nothing matches, we default to "callers on any known edge"
    (both directions, calls only) - covers the most common case
    where the user just names a symbol without a preposition
    ("Router", "retrieve_hybrid?"), matching how they'd read a
    call site in a code review.
    """
    lower = query_text.lower()
    matches: list[tuple[str, str]] = []
    for phrase, direction in sorted(_DIRECTION_KEYWORDS.items(), key=lambda kv: -len(kv[0])):
        if phrase in lower:
            matches.append(direction)
    if matches:
        # dedupe while preserving order (Python 3.7+ dict is ordered)
        return list(dict.fromkeys(matches))
    return [(EDGE_CALLS, "in"), (EDGE_CALLS, "out")]


def _matches_symbol(node_id: str, node_data: dict, symbol: str) -> bool:
    """
    True when a graph node's own name (or leaf name of its dotted
    form) matches the symbol the user named.

    Match rules, in order:
      - internal node's `name` equals symbol (class/function/method
        with an exact match; case-insensitive because Python is
        case-sensitive but users often lowercase in prose)
      - internal node's `qualname` ends with ".symbol" (a method
        named symbol on any class)
      - external node's `name` equals symbol OR ends with ".symbol"
        (matches call/import references written as "module.symbol"
        without needing to resolve the module)
    """
    kind = node_data.get("kind")
    lower_symbol = symbol.lower()

    if kind in (NODE_CLASS, NODE_FUNCTION, NODE_METHOD):
        if str(node_data.get("name", "")).lower() == lower_symbol:
            return True
        qualname = str(node_data.get("qualname", "")).lower()
        if qualname == lower_symbol or qualname.endswith(f".{lower_symbol}"):
            return True
        return False

    if kind == NODE_EXTERNAL:
        name = str(node_data.get("name", "")).lower()
        return name == lower_symbol or name.endswith(f".{lower_symbol}")

    return False


def _seed_nodes(graph: nx.MultiDiGraph, symbol: str) -> list[str]:
    """Every node whose name matches the queried symbol.

    Returns BOTH internal and matching external nodes together, in
    that order. Rationale: the AST call collector records call
    targets by their dotted source-text name, so a call site
    "Router().classify(x)" writes an edge into external:classify
    (not into the internal Router.classify node - resolving that
    would need semantic type inference we deliberately avoid). For
    "who calls classify" to work, the walker needs a seed on the
    external node too, so the caller edge is actually reachable.
    _nodes_to_chunks already handles the mix cleanly (external nodes
    have no document_id and are skipped there).
    """
    internal: list[str] = []
    external: list[str] = []
    for node_id, data in graph.nodes(data=True):
        if _matches_symbol(node_id, data, symbol):
            if data.get("kind") == NODE_EXTERNAL:
                external.append(node_id)
            else:
                internal.append(node_id)
    # Internal first (preferred for "what does X call" - out-edges
    # only exist on the internal definition), external appended so
    # "who calls X" walks pick up the call-target references too.
    return internal + external


def _walk(
    graph: nx.MultiDiGraph,
    seed: str,
    edge_kind: str,
    direction: str,
) -> list[str]:
    """
    One-hop traversal on edges of a given kind from a seed node.

    In (predecessors): every u such that (u -> seed) has edge_kind.
    Out (successors):  every v such that (seed -> v) has edge_kind.

    Deliberately one hop, not transitive. Two-hop expansion balloons
    the shortlist for a typical Python file (a Router class calls
    ~15 things, each of which calls ~15 more) and dilutes the
    reranker's signal. A future "depth" parameter could raise this,
    but one hop already answers the "who calls X" / "what X calls"
    class of question correctly.
    """
    if seed not in graph:
        return []
    neighbors: list[str] = []
    if direction == "in":
        for u, _v, data in graph.in_edges(seed, data=True):
            if data.get("kind") == edge_kind:
                neighbors.append(u)
    else:
        for _u, v, data in graph.out_edges(seed, data=True):
            if data.get("kind") == edge_kind:
                neighbors.append(v)
    return neighbors


def _nodes_to_chunks(graph: nx.MultiDiGraph, node_ids: list[str], owner: str) -> list[dict]:
    """
    Turn graph nodes back into Chroma chunks.

    An internal node carries `document_id` (see code_graph._file_node)
    matching the exact same document_id load_github_repository wrote
    on each file's chunks. We fetch every chunk with that document_id
    - the parent class chunk, its per-method children, and any
    module-level leftover - and let the cross-encoder rerank them by
    textual relevance. External nodes have no chunk to fetch and are
    silently skipped: the reranker sees only real repo content. owner
    is checked alongside document_id (the graph itself is already
    loaded from an owner-namespaced file - see code_graph._graph_path
    - but this chunk lookup is independently scoped too, defense in
    depth rather than relying solely on the caller having loaded the
    right graph).

    Deduplicated by document_id so many nodes in the same file don't
    each pull the same chunks multiple times.
    """
    seen_documents: set[str] = set()
    hits: list[dict] = []
    collection = _get_collection(_KB)

    for node_id in node_ids:
        if node_id not in graph:
            continue
        data = graph.nodes[node_id]
        document_id = data.get("document_id")
        if not document_id or document_id in seen_documents:
            continue
        seen_documents.add(document_id)

        result = collection.get(where={"$and": [{"document_id": document_id}, {"owner": owner}]})
        for chunk_id, content, metadata in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            hits.append(
                {
                    "id": chunk_id,
                    "content": content,
                    "metadata": metadata,
                    # No distance/similarity - graph retrieval isn't a
                    # similarity search. The cross-encoder rerank
                    # downstream will assign a real score.
                    "distance": None,
                    "similarity": None,
                }
            )
    return hits


def retrieve_by_graph(
    query_text: str,
    symbol_name: str,
    repository: str,
    owner: str,
) -> list[dict]:
    """
    Full graph-retrieval path: load graph -> find seeds -> walk in
    every direction the query implies -> materialize touched nodes
    into chunks.

    Returns [] on any of: no graph on disk, no matching seed node,
    every walk empty, no touched node had a document_id. The caller
    (app.retrieval.github_adaptive) is expected to detect the empty
    result and fall back to hybrid.

    Args:
        query_text: The raw user question, used only to pick edge
            directions via keyword match (see _resolve_directions).
            The symbol itself has already been extracted by the
            intent classifier and is passed separately.
        symbol_name: The identifier hint from
            app.routing.github_intent.
        repository: "owner/repo" to look up which graph file to
            load. Required - a graph without a repo scope is
            meaningless (a symbol like "run" could match dozens
            of repos).
        owner: Local account that ingested this repository - selects
            which owner-namespaced graph file to load (see
            code_graph._graph_path) so one account's structural
            queries can never traverse a graph a different account
            privately built for a same-named repository.

    Returns:
        A list of chunk dicts in the standard retrieve() shape (id,
        content, metadata, distance=None, similarity=None), NOT yet
        reranked - the caller is responsible for rerank + top_k
        selection so the same reranker settings are used everywhere.
    """
    graph = load_repository_graph(repository, owner)
    if graph is None:
        logger.info(
            "No graph on disk for '%s' - structural retrieval falling back.", repository
        )
        return []

    seeds = _seed_nodes(graph, symbol_name)
    if not seeds:
        logger.info(
            "Symbol %r not found in graph for '%s' - structural retrieval falling back.",
            symbol_name,
            repository,
        )
        return []

    directions = _resolve_directions(query_text)
    touched: list[str] = list(seeds)  # include the seed itself so
                                      # the definition chunk is
                                      # always in the shortlist
    for seed in seeds:
        for edge_kind, direction in directions:
            touched.extend(_walk(graph, seed, edge_kind, direction))

    hits = _nodes_to_chunks(graph, touched, owner)
    logger.info(
        "Structural retrieval for symbol=%r on '%s': %d seed(s), %d touched node(s), %d chunk(s).",
        symbol_name,
        repository,
        len(seeds),
        len(touched),
        len(hits),
    )
    return hits

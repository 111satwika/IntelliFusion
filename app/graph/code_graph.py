"""
Build and persist a lightweight code graph (calls / imports /
inherits / contains) for one GitHub repository, from the SAME AST
parses the code chunker already runs on every Python file.

Design notes:
- Networkx MultiDiGraph, not Neo4j. This is a single-process Streamlit
  app and the graph fits comfortably in memory (<< 1 MB even for
  large repos) - a full graph database would add operational weight
  for no functional gain, matching the project's "regenerable via
  re-ingestion" philosophy (see .gitignore's data/chroma_db/ note).
- Pickled to data/graphs/{owner}__{repo}.gpickle, one graph per repo,
  so a query for one repo can't be polluted by another repo's nodes.
  If the file is missing (repo ingested BEFORE GraphRAG was added, or
  never ingested at all), retrieval falls back to hybrid instead of
  erroring - same "one missing capability shouldn't break the answer"
  pattern used everywhere else.
- Python-only. The stdlib `ast` module only knows Python; every other
  language ingest supports (JS/TS/Java/Go/...) is silently skipped
  from graph-building, exactly like app.chunking.code_chunker's
  fallback to the generic chunker for non-Python files. A future
  Tree-sitter integration would slot in here without changing the
  retrieval side (see graph_retrieval.py).
- "calls" edges are best-effort NAME-BASED resolution, not full
  semantic type resolution: we record the callee's dotted name as it
  appears in source (e.g. "self.method", "Class.method",
  "module.func", "bare_name"), and match by leaf name at query time.
  A real semantic call graph would need a proper type inferrer
  (e.g. Jedi, Pyright) - overkill for retrieval seeding, where
  "find me anything named foo that this file calls" is already
  enough to bias the reranker.
"""

from __future__ import annotations

import ast
import logging
import pickle
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

# Where per-repo graphs live. Under data/, which is .gitignored - a
# graph is regenerable by re-running ingestion, so committing it
# would just be a stale copy.
_GRAPH_DIR = Path("data/graphs")


# Node "kind" values. Kept as string constants (not an Enum) to stay
# pickle-friendly across code changes: renaming an Enum member would
# invalidate every already-pickled graph, whereas string values
# remain readable/portable if someone opens a graph file directly.
NODE_FILE = "file"
NODE_CLASS = "class"
NODE_FUNCTION = "function"
NODE_METHOD = "method"
NODE_EXTERNAL = "external"  # A referenced name (import target,
                            # inheritance base, call target) that
                            # isn't itself defined in this repo -
                            # kept as a node so edges still terminate
                            # somewhere and traversal doesn't need
                            # special "unresolved" handling.

# Edge "kind" values (stored on each edge under key "kind"). Same
# rationale as node kinds - string constants for pickle stability.
EDGE_CONTAINS = "contains"    # file -> class, class -> method, file -> function
EDGE_IMPORTS = "imports"      # file -> module_name (external node)
EDGE_CALLS = "calls"          # function/method -> called-name (may be
                              # internal or external)
EDGE_INHERITS = "inherits"    # class -> base-class-name

# All edge kinds, useful for callers that want to enumerate.
EDGE_KINDS = (EDGE_CONTAINS, EDGE_IMPORTS, EDGE_CALLS, EDGE_INHERITS)


def _graph_path(owner: str, repository: str) -> Path:
    """Filesystem path for a repository's pickled graph.

    Namespaced under a per-account-owner subdirectory (not to be
    confused with `repository`'s own "owner/repo" GitHub owner) so two
    different local accounts that each ingest a repository of the same
    name get fully independent graphs - one account's structural
    ("who calls X") queries must never traverse a graph another
    account privately built. "owner/repo" has a slash; the on-disk
    name uses '__' so it's a valid single filename on Windows/Linux/
    macOS without needing further nesting.
    """
    safe = repository.replace("/", "__")
    return _GRAPH_DIR / owner / f"{safe}.gpickle"


def _file_node(repository: str, file_path: str) -> str:
    """Stable node id for a file.

    Format matches load_github_repository's document_id
    ("owner/repo:path") so a node can be mapped 1:1 to the Chroma
    chunk that holds that file's content - see
    graph_retrieval._nodes_to_chunks.
    """
    return f"{repository}:{file_path}"


def _class_node(repository: str, file_path: str, class_name: str) -> str:
    """Stable node id for a class definition."""
    return f"{repository}:{file_path}::{class_name}"


def _method_node(
    repository: str, file_path: str, class_name: str, method_name: str
) -> str:
    """Stable node id for a method (dotted qualname under its class)."""
    return f"{repository}:{file_path}::{class_name}.{method_name}"


def _function_node(repository: str, file_path: str, function_name: str) -> str:
    """Stable node id for a top-level function."""
    return f"{repository}:{file_path}::{function_name}"


def _external_node(name: str) -> str:
    """
    Stable node id for a referenced name we don't (or can't) resolve
    to a definition in this repo - an imported third-party module, a
    base class from another package, a called function whose
    definition wasn't parsed. Prefixed with "external:" so it can't
    collide with any real repo node id.
    """
    return f"external:{name}"


class _CallCollector(ast.NodeVisitor):
    """
    Walk a function/method body and record every Call node's callee
    as a dotted string.

    Chosen shapes:
      foo()             -> "foo"
      self.method()     -> "self.method"
      cls.method()      -> "cls.method"
      SomeClass.method()-> "SomeClass.method"
      module.func()     -> "module.func"
      pkg.sub.func()    -> "pkg.sub.func"
      thing()()         -> skipped (not a name we can match on)

    Kept intentionally shallow. A retrieval seed doesn't need full
    semantic resolution to be useful; matching by leaf name against
    the graph's other nodes gets us the right chunk 90% of the time
    (see graph_retrieval._matches_symbol).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _dotted(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._dotted(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return None

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 (AST convention)
        dotted = self._dotted(node.func)
        if dotted:
            self.calls.append(dotted)
        # Walk into arguments so nested calls (e.g. foo(bar())) both
        # register - matches how a reader thinks about "what does this
        # function call?".
        self.generic_visit(node)


def _extract_imports(tree: ast.Module) -> list[str]:
    """
    Every top-level import as a dotted module name.

    Handles both `import x.y` (records "x.y") and `from x.y import z`
    (records "x.y" - the module, not each imported symbol; the symbol
    itself already surfaces as a Name/Attribute reference wherever
    it's used, which the call collector will pick up).
    Relative imports ("from . import foo") record just "." so they're
    distinguishable from a real module named "foo" and don't
    accidentally match an unrelated node.
    """
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or ".")
    return modules


def _extract_base_names(class_node: ast.ClassDef) -> list[str]:
    """Every base class as a dotted string, same convention as _CallCollector."""
    collector = _CallCollector()
    bases: list[str] = []
    for base in class_node.bases:
        name = collector._dotted(base)
        if name:
            bases.append(name)
    return bases


def _add_python_file(
    graph: nx.MultiDiGraph,
    repository: str,
    file_path: str,
    source: str,
) -> None:
    """
    Add every graph-relevant node/edge from one Python source file.

    Silently skips files that don't parse (Python 2 source, template
    with a misleading .py extension, genuinely broken file) - same
    resilience pattern app.chunking.code_chunker uses when it hits an
    unparseable file. A single unparseable file must not abort the
    whole repo's graph build.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        logger.warning(
            "Skipping unparseable Python file '%s' in graph build (%s).",
            file_path,
            error,
        )
        return

    file_id = _file_node(repository, file_path)
    graph.add_node(
        file_id,
        kind=NODE_FILE,
        repository=repository,
        file_path=file_path,
        # document_id matches load_github_repository's chunk metadata
        # so graph_retrieval._nodes_to_chunks can look up chunks
        # without needing to reconstruct the id.
        document_id=file_id,
    )

    for module in _extract_imports(tree):
        module_node = _external_node(module)
        graph.add_node(module_node, kind=NODE_EXTERNAL, name=module)
        graph.add_edge(file_id, module_node, kind=EDGE_IMPORTS)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_id = _class_node(repository, file_path, node.name)
            graph.add_node(
                class_id,
                kind=NODE_CLASS,
                repository=repository,
                file_path=file_path,
                name=node.name,
                document_id=file_id,
            )
            graph.add_edge(file_id, class_id, kind=EDGE_CONTAINS)

            for base_name in _extract_base_names(node):
                base_id = _external_node(base_name)
                graph.add_node(base_id, kind=NODE_EXTERNAL, name=base_name)
                graph.add_edge(class_id, base_id, kind=EDGE_INHERITS)

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_id = _method_node(repository, file_path, node.name, child.name)
                    graph.add_node(
                        method_id,
                        kind=NODE_METHOD,
                        repository=repository,
                        file_path=file_path,
                        name=child.name,
                        qualname=f"{node.name}.{child.name}",
                        class_name=node.name,
                        document_id=file_id,
                    )
                    graph.add_edge(class_id, method_id, kind=EDGE_CONTAINS)
                    _record_calls_from(graph, method_id, child)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_id = _function_node(repository, file_path, node.name)
            graph.add_node(
                function_id,
                kind=NODE_FUNCTION,
                repository=repository,
                file_path=file_path,
                name=node.name,
                qualname=node.name,
                document_id=file_id,
            )
            graph.add_edge(file_id, function_id, kind=EDGE_CONTAINS)
            _record_calls_from(graph, function_id, node)


def _record_calls_from(
    graph: nx.MultiDiGraph,
    caller_id: str,
    body_node: ast.AST,
) -> None:
    """Add one CALLS edge per Call site in body_node.

    Callees are stored as external nodes for now, keyed by their
    dotted name. graph_retrieval resolves those to internal
    class/method/function nodes at query time by leaf-name match,
    so cross-file calls still traverse correctly even though we never
    do full type resolution at build time.
    """
    collector = _CallCollector()
    collector.visit(body_node)
    for call_target in collector.calls:
        target_id = _external_node(call_target)
        graph.add_node(target_id, kind=NODE_EXTERNAL, name=call_target)
        graph.add_edge(caller_id, target_id, kind=EDGE_CALLS)


def build_repository_graph(
    repository: str,
    files: list[tuple[str, str]],
) -> nx.MultiDiGraph:
    """
    Build a fresh graph for one repository from an iterable of
    (file_path, source_text) pairs.

    Non-Python files are silently skipped (they still get chunked and
    embedded upstream - see app.chunking.chunker.chunk_document - so
    they're findable via hybrid; they just don't contribute to the
    structural traversal).

    Args:
        repository: "owner/repo", matches chunk metadata.
        files: (path, decoded_utf8_source) pairs from
            load_github_repository's Documents.

    Returns:
        A freshly-built MultiDiGraph. Not automatically saved - the
        caller (see save_repository_graph) decides when to persist,
        so a failed ingest can't leave a corrupted graph on disk
        that would be loaded by future queries.
    """
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    graph.graph["repository"] = repository

    for file_path, source in files:
        if not file_path.endswith(".py"):
            continue
        _add_python_file(graph, repository, file_path, source)

    logger.info(
        "Built graph for '%s': %d node(s), %d edge(s).",
        repository,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def save_repository_graph(graph: nx.MultiDiGraph, repository: str, owner: str) -> Path:
    """Persist a repository's graph to data/graphs/{owner}/. Creates
    the directory if needed. Returns the path written."""
    path = _graph_path(owner, repository)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(graph, handle)
    logger.info("Saved graph for '%s' (owner=%r) -> %s", repository, owner, path)
    return path


def load_repository_graph(repository: str, owner: str) -> nx.MultiDiGraph | None:
    """
    Load a repository's graph from disk, or None if it hasn't been
    built yet (by this owner).

    Returning None (rather than raising) is deliberate: the caller in
    graph_retrieval treats absence as "fall back to hybrid", which is
    the same behavior as an empty result - the user still gets an
    answer, just without the structural boost.
    """
    path = _graph_path(owner, repository)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, AttributeError) as error:
        # A corrupted pickle (partial write, pickle format changed
        # between graph builds, etc.) is treated the same as a
        # missing graph - fall back to hybrid rather than crashing
        # the query. A subsequent re-ingestion will rewrite it.
        logger.warning(
            "Could not load graph for '%s' (%s) - falling back to hybrid.",
            repository,
            error,
        )
        return None

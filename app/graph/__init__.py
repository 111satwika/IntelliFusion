"""
GraphRAG for the GitHub KB.

app.graph.code_graph builds a per-repository call/import/inherit
graph from the same AST parses the code chunker already runs, stores
it under data/graphs/ (regenerable via re-ingestion, so it lives
under .gitignore like the vector store), and exposes lookup helpers
by symbol name.

app.graph.graph_retrieval is the retrieval strategy the "structural"
intent dispatches to (see app.retrieval.github_adaptive):
    query -> extract symbol_name (already done by the intent
    classifier) -> find matching graph nodes -> traverse callers /
    callees / imports / subclasses -> materialize the touched nodes
    back into their Chroma chunks -> cross-encoder rerank.

Python-only today (mirroring app.chunking.code_chunker's scope), with
a hybrid fallback for every other language.
"""

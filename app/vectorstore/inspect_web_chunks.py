"""
One-off helper to print only the chunks that came from crawled
websites (source_type == "web", e.g. the IBM Docs pages), for manual
inspection. Not part of the pipeline.

Run with:
    python -m app.vectorstore.inspect_web_chunks
    python -m app.vectorstore.inspect_web_chunks "authentication"
        (optional argv[1]: only print chunks whose url or file_name
        contains this substring, case-insensitive)
"""

import sys

from app.vectorstore.store import _get_collection

if __name__ == "__main__":
    filter_text = sys.argv[1].lower() if len(sys.argv) > 1 else None

    collection = _get_collection("web")
    data = collection.get(where={"source_type": "web"}, include=["documents", "metadatas"])

    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    rows = list(zip(ids, documents, metadatas))
    if filter_text:
        rows = [
            (chunk_id, doc, meta)
            for chunk_id, doc, meta in rows
            if filter_text in meta.get("url", "").lower() or filter_text in meta.get("file_name", "").lower()
        ]

    print(f"{len(rows)} website chunk(s) found" + (f" matching '{filter_text}'" if filter_text else "") + ".\n")

    for i, (chunk_id, doc, meta) in enumerate(rows):
        print("=" * 70)
        print(f"#{i} | id: {chunk_id}")
        print(f"page: {meta.get('file_name')}  |  url: {meta.get('url')}")
        print(f"section: {meta.get('section')}  |  content_type: {meta.get('content_type')}")
        print("-" * 70)
        print(doc)
        print()

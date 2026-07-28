"""
One-off helper to print everything currently stored in the Chroma
vector store, for manual inspection. Not part of the pipeline.
"""

from app.vectorstore.store import KB_NAMES, _get_collection

if __name__ == "__main__":
    for kb in KB_NAMES:
        collection = _get_collection(kb)
        # embeddings are excluded by default; include them explicitly.
        data = collection.get(include=["documents", "metadatas", "embeddings"])

        ids = data["ids"]
        documents = data["documents"]
        metadatas = data["metadatas"]
        embeddings = data["embeddings"]

        print(f"\n{'#' * 70}\nKB '{kb}' has {len(ids)} stored chunks.\n{'#' * 70}\n")

        for i, (chunk_id, doc, meta, embedding) in enumerate(
            zip(ids, documents, metadatas, embeddings)
        ):
            print("=" * 70)
            print(f"#{i} | id: {chunk_id}")
            print(f"metadata: {meta}")
            print(f"embedding: dimension={len(embedding)}, first 5 values={list(embedding[:5])}")
            print("-" * 70)
            print(doc)
            print()


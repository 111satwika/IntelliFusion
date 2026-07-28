# Version 1 — Basic RAG

## 1. Architecture

Six components, each a separate concern (so later versions can swap pieces without rewriting everything):

```
┌─────────────────┐
│ Document Loader   │  reads README.md → raw text
└────────┬─────────┘
         ▼
┌─────────────────┐
│ Chunker           │  splits text into overlapping chunks
└────────┬─────────┘
         ▼
┌─────────────────┐
│ Embedder          │  text chunk → vector (list of floats)
└────────┬─────────┘
         ▼
┌─────────────────┐
│ Vector Store       │  Chroma (persistent, cosine distance) — see note below
└────────┬─────────┘
         ▼
┌─────────────────┐          ┌──────────────┐
│ Retriever          │◄───────┤ Query Embedder│
│ (Chroma cosine sim)│          └──────────────┘
└────────┬─────────┘
         ▼
┌─────────────────┐
│ Prompt Builder     │  injects top-K chunks as context
└────────┬─────────┘
         ▼
┌─────────────────┐
│ LLM Generator      │  produces grounded answer
└─────────────────┘
```

Two distinct pipelines share the Embedder: an **ingestion pipeline** (build the index once) and a **query pipeline** (run every time a question is asked).

> Note: this diagram originally planned a plain in-memory list/JSON store
> for V1, with a real vector database arriving in V2. In practice, Chroma
> (a real embedded vector database) was adopted from the start — see
> `app/vectorstore/store.py`'s docstring for why. Manual cosine similarity
> was still implemented and understood first, then swapped for Chroma's
> built-in cosine search, matching the plan's "implement manually first,
> then show how a production tool simplifies it" principle. A `cli.py`
> command-line interface and a Streamlit browser UI (`app_ui.py`) were
> also added, wrapping the same `retrieve → build_prompt → generate_answer`
> pipeline so neither one duplicates pipeline logic.

## 2. Data Flow

**Ingestion (offline, run once or on-demand):**
```
README.md → load text → split into chunks → embed each chunk
    → store [{chunk_text, embedding, metadata}] locally (e.g. JSON/pickle)
```

**Query (runtime, every question):**
```
User question
    → embed question (same embedder as ingestion)
    → compute cosine similarity vs every stored chunk embedding
    → sort, take Top-K
    → build prompt = instructions + retrieved chunks + question
    → send to LLM
    → return answer (ideally with which chunks were used)
```

Key point: embeddings for documents and the query must come from the **same embedding model**, otherwise similarity scores are meaningless.

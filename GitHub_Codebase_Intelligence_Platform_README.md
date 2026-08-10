# GitHub Codebase Intelligence Platform

> A practical, production-oriented RAG project that evolves from a
> beginner-level single-document RAG system into an advanced
> multi-source, code-aware, hybrid, evaluated, graph-based, corrective,
> and self-reflective RAG platform.

------------------------------------------------------------------------

## 🎯 Project Goal

Build one unified AI platform that can understand a software project and
its surrounding knowledge ecosystem.

The platform will ingest:

-   GitHub repositories
-   Source code
-   README and Markdown documentation
-   PDF documents
-   DOCX documents
-   Websites
-   GitHub Issues
-   Pull Requests
-   Commit history
-   REST API data

The system will progressively evolve from:

``` text
Basic RAG
    ↓
Multi-Source RAG
    ↓
Code-Aware RAG
    ↓
Hybrid Search
    ↓
Reranking
    ↓
Query Transformation
    ↓
RAG Evaluation
    ↓
GraphRAG
    ↓
Corrective RAG
    ↓
Self-Reflective RAG
```

------------------------------------------------------------------------

# 🚀 Final Vision

The final system should act as an intelligent codebase knowledge
platform.

Example questions:

-   How is authentication implemented in this repository?
-   Which files are responsible for payment processing?
-   Where is this function used?
-   Why was this function changed?
-   Which issue caused this code change?
-   Have we had a similar bug before?
-   Which pull request fixed this issue?
-   Explain the architecture of this repository.
-   What are the relevant technical documents for this feature?
-   Compare the current implementation with the architecture
    documentation.

The system should answer using relevant context and provide source
references.

------------------------------------------------------------------------

# 🏗️ Overall Architecture

``` text
                         DATA SOURCES
 ┌──────────┐  ┌──────────┐  ┌──────────┐
 │   PDF    │  │   DOCX   │  │ Website  │
 └────┬─────┘  └────┬─────┘  └────┬─────┘
      │              │              │
      └──────────────┼──────────────┘
                     │
 ┌───────────────────▼───────────────────┐
 │            GitHub Ecosystem            │
 │                                       │
 │  Source Code | Issues | PRs | Commits │
 └───────────────────┬───────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Source-Specific       │
          │ Ingestion Loaders     │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ Document Normalizer   │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ Metadata Enrichment   │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ Source-Aware Chunking │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ Embedding Generation  │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ Vector Database       │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ Retrieval Layer       │
          │                      │
          │ Dense Search          │
          │ Sparse Search         │
          │ Hybrid Search         │
          │ Reranking             │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ Query Intelligence    │
          │                      │
          │ Rewriting             │
          │ Expansion             │
          │ Decomposition         │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ LLM Generation        │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ Grounded Answer       │
          │ + Source Citations    │
          └──────────────────────┘
```

------------------------------------------------------------------------

# 📚 Learning Roadmap

## Version 1 --- Basic RAG

### Goal

Build the simplest working RAG system.

### Data Source

Start with:

``` text
README.md
```

### Pipeline

``` text
README.md
    ↓
Load Document
    ↓
Split into Chunks
    ↓
Generate Embeddings
    ↓
Store in Vector Database
    ↓
User Query
    ↓
Generate Query Embedding
    ↓
Similarity Search
    ↓
Retrieve Top-K Chunks
    ↓
LLM
    ↓
Answer
```

### Concepts

-   Document loading
-   Text extraction
-   Chunking
-   Chunk overlap
-   Embeddings
-   Vector representation
-   Cosine similarity
-   Vector dimensions
-   Vector database
-   Similarity search
-   Top-K retrieval
-   Prompt augmentation
-   Context window
-   Grounded generation
-   Hallucination control

### Example

``` text
Question
   ↓
Query Embedding
   ↓
Vector Search
   ↓
Relevant README Chunks
   ↓
LLM
   ↓
Answer
```

------------------------------------------------------------------------

## ✅ Version 1 Implementation Status: Complete

### What's built

| Stage | Module |
|---|---|
| Document loading | [app/ingestion/loader.py](app/ingestion/loader.py) |
| Chunking (heading-aware, structure-aware) | [app/chunking/chunker.py](app/chunking/chunker.py) |
| Embedding (local `sentence-transformers`, `all-MiniLM-L6-v2`, 384-dim) | [app/embeddings/embedder.py](app/embeddings/embedder.py) |
| Vector storage (Chroma, cosine distance, persisted to disk) | [app/vectorstore/store.py](app/vectorstore/store.py) |
| Retrieval (embed query + similarity search) | [app/retrieval/retriever.py](app/retrieval/retriever.py) |
| Prompt building (grounded, token-budget aware) | [app/prompting/prompt_builder.py](app/prompting/prompt_builder.py) |
| LLM generation (local Ollama, `llama3.2:3b`, deterministic decoding) | [app/generation/llm_generator.py](app/generation/llm_generator.py) |
| CLI | [cli.py](cli.py) |
| Browser UI (FastAPI + vanilla JS/HTML — current, primary UI) | [server.py](server.py), [webapp/](webapp/), [api/](api/) |
| Browser UI (Streamlit — earlier build, kept for reference; see its own deprecation notice) | [app_ui.py](app_ui.py) |
| Automated tests | [tests/](tests/) |
| Logging | [app/logging_config.py](app/logging_config.py) |

### How to run it

```powershell
# 0. Install dependencies (once)
pip install -r requirements.txt
playwright install chromium   # only needed for JS-rendered website ingestion (render_js=True)

# 1. Ask a question via the CLI (auto-ingests README.md into the vector
#    store the first time you run anything that touches it)
python cli.py "What is this repository about?"
python cli.py "How does chunking work in this project?" --top-k 5

# 2. Or run the current web UI (FastAPI + webapp/) — the primary way to
#    use the platform: ingest sources, chat, run evaluations, and change
#    settings, all from the browser
python -m uvicorn server:app --reload --port 8000
# then open http://localhost:8000

# 3. Or run the earlier Streamlit UI (kept for reference — see
#    app_ui.py's module docstring for why it's no longer the primary UI)
python -m streamlit run app_ui.py
# then open http://localhost:8501

# 4. Run the automated test suite
python -m pytest -v
```

Requires [Ollama](https://ollama.com) installed locally with `qwen2.5:7b-instruct` pulled (`ollama pull qwen2.5:7b-instruct`) for text generation, and `llava` pulled (`ollama pull llava`) for vision/image questions — every model runs entirely on your machine, no API key needed. (V1 originally shipped against the smaller `llama3.2:3b`; it was later swapped for `qwen2.5:7b-instruct` platform-wide after `llama3.2:3b` was observed hallucinating JSON/code and inventing citations even with the correct chunk retrieved verbatim — see [app/generation/llm_generator.py](app/generation/llm_generator.py).)

Optional environment variables:

- `SEMANTIC_ANSWER_CACHE_ENABLED=1` — turns on the semantic near-duplicate answer cache ([app/generation/semantic_cache.py](app/generation/semantic_cache.py)), which can skip a fresh (60-160+ second on CPU) Ollama generation call for a *rephrased* repeat of an earlier first-turn question, once its resolved-question embedding and retrieved chunk set both closely match a past answer. Off by default — this is a shared, process-global cache (not a per-session setting like CRAG/query-transform/Self-RAG/conversation-memory), and a false-positive match risks serving a wrong answer rather than just a slow one. When a hit occurs it's disclosed in the chat UI's "how was this answer checked" panel.
- `WHISPER_MODEL_SIZE=base` (default) — which local [faster-whisper](https://github.com/SYSTRAN/faster-whisper) model size transcribes audio/video ingestion (see [app/ingestion/loader.py](app/ingestion/loader.py)'s `load_audio_document`/`load_video_document`). `base` is fast with decent accuracy on CPU; `small`/`medium`/`large-v3` trade speed for accuracy (`small` is roughly 3-4x slower on CPU). The model downloads once (~100MB for `base`) on first use, same as the embedding/OCR models.
- `ENABLE_AUDIO_SIMILARITY_SEARCH=1` — turns on native acoustic-similarity search for audio/video ingestion: every ~10s window of a file's audio track is embedded with a local [CLAP](https://github.com/LAION-AI/CLAP) model (`laion/clap-htsat-unfused`, via `transformers`) and stored in its own collection, so a question like "find audio that sounds like X" can match by what a clip *sounds like* rather than what was said in it — a second, additive retrieval path alongside the existing Whisper transcript pipeline, not a replacement for it (see [app/embeddings/audio_embedder.py](app/embeddings/audio_embedder.py)). Off by default — unlike video-frame visual search (which reuses the already-unconditional OCR/vision frame pipeline and stays always-on), this downloads a new multi-hundred-MB model and adds real per-file ingestion cost, so it shouldn't turn on silently for every existing deployment. Sampled video frames are ALSO embedded with the existing CLIP image model for visual-similarity search ("find a frame that looks like X") — that part is always-on, since it's cheap relative to the OCR/vision step that already runs unconditionally on the same frames (see [app/ingestion/ingest.py](app/ingestion/ingest.py)'s `_ingest_video_frames`/`_ingest_audio_clips`). Both native-similarity hits are served for playback/display via a new `/media` route backed by [app/media/media_store.py](app/media/media_store.py).

### How this version improves over "no RAG at all"

Without retrieval, an LLM can only answer from what it memorized during training — it has never seen *this* repository's README, so it can only refuse or, worse, hallucinate a plausible-sounding but wrong answer. V1 fixes this by retrieving the actual relevant passages from `README.md` at query time and instructing the LLM to answer only from that retrieved text, citing which section(s) it used. This is the difference between "the model's guess" and "an answer grounded in this repository's real documentation."

### Limitations of this version

- **Single source only** — just `README.md`; no source code, PDFs, issues, PRs, or commit history yet.
- **No hybrid/keyword search** — pure dense vector similarity only, so an exact keyword or function name isn't specially prioritized over a semantically-similar-but-different passage.
- **No reranking** — Top-K is chosen purely by raw cosine similarity; there's no second-pass relevance model to re-order results.
- **No query rewriting/decomposition** — a single confusing or multi-part question is embedded and retrieved as-is.
- **Small local LLM** — `llama3.2:3b` occasionally needs careful prompt design to avoid false refusals on borderline/ambiguous questions (e.g. self-referential questions like "what is *this* repository"). Deterministic decoding (`temperature=0`) removes sampling-related inconsistency, but doesn't add reasoning ability a bigger model would have.
- **No evaluation harness** — no automated way yet to measure answer quality/faithfulness across many questions; correctness has been verified manually so far.

### What Version 2 will solve

This project already adopted Chroma (a real vector database) rather than a plain in-memory list, ahead of the original plan's staging — see the "Distance metric" and "Provider used" notes in [store.py](app/vectorstore/store.py)'s docstring for why. What's still missing from Version 2's scope is explicitly comparing vector database tradeoffs (FAISS vs pgvector vs Chroma), understanding Approximate Nearest Neighbor search versus exact search, and indexing strategies at larger scale. Version 3 then adds the next new *data source* — PDF ingestion, with page-level metadata (`source_type`, `file_name`, `page_number`, `document_id`).

------------------------------------------------------------------------

# Version 2 --- Multi-Source RAG

Add multiple types of documents.

``` text
GitHub Repository
       │
 ┌─────┼─────────┐
 ↓     ↓         ↓
README Code    Docs
       │         │
       └─────┬───┘
             ↓
      Unified Documents
```

### Sources

``` text
README.md
Source Code
Markdown Documentation
Configuration Files
```

Later:

``` text
PDF
DOCX
Website
REST API
```

### Concepts

-   Multi-source ingestion
-   Document normalization
-   Metadata
-   Metadata filtering
-   Source-aware retrieval
-   Deduplication

------------------------------------------------------------------------

# Version 3 --- PDF Ingestion

Add PDF documents.

``` text
PDF
 ↓
Extract Text
 ↓
Extract Page Metadata
 ↓
Chunk
 ↓
Generate Embeddings
 ↓
Store in Vector Database
```

### Example 
Metadata

``` json
{
  "content": "Authentication is handled using...",
  "metadata": {
    "source_type": "pdf",
    "file_name": "architecture.pdf",
    "page_number": 12
  }
}
```

### Concepts to Learn

-   PDF text extraction
-   Page-level metadata
-   Headers and footers
-   Tables
-   Multi-column layouts
-   PDF parsing challenges
-   Layout-aware parsing

------------------------------------------------------------------------

# Version 4 --- DOCX Ingestion

Add structured Word documents.

``` text
DOCX
 ↓
Paragraphs
 ↓
Headings
 ↓
Tables
 ↓
Sections
 ↓
Structure-Aware Chunking
```

### Example Metadata

``` json
{
  "content": "The authentication service...",
  "metadata": {
    "source_type": "docx",
    "file_name": "technical_design.docx",
    "section": "Authentication"
  }
}
```

### Concepts to Learn

-   Structured document parsing
-   Heading-aware chunking
-   Table extraction
-   Document hierarchy
-   Section-aware metadata

------------------------------------------------------------------------

# Version 5 --- Website Ingestion

Add documentation websites.

``` text
URL
 ↓
HTML
 ↓
Remove Navigation
 ↓
Extract Main Content
 ↓
Clean Text
 ↓
Chunk
 ↓
Embed
```

### Concepts to Learn

-   HTML parsing
-   Web crawling
-   Link discovery
-   URL filtering
-   Duplicate detection
-   Content hashing
-   Incremental crawling

------------------------------------------------------------------------

# Version 6 --- Full GitHub Repository Ingestion

Ingest an entire repository.

``` text
repo/
├── README.md
├── docs/
│   ├── setup.md
│   └── architecture.md
├── src/
│   ├── auth.py
│   └── database.py
├── config/
│   └── config.yaml
└── tests/
```

### Supported Data Types

``` text
Markdown
Python
JavaScript
TypeScript
Java
Go
JSON
YAML
```

------------------------------------------------------------------------

# 🔑 Unified Document Model

Regardless of the source, everything should be normalized into a common
representation.

``` python
class Document:
    content: str
    metadata: dict
```

### PDF

``` json
{
  "content": "Authentication...",
  "metadata": {
    "source_type": "pdf",
    "page": 10
  }
}
```

### DOCX

``` json
{
  "content": "Authentication...",
  "metadata": {
    "source_type": "docx",
    "section": "Security"
  }
}
```

### GitHub Code

``` json
{
  "content": "def authenticate_user(): ...",
  "metadata": {
    "source_type": "code",
    "file_path": "src/auth.py",
    "function": "authenticate_user"
  }
}
```

### Core Principle

``` text
PDF
DOCX
Website
GitHub
API
   ↓
Unified Document
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Database
```

This is a key production RAG architecture pattern.

------------------------------------------------------------------------

# Version 7 --- Structure-Aware Code RAG

## Problem with Generic Chunking

``` text
Python File
    ↓
Every 500 Tokens
```

This can split a function or class in the middle.

## Better Approach

``` text
Python File
    ↓
AST Parser
    ↓
Classes
    ↓
Functions
    ↓
Methods
```

Example:

``` python
def authenticate_user(username, password):
    ...
```

should ideally be stored as one logical chunk.

### Example Metadata

``` json
{
  "content": "def authenticate_user(...): ...",
  "metadata": {
    "source_type": "code",
    "file_path": "src/auth.py",
    "language": "python",
    "function": "authenticate_user"
  }
}
```

### Concepts

-   AST parsing
-   Code-aware chunking
-   Function-level retrieval
-   Class-level retrieval
-   Method-level retrieval
-   Symbol extraction
-   Repository structure

## ✅ Version 7 Implementation Status: Complete

### What's built

| Stage | Module |
|---|---|
| AST-based Python chunking (classes, methods, functions, module-level leftovers) | [app/chunking/code_chunker.py](app/chunking/code_chunker.py) |
| Dispatch from the generic chunker to the AST chunker for `language == "python"` documents | [app/chunking/chunker.py](app/chunking/chunker.py) |
| Whole-class lookup by `class_id` (mirrors `get_table_chunk`) | [app/vectorstore/store.py](app/vectorstore/store.py) |
| Retrieval-time promotion of a retrieved method to its complete parent class | [app/retrieval/retriever.py](app/retrieval/retriever.py) |
| Tests | [tests/test_code_chunker.py](tests/test_code_chunker.py), [tests/test_chunker.py](tests/test_chunker.py), [tests/test_store.py](tests/test_store.py), [tests/test_retriever.py](tests/test_retriever.py) |

A Python file (from local ingestion or `ingest_github_repo`) is parsed with the stdlib `ast` module instead of being split by character/token count:

- Every **class** becomes one atomic, never-split "parent" chunk (`content_type: "class"`) — its entire source (class line, docstring, every method) — plus one "child" chunk per **method** (`content_type: "method"`), each sharing the class's `class_id`. This mirrors the Markdown "whole-table + per-row" pattern already used for tables.
- Every top-level **function** becomes its own chunk (`content_type: "function"`).
- Anything left over (imports, module-level constants, `if __name__ == "__main__":`) is still captured via the ordinary generic chunker, tagged `content_type: "module"`, so nothing is silently dropped.
- Every code chunk gets a `"section"` breadcrumb (`ClassName.method_name` for methods, bare name otherwise) — the same field the existing lexical-overlap reranker already uses, so a query like "how does `authenticate_user` work?" is promoted correctly with no retrieval-side changes needed.
- At retrieval time, a retrieved method is swapped for its complete parent class (via `get_class_chunk`/`_complete_partial_classes`, mirroring `get_table_chunk`/`_complete_partial_tables`) so the LLM sees the constructor and sibling methods too, not just one method in isolation.
- A file that fails to parse as valid Python falls back to the generic chunker for that file, logged as a warning, instead of failing ingestion.

Only Python is AST-parsed today; every other language still uses the generic chunker (a disclosed limitation — Tree-sitter for multi-language parsing is left for a future version, per the original plan).

------------------------------------------------------------------------

# 🔀 Query Routing & Multi-KB Architecture

Built out of the original plan's order, at explicit request, once the underlying retrieval/chunking foundation above was solid enough to make routing worthwhile.

## Problem

A single, unscoped similarity search treats every query the same way, regardless of what kind of answer it actually needs, and — in the original design — regardless of *where* the answer likely lives. A question about a specific function benefits from being scoped to code chunks; a question about a table benefits from being scoped to table chunks; a question about a screenshot needs a completely different (CLIP-based) image search; a question that names its source ("what does the PDF report say...") shouldn't have to search through unrelated websites and GitHub repos at all. Always searching everything, unfiltered, wastes the model's context on irrelevant candidates, gives no chance to bias the search toward the right kind of content, and — when everything lives in one shared collection — can never actually skip work, only filter it after the fact.

## Approach: two independent routing dimensions

`classify_route(query_text)` in [app/routing/router.py](app/routing/router.py) returns one `RouteDecision` with two independent dimensions, each built from the same rule-based + semantic combination:

**1. Content-type routing (`.routes`, `.method`, `.scores`)** — which content-type filter(s) to additionally apply:

1. **Rule-based / keyword routing** — fast substring matching against a small keyword list per route (e.g. "function", "class", "method" → `code`; "table", "row", "column" → `table`; "screenshot", "image", "diagram" → `image`).
2. **Semantic routing** — the query is embedded with the same model used at ingestion time and compared (cosine similarity) against cached exemplar embeddings for each route; a route is matched if similarity clears a threshold. This catches paraphrases with no literal keyword overlap (e.g. "how does the login process work internally" still routes to `code`).
3. **Multi-retriever routing** — every matched *text* route runs as its own, separate `query_embedding()` call scoped with a `content_type` metadata filter (`code` → `class`/`method`/`function`, `table` → `table`/`table_row`), and all routes' results are merged and deduplicated before the existing reranking pipeline runs.

`"general"` (today's unscoped search) is **always** included as a route. A wrong or low-confidence routing decision can therefore only *add* extra, more targeted candidates — it never removes anything the unscoped search would already have found. `"image"` is handled separately: it doesn't scope a text query, it gates whether the (separate, CLIP-based) image search runs at all.

**2. KB routing (`.kbs`)** — which knowledge base(s) to search at all. Unlike content-type routing, this is where routing actually reduces latency: a skipped KB is a `query_embedding()` call that never runs.

## Multi-KB storage

Instead of one shared Chroma collection for every ingested source, [app/vectorstore/store.py](app/vectorstore/store.py) now maintains **seven separate collections**, one per source type:

| KB | Source |
|---|---|
| `markdown` | Standalone local `.md` files |
| `pdf` | Standalone local `.pdf` files |
| `docx` | Standalone local `.docx` files |
| `web` | Ingested websites |
| `github` | Every file from an ingested GitHub repo (including its own README/`.md` files — GitHub ingestion always tags its content `source_type="code"`, regardless of file extension) |
| `audio` | Whisper transcripts of ingested audio files (see Multi-Source RAG: Audio + Video below) |
| `video` | Whisper transcripts + OCR/vision-extracted on-screen text of ingested video files |

Each ingested chunk is routed to its KB automatically by `source_type` metadata at write time (`add_embedded_chunks`); nothing else about ingestion changes. `list_populated_kbs()` reports which KBs actually have data, which is the fallback used whenever a query gives no explicit KB signal.

`classify_route()` sets `.kbs` from the same rule-based-keyword + semantic-exemplar combination as content-type routing does — e.g. "in the PDF report" or "in the codebase" match by keyword; paraphrases match semantically. Unlike `.routes`, `.kbs` has **no** always-included default: an empty list means "no explicit signal", and [app/retrieval/retriever.py](app/retrieval/retriever.py) resolves that by falling back to `list_populated_kbs()` — searching every KB that has data, and nothing else. A query that DOES name its source searches only that KB, skipping every other KB's query entirely.

Point-lookup helpers that don't know in advance which KB an id belongs to (`get_table_chunk`, `get_class_chunk`, `get_chunk_by_document_id`, etc.) check every KB; the `code`/`table` content-type filters from dimension 1 above still apply *within* whichever KB(s) are being searched (the `code` filter is additionally restricted to the `github` KB, since AST-chunked class/method/function content can only ever exist there).

## What's built

| Stage | Module |
|---|---|
| Hybrid route classification (rule-based + semantic + multi-retriever), both content-type and KB dimensions | [app/routing/router.py](app/routing/router.py) |
| Five-collection multi-KB vector store | [app/vectorstore/store.py](app/vectorstore/store.py) |
| Per-KB, per-route, content-type-scoped retrieval with merge/dedup | [app/retrieval/retriever.py](app/retrieval/retriever.py) |
| Skipping CLIP image search entirely when a query isn't routed to `image` | [cli.py](cli.py) |
| Tests | [tests/test_router.py](tests/test_router.py), [tests/test_retriever.py](tests/test_retriever.py), [tests/test_cli.py](tests/test_cli.py), [tests/test_store.py](tests/test_store.py) |

## ✅ Query Routing & Multi-KB Implementation Status: Complete

------------------------------------------------------------------------

# Version 8 --- Hybrid Search

Combine:

``` text
Semantic Search
        +
Keyword Search
```

### Why?

For exact technical terms:

``` text
authenticate_user
```

Keyword search is highly effective.

For semantic questions:

``` text
How does the system validate a user's identity?
```

Vector search is highly effective.

### Architecture

``` text
                  Query
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
     BM25 Search         Vector Search
          ↓                   ↓
          └─────────┬─────────┘
                    ↓
             Result Fusion
                    ↓
                 Top-K
```

### Concepts

-   BM25
-   Sparse retrieval
-   Dense retrieval
-   Reciprocal Rank Fusion (RRF)
-   Hybrid retrieval

## ✅ Version 8 Implementation Status: Complete

Dense (cosine, `all-MiniLM-L6-v2`) and sparse (BM25) search run independently per query and are combined with Reciprocal Rank Fusion — see [app/retrieval/hybrid_retriever.py](app/retrieval/hybrid_retriever.py). The BM25 index is built lazily per knowledge base (first query fetches every chunk in that KB and indexes it in memory), never persisted to disk, and invalidated automatically on any ingest/delete into that KB. Query-transform variants (Version 10, below) each contribute their own dense + BM25 ranked lists into the same fusion, and a HyDE hypothetical-answer paragraph and extracted keywords contribute one dense-only and one BM25-only list respectively — RRF fuses however many ranked lists a given query ends up producing.

------------------------------------------------------------------------

# Version 9 --- Reranking

Initial retrieval:

``` text
100,000 Chunks
       ↓
Vector / Hybrid Search
       ↓
Top 50
       ↓
Reranker
       ↓
Top 5
       ↓
LLM
```

### Example

Query:

``` text
How is authentication implemented?
```

Initial results:

``` text
1. auth.py
2. auth_test.py
3. README.md
4. token_service.py
5. login_controller.py
```

A reranker determines the final relevance order.

### Concepts

-   Candidate retrieval
-   Cross-encoder reranking
-   Precision improvement
-   Retrieval pipeline optimization

## ✅ Version 9 Implementation Status: Complete

The fused (dense + BM25) shortlist is re-scored by a real cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via `sentence-transformers`) that reads the query and each candidate chunk *jointly* rather than comparing precomputed vectors — see `_cross_encoder_rerank()` in [app/retrieval/hybrid_retriever.py](app/retrieval/hybrid_retriever.py). Reranking is always scored against the user's **original** question, never a rewritten query-transform variant, so final ordering never drifts from actual intent. The same reranker is reused everywhere a shortlist needs scoring — including GraphRAG's graph-walk results (see Version 13) and every GitHub-adaptive retrieval strategy (see Version 11) — rather than each retrieval path having its own reranking logic.

------------------------------------------------------------------------

# Version 10 --- Query Intelligence

Users may ask incomplete questions.

``` text
"Login broken"
```

The system can transform this into:

``` text
What authentication-related issues exist in the repository?

Are there previous login-related bugs?

Which files implement login?
```

------------------------------------------------------------------------

## Query Rewriting

``` text
Original Query
      ↓
LLM
      ↓
Improved Query
```

------------------------------------------------------------------------

## Multi-Query Retrieval

``` text
User Query
    ↓
 ┌──┼──┐
 ↓  ↓  ↓
Q1 Q2 Q3
 ↓  ↓  ↓
Search Each
    ↓
Merge Results
```

------------------------------------------------------------------------

## Query Decomposition

Question:

``` text
How does authentication work, where are user credentials stored,
and when was the current implementation introduced?
```

Break into:

``` text
Q1 → Source Code
Q2 → Database Configuration
Q3 → Commit History
```

Then combine the results.

### Concepts

-   Query rewriting
-   Query expansion
-   Multi-query retrieval
-   Query decomposition
-   Intent detection

## ✅ Version 10 Implementation Status: Complete

One Ollama call (when enabled — off by default, toggled per-session from the Settings page) produces a rewritten query, up to 4 sub-questions for compound queries, 2 paraphrases, and a HyDE hypothetical-answer paragraph + keywords — all in a single structured JSON response, so N transformation techniques cost one LLM round trip, not four. See [app/retrieval/query_transform.py](app/retrieval/query_transform.py). A fast-path heuristic skips the LLM entirely for short, simple queries. Results are LRU-cached by `(query, kb)` so a repeated question costs nothing extra within a session.

------------------------------------------------------------------------

# Version 11 --- GitHub Ecosystem RAG

Ingest data beyond repository files.

``` text
┌───────────────┐
│ Source Code   │
├───────────────┤
│ README        │
├───────────────┤
│ Issues        │
├───────────────┤
│ Pull Requests │
├───────────────┤
│ Commits       │
├───────────────┤
│ Discussions    │
└───────┬───────┘
        ↓
   Unified RAG
```

### Example Question

``` text
Why was the authentication function changed?
```

The system retrieves:

``` text
Code
  +
Commit
  +
Pull Request
  +
Issue
```

Potential answer:

``` text
The authentication function was changed because an issue reported
expired token failures. A pull request subsequently modified the
token refresh logic.
```

### Concepts

-   GitHub API ingestion
-   Issues as documents
-   Pull Requests as documents
-   Commit history
-   Discussions
-   Cross-source retrieval
-   Temporal metadata

## ✅ Version 11 Implementation Status: Complete

| Kind | How | Indexed or live? |
|---|---|---|
| Source code + README/Markdown | REST Git Trees API, one call lists the whole file tree | Indexed (chunked, embedded, stored — see Version 6/7) |
| Issues | REST `GET /issues?state=all`, paginated | **Indexed** — [app/ingestion/loader.py](app/ingestion/loader.py)'s `load_github_issues`, [app/ingestion/ingest.py](app/ingestion/ingest.py)'s `ingest_github_activity` |
| Pull Requests | REST `GET /pulls?state=all`, paginated | **Indexed** — `load_github_pull_requests` |
| Discussions | GraphQL (the only API surface with Discussions at all), cursor-paginated, requires `GITHUB_TOKEN` even for public repos | **Indexed** — `load_github_discussions` |
| Commits | REST commits API | **Live only** — [app/retrieval/github_history.py](app/retrieval/github_history.py) queries the API at answer time for the GitHub-KB "history" intent (see Version 13's routing); commit messages are not chunked/embedded/stored |

Issues, PRs, and Discussions land in the same `github` knowledge base as source code (`source_type="code"`, distinguished by a new `content_type` of `"issue"`/`"pull_request"`/`"discussion"`) rather than a separate KB, so a cross-source question ("why was auth changed") retrieves code, issues, and PRs from one search with no special-case merging logic. Each kind is independently toggleable at ingest time (`POST /api/sources/github/activity`, or the "Ingest activity" controls on the Sources page), since Discussions require a token and the other two don't. Deleting a repository (`delete_repository`) already removed chunks by `repository` metadata with no content-type filtering, so it removes a repo's issues/PRs/discussions for free — no new deletion code was needed.

Deliberately out of scope: per-item comment threads (issue/PR/discussion bodies are indexed, replies are not — fetching them would cost one extra API call per item) and a generic "REST API data" connector (the platform talks to GitHub's own REST/GraphQL APIs directly; there's no separate configurable arbitrary-REST-source ingestion path).

------------------------------------------------------------------------

# Version 12 --- RAG Evaluation

Build a proper evaluation dataset.

``` json
[
  {
    "question": "Where is authentication implemented?",
    "expected_source": "src/auth.py"
  },
  {
    "question": "Which PR fixed the token expiration issue?",
    "expected_source": "PR #456"
  }
]
```

Compare:

``` text
Basic Vector Search
        ↓
Hybrid Search
        ↓
Hybrid + Reranking
        ↓
Graph + Vector Search
```

### Retrieval Metrics

-   Precision
-   Recall
-   MRR (Mean Reciprocal Rank)
-   NDCG

### Generation Metrics

-   Faithfulness
-   Answer Relevance
-   Context Relevance
-   Groundedness

### Goal

Measure improvement between versions.

Example:

``` text
┌──────────────────────────┬─────────┐
│ Technique                │ Recall  │
├──────────────────────────┼─────────┤
│ Vector Search            │ 72%     │
│ Hybrid Search            │ 84%     │
│ Hybrid + Reranking       │ 91%     │
└──────────────────────────┴─────────┘
```

## ✅ Version 12 Implementation Status: Complete

| Stage | Module |
|---|---|
| Dataset format (JSONL, one question per line) | [app/evaluation/dataset.py](app/evaluation/dataset.py) |
| Retrieval metrics: hit-rate, MRR, nDCG, **Precision@k, Recall@k** | [app/evaluation/metrics.py](app/evaluation/metrics.py) |
| Generation metrics (LLM-as-judge): Faithfulness, Answer Relevance, Context Relevance | [app/evaluation/metrics.py](app/evaluation/metrics.py) |
| Runner — runs a dataset through the *same* retrieval/CRAG/prompt/generation code paths the chat UI uses | [app/evaluation/runner.py](app/evaluation/runner.py) |
| Report writer (JSON + Markdown) | [app/evaluation/report.py](app/evaluation/report.py) |
| Evaluation page in the web UI (run a dataset as a background job, see aggregate + per-question results) | [api/evaluation.py](api/evaluation.py), [webapp/](webapp/) |
| Tests | [tests/test_metrics.py](tests/test_metrics.py) |

Every retrieval metric ships in both a `document` variant (credits the right page even if the wrong section ranked) and a stricter `chunk` variant. Precision@k/Recall@k were the newest additions, mirroring `hit_rate_at_k`/`mrr`/`ndcg_at_k`'s exact signature shape. LLM-judge metrics return `None` (not a fake zero) on any judge failure, so a partial run's average isn't dragged down by an infrastructure hiccup rather than an actual bad answer. "Groundedness" from the original plan is covered by Faithfulness here (retrieval-time) and by Self-RAG's separate `grounded` critique dimension (generation-time — see Version 15), rather than being a third, redundant metric.

------------------------------------------------------------------------

# Version 13 --- GraphRAG

GitHub data has natural relationships.

``` text
Issue
  ↓
fixed_by
  ↓
Pull Request
  ↓
changed
  ↓
File
  ↓
contains
  ↓
Function
```

Example:

``` text
Issue #123
    ↓
fixed_by
    ↓
PR #456
    ↓
changed
    ↓
auth.py
    ↓
contains
    ↓
authenticate_user()
```

### Question

``` text
Which issue caused authenticate_user() to change?
```

This requires understanding relationships, not just semantic similarity.

### Concepts

-   Entity extraction
-   Relationship extraction
-   Knowledge graphs
-   Graph traversal
-   Vector + graph retrieval
-   GraphRAG

## ✅ Version 13 Implementation Status: Complete

A per-repository call/import/inherit graph (`networkx.MultiDiGraph`) is built from the same AST parse the code chunker already runs, and pickled to `data/graphs/{owner}__{repo}.gpickle` — fully regenerable by re-ingesting, so it's gitignored like the rest of `data/`. See [app/graph/code_graph.py](app/graph/code_graph.py) (build/persist) and [app/graph/graph_retrieval.py](app/graph/graph_retrieval.py) (`retrieve_by_graph` — finds the named symbol's node, figures out which edge direction the question implies from a keyword table, walks exactly one hop, then resolves touched nodes back to real stored chunks and hands them to the same cross-encoder every other retrieval path uses). Only Python is graphed today (matching Version 7's AST-chunker scope); `calls` edges are matched by leaf name, not full type resolution — an explicitly accepted precision/complexity tradeoff since the graph is only ever a retrieval-seeding signal, reranked afterward regardless. Wired into query routing as the GitHub-KB `structural` intent — see [app/routing/github_intent.py](app/routing/github_intent.py), [app/retrieval/github_adaptive.py](app/retrieval/github_adaptive.py).

------------------------------------------------------------------------

# Version 14 --- Corrective RAG

The system evaluates retrieved context.

``` text
Question
   ↓
Retrieve
   ↓
Evaluate Context
   │
   ├── Relevant → Generate
   │
   └── Not Relevant
          ↓
      Rewrite Query
          ↓
      Retrieve Again
```

### Concepts

-   Retrieval evaluation
-   Query correction
-   Fallback retrieval
-   Iterative retrieval

## ✅ Version 14 Implementation Status: Complete

See [app/retrieval/crag.py](app/retrieval/crag.py). After retrieval, one Ollama call grades every retrieved chunk's relevance (`correct`/`partial`/`incorrect`) in a single structured JSON response, so N chunks cost the same LLM time as one. Chunks graded `incorrect` are dropped; if too few survive, **exactly one** supplementary retrieval pass runs using the query-transform rewrite (Version 10) and the results are merged in — deliberately non-looping, since a second corrective pass on already-ambiguous results risks unbounded retries. Off by default (toggled per-session from Settings); on any evaluator failure it falls back to "keep everything as retrieved" rather than dropping chunks the retriever already ranked highly.

------------------------------------------------------------------------

# Version 15 --- Self-Reflective RAG

The system evaluates its own answer.

``` text
Question
    ↓
Retrieve
    ↓
Generate Answer
    ↓
Critic
    ↓
Is Answer Supported?
```

If not:

``` text
Retrieve Again
```

### Concepts

-   Self-reflection
-   Answer verification
-   Claim checking
-   Grounding validation
-   Iterative RAG

## ✅ Version 15 Implementation Status: Complete

See [app/generation/self_reflection.py](app/generation/self_reflection.py). Runs strictly *after* the answer has finished generating (never delays the first streamed token) and scores it on three dimensions from the Self-RAG paper — `grounded` (does every claim trace to the retrieved chunks?), `relevant` (were the retrieved chunks actually right?), `useful` (does the answer address what was asked?) — plus a list of specific unsupported claims, all in one Ollama call. On evaluator failure, a neutral 0.5-across-the-board fallback is used, deliberately avoiding both a falsely-reassuring and a falsely-alarming badge when the check simply couldn't run. Off by default (toggled per-session from Settings); results feed the chat UI's "how was this answer checked" insight panel — see [insight.py](insight.py).

Iterative re-retrieval on a failed self-check (the original plan's "if not: retrieve again") was not implemented — the current design surfaces unsupported claims to the user rather than silently retrying, since a second full generation pass would roughly double answer latency on CPU-bound local inference for a benefit that's easy to just show the user instead.

------------------------------------------------------------------------

# 🔒 Reliability & Multi-Session Support

Built out of the original roadmap's order, like the Query Routing & Multi-KB
section above — not a RAG *technique*, but a correctness fix required once
the platform moved from a single-session Streamlit app to a real FastAPI
server multiple browser tabs/users can hit concurrently.

## Problem

Two gaps existed once concurrency became real:

1. The CRAG / query-transform / Self-RAG toggles (Versions 10/14/15) were
   plain **process-global** Python module flags. One browser tab flipping a
   toggle would silently change what every *other* tab's next question does
   — correct for single-session Streamlit, wrong for a shared server process.
2. `data/pending_documents.json` (tracks which ingested-but-not-yet-saved
   documents belong to which session) was read-modify-written with no lock
   at all. Two concurrent requests could each read the same old state and
   the later write would silently discard the earlier one.

## Fix

- All five per-user settings (`top_k`, `use_vision`, and the three RAG
  toggles) are now stored **per-session** in [session_store.py](session_store.py),
  read/written via [api/settings.py](api/settings.py)'s single `GET`/`POST
  /api/settings`.
- Every RAG-quality module (`crag.py`, `query_transform.py`,
  `self_reflection.py`) gained an explicit `enabled: bool | None = None`
  override parameter, resolved from the requesting session's own settings in
  [api/chat.py](api/chat.py) — with **zero** behavior change for the CLI or
  the evaluation runner, neither of which pass it (they keep using the
  process-global default, which is the correct behavior for a single-run
  batch/CLI context).
- A context-variable-based approach (set once per request, read
  transparently) was considered and rejected: the installed Starlette
  version's own source confirms a streamed chat response drains its
  generator via a *separate* `anyio.to_thread.run_sync()` dispatch per
  yielded token, so consecutive tokens aren't guaranteed to run on the same
  OS thread — a context variable set at stream-start isn't reliably visible
  by the time later tokens generate. Explicit function arguments sidestep
  that ambiguity entirely.
- The pending-document tracker's read-modify-write cycle is now wrapped in a
  real cross-process `filelock.FileLock`, and writes go through a
  temp-file-plus-`os.replace()` atomic swap, so concurrent tabs can no
  longer clobber each other's update and a concurrent read can never
  observe a half-written file.

------------------------------------------------------------------------

# 🎙️ Multi-Source RAG: Audio + Video Ingestion

Two more first-class KBs alongside markdown/pdf/docx/web/github,
following the exact same per-source-type-KB architecture: `audio` and
`video`, so spoken/recorded content becomes searchable and answerable
the same way a PDF is.

## How it works

- **Audio** ([app/ingestion/loader.py](app/ingestion/loader.py)'s
  `load_audio_document`): transcribed locally via
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (a
  CTranslate2 reimplementation of Whisper — much faster than the
  original PyTorch model on CPU, matching this project's "everything
  runs locally, no billed calls" design). The transcript is one line
  per Whisper segment, each prefixed with an inline human-readable
  timestamp (`[01:23] ...`) baked directly into the text — no
  PDF-style page-marker metadata mechanism needed, since a transcript
  is plain prose either way and the LLM can still cite timestamps
  because they're literally part of the content it reads.
- **Video** (`load_video_document`): transcribes the audio track via
  the same Whisper path, AND samples frames at an interval (10s by
  default, widening automatically for long videos rather than
  truncating past a 60-frame cap — see the function's docstring for
  the exact math), running each through
  [app/ocr/frame_analysis.py](app/ocr/frame_analysis.py)'s
  `analyze_frame()` — the same OCR-first, vision-model-only-for-code
  pipeline already used for website screenshot images (see Version 6
  below), extracted into its own reusable module so video frames
  (which have no Document/webpage/CLIP context to hang the old inline
  version off of) share the identical gate instead of a second,
  independently-drifting copy. Transcript lines and frame-text lines
  are merged into ONE Document sorted by timestamp — not stored as
  separate chunks the way website image OCR/vision text is, since a
  video frame (unlike a web image) has no separate embedding space or
  stable URL-based identity to key a separate chunk on, and merging
  means a chunk containing both what was *said* and what was *on
  screen* at that moment retrieves as one coherent unit.
- Both loaders decode via PyAV (faster-whisper's dependency) and
  OpenCV (`opencv-python-headless`), which bundle their own codec
  libraries in the pip wheel — verified live against real WAV/MP4
  files on a machine with no system `ffmpeg` on PATH, so (unlike
  Playwright's browser binary) neither needs a separate install step
  for standard mp3/wav/mp4/m4a files. An unusual codec these bundled
  libs don't cover would still need a system `ffmpeg` install.
- Wired through the same KB machinery every other source uses:
  `app.vectorstore.store.KB_NAMES`/`_SOURCE_TYPE_TO_KB`,
  `app.routing.router`'s keyword + corpus-derived semantic KB routing,
  `app.retrieval.hybrid_retriever._HYBRID_KBS` (dense+BM25+cross-encoder,
  same as web/markdown — audio/video are deliberately excluded from
  parent-child chunking, plain single-tier chunking is enough), and
  the webapp's Sources page (new Audio/Video type-cards, same generic
  file-upload flow as markdown/pdf/docx).

## Cost tradeoff, by design

Video frame analysis reuses the existing OCR-first, vision-only-for-
code gate rather than describing every frame with the vision model —
a frame whose OCR finds nothing and isn't code-classified contributes
nothing. This keeps ingestion tractable on CPU but means purely visual
content with no on-screen text (e.g. a diagram with no labels) isn't
captured. A code-heavy tutorial video, worst case, can still take
30-60+ minutes to ingest if most of its capped 60 frames are
code-classified (each triggering a local vision-model call) — pass a
smaller `max_frames` to `load_video_document()` for a faster, less
thorough ingest if that tradeoff matters more than coverage.

## Native acoustic/visual similarity search

Everything above answers "what was **said** or **shown**" by converting
audio/video to text first. A second, additive retrieval path answers a
different question — "what does this **sound** or **look** like" —
using native embeddings instead of transcribed text:

- **Video frames** ([app/ingestion/ingest.py](app/ingestion/ingest.py)'s
  `_ingest_video_frames`, always on): every sampled frame is ALSO
  embedded with the same CLIP model already used for website images
  ([app/embeddings/image_embedder.py](app/embeddings/image_embedder.py))
  and stored in the same `rag_image_chunks` collection — a video frame
  is just an image in that same vector space, so the existing "show me
  a screenshot of X" chat routing (see `app.routing.router`'s `image`
  content-type route) automatically starts surfacing matched frames
  too, with no new routing code.
- **Audio** (`_ingest_audio_clips`, opt-in via
  `ENABLE_AUDIO_SIMILARITY_SEARCH=1`): every ~10s window of the audio
  track is embedded with CLAP
  ([app/embeddings/audio_embedder.py](app/embeddings/audio_embedder.py))
  and stored in its own `rag_audio_clip_chunks` collection — genuinely
  separate from both the text (MiniLM) and image (CLIP) vector spaces.
  Routed via a new `sound` content-type (distinct from the `audio` KB-
  routing keywords above — one picks *which retriever kind*, the other
  picks *which KB* — see `app.routing.router`).
- Both are DISPLAY/PLAYBACK-only side channels — never fed back into
  the LLM prompt, so they can't affect answer grounding. Frame
  thumbnails and one full copy per ingested audio/video source file
  are saved under `data/media/` and served via a new `/media` static
  route ([app/media/media_store.py](app/media/media_store.py)) so the
  chat UI can actually render/play a hit — filenames are always a hash
  of the document id, never user-controlled, so this can't be used for
  path traversal.

------------------------------------------------------------------------

# 🧠 Final RAG Architecture

``` text
                  DATA SOURCES

   PDF ──────┐
   DOCX ─────┤
   Website ──┤
   GitHub ────┤
   API ───────┘
        │
        ▼
┌────────────────────────┐
│ Source-Specific Loaders │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Document Normalization  │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Metadata Enrichment     │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Source-Aware Chunking   │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Embedding Generation    │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Vector Database         │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Hybrid Retrieval        │
│ + Reranking             │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Query Transformation    │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Graph / Vector Retrieval │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Corrective Retrieval    │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Self-Reflection         │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ LLM + Grounded Answer   │
│ + Source Citations      │
└────────────────────────┘
```

------------------------------------------------------------------------

# 🛠️ Technology Stack

This section was originally written as a forward-looking recommendation
before any code existed. The **✅ Actually used** column reflects what the
platform runs on today; the original column is kept for context on what
was considered.

| Area | Originally recommended | ✅ Actually used |
|---|---|---|
| Backend | FastAPI | FastAPI ([server.py](server.py), [api/](api/)) — implemented as recommended |
| Database | PostgreSQL + pgvector | [Chroma](https://www.trychroma.com/) (local, persisted to `data/chroma_db`, cosine distance) — chosen for a zero-infrastructure local-first tool; seven separate collections instead of one shared table (see Query Routing & Multi-KB section, and Multi-Source RAG: Audio + Video below) |
| Embeddings | Open-source model or API | `sentence-transformers` — `all-MiniLM-L6-v2` (text, 384-dim) + `clip-ViT-B-32` (images, 512-dim) — fully local, no API key |
| LLM | LLM API initially | Local [Ollama](https://ollama.com) — `qwen2.5:7b-instruct` (text) + `llava` (vision) — no API key, no billed calls |
| GitHub | REST + GraphQL API | Both, as recommended — REST for repo files/issues/PRs/commits, GraphQL specifically for Discussions (the only GitHub data with no REST endpoint) |
| Code Parsing | Python AST + Tree-sitter | Python `ast` module only — Tree-sitter (multi-language parsing) remains a disclosed future gap; every non-Python language still uses the generic structure-aware chunker |
| Frontend | React | Hand-rolled vanilla HTML/CSS/JS ([webapp/](webapp/)) — no build step, no framework, no CDN dependency (keeps the whole platform offline-capable) |
| Deployment | Docker | Not yet added — currently runs as a local process (`uvicorn`) |

------------------------------------------------------------------------

# 🗂️ Project Structure

The tree below was originally written as a suggestion before any code
existed. The actual structure (below) evolved differently in a few
places — most notably, chunking/retrieval/generation modules stayed
flatter than proposed (one file per concern rather than one per
technique), a `webapp/` + `api/` pair was added for the FastAPI UI, and
`ui/`/`ui_pages/`/`app_ui.py` is the earlier Streamlit UI (see its
deprecation notice):

``` text
GitHub Codebase Intelligence Platform/
│
├── README.md                    (this file)
├── requirements.txt              ✅ now present
│
├── server.py                     FastAPI app entrypoint (mounts api/*)
├── cli.py                        CLI entrypoint
├── app_ui.py                     Streamlit UI entrypoint (deprecated)
├── session_store.py              Per-session settings + pending-document tracking, file-locked
├── jobs.py                       In-memory background-job registry (used by ingestion/eval)
├── insight.py                    "How was this answer checked" panel logic (framework-agnostic)
│
├── api/                          FastAPI routers
│   ├── chat.py                     POST /api/chat/stream (SSE)
│   ├── sources.py                  Ingestion + source management
│   ├── evaluation.py               Run/inspect evaluation datasets
│   ├── settings.py                 Per-session settings
│   ├── jobs_router.py              Background-job polling
│   └── deps.py                     Session-cookie dependency
│
├── webapp/                       Current browser UI (no build step)
│   ├── index.html
│   └── app.js
│
├── ui/ + ui_pages/                Earlier Streamlit UI (deprecated)
│
├── app/
│   ├── ingestion/                  loader.py (all source types), ingest.py (orchestration)
│   ├── chunking/                   chunker.py (generic), code_chunker.py (AST), parent_child.py
│   ├── embeddings/                 embedder.py (text), image_embedder.py (CLIP)
│   ├── vectorstore/                store.py (Chroma, 5 KBs)
│   ├── routing/                    router.py (KB/route), github_intent.py (6 intents)
│   ├── retrieval/                  retriever.py, hybrid_retriever.py, github_adaptive.py,
│   │                                github_history.py, crag.py, query_transform.py
│   ├── graph/                      code_graph.py (build), graph_retrieval.py (query)
│   ├── prompting/                  prompt_builder.py
│   ├── generation/                 llm_generator.py, self_reflection.py
│   ├── evaluation/                 dataset.py, metrics.py, runner.py, report.py
│   └── ocr/                        image_classifier.py, image_ocr.py, prompts.py
│
├── data/                          gitignored — regenerable from source
│   ├── chroma_db/                  Vector store
│   ├── graphs/                     Pickled per-repo code graphs
│   ├── eval/                       Evaluation datasets + reports
│   └── pending_documents.json      Per-session pending-document tracker
│
└── tests/
```

------------------------------------------------------------------------

# 📅 Recommended Implementation Sequence

``` text
1️⃣ README.md RAG
   ↓
2️⃣ PDF Ingestion
   ↓
3️⃣ DOCX Ingestion
   ↓
4️⃣ Website Ingestion
   ↓
5️⃣ Full GitHub Repository
   ↓
6️⃣ Code-Aware Chunking
   ↓
7️⃣ GitHub Issues
   ↓
8️⃣ Pull Requests
   ↓
9️⃣ Commits
   ↓
🔟 REST APIs
```

Then:

``` text
1️⃣1️⃣ Metadata Filtering
1️⃣2️⃣ Hybrid Search
1️⃣3️⃣ Reranking
1️⃣4️⃣ Query Rewriting
1️⃣5️⃣ Multi-Query Retrieval
1️⃣6️⃣ Query Decomposition
1️⃣7️⃣ RAG Evaluation
1️⃣8️⃣ GraphRAG
1️⃣9️⃣ Corrective RAG
2️⃣0️⃣ Self-Reflective RAG
```

------------------------------------------------------------------------

# 🎓 Learning Method

For every version:

1.  Understand the concept.
2.  Implement it yourself.
3.  Test it with real data.
4.  Create an evaluation dataset.
5.  Compare it with the previous version.
6.  Document the improvement.
7.  Add tests.
8.  Update the architecture.

The goal is not simply to build a chatbot.

The goal is to understand:

> **Why does each RAG technique exist, what problem does it solve, and
> how does it improve the system?**

------------------------------------------------------------------------

# 🏆 Resume Project Description

> **GitHub Codebase Intelligence Platform** --- Built a multi-source RAG
> system that ingests source code, documentation, PDF, DOCX, websites,
> GitHub Issues, Pull Requests, commit history, and API data.
> Implemented source-aware and structure-aware chunking, dense and
> sparse hybrid retrieval, reranking, query rewriting, query
> decomposition, GraphRAG, corrective retrieval, self-reflective RAG,
> and RAG evaluation using precision, recall, MRR, NDCG, faithfulness,
> and answer relevance metrics.

------------------------------------------------------------------------

# 🌱 Long-Term Evolution

This project should eventually evolve into:

``` text
Multi-Source RAG
        ↓
Advanced Retrieval
        ↓
GraphRAG
        ↓
MCP
        ↓
Agentic RAG
        ↓
AI Software Engineering Agent
```

The important principle is:

> **Do not build separate mini-projects for PDF RAG, DOCX RAG, and
> GitHub RAG. Build one unified platform and continuously add new
> ingestion connectors and retrieval capabilities.**

This gives you practical experience with how production-grade RAG
systems are designed.

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
| Browser chat UI (Streamlit) | [app_ui.py](app_ui.py) |
| Automated tests | [tests/](tests/) |
| Logging | [app/logging_config.py](app/logging_config.py) |

### How to run it

```powershell
# 1. Build the index (run once, or again whenever README.md changes)
python -m app.vectorstore.store

# 2. Ask a question via the CLI
python cli.py "What is this repository about?"
python cli.py "How does chunking work in this project?" --top-k 5

# 3. Or ask questions via the browser UI
python -m streamlit run app_ui.py
# then open http://localhost:8501

# 4. Run the automated test suite
python -m pytest -v
```

Requires [Ollama](https://ollama.com) installed locally with the `llama3.2:3b` model pulled (`ollama pull llama3.2:3b`) — the LLM runs entirely on your machine, no API key needed.

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

### Example Metadata

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

Instead of one shared Chroma collection for every ingested source, [app/vectorstore/store.py](app/vectorstore/store.py) now maintains **five separate collections**, one per source type:

| KB | Source |
|---|---|
| `markdown` | Standalone local `.md` files |
| `pdf` | Standalone local `.pdf` files |
| `docx` | Standalone local `.docx` files |
| `web` | Ingested websites |
| `github` | Every file from an ingested GitHub repo (including its own README/`.md` files — GitHub ingestion always tags its content `source_type="code"`, regardless of file extension) |

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

# 🛠️ Recommended Technology Stack

## Backend

``` text
Python
FastAPI
```

## Database

``` text
PostgreSQL
pgvector
```

## Embeddings

``` text
Open-source embedding model or embedding API
```

## LLM

``` text
LLM API initially
```

## GitHub

``` text
GitHub REST API
GitHub GraphQL API
```

## Code Parsing

``` text
Python AST
Tree-sitter
```

## Frontend

``` text
React
```

## Deployment

``` text
Docker
```

------------------------------------------------------------------------

# 🗂️ Suggested Project Structure

``` text
github-codebase-intelligence/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── app/
│   ├── main.py
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   ├── docx_loader.py
│   │   ├── web_loader.py
│   │   ├── github_loader.py
│   │   ├── api_loader.py
│   │   └── normalizer.py
│   │
│   ├── chunking/
│   │   ├── generic_chunker.py
│   │   ├── markdown_chunker.py
│   │   ├── code_chunker.py
│   │   └── structure_aware_chunker.py
│   │
│   ├── embeddings/
│   │   └── embedding_service.py
│   │
│   ├── retrieval/
│   │   ├── vector_search.py
│   │   ├── keyword_search.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   │
│   ├── query/
│   │   ├── query_rewriter.py
│   │   ├── multi_query.py
│   │   └── query_decomposer.py
│   │
│   ├── graph/
│   │   ├── entity_extractor.py
│   │   ├── relationship_extractor.py
│   │   └── graph_retriever.py
│   │
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   └── answer_generator.py
│   │
│   └── evaluation/
│       ├── dataset.py
│       ├── retrieval_metrics.py
│       └── generation_metrics.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── tests/
│
└── frontend/
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

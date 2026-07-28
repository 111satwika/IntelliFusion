I want to build a project called:

GitHub Codebase Intelligence Platform

The goal of this project is to learn and implement Retrieval-Augmented Generation (RAG) from beginner level to advanced production-style RAG.

IMPORTANT:
I do not want you to generate the entire project at once.

I want to build this project incrementally, version by version.

For every version:
1. Explain the RAG concept being introduced.
2. Explain the problem this concept solves.
3. Explain the architecture.
4. Explain the data flow.
5. Explain the design decisions.
6. Help me implement the code step by step.
7. Do not hide important logic behind frameworks unnecessarily.
8. Initially implement important components manually where possible so I understand how they work.
9. After the manual implementation works, show how production frameworks can simplify it.
10. Add tests.
11. Add logging.
12. Update the README.
13. Explain how this version improves over the previous version.
14. Explain the limitations of the current version.
15. Explain what the next version will solve.

Do not skip directly to advanced RAG.

The project should evolve through the following stages.

==================================================
PROJECT OVERVIEW
==================================================

The final system should answer questions about a software repository using multiple sources of knowledge:

1. GitHub repository source code
2. README and Markdown files
3. PDF documents
4. DOCX documents
5. Websites and technical documentation
6. GitHub Issues
7. Pull Requests
8. Commit history
9. REST APIs

The final platform should support:

- Multi-source ingestion
- Document normalization
- Metadata enrichment
- Multiple chunking strategies
- Code-aware chunking
- Embeddings
- Vector search
- Metadata filtering
- Hybrid search
- Reranking
- Query rewriting
- Multi-query retrieval
- Query decomposition
- RAG evaluation
- GraphRAG
- Corrective RAG
- Self-reflective RAG
- Grounded answers
- Source citations
- Incremental ingestion
- Document versioning

==================================================
VERSION 1 — BASIC RAG
==================================================

Start with only one source:

README.md

Build the basic RAG pipeline manually as much as practical:

Document
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embedding
    ↓
Vector Storage
    ↓
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Top-K Context
    ↓
LLM
    ↓
Answer

Teach and implement:

- What is a document?
- What is a chunk?
- Why do we chunk?
- Chunk size
- Chunk overlap
- Tokens versus characters
- What are embeddings?
- How text becomes a vector
- Vector dimensions
- Cosine similarity
- Top-K retrieval
- Context injection
- Context window
- Hallucination
- Grounded generation

Initially implement cosine similarity manually so I understand it.

Use a simple local vector storage approach initially.

Do not use a complex vector database yet.

Create a simple CLI where I can ask:

"What is this repository about?"

==================================================
VERSION 2 — VECTOR DATABASE
==================================================

Replace the simple vector storage with a real vector database.

Teach:

- Why vector databases are needed
- Indexing
- Approximate Nearest Neighbor search
- Exact versus approximate search
- Vector indexes
- Metadata storage

Implement:

Documents
    ↓
Embeddings
    ↓
Vector Database
    ↓
Similarity Search

Explain the tradeoffs between:

- FAISS
- pgvector
- Chroma
- Other vector stores

Choose one technology for the project and explain why.

==================================================
VERSION 3 — PDF INGESTION
==================================================

Add PDF ingestion.

Pipeline:

PDF
    ↓
Text Extraction
    ↓
Page Metadata
    ↓
Chunking
    ↓
Embedding
    ↓
Vector Database

Metadata should include:

- source_type
- file_name
- page_number
- document_id

Teach:

- PDF parsing
- Page metadata
- Headers and footers
- Tables
- Multi-column layouts
- PDF extraction limitations

==================================================
VERSION 4 — DOCX INGESTION
==================================================

Add DOCX ingestion.

Extract:

- Paragraphs
- Headings
- Tables
- Sections

Implement structure-aware chunking.

Metadata should include:

- source_type
- file_name
- section
- heading
- document_id

Teach:

- Structured document parsing
- Document hierarchy
- Heading-aware chunking
- Table extraction

==================================================
VERSION 5 — WEBSITE INGESTION
==================================================

Add website ingestion.

Pipeline:

URL
    ↓
HTML
    ↓
Main Content Extraction
    ↓
Remove Navigation
    ↓
Clean Content
    ↓
Chunk
    ↓
Embed
    ↓
Store

Teach:

- HTML parsing
- Web crawling
- Link discovery
- URL filtering
- Duplicate detection
- Content hashing
- Incremental crawling

==================================================
VERSION 6 — UNIFIED MULTI-SOURCE INGESTION
==================================================

Create a unified ingestion architecture.

All sources:

PDF
DOCX
Website
GitHub
API

must be converted into a common document model:

Document:
    content: str
    metadata: dict

Example:

{
    "content": "...",
    "metadata": {
        "source_type": "pdf",
        "file_name": "architecture.pdf",
        "page_number": 10
    }
}

Implement source-specific loaders:

- PDFLoader
- DOCXLoader
- WebLoader
- GitHubLoader
- APILoader

All loaders should return normalized documents.

Teach:

- Abstraction
- Interfaces
- Adapter pattern
- Source normalization
- Metadata standardization

==================================================
VERSION 7 — FULL GITHUB REPOSITORY INGESTION
==================================================

Ingest a full GitHub repository.

Support:

- README
- Markdown
- Python
- JavaScript
- TypeScript
- Java
- Go
- JSON
- YAML
- Configuration files

Create metadata:

- repository
- branch
- commit_hash
- file_path
- language
- source_type

Teach:

- Repository traversal
- File filtering
- Binary file filtering
- File metadata
- Incremental repository ingestion

==================================================
VERSION 8 — CODE-AWARE CHUNKING
==================================================

Do not chunk source code only by character count.

Implement structure-aware code chunking.

For Python:

- Parse the AST
- Extract classes
- Extract functions
- Extract methods

Example:

def authenticate_user():
    ...

should be stored as a logical code unit whenever possible.

Metadata:

{
    "source_type": "code",
    "file_path": "src/auth.py",
    "language": "python",
    "class_name": "...",
    "function_name": "authenticate_user"
}

Teach:

- AST
- Syntax trees
- Function-level retrieval
- Class-level retrieval
- Symbol extraction
- Code-aware chunking

Later explore Tree-sitter for multi-language parsing.

==================================================
VERSION 9 — METADATA FILTERING
==================================================

Add metadata filtering.

Examples:

source_type = "code"
language = "python"
file_path = "src/auth.py"

Support queries such as:

- Search only Python files
- Search only architecture PDFs
- Search only documentation
- Search only a specific repository

Teach:

- Metadata filtering
- Pre-filtering
- Post-filtering
- Filtering before vector search
- Filtering after retrieval

==================================================
VERSION 10 — HYBRID SEARCH
==================================================

Implement both:

1. Dense vector search
2. Sparse keyword search

Example:

Query:
"authenticate_user"

Keyword search is useful for exact identifiers.

Query:
"How does the system validate a user's identity?"

Vector search is useful for semantic meaning.

Architecture:

Query
    ↓
    ├── Dense Vector Search
    │
    └── Sparse Keyword Search
             ↓
        Result Fusion
             ↓
             Top-K Results

Implement and explain:

- BM25
- Sparse retrieval
- Dense retrieval
- Reciprocal Rank Fusion
- Hybrid retrieval

==================================================
VERSION 11 — RERANKING
==================================================

Implement a two-stage retrieval pipeline.

Stage 1:

Retrieve Top 50

Stage 2:

Reranker

Stage 3:

Return Top 5

Teach:

- Candidate retrieval
- Cross-encoder reranking
- Precision versus recall
- Why retrieving more candidates first can improve final results

Compare:

Vector Search
versus
Vector Search + Reranking

==================================================
VERSION 12 — QUERY TRANSFORMATION
==================================================

Implement:

1. Query rewriting
2. Query expansion
3. Multi-query retrieval
4. Query decomposition

Example:

User Query:

"How does authentication work and when was it introduced?"

Decompose into:

1. How is authentication implemented?
2. Which files contain authentication logic?
3. When was authentication introduced?
4. Which commit introduced it?

Retrieve separately and combine results.

Teach:

- Query rewriting
- Query expansion
- Multi-query retrieval
- Query decomposition
- Intent detection

==================================================
VERSION 13 — GITHUB ECOSYSTEM INGESTION
==================================================

Ingest:

- GitHub Issues
- Pull Requests
- Commit history
- Discussions

Normalize them into the same document model.

Example:

Issue:

{
    "content": "...",
    "metadata": {
        "source_type": "github_issue",
        "issue_number": 123,
        "created_at": "...",
        "labels": []
    }
}

Pull Request:

{
    "content": "...",
    "metadata": {
        "source_type": "pull_request",
        "pr_number": 456
    }
}

Teach:

- API ingestion
- Temporal metadata
- Cross-source retrieval
- Historical context

Support questions such as:

"Why was this function changed?"

The answer may require:

Code
+
Issue
+
Pull Request
+
Commit

==================================================
VERSION 14 — RAG EVALUATION
==================================================

Create a proper evaluation dataset.

Example:

{
    "question": "Where is authentication implemented?",
    "expected_source": "src/auth.py"
}

Evaluate:

1. Retrieval

- Precision
- Recall
- MRR
- NDCG

2. Generation

- Faithfulness
- Answer Relevance
- Context Relevance
- Groundedness

Compare:

Basic Vector Search
versus
Hybrid Search
versus
Hybrid + Reranking

Create an evaluation report showing how each technique improves the system.

==================================================
VERSION 15 — GRAPH RAG
==================================================

Build a knowledge graph from the repository.

Example:

Issue
    ↓
FIXED_BY
    ↓
Pull Request
    ↓
CHANGED
    ↓
File
    ↓
CONTAINS
    ↓
Function

Example:

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

Implement graph-based retrieval.

Combine:

Vector Search
+
Graph Search

Teach:

- Entities
- Relationships
- Knowledge graphs
- Graph traversal
- Vector + graph retrieval
- GraphRAG

==================================================
VERSION 16 — CORRECTIVE RAG
==================================================

The system should evaluate the retrieved context.

Flow:

Question
    ↓
Retrieve
    ↓
Evaluate Context
    ↓
Relevant?
    ├── Yes → Generate
    │
    └── No
          ↓
      Rewrite Query
          ↓
      Retrieve Again

Teach:

- Retrieval evaluation
- Query correction
- Fallback retrieval
- Iterative retrieval

==================================================
VERSION 17 — SELF-REFLECTIVE RAG
==================================================

The system should evaluate its own answer.

Flow:

Question
    ↓
Retrieve
    ↓
Generate Answer
    ↓
Critic
    ↓
Is Answer Supported?
    ├── Yes → Return
    │
    └── No → Retrieve Again

Teach:

- Self-reflection
- Answer verification
- Claim checking
- Grounding validation
- Iterative RAG

==================================================
VERSION 18 — PRODUCTION IMPROVEMENTS
==================================================

Add:

- Caching
- Batch ingestion
- Async processing
- Incremental updates
- Document versioning
- Content hashing
- Retry mechanisms
- Error handling
- Observability
- Logging
- Rate limiting
- Authentication
- API endpoints
- Docker
- Configuration management

==================================================
TECHNOLOGY REQUIREMENTS
==================================================

Use:

Backend:
Python
FastAPI

Frontend:
React

Database:
PostgreSQL
pgvector

Code parsing:
Python AST
Tree-sitter

GitHub:
GitHub REST API

LLM:
Use an LLM API initially.

Embeddings:
Use an embedding model/API.

==================================================
PROJECT STRUCTURE
==================================================

Use a modular architecture:

app/
├── ingestion/
├── chunking/
├── embeddings/
├── storage/
├── retrieval/
├── query/
├── reranking/
├── graph/
├── generation/
├── evaluation/
└── api/

data/
├── raw/
├── processed/
└── evaluation/

tests/

frontend/

==================================================
IMPORTANT LEARNING RULE
==================================================

For every implementation step:

DO NOT simply generate the final code.

First explain:

1. What problem are we solving?
2. Why does this problem exist?
3. What concept is being introduced?
4. What happens internally?
5. What are the alternatives?
6. What are the tradeoffs?
7. What are the limitations?

Then implement the smallest working version.

After implementation:

1. Test it.
2. Explain the output.
3. Show the data flow.
4. Add edge cases.
5. Add tests.
6. Update documentation.

When I say "next", proceed to the next logical implementation step.

When I ask a question, answer the question before moving forward.

Do not jump ahead to advanced RAG techniques until the current version is working and understood.

The goal is not just to build a chatbot.

The goal is to deeply understand how production RAG systems work from beginner to advanced level by building one continuously evolving project.
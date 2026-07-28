"""
Prompt building for the RAG pipeline.

Responsibility: turn a user's question + the retrieved chunks into a
single prompt string ready to send to an LLM. No retrieval and no LLM
call happens here — this module only assembles text.

Design:
- build_prompt() is the single entry point other modules should use,
  so callers never need to know the exact instruction wording or
  formatting used to keep the LLM "grounded" (same "one place per
  concern" pattern used in embedder.py and store.py).
- Grounding: the instructions explicitly tell the LLM to answer ONLY
  from the provided context, and to say so if the answer isn't there,
  instead of relying on its own pretrained knowledge. This is what
  reduces (not eliminates) hallucination.
- Refusal wording matters with small local models: an earlier version
  of these instructions gave the model an exact quoted string to
  return when it couldn't answer ("respond exactly with: ..."). Small
  models (e.g. a 3B local model) are prone to over-using an explicit,
  quotable refusal template as a "safe" default any time the answer
  isn't a single verbatim sentence - even when the context clearly
  contains the answer, just spread across more than one passage. The
  instructions below describe the refusal behavior in plain language
  instead of handing the model a copy-pasteable escape hatch, which
  fixed this false-refusal problem in testing.
- Self-reference ambiguity: even a much larger instruct-tuned model
  (tested: qwen2.5:7b-instruct) refused questions like "What is this
  repository about?" against our own README content, because nothing
  in the retrieved text *explicitly* says "this repository is/does
  X" - the model has to infer that from a section heading, and strict
  grounding instructions make models reluctant to take that inferential
  leap (this is arguably correct anti-hallucination caution, not a bug).
  So this isn't a model-size problem - it's a missing-context problem.
  Fix: _build_context_note() below adds one explicit line stating
  which of this repository's own files (README.md, or any other
  ingested source) the context was extracted from, removing the need
  for the model to infer that connection at all. It's built
  dynamically from the retrieved chunks' file names rather than
  hardcoded, since context can now come from multiple source types
  (see app.ingestion.ingest for multi-source ingestion).
- Completeness: an early version of the instructions told the model to
  "ignore passages that don't help and base your answer on the ones
  that do" - phrasing that, in testing with a small local model
  (llama3.2:3b), encouraged picking just ONE matching passage (often
  the shortest/most quotable one, e.g. a single table row) even when a
  neighboring passage in the SAME prompt had more complete, relevant
  detail that was silently dropped. This is a generation-quality
  problem, not a retrieval problem - the missing detail was already in
  the prompt, the model just didn't use all of it. Fixed by adding an
  explicit instruction to combine details from every relevant passage
  instead of stopping at the first one that matches.
- Context window budget: retrieved chunks are added to the context
  block until a max_context_tokens budget is reached, then the rest
  are dropped. This matters even with a small top_k today, because the
  same function should behave sensibly if top_k or chunk size grows
  later. Uses real token counts via tiktoken if installed, otherwise
  falls back to a word count approximation (mirrors the same fallback
  used in app.chunking.chunker). Default raised from 1000 to 3000:
  code-heavy chunks (full JS/JSON automation-rule snippets) can each be
  600-800+ tokens once JSON-escaping is accounted for, so a top_k=3
  retrieval with a 1000-token budget could silently drop 2 of the 3
  chunks - including, in one real case, the chunk that was the actual
  answer - even though app.retrieval.retriever had already ranked it
  correctly. See app.generation.llm_generator's num_ctx, which was
  raised alongside this so Ollama doesn't just truncate the bigger
  prompt from the other end instead.
- Code paraphrasing problem: even once the correct code-containing
  chunk reaches the model (see above), a small local model
  (llama3.2:3b) tends to describe what a code/JSON snippet does in
  prose instead of reproducing it - unhelpful for automation-rule
  questions, where the actual JSON/JavaScript IS the answer the user
  needs to paste into Targetprocess. The instructions below now
  explicitly tell the model to reproduce any relevant code from the
  context verbatim in a fenced code block rather than only summarizing
  it, since "answer using only the context" alone was not enough to
  stop it from paraphrasing.
- Partial-code problem: even with the verbatim instruction above, the
  model would sometimes reproduce only ONE element of a multi-part
  JSON array verbatim (e.g. just the "action:JavaScript" block) while
  dropping the sibling "source:...EntityChanged" trigger and
  "filter:Relational" blocks that are equally part of the same
  automation rule - each element looked complete on its own so nothing
  about it read as "paraphrased", but the answer was still an
  incomplete automation rule that wouldn't work if pasted as-is. Fixed
  by explicitly telling the model to reproduce the ENTIRE code
  block/array exactly as it appears, not just the part that most
  directly answers the question.
"""

import logging

logger = logging.getLogger(__name__)

_tiktoken_encoder = None
_tiktoken_checked = False

_SYSTEM_INSTRUCTIONS = """Answer the question using only the context provided below. Not every passage will be relevant - ignore any that don't help answer the question. When more than one passage IS relevant, combine the details from ALL of them into one complete answer - do not stop at the first, shortest, or most obviously matching passage if another relevant passage adds more detail. If any relevant passage contains code (JSON, JavaScript, or any other code block), reproduce that ENTIRE code block verbatim in a fenced code block as part of your answer, exactly as it appears in the context - do not paraphrase it, and do not reproduce only the part (e.g. only one element of a JSON array, such as just the action) that seems most relevant while dropping the other elements (such as trigger or filter/condition blocks); the code is only correct and usable if every part of it is included together, unchanged. Do not use outside knowledge or make assumptions beyond what is given in the context. If the context truly contains nothing related to the question, say you don't have enough information to answer it. When useful, mention which section(s) you used."""


def _count_tokens(text: str) -> int:
    """
    Count size units for a piece of text. Uses real token counts via
    tiktoken if installed, otherwise falls back to a word count
    approximation.
    """
    global _tiktoken_encoder, _tiktoken_checked

    if not _tiktoken_checked:
        try:
            import tiktoken

            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _tiktoken_encoder = None
        _tiktoken_checked = True

    if _tiktoken_encoder is not None:
        return len(_tiktoken_encoder.encode(text))
    return len(text.split())


def _build_context_note(retrieved_chunks: list[dict]) -> str:
    """
    Build a sentence stating where the context came from, so the model
    doesn't need to infer that connection itself (see module docstring:
    self-reference ambiguity).

    Crucially, this note must NOT claim that crawled website content
    (source_type == "web", e.g. an external product's documentation
    site) "describes what this repository itself is and does" - that
    claim is simply false for such chunks and actively misleads the
    model. A query like "what is Targetprocess" against IBM Docs
    content crawled about the Targetprocess product previously caused
    a false refusal ("I don't have enough information...") because the
    note told the model the context was about "this repository",
    directly contradicting the actual chunk content about an unrelated
    external product - the model treated that as a sign the context
    must not be relevant. Repo-local files (markdown/pdf/docx, i.e.
    anything with source_type != "web") keep the original "this
    repository's own file(s)" framing; web-sourced chunks instead get a
    neutral "crawled from an external website" framing that doesn't
    assert anything false about what the content describes.

    Built dynamically from the retrieved chunks' file_name (and, for
    web chunks, url) metadata (deduplicated, in first-seen order)
    rather than hardcoded, since context can now come from any ingested
    source (see app.ingestion.ingest for multi-source ingestion).
    """
    repo_file_names: list[str] = []
    web_pages: list[str] = []
    for chunk in retrieved_chunks:
        metadata = chunk["metadata"]
        file_name = metadata.get("file_name")
        if not file_name:
            continue
        if metadata.get("source_type") == "web":
            url = metadata.get("url")
            entry = f"{file_name} ({url})" if url else file_name
            if entry not in web_pages:
                web_pages.append(entry)
        elif file_name not in repo_file_names:
            repo_file_names.append(file_name)

    if not repo_file_names and not web_pages:
        return (
            "The context below is extracted from this repository's own "
            "files, so it describes what this repository itself is and does."
        )

    notes: list[str] = []
    if repo_file_names:
        files_list = ", ".join(repo_file_names)
        notes.append(
            f"The context below is extracted from this repository's own "
            f"file(s) ({files_list}), so it describes what this repository "
            f"itself is and does."
        )
    if web_pages:
        pages_list = ", ".join(web_pages)
        notes.append(
            f"The context also includes page(s) previously crawled from an "
            f"external website ({pages_list}) - these describe that "
            f"external site/product's own subject matter (which may be "
            f"entirely unrelated to this repository), so treat them as "
            f"legitimate context for questions about that subject."
        )
    return " ".join(notes)


def _format_chunk(index: int, chunk: dict) -> str:
    """Format a single retrieved chunk into a labeled context block."""
    metadata = chunk["metadata"]
    file_name = metadata.get("file_name")
    section = metadata.get("section")
    page_number = metadata.get("page_number")
    similarity = chunk.get("similarity")

    parts = []
    if file_name:
        parts.append(file_name)
    if page_number is not None:
        parts.append(f"page {page_number}")
    elif section:
        parts.append(f"section: '{section}'")
    if similarity is not None:
        parts.append(f"similarity: {similarity:.4f}")

    label = f"[Source {index}"
    if parts:
        label += " — " + ", ".join(parts)
    label += "]"
    return f"{label}\n{chunk['content']}"


def build_prompt(
    query_text: str,
    retrieved_chunks: list[dict],
    max_context_tokens: int = 3000,
) -> str:
    """
    Build a single grounded prompt string from a user question and the
    chunks retrieved for it.

    Args:
        query_text: The user's natural-language question.
        retrieved_chunks: Chunks from app.retrieval.retriever.retrieve,
            ordered from most to least similar.
        max_context_tokens: Maximum token budget for the context block.
            Chunks are added in order (most similar first) until adding
            the next one would exceed this budget; remaining chunks are
            dropped rather than truncated mid-text.

    Returns:
        A single prompt string: system instructions + context block +
        the user's question, ready to send to an LLM.
    """
    context_blocks: list[str] = []
    used_tokens = 0

    for i, chunk in enumerate(retrieved_chunks, start=1):
        block = _format_chunk(i, chunk)
        block_tokens = _count_tokens(block)
        if context_blocks and used_tokens + block_tokens > max_context_tokens:
            break
        context_blocks.append(block)
        used_tokens += block_tokens

    context_text = "\n\n".join(context_blocks)

    dropped = len(retrieved_chunks) - len(context_blocks)
    logger.info(
        "Built prompt using %d/%d retrieved chunk(s) (%d dropped for token budget=%d), "
        "~%d tokens used",
        len(context_blocks),
        len(retrieved_chunks),
        dropped,
        max_context_tokens,
        used_tokens,
    )

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"{_build_context_note(retrieved_chunks)}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query_text}\n\n"
        f"Answer:"
    )


if __name__ == "__main__":
    import sys

    from app.retrieval.retriever import retrieve

    query_text = sys.argv[1] if len(sys.argv) > 1 else "What is this repository about?"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    chunks = retrieve(query_text, top_k=top_k)
    prompt = build_prompt(query_text, chunks)

    print(prompt)
    print("\n" + "=" * 70)
    print(f"Prompt token count: {_count_tokens(prompt)}")

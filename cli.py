"""
Simple command-line interface for the Version 1 RAG pipeline.

Usage:
    python cli.py "What is this repository about?"
    python cli.py "How does chunking work in this project?" --top-k 5
    python cli.py --ingest-github owner/repo
    python cli.py --ingest-github https://github.com/owner/repo --branch main "What does this repo do?"
    python cli.py "How does auth work?" --repo owner/repo

Ties together every module built so far, in order:
    retrieve()       -> app.retrieval.retriever
    build_prompt()   -> app.prompting.prompt_builder
    generate_answer()-> app.generation.llm_generator

This file intentionally contains no pipeline logic of its own - it
only parses CLI args and calls the three functions above, so the
pipeline stays testable/usable from other code (e.g. a future web
API) without depending on this CLI.
"""

import argparse
import logging

from app.generation.llm_generator import generate_answer, generate_answer_with_images
from app.logging_config import configure_logging
from app.prompting.prompt_builder import build_prompt
from app.retrieval.retriever import retrieve, retrieve_images
from app.routing.router import classify_route
from app.vectorstore.store import get_chunk_by_document_id

logger = logging.getLogger(__name__)

# Images are matched via a separate multimodal (CLIP) embedding space
# (see app.embeddings.image_embedder), whose absolute cosine-similarity
# scores run lower than all-MiniLM-L6-v2's for genuinely relevant text
# - so this threshold is deliberately loose, just enough to filter out
# clearly-unrelated images rather than to mean "highly confident match".
IMAGE_SIMILARITY_THRESHOLD = 0.2
MAX_IMAGES_FOR_VISION = 2


def ask(query_text: str, top_k: int = 3, repository: str | None = None) -> str:
    """Run the full retrieve -> build_prompt -> generate_answer pipeline."""
    chunks = retrieve(query_text, top_k=top_k, repository=repository)
    prompt = build_prompt(query_text, chunks)
    return generate_answer(prompt)


def _is_code_screenshot(image_url: str) -> bool:
    """
    True if the given image's own OCR/vision text chunk was classified
    as a source-code screenshot (content_type == "image_code") at
    ingestion time (see app.ocr.image_classifier, app.ingestion.ingest),
    as opposed to an ordinary image (content_type == "image_ocr").

    Used to auto-enable the vision model for exactly the screenshots
    it's actually useful for (code rendered as pixels, where OCR alone
    can be garbled/inaccurate), without paying the vision-model cost
    for every loosely CLIP-relevant image.
    """
    chunk = get_chunk_by_document_id(image_url)
    return chunk is not None and chunk["metadata"].get("content_type") == "image_code"


def ask_with_vision(
    query_text: str, top_k: int = 3, use_vision: bool = False, repository: str | None = None
) -> tuple[str, list[dict]]:
    """
    Same pipeline as ask(), but additionally retrieves relevant images
    for DISPLAY, and (only when explicitly opted into via
    `use_vision=True`) sends them to a vision-capable Ollama model (see
    app.generation.llm_generator.generate_answer_with_images) alongside
    the text context - so a question whose real answer only exists
    inside a screenshot (e.g. a code example rendered as an image
    rather than real HTML text) can still be answered by literally
    showing the model the picture, not just its alt text.

    `use_vision` defaults to False: on CPU-only setups the vision model
    (llava) reliably takes 2.5-10+ minutes per call and frequently
    times out (see /memories/repo/rag-platform-notes.md). Worse, a
    slow/timed-out vision call keeps the CPU busy, which then starves
    the text-only fallback call right after it, causing IT to time out
    too - i.e. attempting vision by default made even normally-fast
    text-only answers unreliable. OCR'd text from these same images is
    already part of the normal retrieved text context by the time this
    function runs (stored as an ordinary extra chunk at ingestion time,
    see app.ocr.image_ocr), so most code-screenshot questions are
    already answerable from `chunks` alone without ever calling the
    vision model. Pass `use_vision=True` explicitly to opt back in for
    ALL relevant images.

    Even with `use_vision=False`, vision is still AUTO-enabled for any
    relevant image already classified as a source-code screenshot
    (content_type == "image_code", see _is_code_screenshot) - this is
    exactly the case where plain OCR is most likely to be garbled/
    inaccurate (see app.ocr.image_classifier), so it's worth the extra
    cost even by default, while still not paying that cost for every
    other loosely-relevant image.

    `repository` optionally scopes retrieval to one already-ingested
    "owner/repo" (see app.retrieval.retriever.retrieve's `repository`
    argument) - without it, a question is answered from EVERY source
    ever ingested into the shared vector store, not just the repo the
    user most recently ingested/cares about.

    Returns:
        (answer_text, image_hits_used) - image_hits_used is the list
        of retrieved image dicts (see
        app.retrieval.retriever.retrieve_images) that cleared
        IMAGE_SIMILARITY_THRESHOLD, for callers that want to DISPLAY
        them (e.g. app_ui.py), regardless of whether they were also
        sent to the vision model.
    """
    chunks = retrieve(query_text, top_k=top_k, repository=repository)
    prompt = build_prompt(query_text, chunks)

    # Query routing (see app.routing.router): the image retriever uses
    # a completely separate model/vector space (CLIP) from text
    # retrieval, so - unlike retrieve()'s content_type routes, which
    # only ever ADD extra candidates - it's worth skipping entirely
    # when the query doesn't look at all visual, to avoid paying the
    # CLIP embedding + image-search cost on every single question.
    if "image" in classify_route(query_text).routes:
        try:
            image_hits = retrieve_images(query_text, top_k=MAX_IMAGES_FOR_VISION)
        except Exception:
            logger.exception("Image retrieval failed; continuing text-only.")
            image_hits = []
    else:
        image_hits = []

    relevant_images = [hit for hit in image_hits if hit["similarity"] >= IMAGE_SIMILARITY_THRESHOLD]

    if use_vision:
        images_to_send = relevant_images
    else:
        images_to_send = [hit for hit in relevant_images if _is_code_screenshot(hit["metadata"]["image_url"])]

    if images_to_send:
        image_urls = [hit["metadata"]["image_url"] for hit in images_to_send]
        try:
            answer = generate_answer_with_images(prompt, image_urls)
        except Exception:
            # Vision models are slow on CPU and can time out; the text
            # context already includes any OCR'd text from these same
            # images (see app.ocr.image_ocr), so falling back to the
            # text-only pipeline still gives a useful answer instead
            # of a hard failure.
            logger.exception("Vision generation failed; falling back to text-only answer.")
            answer = generate_answer(prompt)
    else:
        answer = generate_answer(prompt)

    return answer, relevant_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question about this repository (Version 1 RAG pipeline)."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Your natural-language question. Optional when using --ingest-github alone.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many chunks to retrieve as context (default: 3).",
    )
    parser.add_argument(
        "--ingest-github",
        metavar="REPO_URL",
        help=(
            "Ingest a public GitHub repo before answering (e.g. 'owner/repo' or "
            "a full https://github.com/... URL). Can be combined with a question, "
            "or used on its own to just ingest."
        ),
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Branch to ingest with --ingest-github (default: the repo's own default branch).",
    )
    parser.add_argument(
        "--repo",
        metavar="OWNER/REPO",
        default=None,
        help=(
            "Scope the question to one already-ingested GitHub repo (e.g. 'owner/repo'). "
            "Without this, questions are answered from the ENTIRE shared vector store "
            "(every source ever ingested), which can surface the wrong repo's chunks once "
            "more than one repo has been ingested."
        ),
    )
    args = parser.parse_args()

    configure_logging()

    if args.ingest_github:
        from app.ingestion.ingest import ingest_github_repo

        print(f"Ingesting GitHub repo '{args.ingest_github}'...")
        stored = ingest_github_repo(args.ingest_github, branch=args.branch)
        print(f"Stored {stored} chunk(s) from '{args.ingest_github}'.\n")

    if not args.question:
        return

    logger.info("CLI question received: %r (top_k=%d)", args.question, args.top_k)

    print(f"Question: {args.question}\n")
    print("Thinking...\n")
    answer, image_hits = ask_with_vision(args.question, top_k=args.top_k, repository=args.repo)

    print("=" * 70)
    print("Answer:")
    print(answer)
    if image_hits:
        print("\n" + "-" * 70)
        print(f"(Answer used {len(image_hits)} relevant image(s) as visual context:)")
        for hit in image_hits:
            print(f" - {hit['metadata'].get('image_url')}")


if __name__ == "__main__":
    main()

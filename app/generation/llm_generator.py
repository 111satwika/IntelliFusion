"""
LLM generation for the RAG pipeline.

Responsibility: send a finished prompt (from app.prompting.prompt_builder)
to a local LLM and return its generated answer text. No retrieval and
no prompt assembly happens here - this module only calls the model.

Design:
- generate_answer() is the single entry point other modules should
  use, so callers never need to know which LLM provider/API shape is
  behind it (same "one place per concern" pattern used throughout this
  project - embedder.py, store.py, prompt_builder.py).
- Provider used: Ollama, running locally at http://localhost:11434.
  Ollama runs the model entirely on your machine - no API key, no
  network call to a third party, no billing. Same reasoning as the
  local embeddings decision.
- Implemented with a raw HTTP call (requests) rather than the `ollama`
  pip package, so the actual request/response shape is visible instead
  of hidden behind a wrapper. A production version could swap this one
  function for `ollama.generate(...)` (the official client) without
  changing any other module - see the note at the bottom of this file.
- stream=False is used so Ollama returns one complete JSON response
  instead of a stream of partial-token chunks - simpler to handle for
  a first version, at the cost of not showing tokens as they arrive.
- temperature defaults to 0.0 (fully deterministic/greedy decoding).
  Grounded Q&A wants the model to stay close to the provided context,
  not "creatively" wander - and testing showed that even temperature
  0.2 let borderline questions (e.g. same question with/without a
  trailing "?") flip between a correct answer and a false refusal on
  identical retrieved context. temperature=0 removes that source of
  randomness; it does not fix genuine ambiguity in the prompt/context
  itself (see app.prompting.prompt_builder for that class of issue).
"""

import base64
import json
import logging
from collections.abc import Iterator

import requests
from PIL import Image

logger = logging.getLogger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
# qwen2.5:7b-instruct follows retrieved context faithfully on RAG tasks
# (technical docs, exact-code reproduction). The previous default,
# llama3.2:3b, was too small: it routinely ignored the provided context
# and hallucinated plausible-looking but incorrect JSON/code and made
# up source citations, even when the correct answer was verbatim in
# the top retrieved chunk. Both models are similar VRAM/disk footprint
# (~4.7 GB), so this is a pure quality upgrade with no infra change.
_DEFAULT_MODEL = "qwen2.5:7b-instruct"
_DEFAULT_VISION_MODEL = "llava"


def generate_answer(
    prompt: str,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    """
    Send a prompt to the local Ollama model and return its answer text.

    Args:
        prompt: The full prompt string (e.g. from
            app.prompting.prompt_builder.build_prompt).
        model: Which locally-pulled Ollama model to use.
        temperature: Sampling temperature; lower = more deterministic,
            more likely to stick to the given context.

    Returns:
        The model's generated answer text, with surrounding whitespace
        stripped.

    Raises:
        requests.exceptions.ConnectionError: if the Ollama server isn't
            running (start it with `ollama serve`, or just run any
            `ollama run <model>` command once to start it).
    """
    logger.info("Generating answer with model='%s', temperature=%.2f", model, temperature)
    chunks = list(generate_answer_stream(prompt, model=model, temperature=temperature))
    answer = "".join(chunks).strip()
    logger.info("Received answer (%d characters)", len(answer))
    return answer


def generate_answer_stream(
    prompt: str,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Iterator[str]:
    """
    Same as generate_answer() but yields tokens (strings) one at a
    time as Ollama produces them, instead of returning the full answer
    at the end. Callers that want the finished string can just do
    "".join(generate_answer_stream(...)) - which is exactly what
    generate_answer does above.

    Enables live token-by-token rendering in the UI (see app_ui.py's
    use of st.write_stream), so the user sees the answer taking shape
    on CPU-only hardware instead of staring at a spinner for 2-3 min
    while a 7B model reproduces a long JSON/code block verbatim.
    """
    logger.info(
        "Streaming answer with model='%s', temperature=%.2f, prompt_chars=%d",
        model,
        temperature,
        len(prompt),
    )
    try:
        response = requests.post(
            _OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                # Streaming: Ollama emits one NDJSON line per generated
                # token instead of a single JSON at the end. With
                # stream=False the full generation had to finish inside
                # ONE HTTP read timeout window - on CPU-only setups the
                # 7B model reproducing a long JSON block verbatim
                # regularly exceeded the previous 600s ceiling and the
                # client raised ReadTimeout even though Ollama was
                # still generating. Streaming makes the read timeout
                # apply PER token instead of to the whole generation,
                # so as long as tokens keep flowing (a few per second
                # is easy on CPU) the request never times out no matter
                # how long the total answer takes.
                "stream": True,
                # num_ctx set explicitly (Ollama's own default is only
                # 2048) so a full build_prompt() context block - now up
                # to ~3000 tokens on its own, see prompt_builder's
                # max_context_tokens - isn't silently truncated by the
                # model before it even sees the question.
                "options": {"temperature": temperature, "num_ctx": 4096},
            },
            # requests-side streaming so response.iter_lines() yields
            # NDJSON lines as Ollama emits them, without buffering the
            # whole response in memory first.
            stream=True,
            # Per-chunk read timeout: max seconds we're willing to wait
            # between two consecutive tokens (and for the very first
            # token, which also has to cover Ollama's prompt-processing
            # pass - the slowest step on CPU). 300s is generous headroom
            # for a cold-start 7B model chewing through a ~3000-token
            # prompt; steady-state token intervals on CPU are typically
            # well under a second.
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        logger.error("Could not reach Ollama at %s - is it running?", _OLLAMA_URL)
        raise
    except requests.exceptions.RequestException:
        logger.exception("Ollama request failed")
        raise

    # Consume the NDJSON stream: one line per token, plus a final line
    # with "done": true (and no more "response" content). Malformed
    # lines are skipped defensively rather than aborting the whole
    # generation, but note we don't expect any in practice.
    token_count = 0
    first_token_logged = False
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed Ollama stream line: %r", raw_line[:200])
            continue
        # Ollama may return an "error" field instead of "response" if
        # the model isn't loaded, was killed, ran out of memory, etc.
        # Surface it so the caller/UI can see WHY the stream was empty
        # rather than just getting silence.
        error = event.get("error")
        if error:
            logger.error("Ollama returned error mid-stream: %s", error)
            raise RuntimeError(f"Ollama error: {error}")
        token = event.get("response")
        if token:
            if not first_token_logged:
                logger.info("Ollama produced first token; streaming continues.")
                first_token_logged = True
            token_count += 1
            yield token
        if event.get("done"):
            break
    logger.info("Stream finished (%d tokens yielded).", token_count)


def _download_image_bytes(image_url: str, timeout: int = 10) -> bytes | None:
    """
    Download raw image bytes for a vision-model request.

    Deliberately a separate, independent download here rather than
    reusing app.embeddings.image_embedder's downloader: that one
    returns a decoded Pillow Image (for CLIP embedding), not the raw
    bytes Ollama's base64 `images` field needs, and this module should
    not need to depend on the embeddings package just to fetch a URL.
    A URL that fails to download is logged and skipped (returns None)
    rather than raising, so one dead image link can't break the whole
    generation call - same defensive pattern used throughout this
    project (see image_embedder.py's _download_image).
    """
    try:
        response = requests.get(
            image_url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAGGenBot/1.0)"},
        )
        response.raise_for_status()
        return response.content
    except Exception as error:
        logger.warning("Skipping image '%s' for vision generation (could not download): %s", image_url, error)
        return None


def generate_answer_with_images(
    prompt: str,
    image_urls: list[str],
    model: str = _DEFAULT_VISION_MODEL,
    temperature: float = 0.0,
) -> str:
    """
    Send a prompt PLUS one or more images to a local vision-capable
    Ollama model (e.g. llava) and return its generated answer text.

    Unlike generate_answer() (text-only, /api/generate), this calls
    Ollama's /api/chat endpoint, which is what accepts a base64-encoded
    `images` list per message - the only way to let the model actually
    see pixel content, not just an alt-text description. This matters
    for questions whose real answer is rendered as an image on the
    source page (e.g. a code snippet shown as a documentation
    screenshot rather than real HTML text).

    Prefer OCR-at-ingestion-time (see app.ocr.image_ocr, wired into
    app.ingestion.ingest) as the FIRST line of defense for that exact
    scenario: a well-lit code screenshot's OCR'd text is usually more
    exact/reliable than what even a good vision model transcribes back
    as prose, and it's captured once at ingestion time rather than
    re-analyzed on every question. This function is the complementary
    fallback/second opinion for cases OCR doesn't fully cover (messy
    screenshots, diagrams, UI layout questions, etc.) - both can be
    used together (send OCR'd text as context AND let the model see
    the image directly).

    Args:
        prompt: The full text prompt (same shape as generate_answer's,
            e.g. from app.prompting.prompt_builder.build_prompt).
        image_urls: Image URLs to download and attach to the request.
            A URL that fails to download is skipped (logged), so one
            dead image link never fails the whole call.
        model: Which locally-pulled, VISION-CAPABLE Ollama model to
            use. llama3.2:3b/qwen2.5:7b-instruct do NOT understand
            images - only text - so passing a text-only model here
            would silently just ignore the images. Pull a vision model
            once with e.g. `ollama pull llava`.
        temperature: Same meaning as generate_answer's.

    Returns:
        The model's generated answer text, with surrounding whitespace
        stripped.

    Raises:
        requests.exceptions.ConnectionError: if the Ollama server isn't
            running.
    """
    images_b64 = []
    for image_url in image_urls:
        image_bytes = _download_image_bytes(image_url)
        if image_bytes is not None:
            images_b64.append(base64.b64encode(image_bytes).decode("ascii"))

    logger.info(
        "Generating vision answer with model='%s', %d/%d image(s) attached, temperature=%.2f",
        model,
        len(images_b64),
        len(image_urls),
        temperature,
    )

    message: dict = {"role": "user", "content": prompt}
    if images_b64:
        message["images"] = images_b64

    try:
        response = requests.post(
            _OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [message],
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": 4096},
            },
            # Vision models are much slower than text-only ones on CPU
            # (image tokens + typically larger models like llava), so
            # this gets a longer budget than generate_answer()'s.
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        logger.error("Could not reach Ollama at %s - is it running?", _OLLAMA_CHAT_URL)
        raise
    except requests.exceptions.RequestException:
        logger.exception("Ollama vision request failed")
        raise

    answer = response.json()["message"]["content"].strip()
    logger.info("Received vision answer (%d characters)", len(answer))
    return answer


def generate_vision_text(
    image: Image.Image,
    instruction: str,
    model: str = _DEFAULT_VISION_MODEL,
    temperature: float = 0.0,
) -> str:
    """
    Send an ALREADY-DOWNLOADED image plus a strict instruction prompt
    (e.g. app.ocr.prompts.CODE_EXTRACTION_PROMPT) to a local
    vision-capable Ollama model, and return its raw text response.

    Unlike generate_answer_with_images() (which answers a user's
    question using retrieved text context PLUS supporting images),
    this function isn't answering a question - it's asked to
    transcribe/describe ONE specific image on its own, at INGESTION
    time (see app.ingestion.ingest._ingest_images). It also takes an
    already-loaded Pillow Image directly (no re-download) since the
    caller has typically already downloaded the same image once for
    CLIP embedding/OCR (see app.embeddings.image_embedder.download_image),
    and re-fetching the same URL a second time would be wasteful.

    Args:
        image: An already-downloaded/decoded Pillow image.
        instruction: The extraction/description instruction to send
            alongside the image (this is the message content, not a
            question about retrieved context).
        model: Which locally-pulled, VISION-CAPABLE Ollama model to use.
        temperature: Same meaning as generate_answer's.

    Returns:
        The model's raw text response, stripped of surrounding
        whitespace.

    Raises:
        requests.exceptions.ConnectionError: if the Ollama server isn't
            running.
        requests.exceptions.RequestException: on any other request
            failure (including a timeout).
    """
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    logger.info("Generating vision extraction with model='%s', temperature=%.2f", model, temperature)

    message = {"role": "user", "content": instruction, "images": [image_b64]}

    try:
        response = requests.post(
            _OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [message],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        logger.error("Could not reach Ollama at %s - is it running?", _OLLAMA_CHAT_URL)
        raise
    except requests.exceptions.RequestException:
        logger.exception("Ollama vision extraction request failed")
        raise

    text = response.json()["message"]["content"].strip()
    logger.info("Received vision extraction (%d characters)", len(text))
    return text


if __name__ == "__main__":
    import sys

    from app.prompting.prompt_builder import build_prompt
    from app.retrieval.retriever import retrieve

    query_text = sys.argv[1] if len(sys.argv) > 1 else "What is this repository about?"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    chunks = retrieve(query_text, top_k=top_k)
    prompt = build_prompt(query_text, chunks)

    print(f"Query: '{query_text}'\n")
    print("Generating answer...\n")
    answer = generate_answer(prompt)

    print("=" * 70)
    print("Answer:")
    print(answer)

# Production framework note:
# The official `ollama` pip package wraps this same HTTP call:
#     import ollama
#     response = ollama.generate(model=model, prompt=prompt,
#                                 options={"temperature": temperature})
#     return response["response"].strip()
# It adds convenience (streaming iterators, model management helpers)
# but does the same thing under the hood. Swapping to it only requires
# changing the body of generate_answer() above - no other module cares.

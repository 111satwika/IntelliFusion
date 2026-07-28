"""
Shared logging setup for the RAG pipeline.

Responsibility: configure how log messages look and where they go, in
one place, so every module can just do:

    import logging
    logger = logging.getLogger(__name__)

...without each module fighting over logging.basicConfig().

Design:
- Library-style modules (chunker, embedder, store, retriever,
  prompt_builder, llm_generator) only ever call
  `logging.getLogger(__name__)` and log through it - they never call
  `logging.basicConfig()` themselves. This is the standard Python
  logging convention: only the application's entry point configures
  handlers/formatting; library code just emits records.
- configure_logging() is that one entry-point call. cli.py and
  app_ui.py (the two ways a human actually runs this pipeline) call it
  once, at startup, before doing anything else.
- Default level is INFO so normal runs show one line per pipeline
  stage (retrieval, prompt building, generation) without being as
  noisy as DEBUG. Pass level=logging.DEBUG for more detail (e.g. the
  exact prompt token count or dropped chunks).
"""

import logging

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging once for the whole process.

    Safe to call multiple times (e.g. accidentally from both cli.py
    and an imported module) - only the first call takes effect.
    """
    global _configured
    if _configured:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Third-party libraries (model downloads, HTTP clients) are very
    # chatty at INFO level and drown out our own pipeline's logs. Quiet
    # them down to WARNING so "our" log lines stand out; this only
    # affects display, not our own modules' loggers.
    for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _configured = True

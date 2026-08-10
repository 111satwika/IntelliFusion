"""
Evaluation dataset loading.

Datasets are JSONL - one JSON object per line. Each line describes ONE
evaluation item. Only ``question`` is required; everything else is
optional and determines which metrics can be computed for that item.

Minimal item (LLM-judge metrics only):
    {"question": "What is a User Story?"}

Full item (LLM-judge + retrieval metrics):
    {
      "question": "How do I assign a User Story's Feature to a Bug?",
      "kb": "web",
      "expected_answer": "Open the Bug, click Feature, pick the parent User Story's Feature.",
      "expected_document_ids": ["https://help.spiratest.com/.../assign-feature.html"],
      "expected_chunk_ids": ["https://help.spiratest.com/.../assign-feature.html::3"],
      "notes": "Ground truth: 3 parent + 6 child chunks on this page."
    }

Field semantics:
  * ``kb`` - which knowledge base to search. Overrides the runner's
    default_kb for this one item. Useful when a single dataset mixes
    questions from multiple KBs.
  * ``expected_answer`` - free-form reference answer. Not used
    directly today (we don't do ROUGE/BLEU) but preserved in the
    report so a human reviewer can spot-check the model's answer
    against the reference.
  * ``expected_document_ids`` - list of document_ids that SHOULD
    appear in the retrieved set. Used for document-level hit rate.
    Matches the ``document_id`` metadata field on chunks (typically
    a URL for web, a file path for github, etc.).
  * ``expected_chunk_ids`` - list of "document_id::chunk_index"
    keys that SHOULD appear in the retrieved set. Used for
    chunk-level hit-rate/MRR/nDCG. Stricter than document_ids -
    getting the right document but the wrong chunk still counts as
    a miss here.
  * ``notes`` - human notes, ignored by the runner.

Why JSONL and not YAML/CSV: JSONL lets ground-truth ID lists (which
often contain URLs with colons/commas/quotes) live inline without any
escaping gymnastics, and each line is independently parseable so a
malformed item doesn't break the whole file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvalItem:
    """One question from the evaluation dataset."""

    question: str
    kb: str | None = None
    expected_answer: str | None = None
    expected_document_ids: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    notes: str | None = None

    @property
    def has_retrieval_ground_truth(self) -> bool:
        """Retrieval metrics (hit-rate/MRR/nDCG) only make sense when
        we know which chunks/docs SHOULD have been retrieved."""
        return bool(self.expected_document_ids or self.expected_chunk_ids)


def load_dataset(path: str | Path) -> list[EvalItem]:
    """Load a JSONL evaluation dataset from disk.

    Blank lines and lines starting with ``#`` (comment lines) are
    skipped, so contributors can annotate the dataset without breaking
    the parser. A malformed line logs a warning and is skipped rather
    than aborting the whole run - a partial dataset is more useful
    than none when someone is iterating on the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSONL at %s:%d - %s", path, lineno, exc
                )
                continue
            question = (obj.get("question") or "").strip()
            if not question:
                logger.warning(
                    "Skipping %s:%d - missing or empty 'question' field", path, lineno
                )
                continue
            items.append(
                EvalItem(
                    question=question,
                    kb=obj.get("kb") or None,
                    expected_answer=obj.get("expected_answer") or None,
                    expected_document_ids=list(obj.get("expected_document_ids") or []),
                    expected_chunk_ids=list(obj.get("expected_chunk_ids") or []),
                    notes=obj.get("notes") or None,
                )
            )
    logger.info("Loaded %d evaluation item(s) from %s", len(items), path)
    return items

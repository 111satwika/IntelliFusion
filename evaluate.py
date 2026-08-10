"""
CLI entry point for batch evaluation.

Usage:
    python evaluate.py \
        --dataset data/eval/web_eval_sample.jsonl \
        --kb web \
        --top-k 5 \
        --name web-baseline

Common flags:
  * --with-crag / --no-crag           - toggle Corrective RAG
  * --no-query-transform              - disable query transform
  * --no-faithfulness                 - skip that judge call
  * --no-answer-relevance             - skip that judge call
  * --no-context-relevance            - skip that judge call
  * --out-dir data/eval/reports       - where reports land

The report is written as (<timestamp>_<name>.md, .json) pair. The path
of the Markdown file is printed to stdout on success so shell pipelines
can pick it up with $(python evaluate.py …).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.evaluation.dataset import load_dataset
from app.evaluation.report import write_report
from app.evaluation.runner import EvalConfig, run_evaluation
from app.logging_config import configure_logging


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RAG evaluation harness against a JSONL dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSONL evaluation dataset (see app/evaluation/dataset.py).",
    )
    parser.add_argument(
        "--kb",
        default=None,
        help=(
            "Default KB for items that don't specify one. Omit for the "
            "cross-KB router path."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--name",
        default="eval",
        help="Short identifier for the report filename (slugified).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/eval/reports",
        help="Directory for the report pair.",
    )

    parser.add_argument(
        "--with-crag",
        action="store_true",
        help="Enable Corrective RAG for this run (default: off).",
    )
    parser.add_argument(
        "--no-query-transform",
        action="store_true",
        help="Disable query transformation for this run (default: on).",
    )
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Skip the faithfulness judge call.",
    )
    parser.add_argument(
        "--no-answer-relevance",
        action="store_true",
        help="Skip the answer-relevance judge call.",
    )
    parser.add_argument(
        "--no-context-relevance",
        action="store_true",
        help="Skip the context-relevance judge call.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv)

    dataset_path = Path(args.dataset)
    items = load_dataset(dataset_path)
    if not items:
        print(f"No usable items in {dataset_path}.", file=sys.stderr)
        return 1

    config = EvalConfig(
        default_kb=args.kb,
        top_k=args.top_k,
        use_crag=args.with_crag,
        use_query_transform=not args.no_query_transform,
        judge_faithfulness=not args.no_faithfulness,
        judge_answer_relevance=not args.no_answer_relevance,
        judge_context_relevance=not args.no_context_relevance,
    )

    logger = logging.getLogger("evaluate")
    logger.info(
        "Running evaluation: %d items, kb=%s, top_k=%d, crag=%s, qt=%s",
        len(items), config.default_kb, config.top_k,
        config.use_crag, config.use_query_transform,
    )

    def _progress(idx, total, item):
        logger.info("[%d/%d] %s", idx + 1, total, item.question[:80])

    report = run_evaluation(items, config, progress_callback=_progress)
    md_path, json_path = write_report(report, args.out_dir, name=args.name)

    print(str(md_path))  # for shell chaining
    logger.info(
        "Done. %d items, %d failed. Aggregate: %s",
        report.total_items, report.failed_items, report.aggregate(),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

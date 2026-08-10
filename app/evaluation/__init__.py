"""
Evaluation harness for the RAG pipeline.

Public entry points:
  * dataset.load_dataset(path) -> list[EvalItem]
  * runner.run_evaluation(items, config) -> EvalReport
  * report.write_report(report, out_dir) -> (md_path, json_path)

Everything else in this package is an implementation detail of the
harness and should not be imported from application code.

Design notes:
  * The harness runs the SAME retrieval + generation code paths that
    the UI and CLI use - not a parallel implementation. It just
    instruments each call for latency and captures the intermediate
    (chunks, answer) pair so we can score them.
  * All scoring is either deterministic (retrieval metrics against
    ground-truth chunk IDs) or LLM-as-judge (RAGAS-style
    faithfulness / answer-relevance / context-relevance) - no ML
    training, no external services, no API keys.
  * Metrics are computed once per item, then aggregated once per
    run. Individual item results are always preserved so the report
    can show per-question drill-down alongside the summary.
"""

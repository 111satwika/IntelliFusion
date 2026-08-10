"""
Evaluation page: batch-run a JSONL dataset of questions through the
same retrieval + generation pipeline used by the chat pages, and score
each answer on faithfulness / answer-relevance / context-relevance
(LLM-as-judge) and hit-rate / MRR / nDCG (when ground-truth chunk IDs
are provided in the dataset).

*** DEPRECATED (see app_ui.py's module docstring): does not display
the Precision@k/Recall@k metrics app.evaluation.metrics now computes -
those already flow through EvalReport/ItemResult here, but this page's
results table was never updated to show the two new columns; see the
FastAPI webapp's Evaluation page instead.
"""

from pathlib import Path

import streamlit as st

from app.evaluation.dataset import load_dataset as _load_eval_dataset
from app.evaluation.report import write_report as _write_eval_report
from app.evaluation.runner import EvalConfig, run_evaluation
from ui.icons import svg

st.markdown(
    f'<div class="page-title-row">{svg("bar_chart", size=24, stroke="var(--accent)")}<h1>Evaluation</h1></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Batch-run a JSONL dataset of questions through the SAME retrieval "
    "+ generation pipeline used in Chat, and score each answer on "
    "faithfulness / answer-relevance / context-relevance (LLM-as-judge) "
    "and hit-rate / MRR / nDCG (when ground-truth chunk IDs are "
    "provided in the dataset). A report is saved under "
    "`data/eval/reports/`."
)

# Discover ready-to-run datasets in the standard eval dir. Datasets
# live under data/eval/ (git-ignored user data by convention), and a
# starter web dataset is checked in there. This path resolves against
# the process's working directory (wherever `streamlit run` was
# launched from), not __file__, so it's unaffected by which script
# Streamlit is internally executing.
_eval_dir = Path("data/eval")
_eval_reports_dir = _eval_dir / "reports"
_default_datasets: list[Path] = []
if _eval_dir.exists():
    _default_datasets = sorted(_eval_dir.glob("*.jsonl"))

with st.expander("Configure & run evaluation", expanded=True):
    if not _default_datasets:
        st.info(
            "No JSONL datasets found in `data/eval/`. Create one following "
            "the schema in `app/evaluation/dataset.py`."
        )
    else:
        _dataset_options = [str(p.relative_to(Path("."))) for p in _default_datasets]
        _selected_dataset = st.selectbox(
            "Dataset (JSONL)",
            _dataset_options,
            help="One JSON object per line. See `app/evaluation/dataset.py` for the schema.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            _eval_default_kb = st.selectbox(
                "Default KB (for items without a `kb` field)",
                ["web", "markdown", "pdf", "docx", "github", "(auto)"],
                index=0,
                help="Items in the dataset can override this per-question.",
            )
            if _eval_default_kb == "(auto)":
                _eval_default_kb = None
            _eval_top_k = st.number_input(
                "top_k",
                min_value=1, max_value=20, value=5, step=1,
                help="How many chunks to retrieve per question.",
            )
            _eval_name = st.text_input(
                "Report name",
                value="web-baseline",
                help="Slugified into the report filename.",
            )
        with col_b:
            _eval_use_crag = st.checkbox(
                "Use Corrective RAG",
                value=False,
                help=(
                    "Runs the CRAG evaluator on every question. Adds "
                    "~15-25s per question on CPU. Off by default so "
                    "baseline runs finish quickly."
                ),
            )
            _eval_use_qt = st.checkbox(
                "Use query transformation",
                value=True,
                help="Applies the query-rewrite / paraphrase / HyDE expander (recommended).",
            )
            _eval_judge_faith = st.checkbox("Judge faithfulness", value=True)
            _eval_judge_ans = st.checkbox("Judge answer-relevance", value=True)
            _eval_judge_ctx = st.checkbox("Judge context-relevance", value=True)

        st.warning(
            "Each question takes ~30-60 s on CPU (generation + up to 3 "
            "judge calls). Ten questions ≈ 5-10 minutes. Keep the "
            "browser tab open until the report appears."
        )

        if st.button("Run evaluation"):
            _dataset_path = Path(_selected_dataset)
            with st.spinner(f"Loading {_dataset_path.name}…"):
                try:
                    _items = _load_eval_dataset(_dataset_path)
                except Exception as exc:  # noqa: BLE001 - user sees the error
                    st.error(f"Failed to load dataset: {exc}")
                    _items = []

            if _items:
                _config = EvalConfig(
                    default_kb=_eval_default_kb,
                    top_k=int(_eval_top_k),
                    use_crag=bool(_eval_use_crag),
                    use_query_transform=bool(_eval_use_qt),
                    judge_faithfulness=bool(_eval_judge_faith),
                    judge_answer_relevance=bool(_eval_judge_ans),
                    judge_context_relevance=bool(_eval_judge_ctx),
                )
                _progress = st.progress(0.0, text="Starting…")
                _status = st.empty()

                def _progress_cb(idx, total, item):
                    _progress.progress(
                        idx / max(total, 1),
                        text=f"[{idx + 1}/{total}] {item.question[:80]}",
                    )
                    _status.caption(f"Now running question {idx + 1} of {total}.")

                try:
                    _report = run_evaluation(_items, _config, progress_callback=_progress_cb)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Evaluation failed: {exc}")
                    _report = None

                if _report is not None:
                    _progress.progress(1.0, text="Writing report…")
                    _md_path, _json_path = _write_eval_report(
                        _report, _eval_reports_dir, name=_eval_name or "eval"
                    )
                    _progress.empty()
                    _status.empty()

                    # Persist the last report on session state so the
                    # user can page around Streamlit without losing it.
                    st.session_state["last_eval_report"] = _report
                    st.session_state["last_eval_report_md"] = _md_path
                    st.session_state["last_eval_report_json"] = _json_path
                    st.success(
                        f"Done. Report saved to `{_md_path}` and `{_json_path}`."
                    )

# Render whichever report is currently on session state so the user
# can review it without re-running.
_last_report = st.session_state.get("last_eval_report")
if _last_report is not None:
    st.markdown("#### Latest evaluation results")
    _agg = _last_report.aggregate()
    _latencies = [r.latency_total_s for r in _last_report.items if r.latency_total_s is not None]
    _avg_latency = sum(_latencies) / len(_latencies) if _latencies else None

    # (display name, value, is_fraction) - fraction metrics (0-1) get a
    # real proportional bar; latency doesn't share that scale, so its
    # bar is a plain decorative indicator, not a percentage.
    _metric_cards = [(k.replace("_", " ").title(), v, True) for k, v in _agg.items()]
    if _avg_latency is not None:
        _metric_cards.append(("Avg. Latency", _avg_latency, False))

    _cols = st.columns(3)
    for _i, (_name, _value, _is_fraction) in enumerate(_metric_cards):
        with _cols[_i % 3]:
            with st.container(border=True):
                if _value is None:
                    _val_str, _bar_pct, _bar_color = "—", 0, "var(--surface-sunken)"
                elif _is_fraction:
                    _val_str = f"{_value:.2f}"
                    _bar_pct = max(0, min(100, _value * 100))
                    _bar_color = "var(--accent)"
                else:
                    _val_str = f"{_value:.1f}s"
                    _bar_pct = 40
                    _bar_color = "var(--muted-2)"
                st.markdown(
                    f'<div style="font-size:0.75rem; color:var(--muted); font-weight:650; '
                    f'text-transform:uppercase; letter-spacing:0.03em;">{_name}</div>'
                    f'<div class="num" style="font-size:1.5rem; font-weight:600; margin-top:4px;">{_val_str}</div>'
                    f'<div style="height:4px; border-radius:999px; background:var(--surface-sunken); '
                    f'margin-top:10px; overflow:hidden;">'
                    f'<div style="height:100%; width:{_bar_pct}%; background:{_bar_color}; border-radius:999px;"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    with st.expander("Per-question breakdown", expanded=False):
        _rows = [r.metrics_row() for r in _last_report.items]
        st.dataframe(_rows, use_container_width=True)
        for i, r in enumerate(_last_report.items, start=1):
            with st.expander(f"{i}. {r.item.question}", expanded=False):
                if r.error:
                    st.error(f"Errored: {r.error}")
                    continue
                st.markdown(
                    f"**KB used:** `{r.kb_used or '-'}`  ·  "
                    f"**Chunks:** {len(r.chunks)}  ·  "
                    f"**Latency:** {r.latency_total_s:.2f}s "
                    f"(ctx {r.latency_context_prep_s:.2f}s + "
                    f"gen {r.latency_generation_s:.2f}s)"
                )
                st.markdown("**Answer:**")
                st.write(r.answer or "_(empty)_")
                if r.item.expected_answer:
                    st.markdown("**Reference answer (from dataset):**")
                    st.write(r.item.expected_answer)
    _md_path = st.session_state.get("last_eval_report_md")
    if _md_path is not None:
        st.caption(f"Markdown report: `{_md_path}`")

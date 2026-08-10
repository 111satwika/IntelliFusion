"""
CSS layer porting the approved design mockup's token system and
component styles into Streamlit.

Colors/radii/shadows below are copied verbatim from the mockup so the
two stay in sync - if the mockup's palette changes, update the tokens
here to match. Kept LIGHT-theme-only (matching .streamlit/config.toml's
`base = "light"`): Streamlit's native widgets don't auto-flip with the
OS dark-mode preference unless config.toml's base does too, so a
prefers-color-scheme block here would desync custom elements (pills,
icon tiles) from every native widget (buttons, inputs) sitting next to
them - worse than just committing to one theme.

Prefers Streamlit's documented `st-key-<key>` class convention (any
`key=`'d widget/container gets one - see streamlit/elements/widgets/
button.py) over guessed internal selectors wherever possible.
"""

import streamlit as st

_CSS = """
<style>
:root {
    --accent: #0E7C86;
    --accent-strong: #0A5F67;
    --accent-wash: #E3F1F0;
    --paper: #FAF9F6;
    --surface: #FFFFFF;
    --surface-sunken: #F2F1EC;
    --border: #E3E2DC;
    --border-strong: #CFCEC5;
    --ink: #12181A;
    --ink-soft: #3A3F3C;
    --muted: #6B6E68;
    --muted-2: #9A9C95;
    --good: #1F7A4D;
    --good-wash: #E4F2EA;
    --warn: #B8720A;
    --warn-wash: #FAF0DD;
    --bad: #B23B3B;
    --bad-wash: #F8E9E8;
    --radius: 10px;
    --radius-lg: 14px;
    --shadow: 0 1px 2px rgba(18, 24, 26, 0.05);
    --mono: ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono", Consolas, monospace;
}

html { font-size: 116%; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); background: var(--surface-sunken); }
h1 { letter-spacing: -0.015em; }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }

/* Page title with a leading custom-SVG icon (ui_pages/*.py). */
.page-title-row { display: flex; align-items: center; gap: 0.6rem; }
.page-title-row h1 { margin: 0; }
.page-title-row svg { flex-shrink: 0; }

/* ---------- Sidebar nav (ui/sidebar.py) ---------- */
[data-testid="stSidebar"] button {
    border-radius: 8px !important;
    font-size: 1.02rem !important;
    justify-content: flex-start !important;
}
/* Inactive nav rows: plain, borderless (key="nav_<title>"). */
[class*="st-key-nav_"] button {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--ink-soft) !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}
[class*="st-key-nav_"] button:hover {
    background: var(--surface) !important;
    color: var(--ink) !important;
}
/* Active nav row: light surface + accent text (key="navactive_<title>"),
   NOT a solid fill - matches the mockup's subtle active state. */
[class*="st-key-navactive_"] button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent-strong) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow) !important;
}

/* Account footer avatar - see ui/sidebar.py. */
.account-avatar {
    display: inline-flex; align-items: center; justify-content: center;
    width: 2rem; height: 2rem; border-radius: 999px;
    background: var(--accent); color: #fff; font-weight: 650; font-size: 0.9rem;
    flex-shrink: 0;
}

/* ---------- Source-type picker cards (ui_pages/sources.py) ---------- */
[class*="st-key-source_card_"] {
    min-height: 8.5rem;
    text-align: center;
}
[class*="st-key-source_card_"] button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    font-weight: 650 !important; font-size: 0.95rem !important;
}
[class*="st-key-source_card_active_"] {
    border-color: var(--accent) !important;
    background: var(--accent-wash) !important;
}

/* ---------- Status pills (ui/panels.py render_status_pill) ---------- */
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px 3px 8px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 650; white-space: nowrap;
}
.pill::before { content: ""; width: 6px; height: 6px; border-radius: 999px; background: currentColor; }
.pill-good { background: var(--good-wash); color: var(--good); }
.pill-warn { background: var(--warn-wash); color: var(--warn); }
.pill-bad { background: var(--bad-wash); color: var(--bad); }
.pill-muted { background: var(--surface-sunken); color: var(--muted); }

/* Small mono "chip" - query variants, dataset paths, etc. */
.chip {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: var(--mono); font-size: 0.78rem;
    background: var(--surface-sunken); border: 1px solid var(--border);
    border-radius: 6px; padding: 2px 7px;
}

/* ---------- Insight (observability) panel ---------- */
.insight-desc p { margin: 0 0 8px; font-size: 0.92rem; color: var(--ink-soft); }
.insight-desc p:last-child { margin-bottom: 0; }
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

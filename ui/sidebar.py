"""
Sidebar nav rail: brand header, a page switcher built from plain
st.button()/st.switch_page() calls, and a static account footer.

Active/inactive nav styling (see ui/theme.py) is driven by the
button's `key` PREFIX ("navactive_" vs "nav_") rather than Streamlit's
own primary/secondary/tertiary button kinds - this lets the CSS give
the active row the mockup's subtle "light surface + accent text" look
instead of Streamlit's solid-fill primary button style.

There is no authentication in this app - the footer name is just a
cosmetic label for this single-user local tool, not a real session.
"""

import streamlit as st

from ui.icons import svg

_ACCOUNT_NAME = "Satwika"

# Streamlit's built-in Material Symbols icon font - native widgets
# (st.button, st.Page) can only take an emoji or a ":material/xxx:"
# shortcode, not the exact custom SVGs in ui/icons.py, so nav icons
# use the closest Material Symbols equivalent instead of pixel-exact
# mockup icons. Names verified against the installed build's
# streamlit/material_icon_names.py.
_NAV_ICONS = {
    "Sources": ":material/folder:",
    "Chat": ":material/chat_bubble:",
    "Evaluation": ":material/bar_chart:",
    "Settings": ":material/settings:",
}


def render_nav(pages: list, current_title: str) -> None:
    st.sidebar.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.6rem; padding:0 0 1rem;">
            <div style="width:2.1rem; height:2.1rem; border-radius:8px; flex-shrink:0;
                        background:linear-gradient(155deg, var(--accent), var(--accent-strong));
                        display:flex; align-items:center; justify-content:center;">
                {svg("brand", size=17, stroke="#fff")}
            </div>
            <span style="font-weight:650; font-size:1.15rem; letter-spacing:-0.01em;">IntelliFusion</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    for page in pages:
        is_active = page.title == current_title
        # Key PREFIX (not just value) encodes active state for
        # ui/theme.py's CSS - see module docstring.
        key_prefix = "navactive" if is_active else "nav"
        # NOTE: st.switch_page must be called during normal script
        # execution, not from an on_click callback - calling it (or
        # anything that triggers a rerun) inside a callback is a
        # documented no-op that surfaces as a visible warning banner.
        clicked = st.sidebar.button(
            page.title,
            key=f"{key_prefix}_{page.title}",
            icon=_NAV_ICONS.get(page.title),
            use_container_width=True,
        )
        if clicked:
            st.switch_page(page)

    st.sidebar.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.6rem; padding-top:1rem;
                    margin-top:0.5rem; border-top:1px solid var(--border);">
            <span class="account-avatar">{_ACCOUNT_NAME[0].upper()}</span>
            <div style="padding-top:0.6rem;">
                <div style="font-weight:600; font-size:0.95rem;">{_ACCOUNT_NAME}</div>
                <div style="font-size:0.78rem; color:var(--muted);">Local workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

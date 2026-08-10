"""
Shared inline-SVG icon library, ported verbatim (path data) from the
approved design mockup so every raw-HTML render in the app (source
rows, KB status cards, chat avatars, insight panels) uses the exact
same marks as the reference design.

These are only usable in raw-HTML contexts (st.markdown with
unsafe_allow_html=True) - native Streamlit widgets (st.button,
st.Page) can only take an emoji or a ":material/xxx:" shortcode for
their `icon=`, not an arbitrary SVG, so the sidebar nav (ui/sidebar.py)
still uses Streamlit's Material Symbols font for its buttons - the
closest available match, not pixel-identical to the mockup. Page
titles and everything else render via raw st.markdown, so THOSE use
these exact SVGs.
"""

# Inner <path>/<circle>/... markup only (no outer <svg> tag), so callers
# can wrap it with svg() below at whatever size/color a given spot needs.
PATHS = {
    "brand": (
        '<rect x="8" y="2.5" width="12" height="15" rx="1.6" stroke-width="1.3" opacity="0.35"/>'
        '<rect x="5.5" y="4.5" width="12" height="15" rx="1.6" stroke-width="1.5" opacity="0.65"/>'
        '<rect x="3" y="6.5" width="12" height="15" rx="1.8" stroke-width="2"/>'
    ),
    "markdown": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>'
        '<path d="M9 15v-2l1.5 2L12 13v2"/><path d="M15 15v-2l1.5 2"/>'
    ),
    "pdf": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>'
        '<path d="M9.5 17v-4h1a1.5 1.5 0 0 1 0 3h-1"/><path d="M13.5 17v-4h1.5"/><path d="M13.5 15h1.2"/>'
    ),
    "docx": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>'
        '<path d="M8.5 13l1 4 1.2-4 1.2 4 1-4"/>'
    ),
    "web": (
        '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/>'
        '<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'
    ),
    "github": (
        '<path d="M15 22v-4.5c0-.63-.25-1.17-.66-1.58C17.14 15.44 19 13.4 19 10.5 19 6.36 15.87 3 12 3S5 6.36 5 10.5c0 2.9 1.86 4.94 4.66 5.42-.41.41-.66.95-.66 1.58V22"/>'
        '<path d="M9 19c-1.5.5-2.7-.5-3-1.5"/>'
    ),
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "bar_chart": (
        '<path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/>'
        '<rect x="12.5" y="8" width="3" height="10"/><rect x="18" y="5" width="3" height="13"/>'
    ),
    "settings": (
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 '
        '1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1'
        '-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a'
        '1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51'
        'V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65'
        ' 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>'
    ),
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "close": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "send": '<line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>',
}


def svg(name: str, *, size: int = 18, stroke: str = "currentColor", fill: str = "none") -> str:
    """Return a standalone <svg> element for one of the PATHS keys."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{PATHS[name]}</svg>'
    )


def icon_tile(name: str, *, tile: int = 38, radius: int = 9, bg: str, fg: str, icon_size: int = 19) -> str:
    """A rounded icon tile (colored background + centered icon) - used for
    source-list rows and KB status cards."""
    return (
        f'<div style="width:{tile}px; height:{tile}px; border-radius:{radius}px; background:{bg}; '
        f'display:flex; align-items:center; justify-content:center; flex-shrink:0;">'
        f'{svg(name, size=icon_size, stroke=fg)}</div>'
    )


def avatar_data_uri(name: str, *, bg_hex: str, fg_hex: str = "#ffffff") -> str:
    """
    A standalone base64 SVG data: URI for st.chat_message's `avatar=`
    param (which accepts anything st.image does, including data URIs).

    Uses LITERAL hex colors, not CSS var(--...) references: this SVG is
    decoded as its own standalone image document, with no access to the
    host page's CSS custom properties - var() would silently fail to
    resolve there even though it works fine in every other icon helper
    in this module (which render inline in the page's own DOM).
    """
    import base64

    full_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">'
        f'<circle cx="16" cy="16" r="16" fill="{bg_hex}"/>'
        f'<g transform="translate(7.5,7.5)">'
        f'<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="{fg_hex}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{PATHS[name]}</svg>'
        f'</g></svg>'
    )
    encoded = base64.b64encode(full_svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"

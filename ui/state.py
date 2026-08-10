"""
Session/tracker state shared across every page.

Pending-document tracking: documents ingested through the KB tabs
land in Chroma immediately (so the user can chat against them right
away), but are also recorded in data/pending_documents.json keyed by
Streamlit session id. A "Save" button per document removes it from
that tracker, promoting it to a normal permanent KB entry. A
"Discard" button both removes it from the tracker AND deletes its
chunks from the KB. On every app startup, any session in the tracker
whose last_seen_at is older than _ABANDONED_SESSION_MINUTES is
considered abandoned and its pending chunks are auto-deleted - this
is our stand-in for a real "session ended" hook (Streamlit doesn't
provide one).
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from app.vectorstore.store import delete_document, delete_repository

logger = logging.getLogger(__name__)

# ui/state.py -> ui/ -> repo root, so .parent.parent (NOT .parent) is
# required here to land on <repo>/data/pending_documents.json - this
# file lives one directory deeper than the old app_ui.py did.
_PENDING_TRACKER_PATH = Path(__file__).resolve().parent.parent / "data" / "pending_documents.json"
_ABANDONED_SESSION_MINUTES = 30

KB_LABELS = {
    "markdown": "Markdown",
    "pdf": "PDF",
    "docx": "DOCX",
    "web": "Web",
    "github": "GitHub",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_tracker() -> dict:
    """Load the pending-doc tracker, returning an empty structure if absent."""
    if not _PENDING_TRACKER_PATH.exists():
        return {"sessions": {}}
    try:
        return json.loads(_PENDING_TRACKER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Pending-doc tracker unreadable; starting fresh.")
        return {"sessions": {}}


def _save_tracker(tracker: dict) -> None:
    _PENDING_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PENDING_TRACKER_PATH.write_text(json.dumps(tracker, indent=2), encoding="utf-8")


def _empty_pending() -> dict:
    return {kb: [] for kb in KB_LABELS}


def _cleanup_abandoned_sessions(tracker: dict, current_session_id: str) -> dict:
    """
    Delete pending chunks from any session whose last_seen_at is older
    than _ABANDONED_SESSION_MINUTES (and isn't the current session).
    Returns the updated tracker. The threshold protects against
    accidentally nuking a still-open session in another browser tab.
    """
    threshold = datetime.now(timezone.utc) - timedelta(minutes=_ABANDONED_SESSION_MINUTES)
    surviving: dict[str, dict] = {}
    for session_id, session_data in tracker.get("sessions", {}).items():
        if session_id == current_session_id:
            surviving[session_id] = session_data
            continue
        try:
            last_seen = datetime.fromisoformat(session_data.get("last_seen_at", ""))
        except ValueError:
            last_seen = datetime.min.replace(tzinfo=timezone.utc)
        if last_seen >= threshold:
            surviving[session_id] = session_data
            continue
        # Abandoned - purge its pending chunks from the KBs.
        pending = session_data.get("pending", {})
        for kb, entries in pending.items():
            for entry in entries:
                if entry.get("repository"):
                    delete_repository(entry["repository"])
                elif entry.get("document_id"):
                    delete_document(entry["document_id"], kb=kb)
        logger.info("Cleaned up abandoned session %s (%d KB(s))", session_id, len(pending))
    tracker["sessions"] = surviving
    return tracker


def _ensure_session(tracker: dict, session_id: str) -> dict:
    """Make sure the current session has an entry in the tracker."""
    sessions = tracker.setdefault("sessions", {})
    if session_id not in sessions:
        sessions[session_id] = {
            "started_at": _now_iso(),
            "last_seen_at": _now_iso(),
            "pending": _empty_pending(),
        }
    else:
        sessions[session_id]["last_seen_at"] = _now_iso()
        # Defensively make sure every KB key exists (in case KB_LABELS
        # gained a new KB after the tracker file was first written).
        for kb in KB_LABELS:
            sessions[session_id]["pending"].setdefault(kb, [])
    return tracker


def get_pending(kb: str) -> list[dict]:
    tracker = _load_tracker()
    session = tracker.get("sessions", {}).get(st.session_state["session_id"], {})
    return session.get("pending", {}).get(kb, [])


def add_pending(kb: str, entry: dict) -> None:
    tracker = _load_tracker()
    tracker = _ensure_session(tracker, st.session_state["session_id"])
    tracker["sessions"][st.session_state["session_id"]]["pending"][kb].append(entry)
    _save_tracker(tracker)


def remove_pending(kb: str, entry_key: str, entry_value: str) -> None:
    tracker = _load_tracker()
    session = tracker.get("sessions", {}).get(st.session_state["session_id"])
    if not session:
        return
    session["pending"][kb] = [
        entry for entry in session["pending"][kb] if entry.get(entry_key) != entry_value
    ]
    _save_tracker(tracker)


def init_session() -> None:
    """
    Give this browser session a stable id, then run the "clean up
    sessions that were abandoned > _ABANDONED_SESSION_MINUTES ago"
    pass exactly once per new session (that's what the "session_id
    not yet in state" check gates) - the app_ui.py entry script
    re-executes on every interaction, but this guard keeps the
    expensive cleanup a one-time-per-session cost regardless.

    Also initializes the per-KB and cross-KB chat history lists.
    """
    import uuid

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex
        tracker = _load_tracker()
        tracker = _cleanup_abandoned_sessions(tracker, st.session_state["session_id"])
        tracker = _ensure_session(tracker, st.session_state["session_id"])
        _save_tracker(tracker)
    else:
        # Keep the current session's last_seen_at fresh, so a long chat
        # session (with no ingest activity) isn't mistaken for abandoned
        # by the next browser tab that opens the app.
        tracker = _ensure_session(_load_tracker(), st.session_state["session_id"])
        _save_tracker(tracker)

    if "messages_by_kb" not in st.session_state:
        st.session_state["messages_by_kb"] = {kb: [] for kb in KB_LABELS}
    if "messages_global" not in st.session_state:
        st.session_state["messages_global"] = []

    # Settings page owns these widgets (key="top_k"/"use_vision"), but
    # Chat/Sources read them from session_state too and may run before
    # the user has ever visited Settings in this session - seed sane
    # defaults so that works.
    st.session_state.setdefault("top_k", 3)
    st.session_state.setdefault("use_vision", False)

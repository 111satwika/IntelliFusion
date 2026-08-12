"""
Cookie-session-keyed pending-document tracker - FastAPI port of
ui/state.py's Streamlit-session-keyed version (same JSON schema, same
30-minute abandonment sweep), plus per-session settings (top_k,
use_vision, the RAG-quality toggles) and per-thread chat history that
used to live in st.session_state.

Pending-document tracking: documents ingested through the Sources
endpoints land in Chroma immediately (so they're queryable right
away), but are also recorded in data/pending_documents.json keyed by
an HttpOnly session cookie (see api/deps.py). A "save" action removes
a document from that tracker, promoting it to a normal permanent KB
entry. A "discard" action both removes it from the tracker AND
deletes its chunks from the KB. Any session whose last_seen_at is
older than _ABANDONED_SESSION_MINUTES is swept (its pending chunks
auto-deleted) whenever a brand-new session cookie is issued - this is
our stand-in for a real "browser tab closed" signal, which HTTP has no
way to observe directly.

Chat history: each session can hold up to 8 independent conversation
threads - one per KB tab on the Sources page ("markdown"/"pdf"/"docx"/
"web"/"github"/"audio"/"video") plus one for the cross-KB Chat page
(thread_key "__cross__", matching webapp/app.js's exact thread-identity
scheme).
Only the last _MAX_HISTORY_TURNS question/answer pairs are kept per
thread - this is context feeding retrieval/prompt-building (see
app.retrieval.contextualize, app.prompting.prompt_builder), not a
durable transcript, so unbounded growth would only cost token budget
for no benefit.

Concurrency: every load-mutate-save sequence below is wrapped in a
cross-process FileLock (data/pending_documents.json.lock), because
multiple browser tabs/sessions can hit these functions concurrently
under the FastAPI server (unlike the old single-session Streamlit
build). Without it, two concurrent read-modify-writes could each read
the same old state and the later write would silently discard the
earlier one (e.g. one tab's add_pending() lost because another tab's
set_settings() wrote over it). Writes are additionally staged to a
temp file and moved into place with os.replace() (atomic on both
Windows and POSIX), so a concurrent READ (which isn't itself covered
by the lock, since reads don't need it for correctness beyond this)
can never observe a half-written file.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from filelock import FileLock

logger = logging.getLogger(__name__)

_PENDING_TRACKER_PATH = Path(__file__).resolve().parent / "data" / "pending_documents.json"
_PENDING_TRACKER_LOCK_PATH = _PENDING_TRACKER_PATH.with_suffix(".json.lock")
_ABANDONED_SESSION_MINUTES = 30

# Matches app.vectorstore.store._DEFAULT_OWNER's value (duplicated
# rather than imported - this module is intentionally lightweight/
# dependency-free at import time, see the lazy import inside
# _cleanup_abandoned_sessions).
_DEFAULT_OWNER = "local"

# How many question/answer pairs are kept per conversation thread (see
# module docstring) - old turns are dropped oldest-first once a thread
# exceeds this.
_MAX_HISTORY_TURNS = 8

KB_LABELS = {
    "markdown": "Markdown",
    "pdf": "PDF",
    "docx": "DOCX",
    "web": "Web",
    "github": "GitHub",
    "audio": "Audio",
    "video": "Video",
}

_DEFAULT_SETTINGS = {
    "top_k": 3,
    "use_vision": False,
    "crag_enabled": False,
    "query_transform_enabled": False,
    "self_rag_enabled": False,
    "conversation_memory_enabled": False,
}


def _tracker_lock() -> FileLock:
    """One FileLock per call, guarding the load-mutate-save sequence
    in every mutating function below. A fresh FileLock instance is
    cheap (it just wraps the lock file path) and safe to create per
    call - re-entrancy isn't needed since nothing here nests locks."""
    _PENDING_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(_PENDING_TRACKER_LOCK_PATH))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_tracker() -> dict:
    if not _PENDING_TRACKER_PATH.exists():
        return {"sessions": {}}
    try:
        return json.loads(_PENDING_TRACKER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Pending-doc tracker unreadable; starting fresh.")
        return {"sessions": {}}


def _save_tracker(tracker: dict) -> None:
    """Write via a temp file + os.replace() so a concurrent reader can
    never see a partially-written file - write_text() alone can be
    interrupted mid-write (by another process's read, or a crash),
    while os.replace() is an atomic rename on both Windows and POSIX.
    """
    _PENDING_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(_PENDING_TRACKER_PATH.parent), prefix=".pending_documents_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2)
        os.replace(tmp_path, _PENDING_TRACKER_PATH)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _empty_pending() -> dict:
    return {kb: [] for kb in KB_LABELS}


def _cleanup_abandoned_sessions(tracker: dict, current_session_id: str) -> dict:
    """
    Delete pending chunks from any session whose last_seen_at is older
    than _ABANDONED_SESSION_MINUTES (and isn't the current session).
    """
    # Imported here (not at module scope) so importing session_store
    # doesn't eagerly pull in app.vectorstore.store's heavy deps
    # (chromadb, embedding models) for callers that never trigger a
    # cleanup sweep - same lazy-import pattern used elsewhere in app/.
    from app.vectorstore.store import delete_document, delete_repository

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
        pending = session_data.get("pending", {})
        for kb, entries in pending.items():
            for entry in entries:
                # "owner" is stamped onto every pending entry by
                # api/sources.py at add_pending() time - defaulting to
                # _DEFAULT_OWNER covers entries added before per-user
                # ownership existed, so an old pending record doesn't
                # crash the sweep on upgrade.
                entry_owner = entry.get("owner", _DEFAULT_OWNER)
                if entry.get("repository"):
                    delete_repository(entry["repository"], entry_owner)
                elif entry.get("document_id"):
                    delete_document(entry["document_id"], entry_owner, kb=kb)
        logger.info("Cleaned up abandoned session %s (%d KB(s))", session_id, len(pending))
    tracker["sessions"] = surviving
    return tracker


def _ensure_session(tracker: dict, session_id: str) -> dict:
    """Pending-document tracking ONLY - see module docstring's split
    between this (anonymous, browser-tab-scoped, subject to the
    30-minute abandonment sweep) and _ensure_user_record below
    (permanent account settings/history, never swept)."""
    sessions = tracker.setdefault("sessions", {})
    if session_id not in sessions:
        sessions[session_id] = {
            "started_at": _now_iso(),
            "last_seen_at": _now_iso(),
            "pending": _empty_pending(),
        }
    else:
        sessions[session_id]["last_seen_at"] = _now_iso()
        for kb in KB_LABELS:
            sessions[session_id]["pending"].setdefault(kb, [])
    return tracker


def _ensure_user_record(tracker: dict, owner: str) -> dict:
    """Settings + chat history ONLY, keyed by the real logged-in
    username (or LOCAL_OWNER when auth is disabled - see api.deps) -
    deliberately a separate top-level key from "sessions" above, and
    deliberately never touched by _cleanup_abandoned_sessions: an
    account's settings/history are meant to persist indefinitely, not
    expire after 30 minutes of inactivity the way an anonymous
    browser-tab session's in-progress pending uploads should."""
    users = tracker.setdefault("users", {})
    if owner not in users:
        users[owner] = {"settings": dict(_DEFAULT_SETTINGS), "history": {}}
    else:
        users[owner].setdefault("settings", dict(_DEFAULT_SETTINGS))
        users[owner].setdefault("history", {})
    return tracker


def new_session() -> str:
    """
    Mint a brand-new session id, sweep any abandoned sessions (the
    30-minute cleanup - see module docstring), and register the new
    session. Called once per browser that has no session cookie yet.
    """
    import uuid

    session_id = uuid.uuid4().hex
    with _tracker_lock():
        tracker = _load_tracker()
        tracker = _cleanup_abandoned_sessions(tracker, session_id)
        tracker = _ensure_session(tracker, session_id)
        _save_tracker(tracker)
    return session_id


def touch_session(session_id: str) -> None:
    """Bump last_seen_at for an existing session cookie (no cleanup sweep - see new_session())."""
    with _tracker_lock():
        tracker = _ensure_session(_load_tracker(), session_id)
        _save_tracker(tracker)


def get_pending(session_id: str, kb: str) -> list[dict]:
    tracker = _load_tracker()
    session = tracker.get("sessions", {}).get(session_id, {})
    return session.get("pending", {}).get(kb, [])


def add_pending(session_id: str, kb: str, entry: dict) -> None:
    with _tracker_lock():
        tracker = _load_tracker()
        tracker = _ensure_session(tracker, session_id)
        tracker["sessions"][session_id]["pending"][kb].append(entry)
        _save_tracker(tracker)


def remove_pending(session_id: str, kb: str, entry_key: str, entry_value: str) -> None:
    with _tracker_lock():
        tracker = _load_tracker()
        session = tracker.get("sessions", {}).get(session_id)
        if not session:
            return
        session["pending"][kb] = [
            entry for entry in session["pending"][kb] if entry.get(entry_key) != entry_value
        ]
        _save_tracker(tracker)


def get_settings(owner: str) -> dict:
    """owner: the logged-in username, or LOCAL_OWNER when auth is
    disabled (see api.deps.get_current_user) - settings now follow the
    account across browsers/devices, not one anonymous browser tab."""
    tracker = _load_tracker()
    user = tracker.get("users", {}).get(owner, {})
    return {**_DEFAULT_SETTINGS, **user.get("settings", {})}


def set_settings(owner: str, **updates) -> dict:
    with _tracker_lock():
        tracker = _load_tracker()
        tracker = _ensure_user_record(tracker, owner)
        tracker["users"][owner]["settings"].update(updates)
        _save_tracker(tracker)
        return tracker["users"][owner]["settings"]


def get_history(owner: str, thread_key: str) -> list[dict]:
    """
    Return this account's conversation turns for one thread
    (thread_key is a KB name for a Sources-page thread, or "__cross__"
    for the cross-KB Chat page - see module docstring), oldest first.
    Empty list for an account/thread that has never had a turn appended.
    """
    tracker = _load_tracker()
    user = tracker.get("users", {}).get(owner, {})
    return user.get("history", {}).get(thread_key, [])


def append_history(owner: str, thread_key: str, question: str, answer: str) -> None:
    """
    Record one completed question/answer turn for a thread, then trim
    to the last _MAX_HISTORY_TURNS turns (oldest dropped first). Called
    only after an answer has fully finished streaming (see api/chat.py)
    so a client that disconnects mid-stream never leaves a partial/
    empty answer in history.
    """
    with _tracker_lock():
        tracker = _load_tracker()
        tracker = _ensure_user_record(tracker, owner)
        history = tracker["users"][owner]["history"].setdefault(thread_key, [])
        history.append({"question": question, "answer": answer, "at": _now_iso()})
        del history[:-_MAX_HISTORY_TURNS]
        _save_tracker(tracker)


def clear_history(owner: str, thread_key: str) -> None:
    """Empty one thread's history, leaving every other thread (and every other account) untouched."""
    with _tracker_lock():
        tracker = _load_tracker()
        user = tracker.get("users", {}).get(owner)
        if not user:
            return
        user.setdefault("history", {})[thread_key] = []
        _save_tracker(tracker)

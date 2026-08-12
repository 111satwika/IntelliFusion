"""Session-cookie and current-user dependencies shared by every router."""

from fastapi import HTTPException, Request, Response

import session_store
from api import auth

SESSION_COOKIE_NAME = "session_id"
_SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

# Owner value used for every document/query/setting when no real login
# is required (auth.auth_enabled() is False - the default, matching
# plain local `uvicorn server:app` usage and every existing test). Lets
# ingestion/retrieval code always have SOME owner to scope by instead
# of needing a separate "multi-user mode?" branch everywhere.
LOCAL_OWNER = "local"


def get_session_id(request: Request, response: Response) -> str:
    """
    Read the session cookie, minting a new one (and running the
    30-minute abandoned-session sweep - see session_store.new_session)
    if the browser doesn't have one yet. Existing sessions just get
    their last_seen_at bumped.

    Deliberately untouched by the addition of get_current_user() below -
    this cookie stays anonymous/browser-scoped and keeps backing only
    the pending-document tracker (see session_store.py), which isn't
    privacy-sensitive (nothing in it is visible to anyone until saved).
    Settings/history moved to being keyed by the real username instead -
    see session_store.py's own docstring for the split.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id is None:
        session_id = session_store.new_session()
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            httponly=True,
            samesite="lax",
            max_age=_SESSION_COOKIE_MAX_AGE,
        )
    else:
        session_store.touch_session(session_id)
    return session_id


def get_current_user(request: Request) -> str:
    """
    The logged-in username, for every route that needs a real identity
    (settings, chat history, document ownership). Returns LOCAL_OWNER
    when auth.auth_enabled() is False, so callers never need an
    "if multi-user mode" branch - there's always some owner value.

    When auth IS enabled, server.py's auth_gate middleware has already
    rejected any request without a valid cookie before it reaches this
    dependency - the 401 raised here is a defense-in-depth backstop,
    not the primary enforcement point.
    """
    if not auth.auth_enabled():
        return LOCAL_OWNER
    username = auth.get_authenticated_user(request.cookies.get(auth.AUTH_COOKIE_NAME))
    if username is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return username

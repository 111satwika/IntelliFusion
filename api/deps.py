"""Session-cookie dependency shared by every router."""

from fastapi import Request, Response

import session_store

SESSION_COOKIE_NAME = "session_id"
_SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def get_session_id(request: Request, response: Response) -> str:
    """
    Read the session cookie, minting a new one (and running the
    30-minute abandoned-session sweep - see session_store.new_session)
    if the browser doesn't have one yet. Existing sessions just get
    their last_seen_at bumped.
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

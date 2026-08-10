"""
Optional single-password auth gate for deployments reachable beyond
localhost (see DEPLOYMENT.md).

Design:
- Fully opt-in via the APP_PASSWORD env var. If it's unset (the
  default - matches plain `uvicorn server:app` local usage and every
  existing test), auth_enabled() is False and nothing in this module
  changes any request's behavior - mirrors the same opt-in-via-env-var
  pattern already used for CRAG/query-transform/Self-RAG toggles and
  ENABLE_AUDIO_SIMILARITY_SEARCH elsewhere in this codebase.
- This is a single shared password for one trusted user/household, not
  a multi-user account system - deliberately as simple as the threat
  model (a self-hosted tool on your own LAN/server) calls for. See
  DEPLOYMENT.md for when you need more than this.
- The auth_token cookie is signed with HMAC-SHA256 (stdlib hashlib/hmac
  only - no new dependency) rather than reusing api.deps's session_id
  cookie, which is an unsigned random id used purely for RAG session
  *state* (pending docs, settings) and was never meant to prove
  identity.
"""

import hashlib
import hmac
import os
import time

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse

AUTH_COOKIE_NAME = "auth_token"
_AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days, matches api.deps's session cookie

router = APIRouter(tags=["auth"])


def auth_enabled() -> bool:
    return bool(os.environ.get("APP_PASSWORD"))


def _secret_key() -> str:
    # Falls back to the password itself if APP_SECRET_KEY isn't set -
    # works, but every process restart invalidates existing sessions
    # since nothing else is stable to derive a key from. DEPLOYMENT.md
    # recommends setting APP_SECRET_KEY explicitly to avoid that.
    return os.environ.get("APP_SECRET_KEY") or os.environ.get("APP_PASSWORD", "")


def _sign(expiry_ts: int) -> str:
    material = str(expiry_ts).encode("utf-8")
    return hmac.new(_secret_key().encode("utf-8"), material, hashlib.sha256).hexdigest()


def _make_token(ttl_seconds: int = _AUTH_COOKIE_MAX_AGE) -> str:
    expiry_ts = int(time.time()) + ttl_seconds
    return f"{expiry_ts}.{_sign(expiry_ts)}"


def is_authenticated(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    expiry_str, _, signature = token.partition(".")
    try:
        expiry_ts = int(expiry_str)
    except ValueError:
        return False
    if expiry_ts < int(time.time()):
        return False
    return hmac.compare_digest(_sign(expiry_ts), signature)


def _cookie_secure() -> bool:
    return os.environ.get("COOKIE_SECURE", "0") == "1"


_LOGIN_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>IntelliFusion — Sign in</title>
<style>
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: #0e1114; color: #e9ebee; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  form {{
    background: #161a1f; border: 1px solid #272d35; border-radius: 12px; padding: 32px 30px;
    width: 280px; box-shadow: 0 12px 32px rgba(0,0,0,0.45);
  }}
  h1 {{ font-size: 18px; margin: 0 0 18px; }}
  input {{
    width: 100%; box-sizing: border-box; padding: 9px 11px; border-radius: 8px; border: 1px solid #363d47;
    background: #1c2127; color: #e9ebee; font-size: 14px; margin-bottom: 14px;
  }}
  button {{
    width: 100%; padding: 9px 11px; border-radius: 8px; border: none; background: #57b8c8; color: #0e1114;
    font-weight: 700; font-size: 14px; cursor: pointer;
  }}
  .error {{ color: #e08585; font-size: 13px; margin: -6px 0 14px; }}
</style>
</head>
<body>
  <form method="post" action="/login">
    <h1>IntelliFusion</h1>
    {error}
    <input type="password" name="password" placeholder="Password" autofocus />
    <button type="submit">Sign in</button>
  </form>
</body>
</html>
"""


@router.get("/login")
def login_page():
    return HTMLResponse(_LOGIN_PAGE.format(error=""))


@router.post("/login")
def login_submit(password: str = Form(...)):
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or not hmac.compare_digest(password, expected):
        html = _LOGIN_PAGE.format(error='<div class="error">Incorrect password.</div>')
        return HTMLResponse(html, status_code=401)

    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        AUTH_COOKIE_NAME,
        _make_token(),
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(),
        max_age=_AUTH_COOKIE_MAX_AGE,
    )
    return response


@router.post("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(AUTH_COOKIE_NAME)
    return response

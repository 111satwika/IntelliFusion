"""
Optional per-user login gate for deployments reachable beyond
localhost (see DEPLOYMENT.md).

Design:
- Fully opt-in via the APP_PASSWORD env var - its VALUE is no longer a
  password (real per-account passwords now live in user_store.json,
  see that module), it's purely an on/off switch kept under its
  original name so an existing deployment's .env doesn't need to
  change to keep the gate enabled. If unset (the default - matches
  plain `uvicorn server:app` local usage and every existing test),
  auth_enabled() is False and nothing in this module changes any
  request's behavior - mirrors the same opt-in-via-env-var pattern
  used for CRAG/query-transform/Self-RAG toggles and
  ENABLE_AUDIO_SIMILARITY_SEARCH elsewhere in this codebase.
- Real, individual accounts (username + password each), not the single
  shared password this module started as - see user_store.py for
  account storage. Self-serve signup is open to anyone who can reach
  the server, same threat model as the shared password it replaces.
- The auth_token cookie is signed with HMAC-SHA256 (stdlib hashlib/hmac
  only - no new dependency) and now carries the username as part of
  the signed material (f"{username}.{expiry_ts}.{signature}") so a
  token proves WHICH account is authenticated, not just THAT some
  account is. Usernames are restricted to a delimiter-safe charset
  (see _USERNAME_RE) specifically so this dot-delimited format can
  never be ambiguous to parse. Deliberately a separate cookie from
  api.deps's session_id cookie, which is an unsigned random id used
  purely for RAG session *state* (pending docs) and was never meant to
  prove identity.
"""

import hashlib
import hmac
import os
import re
import time

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import user_store

AUTH_COOKIE_NAME = "auth_token"
_AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days, matches api.deps's session cookie

_USERNAME_RE = re.compile(r"^[a-z0-9_-]{3,32}$")
_MIN_PASSWORD_LENGTH = 8

router = APIRouter(tags=["auth"])


def auth_enabled() -> bool:
    return bool(os.environ.get("APP_PASSWORD"))


def _secret_key() -> str:
    # Falls back to APP_PASSWORD's own value if APP_SECRET_KEY isn't
    # set - works (it's just an on/off flag's value pressed into
    # service as a signing key, still a real secret string), but every
    # process restart invalidates existing sessions since nothing else
    # is stable to derive a key from. DEPLOYMENT.md recommends setting
    # APP_SECRET_KEY explicitly to avoid that.
    return os.environ.get("APP_SECRET_KEY") or os.environ.get("APP_PASSWORD", "")


def _normalize_username(username: str) -> str:
    return username.strip().lower()


def _sign(username: str, expiry_ts: int) -> str:
    material = f"{username}.{expiry_ts}".encode("utf-8")
    return hmac.new(_secret_key().encode("utf-8"), material, hashlib.sha256).hexdigest()


def _make_token(username: str, ttl_seconds: int = _AUTH_COOKIE_MAX_AGE) -> str:
    expiry_ts = int(time.time()) + ttl_seconds
    return f"{username}.{expiry_ts}.{_sign(username, expiry_ts)}"


def get_authenticated_user(token: str | None) -> str | None:
    """Return the signed-in username, or None if the token is missing,
    malformed, expired, or its signature doesn't match."""
    if not token or token.count(".") != 2:
        return None
    username, expiry_str, signature = token.split(".", 2)
    try:
        expiry_ts = int(expiry_str)
    except ValueError:
        return None
    if expiry_ts < int(time.time()):
        return None
    if not hmac.compare_digest(_sign(username, expiry_ts), signature):
        return None
    return username


def _cookie_secure() -> bool:
    return os.environ.get("COOKIE_SECURE", "0") == "1"


def _set_auth_cookie(response, username: str) -> None:
    response.set_cookie(
        AUTH_COOKIE_NAME,
        _make_token(username),
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(),
        max_age=_AUTH_COOKIE_MAX_AGE,
    )


_PAGE_STYLE = """
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
  .switch {{ margin-top: 14px; font-size: 12.5px; color: #8b929d; text-align: center; }}
  .switch a {{ color: #57b8c8; text-decoration: none; }}
"""

_LOGIN_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>IntelliFusion — Sign in</title>
<style>{style}</style>
</head>
<body>
  <form method="post" action="/login">
    <h1>IntelliFusion</h1>
    {error}
    <input type="text" name="username" placeholder="Username" autofocus autocapitalize="off" />
    <input type="password" name="password" placeholder="Password" />
    <button type="submit">Sign in</button>
    <div class="switch">No account? <a href="/signup">Create one</a></div>
  </form>
</body>
</html>
""".format(style=_PAGE_STYLE, error="{error}")

_SIGNUP_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>IntelliFusion — Create account</title>
<style>{style}</style>
</head>
<body>
  <form method="post" action="/signup">
    <h1>Create your account</h1>
    {error}
    <input type="text" name="username" placeholder="Username (3-32 chars, a-z 0-9 _ -)" autofocus autocapitalize="off" />
    <input type="password" name="password" placeholder="Password (min 8 characters)" />
    <input type="password" name="confirm_password" placeholder="Confirm password" />
    <button type="submit">Create account</button>
    <div class="switch">Already have an account? <a href="/login">Sign in</a></div>
  </form>
</body>
</html>
""".format(style=_PAGE_STYLE, error="{error}")


@router.get("/login")
def login_page():
    return HTMLResponse(_LOGIN_PAGE.format(error=""))


@router.post("/login")
def login_submit(username: str = Form(...), password: str = Form(...)):
    normalized = _normalize_username(username)
    if not user_store.verify_user(normalized, password):
        html = _LOGIN_PAGE.format(error='<div class="error">Incorrect username or password.</div>')
        return HTMLResponse(html, status_code=401)

    response = RedirectResponse(url="/", status_code=303)
    _set_auth_cookie(response, normalized)
    return response


@router.get("/signup")
def signup_page():
    return HTMLResponse(_SIGNUP_PAGE.format(error=""))


@router.post("/signup")
def signup_submit(
    username: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)
):
    normalized = _normalize_username(username)

    error = None
    if not _USERNAME_RE.match(normalized):
        error = "Username must be 3-32 characters: lowercase letters, numbers, underscore, or hyphen only."
    elif len(password) < _MIN_PASSWORD_LENGTH:
        error = f"Password must be at least {_MIN_PASSWORD_LENGTH} characters."
    elif password != confirm_password:
        error = "Passwords don't match."
    elif user_store.user_exists(normalized):
        error = "That username is already taken."

    if error:
        html = _SIGNUP_PAGE.format(error=f'<div class="error">{error}</div>')
        return HTMLResponse(html, status_code=400)

    user_store.create_user(normalized, password)
    response = RedirectResponse(url="/", status_code=303)
    _set_auth_cookie(response, normalized)
    return response


@router.post("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(AUTH_COOKIE_NAME)
    return response


@router.get("/api/auth/me")
def whoami(request: Request):
    if not auth_enabled():
        return JSONResponse({"username": "local"})
    username = get_authenticated_user(request.cookies.get(AUTH_COOKIE_NAME))
    return JSONResponse({"username": username})

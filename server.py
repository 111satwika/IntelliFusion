"""
FastAPI entry point for IntelliFusion.

Run with:
    uvicorn server:app --reload --port 8000

Then open http://localhost:8000

Responsibility: this file only wires routers and serves the frontend.
No retrieval/prompting/generation logic lives here - every route
module under api/ calls into the same app/ and cli.py pipeline the
original CLI used, so nothing here can drift from that pipeline.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.logging_config import configure_logging
from app.media.media_store import MEDIA_DIR
from api import auth, chat, evaluation, jobs_router, settings, sources

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="IntelliFusion")

if not auth.auth_enabled():
    logger.warning(
        "Starting with no login gate (APP_PASSWORD is not set) - anyone "
        "who can reach this server's port can use and modify your data. "
        "Fine for localhost-only use; set APP_PASSWORD in .env before "
        "exposing this beyond your own machine. See DEPLOYMENT.md."
    )

# Optional single-password gate (see api/auth.py + DEPLOYMENT.md) - a
# no-op unless APP_PASSWORD is set, so plain local `uvicorn
# server:app` usage and the test suite are unaffected. This is the
# only middleware in the app; everything else is per-route Depends()
# (see api/deps.py's get_session_id).
@app.middleware("http")
async def auth_gate(request: Request, call_next):
    if not auth.auth_enabled() or request.url.path == "/login":
        return await call_next(request)

    if auth.is_authenticated(request.cookies.get(auth.AUTH_COOKIE_NAME)):
        return await call_next(request)

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/login", status_code=303)
    return JSONResponse({"detail": "Not authenticated"}, status_code=401)


app.include_router(auth.router)
app.include_router(sources.router)
app.include_router(chat.router)
app.include_router(evaluation.router)
app.include_router(settings.router)
app.include_router(jobs_router.router)

_WEBAPP_DIR = Path(__file__).resolve().parent / "webapp"
_INDEX_PATH = _WEBAPP_DIR / "index.html"
_APP_JS_PATH = _WEBAPP_DIR / "app.js"

app.mount("/static", StaticFiles(directory=_WEBAPP_DIR), name="static")

# Serves video frame thumbnails + full audio/video source copies saved
# by app.media.media_store at ingest time, for native acoustic/visual
# similarity search result playback/display (see that module's
# docstring - website images already have a real external URL to
# render directly, local audio/video content doesn't). Must exist on
# disk before StaticFiles will mount it - created lazily by
# media_store's own save functions too, but guaranteed here so a
# fresh checkout with no media ingested yet still starts cleanly.
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")


@app.get("/")
def index():
    # Stamp app.js's own last-modified time onto its <script src>, so
    # editing app.js changes the URL the browser sees and a plain
    # refresh can never serve a stale cached copy - no manual
    # cache-busting version number to remember to bump.
    html = _INDEX_PATH.read_text(encoding="utf-8")
    version = int(_APP_JS_PATH.stat().st_mtime)
    html = html.replace('src="/static/app.js"', f'src="/static/app.js?v={version}"')
    return HTMLResponse(html)

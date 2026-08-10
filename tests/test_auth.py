"""Tests for api.auth - the opt-in single-password gate."""

import pytest
from fastapi.testclient import TestClient

from api import auth


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    monkeypatch.delenv("COOKIE_SECURE", raising=False)


def test_auth_disabled_when_app_password_unset():
    assert auth.auth_enabled() is False


def test_auth_enabled_when_app_password_set(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    assert auth.auth_enabled() is True


def test_token_roundtrip_is_valid(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    token = auth._make_token()
    assert auth.is_authenticated(token) is True


def test_none_or_missing_token_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    assert auth.is_authenticated(None) is False
    assert auth.is_authenticated("") is False


def test_expired_token_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    token = auth._make_token(ttl_seconds=-10)  # already expired
    assert auth.is_authenticated(token) is False


def test_tampered_signature_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    token = auth._make_token()
    expiry_str, _, signature = token.partition(".")
    tampered = f"{expiry_str}.{signature[:-1]}{'0' if signature[-1] != '0' else '1'}"
    assert auth.is_authenticated(tampered) is False


def test_token_signed_with_a_different_secret_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "secret-a")
    token = auth._make_token()

    monkeypatch.setenv("APP_SECRET_KEY", "secret-b")
    assert auth.is_authenticated(token) is False


def _build_test_app():
    """A minimal FastAPI app wired exactly like server.py's auth gate,
    without importing server.py itself (which eagerly loads the full
    RAG pipeline's routers)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, RedirectResponse

    app = FastAPI()

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

    @app.get("/")
    def protected_root():
        return {"ok": True}

    return app


def test_unauthenticated_html_request_redirects_to_login(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())

    response = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_unauthenticated_api_request_gets_401_json(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())

    response = client.get("/", headers={"accept": "application/json"})

    assert response.status_code == 401


def test_wrong_password_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())

    response = client.post("/login", data={"password": "wrong"})

    assert response.status_code == 401
    assert auth.AUTH_COOKIE_NAME not in response.cookies


def test_correct_password_logs_in_and_grants_access(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())

    login_response = client.post("/login", data={"password": "hunter2"}, follow_redirects=False)
    assert login_response.status_code == 303
    assert auth.AUTH_COOKIE_NAME in login_response.cookies

    protected_response = client.get("/")
    assert protected_response.status_code == 200
    assert protected_response.json() == {"ok": True}


def test_request_is_allowed_through_untouched_when_auth_disabled():
    client = TestClient(_build_test_app())

    response = client.get("/")

    assert response.status_code == 200

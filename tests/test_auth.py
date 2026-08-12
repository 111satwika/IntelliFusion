"""Tests for api.auth - the opt-in per-user login gate."""

import pytest
from fastapi.testclient import TestClient

import user_store
from api import auth


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    monkeypatch.delenv("COOKIE_SECURE", raising=False)


@pytest.fixture(autouse=True)
def _isolated_user_store(tmp_path, monkeypatch):
    """Every test gets a fresh, empty users.json - never touches the
    real data/users.json."""
    users_path = tmp_path / "users.json"
    monkeypatch.setattr(user_store, "_USERS_PATH", users_path)
    monkeypatch.setattr(user_store, "_USERS_LOCK_PATH", users_path.with_suffix(".json.lock"))


def test_auth_disabled_when_app_password_unset():
    assert auth.auth_enabled() is False


def test_auth_enabled_when_app_password_set(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    assert auth.auth_enabled() is True


def test_token_roundtrip_is_valid(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    token = auth._make_token("alice")
    assert auth.get_authenticated_user(token) == "alice"


def test_none_or_missing_token_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    assert auth.get_authenticated_user(None) is None
    assert auth.get_authenticated_user("") is None


def test_expired_token_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    token = auth._make_token("alice", ttl_seconds=-10)  # already expired
    assert auth.get_authenticated_user(token) is None


def test_tampered_signature_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    token = auth._make_token("alice")
    username, expiry_str, signature = token.split(".", 2)
    tampered = f"{username}.{expiry_str}.{signature[:-1]}{'0' if signature[-1] != '0' else '1'}"
    assert auth.get_authenticated_user(tampered) is None


def test_token_signed_with_a_different_secret_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "secret-a")
    token = auth._make_token("alice")

    monkeypatch.setenv("APP_SECRET_KEY", "secret-b")
    assert auth.get_authenticated_user(token) is None


def test_a_token_cannot_be_replayed_under_a_different_username(monkeypatch):
    # Someone can't take a valid token and just relabel the username
    # part - the signature covers both username and expiry together.
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    token = auth._make_token("alice")
    _, expiry_str, signature = token.split(".", 2)
    forged = f"bob.{expiry_str}.{signature}"
    assert auth.get_authenticated_user(forged) is None


def _build_test_app():
    """A minimal FastAPI app wired exactly like server.py's auth gate,
    without importing server.py itself (which eagerly loads the full
    RAG pipeline's routers)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, RedirectResponse

    app = FastAPI()
    unauthenticated_paths = {"/login", "/signup"}

    @app.middleware("http")
    async def auth_gate(request: Request, call_next):
        if not auth.auth_enabled() or request.url.path in unauthenticated_paths:
            return await call_next(request)
        if auth.get_authenticated_user(request.cookies.get(auth.AUTH_COOKIE_NAME)) is not None:
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


def test_signup_page_and_login_page_are_reachable_without_a_cookie(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())

    assert client.get("/login").status_code == 200
    assert client.get("/signup").status_code == 200


def test_signup_creates_an_account_and_logs_in(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())

    response = client.post(
        "/signup",
        data={"username": "Alice", "password": "correcthorse", "confirm_password": "correcthorse"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert auth.AUTH_COOKIE_NAME in response.cookies
    assert user_store.user_exists("alice")  # normalized to lowercase


def test_signup_rejects_duplicate_username_case_insensitively(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())
    client.post("/signup", data={"username": "alice", "password": "correcthorse", "confirm_password": "correcthorse"})

    response = client.post(
        "/signup",
        data={"username": "ALICE", "password": "differentpass", "confirm_password": "differentpass"},
    )

    assert response.status_code == 400
    assert "taken" in response.text.lower()


def test_signup_rejects_mismatched_passwords(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())

    response = client.post(
        "/signup",
        data={"username": "alice", "password": "correcthorse", "confirm_password": "somethingelse"},
    )

    assert response.status_code == 400
    assert auth.AUTH_COOKIE_NAME not in response.cookies


def test_signup_rejects_short_password(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())

    response = client.post(
        "/signup", data={"username": "alice", "password": "short", "confirm_password": "short"}
    )

    assert response.status_code == 400
    assert not user_store.user_exists("alice")


def test_signup_rejects_invalid_username_characters(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())

    response = client.post(
        "/signup",
        data={"username": "al ice!", "password": "correcthorse", "confirm_password": "correcthorse"},
    )

    assert response.status_code == 400


def test_wrong_password_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())
    client.post("/signup", data={"username": "alice", "password": "correcthorse", "confirm_password": "correcthorse"})
    client.post("/logout")

    response = client.post("/login", data={"username": "alice", "password": "wrong"})

    assert response.status_code == 401
    assert auth.AUTH_COOKIE_NAME not in response.cookies


def test_login_for_unknown_username_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())

    response = client.post("/login", data={"username": "nobody", "password": "whatever1"})

    assert response.status_code == 401


def test_correct_password_logs_in_and_grants_access(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())
    client.post("/signup", data={"username": "alice", "password": "correcthorse", "confirm_password": "correcthorse"})
    client.post("/logout")

    login_response = client.post(
        "/login", data={"username": "alice", "password": "correcthorse"}, follow_redirects=False
    )
    assert login_response.status_code == 303
    assert auth.AUTH_COOKIE_NAME in login_response.cookies

    protected_response = client.get("/")
    assert protected_response.status_code == 200
    assert protected_response.json() == {"ok": True}


def test_whoami_reflects_the_logged_in_user(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    client = TestClient(_build_test_app())
    client.post("/signup", data={"username": "alice", "password": "correcthorse", "confirm_password": "correcthorse"})

    response = client.get("/api/auth/me")

    assert response.json() == {"username": "alice"}


def test_whoami_returns_local_when_auth_disabled():
    client = TestClient(_build_test_app())

    response = client.get("/api/auth/me")

    assert response.json() == {"username": "local"}


def test_request_is_allowed_through_untouched_when_auth_disabled():
    client = TestClient(_build_test_app())

    response = client.get("/")

    assert response.status_code == 200

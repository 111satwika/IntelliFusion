"""
Real per-user accounts, backing api/auth.py's login/signup once real
usernames replace the old single shared APP_PASSWORD.

Mirrors session_store.py's exact pattern deliberately (same FileLock +
load-mutate-save + atomic-tempfile-os.replace() shape) rather than
inventing a second storage convention for what's structurally the same
problem: a small JSON file mutated by concurrent browser requests under
FastAPI.

Password hashing: stdlib hashlib.pbkdf2_hmac (no bcrypt/passlib
dependency - matches api/auth.py's own stdlib-only HMAC token signing).
200_000 iterations is a reasonable 2024+ baseline cost for PBKDF2-SHA256
on a local, non-mass-scale tool. A random 16-byte salt per user means
two users with the same password never produce the same stored hash.
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

logger = logging.getLogger(__name__)

_USERS_PATH = Path(__file__).resolve().parent / "data" / "users.json"
_USERS_LOCK_PATH = _USERS_PATH.with_suffix(".json.lock")

_PBKDF2_ITERATIONS = 200_000


def _users_lock() -> FileLock:
    _USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(_USERS_LOCK_PATH))


def _load_users() -> dict:
    if not _USERS_PATH.exists():
        return {"users": {}}
    try:
        return json.loads(_USERS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("User store unreadable; starting fresh.")
        return {"users": {}}


def _save_users(data: dict) -> None:
    _USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(_USERS_PATH.parent), prefix=".users_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, _USERS_PATH)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS).hex()


def user_exists(username: str) -> bool:
    return username in _load_users().get("users", {})


def create_user(username: str, password: str) -> bool:
    """
    Create a new account. Returns False (no-op) if the username is
    already taken - callers check this to show a "username taken"
    error rather than silently overwriting an existing account's
    password.
    """
    with _users_lock():
        data = _load_users()
        users = data.setdefault("users", {})
        if username in users:
            return False
        salt = secrets.token_bytes(16)
        users[username] = {
            "password_hash": _hash_password(password, salt),
            "salt": salt.hex(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_users(data)
        return True


def verify_user(username: str, password: str) -> bool:
    """True only if the username exists AND the password matches -
    hmac.compare_digest avoids a timing side-channel on the comparison
    itself (the same reasoning api/auth.py already applies to its own
    password check)."""
    record = _load_users().get("users", {}).get(username)
    if record is None:
        return False
    salt = bytes.fromhex(record["salt"])
    expected = _hash_password(password, salt)
    return hmac.compare_digest(expected, record["password_hash"])

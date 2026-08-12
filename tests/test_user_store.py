"""Tests for user_store.py - real per-user account storage."""

import pytest

import user_store


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    users_path = tmp_path / "users.json"
    monkeypatch.setattr(user_store, "_USERS_PATH", users_path)
    monkeypatch.setattr(user_store, "_USERS_LOCK_PATH", users_path.with_suffix(".json.lock"))


def test_user_does_not_exist_before_creation():
    assert user_store.user_exists("alice") is False


def test_create_user_then_exists():
    user_store.create_user("alice", "hunter2pass")

    assert user_store.user_exists("alice") is True


def test_create_user_returns_false_for_duplicate():
    assert user_store.create_user("alice", "hunter2pass") is True
    assert user_store.create_user("alice", "differentpass") is False


def test_verify_user_accepts_correct_password():
    user_store.create_user("alice", "hunter2pass")

    assert user_store.verify_user("alice", "hunter2pass") is True


def test_verify_user_rejects_wrong_password():
    user_store.create_user("alice", "hunter2pass")

    assert user_store.verify_user("alice", "wrongpass") is False


def test_verify_user_rejects_unknown_username():
    assert user_store.verify_user("nobody", "whatever") is False


def test_two_users_with_the_same_password_get_different_hashes():
    user_store.create_user("alice", "sharedpassword")
    user_store.create_user("bob", "sharedpassword")

    data = user_store._load_users()
    assert data["users"]["alice"]["password_hash"] != data["users"]["bob"]["password_hash"]
    assert data["users"]["alice"]["salt"] != data["users"]["bob"]["salt"]


def test_store_survives_a_fresh_load_from_disk():
    user_store.create_user("alice", "hunter2pass")

    # Simulate a new process reading the same file - no in-memory state to rely on.
    assert user_store.verify_user("alice", "hunter2pass") is True

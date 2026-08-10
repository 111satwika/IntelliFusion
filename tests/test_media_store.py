"""Tests for app.media.media_store.

MEDIA_DIR/FRAMES_DIR/SOURCES_DIR are monkeypatched to a tmp_path per
test, so nothing is written under the real repo's data/media/ during
the test run.
"""

from PIL import Image

from app.media import media_store


def _isolated_dirs(tmp_path, monkeypatch):
    media_dir = tmp_path / "media"
    frames_dir = media_dir / "frames"
    sources_dir = media_dir / "sources"
    monkeypatch.setattr(media_store, "MEDIA_DIR", media_dir)
    monkeypatch.setattr(media_store, "FRAMES_DIR", frames_dir)
    monkeypatch.setattr(media_store, "SOURCES_DIR", sources_dir)
    return frames_dir, sources_dir


def test_save_frame_thumbnail_writes_a_jpeg_and_returns_a_media_url(tmp_path, monkeypatch):
    frames_dir, _sources_dir = _isolated_dirs(tmp_path, monkeypatch)
    image = Image.new("RGB", (16, 16))

    url = media_store.save_frame_thumbnail("tutorial.mp4", 65.0, image)

    assert url.startswith("/media/frames/")
    assert url.endswith(".jpg")
    saved_path = frames_dir / url.rsplit("/", 1)[-1]
    assert saved_path.exists()


def test_save_frame_thumbnail_is_deterministic_per_document_and_timestamp(tmp_path, monkeypatch):
    _isolated_dirs(tmp_path, monkeypatch)
    image = Image.new("RGB", (16, 16))

    url_a = media_store.save_frame_thumbnail("tutorial.mp4", 65.0, image)
    url_b = media_store.save_frame_thumbnail("tutorial.mp4", 65.0, image)
    url_c = media_store.save_frame_thumbnail("tutorial.mp4", 70.0, image)

    assert url_a == url_b  # same document + timestamp -> same file, overwrite not duplicate
    assert url_a != url_c


def test_save_media_copy_copies_the_file_and_returns_a_media_url(tmp_path, monkeypatch):
    _sources_dir_unused, sources_dir = _isolated_dirs(tmp_path, monkeypatch)
    source_file = tmp_path / "meeting.mp3"
    source_file.write_bytes(b"fake audio bytes")

    url = media_store.save_media_copy("meeting.mp3", str(source_file))

    assert url.startswith("/media/sources/")
    assert url.endswith(".mp3")
    saved_path = sources_dir / url.rsplit("/", 1)[-1]
    assert saved_path.read_bytes() == b"fake audio bytes"


def test_save_media_copy_is_idempotent_skips_existing_target(tmp_path, monkeypatch):
    _sources_dir_unused, sources_dir = _isolated_dirs(tmp_path, monkeypatch)
    source_file = tmp_path / "meeting.mp3"
    source_file.write_bytes(b"original bytes")

    first_url = media_store.save_media_copy("meeting.mp3", str(source_file))
    saved_path = sources_dir / first_url.rsplit("/", 1)[-1]
    saved_path.write_bytes(b"pretend this was never touched again")

    source_file.write_bytes(b"changed bytes - should NOT be re-copied")
    second_url = media_store.save_media_copy("meeting.mp3", str(source_file))

    assert second_url == first_url
    assert saved_path.read_bytes() == b"pretend this was never touched again"


def test_save_media_copy_rejects_unrecognized_extension(tmp_path, monkeypatch):
    _isolated_dirs(tmp_path, monkeypatch)
    source_file = tmp_path / "notes.txt"
    source_file.write_bytes(b"not audio or video")

    url = media_store.save_media_copy("notes.txt", str(source_file))

    assert url is None


def test_delete_media_for_document_removes_frames_and_source_copy(tmp_path, monkeypatch):
    frames_dir, sources_dir = _isolated_dirs(tmp_path, monkeypatch)
    image = Image.new("RGB", (16, 16))
    source_file = tmp_path / "tutorial.mp4"
    source_file.write_bytes(b"fake video bytes")

    media_store.save_frame_thumbnail("tutorial.mp4", 0.0, image)
    media_store.save_frame_thumbnail("tutorial.mp4", 10.0, image)
    media_store.save_media_copy("tutorial.mp4", str(source_file))

    deleted = media_store.delete_media_for_document("tutorial.mp4")

    assert deleted == 3  # 2 frames + 1 source copy
    assert list(frames_dir.glob("*")) == []
    assert list(sources_dir.glob("*")) == []


def test_delete_media_for_document_leaves_other_documents_untouched(tmp_path, monkeypatch):
    frames_dir, sources_dir = _isolated_dirs(tmp_path, monkeypatch)
    image = Image.new("RGB", (16, 16))
    keep_file = tmp_path / "keep.mp4"
    keep_file.write_bytes(b"keep me")
    delete_file = tmp_path / "delete_me.mp4"
    delete_file.write_bytes(b"delete me")

    media_store.save_frame_thumbnail("keep.mp4", 0.0, image)
    media_store.save_media_copy("keep.mp4", str(keep_file))
    media_store.save_frame_thumbnail("delete_me.mp4", 0.0, image)
    media_store.save_media_copy("delete_me.mp4", str(delete_file))

    deleted = media_store.delete_media_for_document("delete_me.mp4")

    assert deleted == 2
    remaining_frames = list(frames_dir.glob("*"))
    remaining_sources = list(sources_dir.glob("*"))
    assert len(remaining_frames) == 1
    assert len(remaining_sources) == 1


def test_delete_media_for_document_is_a_noop_for_an_unknown_document(tmp_path, monkeypatch):
    _isolated_dirs(tmp_path, monkeypatch)

    deleted = media_store.delete_media_for_document("never-ingested.mp4")

    assert deleted == 0

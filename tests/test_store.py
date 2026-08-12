"""Tests for app.vectorstore.store.

Each test points Chroma at a fresh tmp_path directory instead of the
real data/chroma_db/, so tests never touch (or depend on) real
project data and are safe to run repeatedly/in any order.
"""

import pytest

from app.embeddings.embedder import EmbeddedChunk
from app.vectorstore import store


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CHROMA_DB_DIR", str(tmp_path))
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store, "_collections", {})
    monkeypatch.setattr(store, "_image_collection", None)
    monkeypatch.setattr(store, "_audio_clip_collection", None)
    yield store
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store, "_collections", {})
    monkeypatch.setattr(store, "_image_collection", None)
    monkeypatch.setattr(store, "_audio_clip_collection", None)


def test_add_and_query_roundtrip(isolated_store):
    chunks = [
        EmbeddedChunk(
            content="cats are great pets",
            embedding=[1.0, 0.0],
            metadata={"file_name": "doc.md", "chunk_index": 0},
        ),
        EmbeddedChunk(
            content="dogs are loyal companions",
            embedding=[0.0, 1.0],
            metadata={"file_name": "doc.md", "chunk_index": 1},
        ),
    ]

    isolated_store.add_embedded_chunks(chunks)

    assert isolated_store.count() == 2

    results = isolated_store.query_embedding([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["content"] == "cats are great pets"
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-6)


def test_add_embedded_chunks_with_empty_list_is_a_no_op(isolated_store):
    isolated_store.add_embedded_chunks([])
    # Collection is never even created, but count() should still work.
    assert isolated_store.count() == 0


def test_add_embedded_chunks_routes_each_chunk_to_its_kb_by_source_type(isolated_store):
    # Chunks with different source_type metadata must land in
    # DIFFERENT Chroma collections (KBs) - see store's module
    # docstring - not all pooled into one shared collection.
    chunks = [
        EmbeddedChunk(
            content="a markdown chunk",
            embedding=[1.0, 0.0],
            metadata={"document_id": "doc.md", "chunk_index": 0, "source_type": "markdown"},
        ),
        EmbeddedChunk(
            content="a pdf chunk",
            embedding=[1.0, 0.0],
            metadata={"document_id": "doc.pdf", "chunk_index": 0, "source_type": "pdf"},
        ),
        EmbeddedChunk(
            content="a docx chunk",
            embedding=[1.0, 0.0],
            metadata={"document_id": "doc.docx", "chunk_index": 0, "source_type": "docx"},
        ),
        EmbeddedChunk(
            content="a web chunk",
            embedding=[1.0, 0.0],
            metadata={"document_id": "https://example.com", "chunk_index": 0, "source_type": "web"},
        ),
        # GitHub-sourced files always get source_type="code" (see
        # app.ingestion.loader.load_github_repository), even a repo's
        # own README.md - they still land in the "github" KB, not
        # "markdown".
        EmbeddedChunk(
            content="a github file chunk",
            embedding=[1.0, 0.0],
            metadata={"document_id": "owner/repo:README.md", "chunk_index": 0, "source_type": "code"},
        ),
        EmbeddedChunk(
            content="an audio transcript chunk",
            embedding=[1.0, 0.0],
            metadata={"document_id": "meeting.mp3", "chunk_index": 0, "source_type": "audio"},
        ),
        EmbeddedChunk(
            content="a video transcript+frame chunk",
            embedding=[1.0, 0.0],
            metadata={"document_id": "tutorial.mp4", "chunk_index": 0, "source_type": "video"},
        ),
    ]

    isolated_store.add_embedded_chunks(chunks)

    assert isolated_store.count(kb="markdown") == 1
    assert isolated_store.count(kb="pdf") == 1
    assert isolated_store.count(kb="docx") == 1
    assert isolated_store.count(kb="web") == 1
    assert isolated_store.count(kb="github") == 1
    assert isolated_store.count(kb="audio") == 1
    assert isolated_store.count(kb="video") == 1
    assert isolated_store.count() == 7  # total across every KB


def test_add_embedded_chunks_falls_back_to_markdown_kb_for_unrecognized_source_type(isolated_store):
    chunks = [
        EmbeddedChunk(content="no source_type at all", embedding=[1.0, 0.0], metadata={"chunk_index": 0}),
    ]

    isolated_store.add_embedded_chunks(chunks)

    assert isolated_store.count(kb="markdown") == 1
    assert isolated_store.count(kb="pdf") == 0


def test_query_embedding_only_searches_the_requested_kb(isolated_store):
    isolated_store.add_embedded_chunks(
        [
            EmbeddedChunk(
                content="markdown content",
                embedding=[1.0, 0.0],
                metadata={"document_id": "doc.md", "chunk_index": 0, "source_type": "markdown"},
            ),
            EmbeddedChunk(
                content="pdf content",
                embedding=[1.0, 0.0],
                metadata={"document_id": "doc.pdf", "chunk_index": 0, "source_type": "pdf"},
            ),
        ]
    )

    markdown_results = isolated_store.query_embedding([1.0, 0.0], top_k=5, kb="markdown")
    pdf_results = isolated_store.query_embedding([1.0, 0.0], top_k=5, kb="pdf")

    assert [hit["content"] for hit in markdown_results] == ["markdown content"]
    assert [hit["content"] for hit in pdf_results] == ["pdf content"]


def test_list_populated_kbs_only_returns_kbs_with_data(isolated_store):
    assert isolated_store.list_populated_kbs("local") == []

    isolated_store.add_embedded_chunks(
        [
            EmbeddedChunk(
                content="pdf content",
                embedding=[1.0, 0.0],
                metadata={"document_id": "doc.pdf", "chunk_index": 0, "source_type": "pdf"},
            ),
        ]
    )

    assert isolated_store.list_populated_kbs("local") == ["pdf"]


def test_query_embedding_with_where_filter_scopes_to_matching_metadata(isolated_store):
    chunks = [
        EmbeddedChunk(
            content="repo A content",
            embedding=[1.0, 0.0],
            metadata={"document_id": "owner/repo-a:foo.py", "chunk_index": 0, "repository": "owner/repo-a"},
        ),
        EmbeddedChunk(
            content="repo B content",
            embedding=[1.0, 0.0],
            metadata={"document_id": "owner/repo-b:foo.py", "chunk_index": 0, "repository": "owner/repo-b"},
        ),
    ]

    isolated_store.add_embedded_chunks(chunks)

    results = isolated_store.query_embedding([1.0, 0.0], top_k=5, where={"repository": "owner/repo-b"})

    assert len(results) == 1
    assert results[0]["content"] == "repo B content"


def test_list_repositories_returns_distinct_repositories_only(isolated_store):
    chunks = [
        EmbeddedChunk(
            content="repo A content",
            embedding=[1.0, 0.0],
            metadata={"document_id": "owner/repo-a:foo.py", "chunk_index": 0, "repository": "owner/repo-a"},
        ),
        EmbeddedChunk(
            content="repo A content 2",
            embedding=[0.9, 0.1],
            metadata={"document_id": "owner/repo-a:bar.py", "chunk_index": 0, "repository": "owner/repo-a"},
        ),
        EmbeddedChunk(
            content="non-github content",
            embedding=[0.0, 1.0],
            metadata={"document_id": "README.md", "chunk_index": 0},
        ),
    ]

    isolated_store.add_embedded_chunks(chunks)

    assert isolated_store.list_repositories("local") == ["owner/repo-a"]


def test_list_repositories_returns_empty_list_when_none_ingested(isolated_store):
    assert isolated_store.list_repositories("local") == []


def test_multi_page_chunks_get_distinct_ids_instead_of_colliding(isolated_store):
    # Regression test: chunk_index restarts at 0 for every page of a
    # multi-page source (e.g. a PDF chunked one page at a time), so the
    # id must also include page_number or page 1's chunk 0 and page 2's
    # chunk 0 would overwrite each other under the same id.
    chunks = [
        EmbeddedChunk(
            content="page one content",
            embedding=[1.0, 0.0],
            metadata={"document_id": "sample.pdf", "file_name": "sample.pdf", "page_number": 1, "chunk_index": 0},
        ),
        EmbeddedChunk(
            content="page two content",
            embedding=[0.0, 1.0],
            metadata={"document_id": "sample.pdf", "file_name": "sample.pdf", "page_number": 2, "chunk_index": 0},
        ),
    ]

    isolated_store.add_embedded_chunks(chunks)

    assert isolated_store.count() == 2


def test_sanitize_metadata_joins_lists_and_drops_none_values():
    raw = {
        "block_types": ["heading", "table"],
        "note": None,
        "chunk_index": 2,
        "is_valid": True,
    }

    clean = store._sanitize_metadata(raw)

    assert clean["block_types"] == "heading, table"
    assert "note" not in clean
    assert clean["chunk_index"] == 2
    assert clean["is_valid"] is True


def test_get_table_chunk_returns_the_whole_table_not_a_row(isolated_store):
    chunks = [
        EmbeddedChunk(
            content="whole table content",
            embedding=[1.0, 0.0],
            metadata={
                "file_name": "doc.md",
                "chunk_index": 0,
                "content_type": "table",
                "table_id": "doc.md::table_0",
            },
        ),
        EmbeddedChunk(
            content="row 1 content",
            embedding=[0.9, 0.1],
            metadata={
                "file_name": "doc.md",
                "chunk_index": 1,
                "content_type": "table_row",
                "table_id": "doc.md::table_0",
            },
        ),
    ]

    isolated_store.add_embedded_chunks(chunks)

    result = isolated_store.get_table_chunk("doc.md::table_0", "local")

    assert result is not None
    assert result["content"] == "whole table content"
    assert result["metadata"]["content_type"] == "table"


def test_get_table_chunk_returns_none_when_no_match(isolated_store):
    assert isolated_store.get_table_chunk("nonexistent", "local") is None


def test_get_class_chunk_returns_the_whole_class_not_a_method(isolated_store):
    chunks = [
        EmbeddedChunk(
            content="[Class: Foo]\n\nclass Foo:\n    ...",
            embedding=[1.0, 0.0],
            metadata={
                "file_name": "foo.py",
                "chunk_index": 0,
                "content_type": "class",
                "class_id": "foo.py::class_0",
            },
        ),
        EmbeddedChunk(
            content="[Method: Foo.bar]\n\ndef bar(self):\n    ...",
            embedding=[0.9, 0.1],
            metadata={
                "file_name": "foo.py",
                "chunk_index": 1,
                "content_type": "method",
                "class_id": "foo.py::class_0",
            },
        ),
    ]

    isolated_store.add_embedded_chunks(chunks)

    result = isolated_store.get_class_chunk("foo.py::class_0", "local")

    assert result is not None
    assert result["content"] == "[Class: Foo]\n\nclass Foo:\n    ..."
    assert result["metadata"]["content_type"] == "class"


def test_get_class_chunk_returns_none_when_no_match(isolated_store):
    assert isolated_store.get_class_chunk("nonexistent", "local") is None


def test_get_chunks_by_content_type_returns_matching_chunks_only(isolated_store):
    chunks = [
        EmbeddedChunk(
            content="ocr text from image one",
            embedding=[1.0, 0.0],
            metadata={"document_id": "https://example.com/a.png", "chunk_index": 0, "content_type": "image_ocr"},
        ),
        EmbeddedChunk(
            content="ordinary page text",
            embedding=[0.0, 1.0],
            metadata={"document_id": "doc.md", "chunk_index": 0, "content_type": "text"},
        ),
    ]

    isolated_store.add_embedded_chunks(chunks)

    results = isolated_store.get_chunks_by_content_type("image_ocr")

    assert len(results) == 1
    assert results[0]["content"] == "ocr text from image one"
    assert results[0]["metadata"]["content_type"] == "image_ocr"
    # Storage id is owner-prefixed (see store.add_embedded_chunks) -
    # document_id itself (the metadata VALUE) stays untouched, only
    # the internal Chroma id changed shape.
    assert results[0]["id"] == "local::https://example.com/a.png::chunk_0"


def test_get_chunks_by_content_type_returns_empty_list_when_no_match(isolated_store):
    assert isolated_store.get_chunks_by_content_type("image_ocr") == []


def test_add_and_query_image_roundtrip(isolated_store):
    image_chunks = [
        {
            "content": "Authentication flow diagram",
            "embedding": [1.0, 0.0],
            "metadata": {
                "image_url": "https://example.com/auth-flow.png",
                "alt_text": "Authentication flow diagram",
                "page_url": "https://example.com/docs/api",
                "page_title": "API Authentication",
            },
        },
        {
            "content": "Dashboard screenshot",
            "embedding": [0.0, 1.0],
            "metadata": {
                "image_url": "https://example.com/dashboard.png",
                "alt_text": "Dashboard screenshot",
                "page_url": "https://example.com/docs/dashboard",
                "page_title": "Dashboard",
            },
        },
    ]

    isolated_store.add_image_chunks(image_chunks)

    assert isolated_store.image_count() == 2
    # Text-chunk collection stays untouched by image storage.
    assert isolated_store.count() == 0

    results = isolated_store.query_image_embedding([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["content"] == "Authentication flow diagram"
    assert results[0]["metadata"]["image_url"] == "https://example.com/auth-flow.png"
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-6)


def test_add_image_chunks_with_empty_list_is_a_no_op(isolated_store):
    isolated_store.add_image_chunks([])
    assert isolated_store.image_count() == 0


def test_add_image_chunks_upserts_by_image_url(isolated_store):
    image_chunk = {
        "content": "old alt text",
        "embedding": [1.0, 0.0],
        "metadata": {"image_url": "https://example.com/pic.png", "alt_text": "old alt text"},
    }
    isolated_store.add_image_chunks([image_chunk])

    updated_chunk = {
        "content": "new alt text",
        "embedding": [0.0, 1.0],
        "metadata": {"image_url": "https://example.com/pic.png", "alt_text": "new alt text"},
    }
    isolated_store.add_image_chunks([updated_chunk])

    assert isolated_store.image_count() == 1
    results = isolated_store.query_image_embedding([0.0, 1.0], top_k=1)
    assert results[0]["content"] == "new alt text"


def test_query_image_embedding_where_filter_scopes_by_source_kb(isolated_store):
    # Regression test: without a working where filter, a web
    # screenshot (or a frame from an unrelated video) can outrank/
    # replace the correct frame for a KB-scoped video question.
    image_chunks = [
        {
            "content": "video frame",
            "embedding": [1.0, 0.0],
            "metadata": {"image_url": "/media/frames/a.jpg", "source_kb": "video"},
        },
        {
            "content": "web screenshot",
            "embedding": [1.0, 0.0],
            "metadata": {"image_url": "https://example.com/b.png", "source_kb": "web"},
        },
    ]
    isolated_store.add_image_chunks(image_chunks)

    results = isolated_store.query_image_embedding([1.0, 0.0], top_k=5, where={"source_kb": "video"})

    assert len(results) == 1
    assert results[0]["content"] == "video frame"


def test_add_and_query_audio_clip_roundtrip(isolated_store):
    clip_chunks = [
        {
            "content": "Clip from meeting.mp3 at 00:00",
            "embedding": [1.0, 0.0],
            "metadata": {"document_id": "meeting.mp3", "start_seconds": 0.0, "end_seconds": 10.0},
        },
        {
            "content": "Clip from meeting.mp3 at 00:10",
            "embedding": [0.0, 1.0],
            "metadata": {"document_id": "meeting.mp3", "start_seconds": 10.0, "end_seconds": 20.0},
        },
    ]

    isolated_store.add_audio_clip_chunks(clip_chunks)

    assert isolated_store.audio_clip_count() == 2
    # Neither the text-chunk nor the image collection is touched by
    # audio-clip storage - three genuinely separate embedding spaces.
    assert isolated_store.count() == 0
    assert isolated_store.image_count() == 0

    results = isolated_store.query_audio_clip_embedding([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["content"] == "Clip from meeting.mp3 at 00:00"
    assert results[0]["metadata"]["document_id"] == "meeting.mp3"
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-6)


def test_query_audio_clip_embedding_where_filter_scopes_by_source_kb(isolated_store):
    # Regression test: without a working where filter, a clip from an
    # unrelated file (different source_kb) can outrank/replace the
    # correct clip for a KB-scoped question.
    clip_chunks = [
        {
            "content": "video clip",
            "embedding": [1.0, 0.0],
            "metadata": {"document_id": "surf.mp4", "start_seconds": 0.0, "source_kb": "video"},
        },
        {
            "content": "unrelated audio clip",
            "embedding": [1.0, 0.0],
            "metadata": {"document_id": "unrelated.mp3", "start_seconds": 0.0, "source_kb": "audio"},
        },
    ]
    isolated_store.add_audio_clip_chunks(clip_chunks)

    results = isolated_store.query_audio_clip_embedding([1.0, 0.0], top_k=5, where={"source_kb": "video"})

    assert len(results) == 1
    assert results[0]["content"] == "video clip"


def test_add_audio_clip_chunks_with_empty_list_is_a_no_op(isolated_store):
    isolated_store.add_audio_clip_chunks([])
    assert isolated_store.audio_clip_count() == 0


def test_add_audio_clip_chunks_upserts_by_document_id_and_start_seconds(isolated_store):
    clip_chunk = {
        "content": "old label",
        "embedding": [1.0, 0.0],
        "metadata": {"document_id": "meeting.mp3", "start_seconds": 0.0},
    }
    isolated_store.add_audio_clip_chunks([clip_chunk])

    updated_chunk = {
        "content": "new label",
        "embedding": [0.0, 1.0],
        "metadata": {"document_id": "meeting.mp3", "start_seconds": 0.0},
    }
    isolated_store.add_audio_clip_chunks([updated_chunk])

    assert isolated_store.audio_clip_count() == 1
    results = isolated_store.query_audio_clip_embedding([0.0, 1.0], top_k=1)
    assert results[0]["content"] == "new label"


def test_sample_kb_embeddings_returns_empty_list_for_unpopulated_kb(isolated_store):
    assert isolated_store.sample_kb_embeddings("markdown") == []


def test_sample_kb_embeddings_returns_stored_vectors(isolated_store):
    chunks = [
        EmbeddedChunk(content="a", embedding=[1.0, 0.0], metadata={"file_name": "doc.md", "chunk_index": 0}),
        EmbeddedChunk(content="b", embedding=[0.0, 1.0], metadata={"file_name": "doc.md", "chunk_index": 1}),
    ]
    isolated_store.add_embedded_chunks(chunks)

    sample = isolated_store.sample_kb_embeddings("markdown", limit=10)

    assert len(sample) == 2
    assert sorted(tuple(v) for v in sample) == [(0.0, 1.0), (1.0, 0.0)]


def test_sample_kb_embeddings_caps_at_limit(isolated_store):
    chunks = [
        EmbeddedChunk(content=f"chunk {i}", embedding=[float(i), 0.0], metadata={"file_name": "doc.md", "chunk_index": i})
        for i in range(10)
    ]
    isolated_store.add_embedded_chunks(chunks)

    sample = isolated_store.sample_kb_embeddings("markdown", limit=3)

    assert len(sample) == 3


def test_sample_kb_embeddings_only_looks_at_the_requested_kb(isolated_store):
    isolated_store.add_embedded_chunks([
        EmbeddedChunk(content="md", embedding=[1.0, 0.0], metadata={"source_type": "markdown", "file_name": "a.md", "chunk_index": 0}),
        EmbeddedChunk(content="pdf", embedding=[0.0, 1.0], metadata={"source_type": "pdf", "file_name": "b.pdf", "chunk_index": 0}),
    ])

    assert len(isolated_store.sample_kb_embeddings("markdown", limit=10)) == 1
    assert len(isolated_store.sample_kb_embeddings("pdf", limit=10)) == 1


def test_sample_kb_embeddings_returns_native_python_floats_not_numpy_scalars(isolated_store):
    # Regression test: Chroma stores/returns embeddings as numpy float32
    # arrays, and a bare `list(vector)` keeps numpy.float32 scalars in
    # the returned list rather than converting to native floats. That's
    # invisible to == comparisons (numpy.float32(1.0) == 1.0 is True,
    # see the roundtrip test above) but poisons every score derived
    # from these vectors downstream (app.routing.router's cosine
    # similarities) with numpy scalar types - in particular,
    # `numpy_float >= threshold` produces numpy.bool_, which (unlike a
    # native bool) json.dumps cannot serialize. This broke the entire
    # /api/chat/stream response with a 500 mid-stream once routing
    # scores reached insight.py's JSON payload.
    isolated_store.add_embedded_chunks([
        EmbeddedChunk(content="a", embedding=[1.0, 0.0], metadata={"file_name": "doc.md", "chunk_index": 0}),
    ])

    sample = isolated_store.sample_kb_embeddings("markdown", limit=10)

    assert all(type(component) is float for vector in sample for component in vector)


def test_delete_document_removes_matching_text_chunks(isolated_store):
    chunks = [
        EmbeddedChunk(content="a", embedding=[1.0, 0.0], metadata={"document_id": "keep.md", "chunk_index": 0}),
        EmbeddedChunk(content="b", embedding=[1.0, 0.0], metadata={"document_id": "delete.md", "chunk_index": 0}),
    ]
    isolated_store.add_embedded_chunks(chunks)

    deleted = isolated_store.delete_document("delete.md", "local")

    assert deleted == 1
    assert isolated_store.count() == 1


def test_delete_document_removes_video_frames_from_the_image_collection(isolated_store, monkeypatch):
    # Regression test: before this, deleting a video's text chunks left
    # its frames orphaned in the image collection forever - there was
    # no cleanup path for them at all.
    monkeypatch.setattr("app.media.media_store.delete_media_for_document", lambda document_id, owner: 0)
    isolated_store.add_embedded_chunks([
        EmbeddedChunk(content="transcript", embedding=[1.0, 0.0], metadata={"document_id": "tutorial.mp4", "chunk_index": 0, "source_type": "video"}),
    ])
    isolated_store.add_image_chunks([
        {
            "content": "Frame from tutorial.mp4 at 00:00",
            "embedding": [1.0, 0.0],
            "metadata": {"image_url": "/media/frames/abc_0.jpg", "video_document_id": "tutorial.mp4", "origin": "video_frame"},
        },
        {
            "content": "an unrelated web screenshot",
            "embedding": [1.0, 0.0],
            "metadata": {"image_url": "https://example.com/other.png", "origin": "web_image"},
        },
    ])

    isolated_store.delete_document("tutorial.mp4", "local")

    assert isolated_store.image_count() == 1  # only the unrelated web image survives
    remaining = isolated_store.query_image_embedding([1.0, 0.0], top_k=5)
    assert all(hit["metadata"].get("video_document_id") != "tutorial.mp4" for hit in remaining)


def test_delete_document_removes_web_page_images_by_page_url(isolated_store, monkeypatch):
    monkeypatch.setattr("app.media.media_store.delete_media_for_document", lambda document_id, owner: 0)
    isolated_store.add_image_chunks([
        {
            "content": "diagram",
            "embedding": [1.0, 0.0],
            "metadata": {"image_url": "https://example.com/diagram.png", "page_url": "https://example.com/docs", "origin": "web_image"},
        },
    ])

    isolated_store.delete_document("https://example.com/docs", "local")

    assert isolated_store.image_count() == 0


def test_delete_document_removes_audio_clips(isolated_store, monkeypatch):
    monkeypatch.setattr("app.media.media_store.delete_media_for_document", lambda document_id, owner: 0)
    isolated_store.add_audio_clip_chunks([
        {
            "content": "clip",
            "embedding": [1.0, 0.0],
            "metadata": {"document_id": "meeting.mp3", "start_seconds": 0.0},
        },
    ])

    isolated_store.delete_document("meeting.mp3", "local")

    assert isolated_store.audio_clip_count() == 0


def test_delete_document_calls_media_store_cleanup_with_the_document_id(isolated_store, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.media.media_store.delete_media_for_document",
        lambda document_id, owner: calls.append((document_id, owner)) or 0,
    )

    isolated_store.delete_document("tutorial.mp4", "local")

    assert calls == [("tutorial.mp4", "local")]

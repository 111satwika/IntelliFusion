"""Tests for app.ingestion.ingest.

chunk_with_parent_child, embed_chunks, and add_embedded_chunks are
all faked out here so this only verifies ingest.py's own logic:
discovering files by extension, using the right loader for each,
flattening single-Document vs list[Document] loader results, skipping
unsupported file types, reading URLs from a urls file, and skipping
unreachable URLs - without the cost of loading a real embedding model,
touching a real vector store, or making a real HTTP request.
"""

import requests

from app.ingestion import ingest
from app.ingestion.loader import Document
from app.ocr.frame_analysis import FrameAnalysis
from app.ocr.image_classifier import CODE, OTHER


def test_load_all_documents_uses_matching_loader_per_extension(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("markdown content", encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"fake pdf bytes")
    (tmp_path / "c.txt").write_text("unsupported", encoding="utf-8")

    def fake_load_markdown(path):
        return Document(content="markdown content", metadata={"file_name": "a.md"})

    def fake_load_pdf(path):
        return [
            Document(content="page 1", metadata={"file_name": "b.pdf", "page_number": 1}),
            Document(content="page 2", metadata={"file_name": "b.pdf", "page_number": 2}),
        ]

    monkeypatch.setitem(ingest._LOADERS_BY_EXTENSION, ".md", fake_load_markdown)
    monkeypatch.setitem(ingest._LOADERS_BY_EXTENSION, ".pdf", fake_load_pdf)

    documents = ingest._load_all_documents(str(tmp_path))

    assert len(documents) == 3
    assert documents[0].metadata["file_name"] == "a.md"
    assert documents[1].metadata["page_number"] == 1
    assert documents[2].metadata["page_number"] == 2


def test_loaders_by_extension_includes_audio_and_video():
    from app.ingestion.loader import load_audio_document, load_video_document

    for ext in (".mp3", ".wav", ".m4a", ".flac", ".ogg"):
        assert ingest._LOADERS_BY_EXTENSION[ext] is load_audio_document
    for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
        assert ingest._LOADERS_BY_EXTENSION[ext] is load_video_document


def test_ingest_all_chunks_embeds_and_stores_every_document(tmp_path, monkeypatch):
    (tmp_path / "only.md").write_text("content", encoding="utf-8")

    fake_documents = [Document(content="content", metadata={"file_name": "only.md"})]
    calls = []

    monkeypatch.setattr(ingest, "_load_all_documents", lambda raw_dir: fake_documents)
    monkeypatch.setattr(ingest, "_load_all_websites", lambda urls_file: [])
    # chunk_with_parent_child is a drop-in replacement for chunk_document
    # that also emits child sentence windows for github/web KBs; ingest.py
    # now calls it in place of chunk_document, so tests mock it directly.
    monkeypatch.setattr(ingest, "chunk_with_parent_child", lambda doc, chunk_size, chunk_overlap: ["chunk1", "chunk2"])
    monkeypatch.setattr(ingest, "embed_chunks", lambda chunks: calls.append(("embed_chunks", chunks)) or ["embedded1", "embedded2"])
    monkeypatch.setattr(ingest, "add_embedded_chunks", lambda embedded: calls.append(("add_embedded_chunks", embedded)))
    monkeypatch.setattr(ingest, "count", lambda: 2)

    # urls_file points at a nonexistent path in tmp_path (rather than
    # the real default data/urls.txt) so this test can't accidentally
    # make a real network request against whatever's configured there.
    total = ingest.ingest_all(raw_dir=str(tmp_path), urls_file=str(tmp_path / "urls.txt"))

    assert total == 2
    assert calls == [
        ("embed_chunks", ["chunk1", "chunk2"]),
        ("add_embedded_chunks", ["embedded1", "embedded2"]),
    ]


def test_read_urls_skips_blank_lines_and_comments(tmp_path):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text(
        "https://example.com/one\n"
        "\n"
        "# a comment line\n"
        "https://example.com/two\n",
        encoding="utf-8",
    )

    urls = ingest._read_urls(str(urls_file))

    assert urls == ["https://example.com/one", "https://example.com/two"]


def test_ingest_github_repo_chunks_embeds_and_stores_every_file(monkeypatch):
    fake_documents = [
        Document(content="print('hi')", metadata={"document_id": "owner/repo:a.py"}),
        Document(content="# Readme", metadata={"document_id": "owner/repo:README.md"}),
    ]
    calls = []

    monkeypatch.setattr(
        ingest,
        "load_github_repository",
        lambda repo_url, branch, max_files: calls.append(("load", repo_url, branch, max_files)) or fake_documents,
    )
    monkeypatch.setattr(ingest, "chunk_with_parent_child", lambda doc, chunk_size, chunk_overlap: ["chunk"])
    monkeypatch.setattr(
        ingest,
        "embed_chunks",
        lambda chunks: calls.append(("embed_chunks", chunks)) or ["embedded"],
    )
    monkeypatch.setattr(
        ingest,
        "add_embedded_chunks",
        lambda embedded: calls.append(("add_embedded_chunks", embedded)),
    )
    monkeypatch.setattr(ingest, "count", lambda: 2)

    total = ingest.ingest_github_repo("owner/repo", branch="main")

    assert total == 2
    assert calls[0] == ("load", "owner/repo", "main", 300)
    assert calls.count(("embed_chunks", ["chunk"])) == 2
    assert calls.count(("add_embedded_chunks", ["embedded"])) == 2


def test_ingest_github_activity_chunks_embeds_and_stores_each_kind(monkeypatch):
    issue_docs = [Document(content="issue text", metadata={"document_id": "owner/repo:issue:1"})]
    pr_docs = [Document(content="pr text", metadata={"document_id": "owner/repo:pr:1"})]
    discussion_docs = [
        Document(content="d text", metadata={"document_id": "owner/repo:discussion:1"})
    ]
    calls = []

    monkeypatch.setattr(ingest, "load_github_issues", lambda owner, repo, max_items: issue_docs)
    monkeypatch.setattr(ingest, "load_github_pull_requests", lambda owner, repo, max_items: pr_docs)
    monkeypatch.setattr(ingest, "load_github_discussions", lambda owner, repo, max_items: discussion_docs)
    monkeypatch.setattr(ingest, "chunk_with_parent_child", lambda doc: ["chunk"])
    monkeypatch.setattr(
        ingest, "embed_chunks", lambda chunks: calls.append(("embed_chunks", chunks)) or ["embedded"]
    )
    monkeypatch.setattr(
        ingest, "add_embedded_chunks", lambda embedded: calls.append(("add_embedded_chunks", embedded))
    )

    result = ingest.ingest_github_activity("owner/repo")

    assert result == {
        "repository": "owner/repo",
        "issues": 1,
        "pull_requests": 1,
        "discussions": 1,
        "chunks": 3,
    }
    assert calls.count(("embed_chunks", ["chunk"])) == 3
    assert calls.count(("add_embedded_chunks", ["embedded"])) == 3


def test_ingest_github_activity_respects_disabled_toggles(monkeypatch):
    calls = []

    monkeypatch.setattr(
        ingest,
        "load_github_issues",
        lambda owner, repo, max_items: calls.append("issues")
        or [Document(content="i", metadata={"document_id": "x"})],
    )
    monkeypatch.setattr(
        ingest,
        "load_github_pull_requests",
        lambda owner, repo, max_items: calls.append("prs") or [],
    )
    monkeypatch.setattr(
        ingest,
        "load_github_discussions",
        lambda owner, repo, max_items: calls.append("discussions") or [],
    )
    monkeypatch.setattr(ingest, "chunk_with_parent_child", lambda doc: ["chunk"])
    monkeypatch.setattr(ingest, "embed_chunks", lambda chunks: ["embedded"])
    monkeypatch.setattr(ingest, "add_embedded_chunks", lambda embedded: None)

    result = ingest.ingest_github_activity(
        "owner/repo", include_issues=True, include_prs=False, include_discussions=False
    )

    assert calls == ["issues"]
    assert result["pull_requests"] == 0
    assert result["discussions"] == 0


def test_read_urls_returns_empty_list_when_file_missing(tmp_path):
    assert ingest._read_urls(str(tmp_path / "does_not_exist.txt")) == []


def test_load_all_websites_skips_unreachable_urls(tmp_path, monkeypatch):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text(
        "https://example.com/good\nhttps://example.com/bad\n", encoding="utf-8"
    )

    def fake_load_website(url):
        if url.endswith("/bad"):
            raise requests.exceptions.RequestException("boom")
        return Document(content="good content", metadata={"file_name": url})

    monkeypatch.setattr(ingest, "load_website_document", fake_load_website)

    documents = ingest._load_all_websites(str(urls_file))

    assert len(documents) == 1
    assert documents[0].metadata["file_name"] == "https://example.com/good"


def test_load_all_websites_routes_crawl_prefixed_lines_to_crawl_website(tmp_path, monkeypatch):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text(
        "https://example.com/single\ncrawl:https://example.com/docs/hub\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        ingest,
        "load_website_document",
        lambda url: Document(content="single page", metadata={"file_name": url}),
    )

    crawl_calls = []

    def fake_crawl_website(seed_url):
        crawl_calls.append(seed_url)
        return [
            Document(content="hub page 1", metadata={"file_name": "hub-1"}),
            Document(content="hub page 2", metadata={"file_name": "hub-2"}),
        ]

    monkeypatch.setattr(ingest, "crawl_website", fake_crawl_website)

    documents = ingest._load_all_websites(str(urls_file))

    assert crawl_calls == ["https://example.com/docs/hub"]
    assert [doc.metadata["file_name"] for doc in documents] == [
        "https://example.com/single",
        "hub-1",
        "hub-2",
    ]


def test_ingest_all_includes_websites_alongside_files(tmp_path, monkeypatch):
    fake_file_documents = [Document(content="content", metadata={"file_name": "only.md"})]
    fake_web_documents = [Document(content="web content", metadata={"file_name": "https://example.com"})]
    calls = []

    monkeypatch.setattr(ingest, "_load_all_documents", lambda raw_dir: fake_file_documents)
    monkeypatch.setattr(ingest, "_load_all_websites", lambda urls_file: fake_web_documents)
    monkeypatch.setattr(ingest, "chunk_with_parent_child", lambda doc, chunk_size, chunk_overlap: ["chunk1"])
    monkeypatch.setattr(ingest, "embed_chunks", lambda chunks: calls.append(("embed_chunks", chunks)) or ["embedded1"])
    monkeypatch.setattr(ingest, "add_embedded_chunks", lambda embedded: calls.append(("add_embedded_chunks", embedded)))
    monkeypatch.setattr(ingest, "count", lambda: 2)

    total = ingest.ingest_all(raw_dir=str(tmp_path), urls_file=str(tmp_path / "urls.txt"))

    # One document from files + one from websites = 2 chunk_with_parent_child calls.
    assert total == 2
    assert calls.count(("embed_chunks", ["chunk1"])) == 2


def test_ingest_images_stores_image_chunks_and_ocr_text_chunks(monkeypatch):
    document = Document(content="doc", metadata={"file_name": "page.html", "url": "https://example.com/page"})
    records = [
        {
            "image_url": "https://example.com/code.png",
            "alt_text": "Close a Request example",
            "page_url": "https://example.com/page",
            "page_title": "page.html",
        }
    ]
    fake_image = object()
    calls = []

    monkeypatch.setattr(ingest, "extract_image_records", lambda doc: records)
    monkeypatch.setattr(ingest, "download_image", lambda url: fake_image if url == records[0]["image_url"] else None)
    monkeypatch.setattr(ingest, "embed_images", lambda images: [[0.1, 0.2]])
    monkeypatch.setattr(ingest, "add_image_chunks", lambda chunks: calls.append(("add_image_chunks", chunks)))
    monkeypatch.setattr(ingest, "analyze_frame", lambda image: FrameAnalysis(text="def close_request(): ...", image_type=OTHER))
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.9, 0.8]])
    monkeypatch.setattr(ingest, "add_embedded_chunks", lambda chunks: calls.append(("add_embedded_chunks", chunks)))

    stored = ingest._ingest_images(document, "local")

    assert stored == 1
    assert calls[0][0] == "add_image_chunks"
    assert calls[0][1][0]["metadata"]["image_url"] == "https://example.com/code.png"
    assert calls[1][0] == "add_embedded_chunks"
    ocr_chunk = calls[1][1][0]
    assert ocr_chunk.metadata["content_type"] == "image_ocr"
    assert ocr_chunk.metadata["image_url"] == "https://example.com/code.png"
    assert "def close_request" in ocr_chunk.content
    assert ocr_chunk.embedding == [0.9, 0.8]


def test_ingest_images_skips_ocr_text_chunk_when_no_text_found(monkeypatch):
    document = Document(content="doc", metadata={})
    records = [{"image_url": "https://example.com/icon.png", "alt_text": "icon", "page_url": None, "page_title": None}]
    fake_image = object()
    calls = []

    monkeypatch.setattr(ingest, "extract_image_records", lambda doc: records)
    monkeypatch.setattr(ingest, "download_image", lambda url: fake_image)
    monkeypatch.setattr(ingest, "embed_images", lambda images: [[0.1, 0.2]])
    monkeypatch.setattr(ingest, "add_image_chunks", lambda chunks: calls.append(("add_image_chunks", chunks)))
    monkeypatch.setattr(ingest, "analyze_frame", lambda image: FrameAnalysis(text="", image_type=OTHER))
    monkeypatch.setattr(ingest, "add_embedded_chunks", lambda chunks: calls.append(("add_embedded_chunks", chunks)))

    stored = ingest._ingest_images(document, "local")

    assert stored == 1
    assert [call[0] for call in calls] == ["add_image_chunks"]  # no OCR text chunk was stored


def test_ingest_images_returns_zero_for_document_with_no_images(monkeypatch):
    monkeypatch.setattr(ingest, "extract_image_records", lambda doc: [])

    assert ingest._ingest_images(Document(content="no images", metadata={}), "local") == 0


def test_ingest_images_skips_undownloadable_images(monkeypatch):
    records = [{"image_url": "https://example.com/broken.png", "alt_text": "x", "page_url": None, "page_title": None}]

    monkeypatch.setattr(ingest, "extract_image_records", lambda doc: records)
    monkeypatch.setattr(ingest, "download_image", lambda url: None)

    assert ingest._ingest_images(Document(content="doc", metadata={}), "local") == 0


def test_ingest_images_marks_new_web_image_chunks_with_origin(monkeypatch):
    # Regression guard: origin="web_image" must be written explicitly
    # on every new web-image chunk (see app.ingestion.ingest's edit
    # alongside this feature) so metadata.get("origin", "web_image")
    # is a safe read pattern everywhere now that video frames can also
    # land in the same collection with origin="video_frame".
    document = Document(content="doc", metadata={"file_name": "page.html", "url": "https://example.com/page"})
    records = [{"image_url": "https://example.com/icon.png", "alt_text": "icon", "page_url": None, "page_title": None}]
    calls = []

    monkeypatch.setattr(ingest, "extract_image_records", lambda doc: records)
    monkeypatch.setattr(ingest, "download_image", lambda url: object())
    monkeypatch.setattr(ingest, "embed_images", lambda images: [[0.1, 0.2]])
    monkeypatch.setattr(ingest, "add_image_chunks", lambda chunks: calls.append(chunks))
    monkeypatch.setattr(ingest, "analyze_frame", lambda image: FrameAnalysis(text="", image_type=OTHER))

    ingest._ingest_images(document, "local")

    assert calls[0][0]["metadata"]["origin"] == "web_image"


def test_ingest_video_frames_embeds_and_stores_each_sampled_frame(monkeypatch):
    document = Document(content="transcript", metadata={"document_id": "tutorial.mp4", "source_type": "video"})
    fake_frames = [(0.0, "frame_at_0s"), (10.0, "frame_at_10s")]

    monkeypatch.setattr(ingest, "_video_duration_seconds", lambda path: 15.0)
    monkeypatch.setattr(ingest, "_sample_video_frames", lambda path, interval, duration: iter(fake_frames))
    monkeypatch.setattr(ingest, "embed_images", lambda frames: [[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(
        ingest.media_store, "save_frame_thumbnail",
        lambda document_id, owner, timestamp, frame: f"/media/frames/{document_id}_{int(timestamp)}.jpg",
    )
    calls = []
    monkeypatch.setattr(ingest, "add_image_chunks", lambda chunks: calls.append(chunks))

    stored = ingest._ingest_video_frames(document, "tutorial.mp4", "local")

    assert stored == 2
    chunks = calls[0]
    assert chunks[0]["metadata"]["image_url"] == "/media/frames/tutorial.mp4_0.jpg"
    assert chunks[0]["metadata"]["origin"] == "video_frame"
    assert chunks[0]["metadata"]["video_document_id"] == "tutorial.mp4"
    assert chunks[0]["metadata"]["timestamp_seconds"] == 0.0
    assert chunks[1]["metadata"]["timestamp_seconds"] == 10.0


def test_ingest_video_frames_returns_zero_when_no_frames_sampled(monkeypatch):
    document = Document(content="transcript", metadata={"document_id": "silent.mp4", "source_type": "video"})

    monkeypatch.setattr(ingest, "_video_duration_seconds", lambda path: 0.0)
    monkeypatch.setattr(ingest, "_sample_video_frames", lambda path, interval, duration: iter([]))
    calls = []
    monkeypatch.setattr(ingest, "add_image_chunks", lambda chunks: calls.append(chunks))

    stored = ingest._ingest_video_frames(document, "silent.mp4", "local")

    assert stored == 0
    assert not calls


def test_ingest_audio_clips_is_a_noop_when_disabled(monkeypatch):
    # Default state (ENABLE_AUDIO_SIMILARITY_SEARCH unset) - confirms
    # the opt-in gate does no work at all, not even sampling.
    monkeypatch.setattr(ingest, "_ENABLE_AUDIO_SIMILARITY_SEARCH", False)
    document = Document(content="transcript", metadata={"document_id": "meeting.mp3", "source_type": "audio"})
    calls = []
    monkeypatch.setattr(ingest, "_sample_audio_clips", lambda path: calls.append(1) or iter([]))

    stored = ingest._ingest_audio_clips(document, "meeting.mp3", "local")

    assert stored == 0
    assert not calls


def test_ingest_audio_clips_embeds_and_stores_each_sampled_clip(monkeypatch):
    monkeypatch.setattr(ingest, "_ENABLE_AUDIO_SIMILARITY_SEARCH", True)
    document = Document(content="transcript", metadata={"document_id": "meeting.mp3", "source_type": "audio"})
    fake_clips = [(0.0, 10.0, "waveform_a"), (10.0, 17.0, "waveform_b")]

    monkeypatch.setattr(ingest, "_sample_audio_clips", lambda path: iter(fake_clips))
    monkeypatch.setattr(ingest, "embed_audio_clips", lambda waveforms: [[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(
        ingest.media_store, "save_media_copy",
        lambda document_id, owner, path: f"/media/sources/{document_id}.mp3",
    )
    calls = []
    monkeypatch.setattr(ingest, "add_audio_clip_chunks", lambda chunks: calls.append(chunks))

    stored = ingest._ingest_audio_clips(document, "meeting.mp3", "local")

    assert stored == 2
    chunks = calls[0]
    assert chunks[0]["metadata"]["document_id"] == "meeting.mp3"
    assert chunks[0]["metadata"]["start_seconds"] == 0.0
    assert chunks[0]["metadata"]["end_seconds"] == 10.0
    assert chunks[0]["metadata"]["source_audio_url"] == "/media/sources/meeting.mp3.mp3"
    assert chunks[1]["metadata"]["start_seconds"] == 10.0


def test_ingest_audio_clips_returns_zero_when_no_clips_sampled(monkeypatch):
    monkeypatch.setattr(ingest, "_ENABLE_AUDIO_SIMILARITY_SEARCH", True)
    document = Document(content="transcript", metadata={"document_id": "silent.mp4", "source_type": "video"})

    monkeypatch.setattr(ingest, "_sample_audio_clips", lambda path: iter([]))
    calls = []
    monkeypatch.setattr(ingest, "add_audio_clip_chunks", lambda chunks: calls.append(chunks))

    stored = ingest._ingest_audio_clips(document, "silent.mp4", "local")

    assert stored == 0
    assert not calls


def test_ingest_images_runs_vision_extraction_for_code_classified_images(monkeypatch):
    document = Document(content="doc", metadata={"file_name": "page.html", "url": "https://example.com/page"})
    records = [
        {
            "image_url": "https://example.com/code.png",
            "alt_text": "Close a Request example",
            "page_url": "https://example.com/page",
            "page_title": "page.html",
        }
    ]
    fake_image = object()
    calls = []

    monkeypatch.setattr(ingest, "extract_image_records", lambda doc: records)
    monkeypatch.setattr(ingest, "download_image", lambda url: fake_image)
    monkeypatch.setattr(ingest, "embed_images", lambda images: [[0.1, 0.2]])
    monkeypatch.setattr(ingest, "add_image_chunks", lambda chunks: calls.append(("add_image_chunks", chunks)))
    monkeypatch.setattr(ingest, "analyze_frame", lambda image: FrameAnalysis(text="COMBINED_TEXT", image_type=CODE))
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.9, 0.8]])
    monkeypatch.setattr(ingest, "add_embedded_chunks", lambda chunks: calls.append(("add_embedded_chunks", chunks)))

    stored = ingest._ingest_images(document, "local")

    assert stored == 1
    code_chunk = calls[1][1][0]
    assert code_chunk.metadata["content_type"] == "image_code"
    assert "COMBINED_TEXT" in code_chunk.content


# Vision-failure fallback (vision call raises -> keep OCR-only text) is
# now app.ocr.frame_analysis.analyze_frame's own responsibility, not
# _ingest_images's - see tests/test_frame_analysis.py for that case.

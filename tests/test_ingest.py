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
    monkeypatch.setattr(ingest, "extract_text_from_image", lambda image: "def close_request(): ...")
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.9, 0.8]])
    monkeypatch.setattr(ingest, "add_embedded_chunks", lambda chunks: calls.append(("add_embedded_chunks", chunks)))

    stored = ingest._ingest_images(document)

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
    monkeypatch.setattr(ingest, "extract_text_from_image", lambda image: "")
    monkeypatch.setattr(ingest, "add_embedded_chunks", lambda chunks: calls.append(("add_embedded_chunks", chunks)))

    stored = ingest._ingest_images(document)

    assert stored == 1
    assert [call[0] for call in calls] == ["add_image_chunks"]  # no OCR text chunk was stored


def test_ingest_images_returns_zero_for_document_with_no_images(monkeypatch):
    monkeypatch.setattr(ingest, "extract_image_records", lambda doc: [])

    assert ingest._ingest_images(Document(content="no images", metadata={})) == 0


def test_ingest_images_skips_undownloadable_images(monkeypatch):
    records = [{"image_url": "https://example.com/broken.png", "alt_text": "x", "page_url": None, "page_title": None}]

    monkeypatch.setattr(ingest, "extract_image_records", lambda doc: records)
    monkeypatch.setattr(ingest, "download_image", lambda url: None)

    assert ingest._ingest_images(Document(content="doc", metadata={})) == 0


def _code_like_ocr_text():
    # Classifies as "code" under app.ocr.image_classifier's real heuristic.
    return "const request = context.GetEntity();\nif (request.EntityState.Name === 'Closed') {\n  return true;\n}"


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
    monkeypatch.setattr(ingest, "extract_text_from_image", lambda image: _code_like_ocr_text())
    monkeypatch.setattr(
        ingest, "generate_vision_text", lambda image, instruction: "const request = context.GetEntity();"
    )
    monkeypatch.setattr(ingest, "combine_ocr_and_vision", lambda ocr_text, vision_text: "COMBINED_TEXT")
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.9, 0.8]])
    monkeypatch.setattr(ingest, "add_embedded_chunks", lambda chunks: calls.append(("add_embedded_chunks", chunks)))

    stored = ingest._ingest_images(document)

    assert stored == 1
    code_chunk = calls[1][1][0]
    assert code_chunk.metadata["content_type"] == "image_code"
    assert "COMBINED_TEXT" in code_chunk.content


def test_ingest_images_falls_back_to_ocr_only_when_vision_extraction_fails(monkeypatch):
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

    def failing_vision(image, instruction):
        raise TimeoutError("vision model too slow")

    monkeypatch.setattr(ingest, "extract_image_records", lambda doc: records)
    monkeypatch.setattr(ingest, "download_image", lambda url: fake_image)
    monkeypatch.setattr(ingest, "embed_images", lambda images: [[0.1, 0.2]])
    monkeypatch.setattr(ingest, "add_image_chunks", lambda chunks: calls.append(("add_image_chunks", chunks)))
    monkeypatch.setattr(ingest, "extract_text_from_image", lambda image: _code_like_ocr_text())
    monkeypatch.setattr(ingest, "generate_vision_text", failing_vision)
    monkeypatch.setattr(ingest, "combine_ocr_and_vision", lambda ocr_text, vision_text: ocr_text)
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.9, 0.8]])
    monkeypatch.setattr(ingest, "add_embedded_chunks", lambda chunks: calls.append(("add_embedded_chunks", chunks)))

    stored = ingest._ingest_images(document)

    assert stored == 1
    code_chunk = calls[1][1][0]
    assert code_chunk.metadata["content_type"] == "image_code"
    assert "const request" in code_chunk.content

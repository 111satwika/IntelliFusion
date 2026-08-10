"""Tests for app.ingestion.loader."""

from unittest.mock import Mock

import docx
import pytest
import requests
from fpdf import FPDF

from app.ingestion import loader
from app.ingestion.loader import (
    _flatten_ibm_docs_topics,
    _ibm_docs_product_path,
    _is_ibm_docs_url,
    _parse_github_repo_url,
    crawl_ibm_docs,
    crawl_website,
    extract_image_records,
    load_audio_document,
    load_docx_document,
    load_github_discussions,
    load_github_issues,
    load_github_pull_requests,
    load_github_repository,
    load_markdown_document,
    load_pdf_document,
    load_video_document,
    load_website_document,
)
from app.ocr.frame_analysis import FrameAnalysis


def _make_pdf(path, pages_text: list[str]) -> None:
    """Create a small real PDF at `path` with one page per string in pages_text."""
    pdf = FPDF()
    for text in pages_text:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 8, text)
    pdf.output(str(path))


def _make_pdf_with_table(path, rows: list[list[str]]) -> None:
    """Create a single-page PDF containing a real table (first row = header)."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    with pdf.table() as table:
        for row_data in rows:
            row = table.row()
            for cell in row_data:
                row.cell(cell)
    pdf.output(str(path))


def _make_pdf_with_long_table(path, header: list[str], data_rows: list[list[str]]) -> None:
    """
    Create a PDF whose table has enough rows to overflow onto a second
    physical page. fpdf2 automatically repeats the header row on every
    continuation page, which is the exact signal load_pdf_document()
    relies on to detect and merge the continuation.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    with pdf.table() as table:
        header_row = table.row()
        for cell in header:
            header_row.cell(cell)
        for row_data in data_rows:
            row = table.row()
            for cell in row_data:
                row.cell(cell)
    pdf.output(str(path))


def test_load_markdown_document_reads_content_and_metadata(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\n\nSome content.", encoding="utf-8")

    doc = load_markdown_document(str(file_path))

    assert doc.content == "# Title\n\nSome content."
    assert doc.metadata["file_name"] == "sample.md"
    assert doc.metadata["source_type"] == "markdown"
    assert doc.metadata["file_path"] == str(file_path)
    assert doc.metadata["document_id"] == "sample.md"


def test_load_markdown_document_normalizes_crlf_line_endings(tmp_path):
    file_path = tmp_path / "crlf.md"
    file_path.write_bytes(b"line one\r\nline two\r\n")

    doc = load_markdown_document(str(file_path))

    assert "\r\n" not in doc.content
    assert doc.content == "line one\nline two\n"


def test_load_markdown_document_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.md"

    with pytest.raises(FileNotFoundError):
        load_markdown_document(str(missing_path))


def test_load_pdf_document_returns_a_single_document_for_the_whole_file(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, ["First page text.", "Second page text.", "Third page text."])

    doc = load_pdf_document(str(pdf_path))

    # No pagination: the whole file is one Document, same as DOCX, even
    # though it spans 3 physical pages.
    assert "page_number" not in doc.metadata
    assert doc.metadata["total_pages"] == 3
    assert "First page text." in doc.content
    assert "Second page text." in doc.content
    assert "Third page text." in doc.content


def test_load_pdf_document_attaches_document_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, ["Page one.", "Page two."])

    doc = load_pdf_document(str(pdf_path))

    assert doc.metadata["source_type"] == "pdf"
    assert doc.metadata["file_name"] == "sample.pdf"
    assert doc.metadata["document_id"] == "sample.pdf"
    assert doc.metadata["total_pages"] == 2


def test_load_pdf_document_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf_document(str(missing_path))


def test_load_pdf_document_serializes_tables_as_markdown(tmp_path):
    pdf_path = tmp_path / "table.pdf"
    _make_pdf_with_table(
        pdf_path,
        [
            ["Role", "Deploy to Production"],
            ["Viewer", "No"],
            ["Administrator", "Yes"],
        ],
    )

    doc = load_pdf_document(str(pdf_path))

    content = doc.content
    # Header + separator + one row per data row, all as pipe-delimited lines.
    assert "| Role | Deploy to Production |" in content
    assert "| --- | --- |" in content
    assert "| Viewer | No |" in content
    assert "| Administrator | Yes |" in content


def test_load_pdf_document_merges_table_continued_across_pages(tmp_path):
    pdf_path = tmp_path / "long_table.pdf"
    header = ["ID", "Name"]
    data_rows = [[str(n), f"Item {n}"] for n in range(1, 60)]
    _make_pdf_with_long_table(pdf_path, header, data_rows)

    doc = load_pdf_document(str(pdf_path))

    # The table overflowed onto a second physical page, but since the whole
    # file is already one Document, the continuation rows are merged into
    # the same table rather than appearing as a second, incomplete one.
    assert doc.metadata["total_pages"] == 2

    content = doc.content
    assert "| ID | Name |" in content
    assert "| --- | --- |" in content
    assert "| 1 | Item 1 |" in content
    assert "| 59 | Item 59 |" in content
    # The header row must not be duplicated by the merge.
    assert content.count("| ID | Name |") == 1


def test_load_docx_document_returns_a_single_document(tmp_path):
    path = tmp_path / "sample.docx"
    word_document = docx.Document()
    word_document.add_heading("Section One", level=1)
    word_document.add_paragraph("First paragraph of content.")
    word_document.add_page_break()
    word_document.add_heading("Section Two (continued)", level=1)
    word_document.add_paragraph("Content that would render on a second page.")
    word_document.save(str(path))

    doc = load_docx_document(str(path))

    # No pagination: the whole file is one Document, unlike PDF, even
    # though it contains an explicit page break.
    assert doc.metadata["file_name"] == "sample.docx"
    assert doc.metadata["source_type"] == "docx"
    assert "page_number" not in doc.metadata
    assert "# Section One" in doc.content
    assert "# Section Two (continued)" in doc.content
    assert "First paragraph of content." in doc.content
    assert "Content that would render on a second page." in doc.content


def test_load_docx_document_converts_heading_styles_to_markdown_headings(tmp_path):
    path = tmp_path / "headings.docx"
    word_document = docx.Document()
    word_document.add_heading("Top Level", level=1)
    word_document.add_heading("Nested", level=2)
    word_document.add_paragraph("Plain paragraph text.")
    word_document.save(str(path))

    doc = load_docx_document(str(path))

    assert "# Top Level" in doc.content
    assert "## Nested" in doc.content
    assert "Plain paragraph text." in doc.content
    # Plain paragraphs must not get a stray heading prefix.
    assert "# Plain paragraph text." not in doc.content


def test_load_docx_document_serializes_tables_as_markdown(tmp_path):
    path = tmp_path / "table.docx"
    word_document = docx.Document()
    word_document.add_heading("Roles", level=1)
    table = word_document.add_table(rows=3, cols=2)
    header_cells = table.rows[0].cells
    header_cells[0].text = "Role"
    header_cells[1].text = "Deploy to Production"
    table.rows[1].cells[0].text = "Viewer"
    table.rows[1].cells[1].text = "No"
    table.rows[2].cells[0].text = "Administrator"
    table.rows[2].cells[1].text = "Yes"
    word_document.save(str(path))

    doc = load_docx_document(str(path))

    assert "| Role | Deploy to Production |" in doc.content
    assert "| --- | --- |" in doc.content
    assert "| Viewer | No |" in doc.content
    assert "| Administrator | Yes |" in doc.content


def test_load_docx_document_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_docx_document("does_not_exist.docx")


# ---------------------------------------------------------------------
# Audio / video loaders - the real Whisper model / OpenCV frame decode
# are NOT used here. _transcribe_audio, _sample_video_frames, and
# _video_duration_seconds are monkeypatched directly on the loader
# module (matching this file's existing "fake the network/model call,
# assert on the returned Document" convention), so these tests run
# fast and deterministically instead of loading a real model per run.
# ---------------------------------------------------------------------


class _FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


def test_load_audio_document_builds_timestamped_transcript(tmp_path, monkeypatch):
    audio_path = tmp_path / "meeting.mp3"
    audio_path.write_bytes(b"fake audio bytes")

    segments = [_FakeSegment(0.0, 2.5, "hello world"), _FakeSegment(65.0, 68.0, "this is a test")]
    monkeypatch.setattr(loader, "_transcribe_audio", lambda path: (segments, 68.0))

    document = load_audio_document(str(audio_path))

    assert "[00:00] hello world" in document.content
    assert "[01:05] this is a test" in document.content
    assert document.metadata["source_type"] == "audio"
    assert document.metadata["file_name"] == "meeting.mp3"
    assert document.metadata["document_id"] == "meeting.mp3"
    assert document.metadata["duration_seconds"] == 68.0


def test_load_audio_document_empty_transcript_when_no_speech_found(tmp_path, monkeypatch):
    audio_path = tmp_path / "silence.wav"
    audio_path.write_bytes(b"fake audio bytes")

    monkeypatch.setattr(loader, "_transcribe_audio", lambda path: ([], 2.0))

    document = load_audio_document(str(audio_path))

    assert document.content == ""
    assert document.metadata["duration_seconds"] == 2.0


def test_load_audio_document_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_audio_document("does_not_exist.mp3")


def test_load_video_document_merges_transcript_and_frame_text_in_timestamp_order(tmp_path, monkeypatch):
    video_path = tmp_path / "tutorial.mp4"
    video_path.write_bytes(b"fake video bytes")

    segments = [_FakeSegment(0.0, 3.0, "let me show you this function")]
    monkeypatch.setattr(loader, "_transcribe_audio", lambda path: (segments, 20.0))
    monkeypatch.setattr(loader, "_video_duration_seconds", lambda path: 20.0)

    fake_frames = [(1.0, "frame_at_1s"), (2.0, "frame_at_2s")]
    monkeypatch.setattr(
        loader, "_sample_video_frames", lambda path, interval, duration: iter(fake_frames)
    )

    def fake_analyze(frame):
        # frame is whatever _sample_video_frames yielded as the second
        # tuple element (a placeholder string here, a real PIL.Image in
        # production) - analyze_frame's real signature doesn't care.
        if frame == "frame_at_1s":
            return FrameAnalysis(text="def close_request():", image_type="code")
        return FrameAnalysis(text="", image_type="other")  # no meaningful text at 2s

    monkeypatch.setattr(loader, "analyze_frame", fake_analyze)

    document = load_video_document(str(video_path), frame_interval_seconds=1, max_frames=60)

    lines = document.content.splitlines()
    assert lines == [
        "[00:00] let me show you this function",
        "[00:01] [Frame text: def close_request():]",
    ]
    assert document.metadata["source_type"] == "video"
    assert document.metadata["duration_seconds"] == 20.0
    assert document.metadata["frame_count_sampled"] == 2  # both sampled frames counted, even the empty one


def test_load_video_document_widens_interval_for_long_videos_instead_of_truncating(tmp_path, monkeypatch):
    video_path = tmp_path / "long.mp4"
    video_path.write_bytes(b"fake video bytes")

    monkeypatch.setattr(loader, "_transcribe_audio", lambda path: ([], 1200.0))  # 20-minute video
    monkeypatch.setattr(loader, "_video_duration_seconds", lambda path: 1200.0)

    captured = {}

    def fake_sample(path, interval, duration):
        captured["interval"] = interval
        captured["duration"] = duration
        return iter([])

    monkeypatch.setattr(loader, "_sample_video_frames", fake_sample)

    load_video_document(str(video_path), frame_interval_seconds=10, max_frames=60)

    # 1200s / 60 frames = 20s/frame, wider than the 10s default - the
    # interval must widen automatically rather than sampling only the
    # first 600s (60 frames * 10s) and silently dropping the rest.
    assert captured["interval"] == 20.0
    assert captured["duration"] == 1200.0


def test_load_video_document_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_video_document("does_not_exist.mp4")


def _mock_response(html: str, status_ok: bool = True):
    response = Mock()
    response.text = html
    if status_ok:
        response.raise_for_status = Mock()
    else:
        response.raise_for_status = Mock(side_effect=requests.exceptions.HTTPError("404"))
    return response


def test_load_website_document_extracts_headings_prose_and_title(monkeypatch):
    html = """
    <html>
      <head><title>Example Page</title></head>
      <body>
        <h1>Main Heading</h1>
        <p>First paragraph of content.</p>
        <h2>Subsection</h2>
        <p>Second paragraph of content.</p>
      </body>
    </html>
    """
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_response(html))
    )

    doc = load_website_document("https://example.com/page")

    assert doc.metadata["source_type"] == "web"
    assert doc.metadata["url"] == "https://example.com/page"
    assert doc.metadata["document_id"] == "https://example.com/page"
    assert doc.metadata["file_name"] == "Example Page"
    assert "# Main Heading" in doc.content
    assert "## Subsection" in doc.content
    assert "First paragraph of content." in doc.content
    assert "Second paragraph of content." in doc.content


def test_load_website_document_falls_back_to_url_when_no_title(monkeypatch):
    html = "<html><body><p>No title here.</p></body></html>"
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_response(html))
    )

    doc = load_website_document("https://example.com/no-title")

    assert doc.metadata["file_name"] == "https://example.com/no-title"


def test_load_website_document_serializes_tables_as_markdown(monkeypatch):
    html = """
    <html><body>
      <h1>Roles</h1>
      <table>
        <tr><th>Role</th><th>Deploy to Production</th></tr>
        <tr><td>Viewer</td><td>No</td></tr>
        <tr><td>Administrator</td><td>Yes</td></tr>
      </table>
    </body></html>
    """
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_response(html))
    )

    doc = load_website_document("https://example.com/roles")

    assert "| Role | Deploy to Production |" in doc.content
    assert "| --- | --- |" in doc.content
    assert "| Viewer | No |" in doc.content
    assert "| Administrator | Yes |" in doc.content


def test_load_website_document_strips_boilerplate_tags(monkeypatch):
    html = """
    <html><body>
      <nav>Home | About | Contact</nav>
      <header>Site Header</header>
      <script>trackVisitor();</script>
      <style>.hidden { display: none; }</style>
      <article>
        <h1>Real Content</h1>
        <p>This is the actual page content.</p>
      </article>
      <footer>Copyright 2026</footer>
    </body></html>
    """
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_response(html))
    )

    doc = load_website_document("https://example.com/article")

    assert "Real Content" in doc.content
    assert "This is the actual page content." in doc.content
    assert "Home | About | Contact" not in doc.content
    assert "Site Header" not in doc.content
    assert "trackVisitor" not in doc.content
    assert "Copyright 2026" not in doc.content


def test_load_website_document_raises_for_http_error(monkeypatch):
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get",
        Mock(return_value=_mock_response("", status_ok=False)),
    )

    with pytest.raises(requests.exceptions.HTTPError):
        load_website_document("https://example.com/missing")


def test_load_website_document_includes_meaningful_images_with_absolute_url(monkeypatch):
    html = """
    <html><body>
      <h1>API Authentication</h1>
      <p>The API requires authentication.</p>
      <img src="/images/auth-flow.png" alt="Authentication flow diagram">
      <p>The token must be included in the request header.</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_response(html))
    )

    doc = load_website_document("https://example.com/docs/api")

    assert "[Image: Authentication flow diagram] (https://example.com/images/auth-flow.png)" in doc.content
    # Order preserved: image marker stays between its surrounding paragraphs.
    before_index = doc.content.index("The API requires authentication.")
    image_index = doc.content.index("[Image: Authentication flow diagram]")
    after_index = doc.content.index("The token must be included")
    assert before_index < image_index < after_index


def test_load_website_document_skips_decorative_images(monkeypatch):
    html = """
    <html><body>
      <h1>Page</h1>
      <img src="/logo.png" alt="">
      <img src="/spacer.gif">
      <p>Real content.</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_response(html))
    )

    doc = load_website_document("https://example.com/page")

    assert "[Image:" not in doc.content
    assert "logo.png" not in doc.content
    assert "spacer.gif" not in doc.content
    assert "Real content." in doc.content


def test_load_website_document_captures_pre_blocks_as_fenced_code(monkeypatch):
    # Regression test: <pre> was previously MISSING from _HTML_CONTENT_TAGS,
    # so every literal code sample on a crawled page was silently dropped
    # (never even stored as text) - see /memories/repo/rag-platform-notes.md.
    html = """
    <html><body>
      <h1>Rename an existing tab</h1>
      <p>To rename a tab, modify the value of the title part.</p>
      <pre><code>{
   "title": {
      "type": "string",
      "value": "Change Log"
   }
}</code></pre>
      <p>Then click Apply changes.</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_response(html))
    )

    doc = load_website_document("https://example.com/rename-tab")

    assert '"value": "Change Log"' in doc.content
    assert "```" in doc.content
    # Order preserved: code block stays between its surrounding paragraphs.
    before_index = doc.content.index("To rename a tab")
    code_index = doc.content.index('"value": "Change Log"')
    after_index = doc.content.index("Then click Apply changes.")
    assert before_index < code_index < after_index


def test_load_website_document_skips_empty_pre_blocks(monkeypatch):
    html = """
    <html><body>
      <h1>Page</h1>
      <pre>   </pre>
      <p>Real content.</p>
    </body></html>
    """
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_response(html))
    )

    doc = load_website_document("https://example.com/page")

    assert "```" not in doc.content
    assert "Real content." in doc.content


def test_extract_image_records_parses_markers_from_content(monkeypatch):
    html = """
    <html><body>
      <h1>API Authentication</h1>
      <img src="/images/auth-flow.png" alt="Authentication flow diagram">
    </body></html>
    """
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_response(html))
    )

    doc = load_website_document("https://example.com/docs/api")

    records = extract_image_records(doc)

    assert records == [
        {
            "image_url": "https://example.com/images/auth-flow.png",
            "alt_text": "Authentication flow diagram",
            "page_url": "https://example.com/docs/api",
            "page_title": doc.metadata.get("file_name"),
        }
    ]


def test_extract_image_records_returns_empty_list_for_document_without_images(monkeypatch):
    html = "<html><body><h1>Page</h1><p>No images here.</p></body></html>"
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_response(html))
    )

    doc = load_website_document("https://example.com/page")

    assert extract_image_records(doc) == []


def _mock_pages(pages: dict[str, str]):
    """
    Build a fake `requests.get` that returns _mock_response(html) for
    whichever URL from `pages` is requested, so a test can simulate a
    small multi-page site for crawl_website().
    """

    def fake_get(url, **kwargs):
        return _mock_response(pages[url])

    return Mock(side_effect=fake_get)


def test_crawl_website_follows_in_scope_links_and_collects_every_page(monkeypatch):
    pages = {
        "https://example.com/docs/hub": """
            <html><head><title>Hub</title></head><body>
              <nav>
                <a href="/docs/hub/getting-started">Getting Started</a>
                <a href="/docs/hub/api">API</a>
                <a href="https://example.com/blog">Blog</a>
              </nav>
              <h1>Hub</h1><p>Hub landing content.</p>
            </body></html>
        """,
        "https://example.com/docs/hub/getting-started": """
            <html><head><title>Getting Started</title></head><body>
              <h1>Getting Started</h1><p>Getting started content.</p>
            </body></html>
        """,
        "https://example.com/docs/hub/api": """
            <html><head><title>API</title></head><body>
              <h1>API</h1><p>API reference content.</p>
            </body></html>
        """,
    }
    monkeypatch.setattr("app.ingestion.loader.requests.get", _mock_pages(pages))

    documents = crawl_website("https://example.com/docs/hub")

    urls = {doc.metadata["url"] for doc in documents}
    assert urls == {
        "https://example.com/docs/hub",
        "https://example.com/docs/hub/getting-started",
        "https://example.com/docs/hub/api",
    }
    # The out-of-scope /blog link (different path prefix) must not be crawled.
    assert "https://example.com/blog" not in urls


def test_crawl_website_does_not_follow_out_of_scope_or_off_site_links(monkeypatch):
    pages = {
        "https://example.com/docs/hub": """
            <html><body>
              <a href="/docs/other-product">Other product docs</a>
              <a href="https://other-site.com/docs/hub/page">Off-site lookalike path</a>
              <h1>Hub</h1>
            </body></html>
        """,
    }
    monkeypatch.setattr("app.ingestion.loader.requests.get", _mock_pages(pages))

    documents = crawl_website("https://example.com/docs/hub")

    assert len(documents) == 1
    assert documents[0].metadata["url"] == "https://example.com/docs/hub"


def test_crawl_website_respects_max_pages_cap(monkeypatch):
    pages = {
        "https://example.com/docs/hub": '<html><body><a href="/docs/hub/a">a</a><a href="/docs/hub/b">b</a></body></html>',
        "https://example.com/docs/hub/a": "<html><body>Page A</body></html>",
        "https://example.com/docs/hub/b": "<html><body>Page B</body></html>",
    }
    monkeypatch.setattr("app.ingestion.loader.requests.get", _mock_pages(pages))

    documents = crawl_website("https://example.com/docs/hub", max_pages=2)

    assert len(documents) == 2


def test_crawl_website_skips_unreachable_pages_and_keeps_going(monkeypatch):
    def fake_get(url, **kwargs):
        if url == "https://example.com/docs/hub/broken":
            raise requests.exceptions.ConnectionError("boom")
        pages = {
            "https://example.com/docs/hub": (
                '<html><body><a href="/docs/hub/broken">broken</a>'
                '<a href="/docs/hub/ok">ok</a></body></html>'
            ),
            "https://example.com/docs/hub/ok": "<html><body>Fine</body></html>",
        }
        return _mock_response(pages[url])

    monkeypatch.setattr("app.ingestion.loader.requests.get", Mock(side_effect=fake_get))

    documents = crawl_website("https://example.com/docs/hub")

    urls = {doc.metadata["url"] for doc in documents}
    assert urls == {"https://example.com/docs/hub", "https://example.com/docs/hub/ok"}


def test_is_ibm_docs_url_matches_only_ibm_docs_paths():
    assert _is_ibm_docs_url("https://www.ibm.com/docs/en/targetprocess/tp-dev-hub/saas")
    assert not _is_ibm_docs_url("https://www.ibm.com/support/pages/foo")
    assert not _is_ibm_docs_url("https://example.com/docs/en/targetprocess/tp-dev-hub/saas")


def _mock_json_response(data: dict, status_ok: bool = True):
    response = Mock()
    response.json = Mock(return_value=data)
    if status_ok:
        response.raise_for_status = Mock()
    else:
        response.raise_for_status = Mock(side_effect=requests.exceptions.HTTPError("error"))
    return response


def _mock_raw_file_response(content: bytes):
    response = Mock()
    response.raise_for_status = Mock()
    response.content = content
    return response


@pytest.mark.parametrize(
    "repo_url,expected",
    [
        ("owner/repo", ("owner", "repo", None)),
        ("https://github.com/owner/repo", ("owner", "repo", None)),
        ("https://github.com/owner/repo.git", ("owner", "repo", None)),
        ("https://github.com/owner/repo/tree/dev", ("owner", "repo", "dev")),
        ("https://github.com/owner/repo/tree/feature/nested", ("owner", "repo", "feature/nested")),
    ],
)
def test_parse_github_repo_url_accepts_various_forms(repo_url, expected):
    assert _parse_github_repo_url(repo_url) == expected


def test_parse_github_repo_url_rejects_a_bare_owner():
    with pytest.raises(ValueError):
        _parse_github_repo_url("just-an-owner")


def test_load_github_repository_filters_and_loads_ingestible_files(monkeypatch):
    tree = {
        "sha": "abc123",
        "truncated": False,
        "tree": [
            {"path": "README.md", "type": "blob", "size": 20},
            {"path": "src/main.py", "type": "blob", "size": 30},
            {"path": "node_modules/lib/index.js", "type": "blob", "size": 10},
            {"path": "package-lock.json", "type": "blob", "size": 500},
            {"path": "assets/logo.png", "type": "blob", "size": 40},
            {"path": "src", "type": "tree", "size": 0},
        ],
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        if url == "https://api.github.com/repos/owner/repo/git/trees/main":
            return _mock_json_response(tree)
        if url == "https://raw.githubusercontent.com/owner/repo/main/README.md":
            return _mock_raw_file_response(b"# Hello")
        if url == "https://raw.githubusercontent.com/owner/repo/main/src/main.py":
            return _mock_raw_file_response(b"print('hi')")
        raise AssertionError(f"Unexpected URL requested: {url}")

    monkeypatch.setattr("app.ingestion.loader.requests.get", Mock(side_effect=fake_get))

    documents = load_github_repository("https://github.com/owner/repo", branch="main")

    doc_by_path = {doc.metadata["file_path"]: doc for doc in documents}
    # node_modules/, package-lock.json, and the unsupported .png are excluded.
    assert set(doc_by_path) == {"README.md", "src/main.py"}
    assert doc_by_path["README.md"].content == "# Hello"
    assert doc_by_path["README.md"].metadata["language"] == "markdown"
    assert doc_by_path["README.md"].metadata["repository"] == "owner/repo"
    assert doc_by_path["README.md"].metadata["branch"] == "main"
    assert doc_by_path["README.md"].metadata["commit_hash"] == "abc123"
    assert doc_by_path["README.md"].metadata["document_id"] == "owner/repo:README.md"
    assert doc_by_path["src/main.py"].metadata["language"] == "python"


def test_load_github_repository_resolves_default_branch_when_not_specified(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        if url == "https://api.github.com/repos/owner/repo":
            return _mock_json_response({"default_branch": "trunk"})
        if url == "https://api.github.com/repos/owner/repo/git/trees/trunk":
            return _mock_json_response({"sha": "s1", "truncated": False, "tree": []})
        raise AssertionError(f"Unexpected URL requested: {url}")

    monkeypatch.setattr("app.ingestion.loader.requests.get", Mock(side_effect=fake_get))

    documents = load_github_repository("owner/repo")

    assert documents == []


def test_load_github_repository_skips_unreadable_files_and_keeps_going(monkeypatch):
    tree = {
        "sha": "abc123",
        "truncated": False,
        "tree": [
            {"path": "good.py", "type": "blob", "size": 10},
            {"path": "broken.py", "type": "blob", "size": 10},
        ],
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        if url == "https://api.github.com/repos/owner/repo/git/trees/main":
            return _mock_json_response(tree)
        if url == "https://raw.githubusercontent.com/owner/repo/main/good.py":
            return _mock_raw_file_response(b"print('ok')")
        if url == "https://raw.githubusercontent.com/owner/repo/main/broken.py":
            raise requests.exceptions.ConnectionError("boom")
        raise AssertionError(f"Unexpected URL requested: {url}")

    monkeypatch.setattr("app.ingestion.loader.requests.get", Mock(side_effect=fake_get))

    documents = load_github_repository("owner/repo", branch="main")

    assert [doc.metadata["file_path"] for doc in documents] == ["good.py"]


def test_load_github_repository_caps_at_max_files(monkeypatch):
    tree = {
        "sha": "abc123",
        "truncated": False,
        "tree": [{"path": f"file{i}.py", "type": "blob", "size": 5} for i in range(5)],
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        if url == "https://api.github.com/repos/owner/repo/git/trees/main":
            return _mock_json_response(tree)
        assert url.startswith("https://raw.githubusercontent.com/owner/repo/main/file")
        return _mock_raw_file_response(b"content")

    monkeypatch.setattr("app.ingestion.loader.requests.get", Mock(side_effect=fake_get))

    documents = load_github_repository("owner/repo", branch="main", max_files=2)

    assert len(documents) == 2


def test_load_github_issues_filters_out_pull_requests_and_paginates(monkeypatch):
    page1 = [
        {"number": 2, "title": "Second issue", "body": "b2", "state": "open", "user": {"login": "a"}, "html_url": "u2"},
        {"number": 1, "title": "A PR", "body": "b1", "state": "open", "pull_request": {"url": "x"}},
    ]
    page2 = [
        {"number": 3, "title": "Third issue", "body": None, "state": "closed", "user": {"login": "b"}, "html_url": "u3"},
    ]

    def fake_get(url, headers=None, params=None, timeout=None):
        assert url == "https://api.github.com/repos/owner/repo/issues"
        if params["page"] == 1:
            return _mock_json_response(page1)
        if params["page"] == 2:
            return _mock_json_response(page2)
        return _mock_json_response([])

    monkeypatch.setattr("app.ingestion.loader.requests.get", Mock(side_effect=fake_get))
    # Force pagination to advance one page at a time regardless of the
    # real API's 100/page cap, so this test doesn't need 100 fake items.
    monkeypatch.setattr("app.ingestion.loader._ACTIVITY_API_MAX_PER_PAGE", 2)

    documents = load_github_issues("owner", "repo")

    assert [doc.metadata["number"] for doc in documents] == [2, 3]
    assert documents[0].metadata["content_type"] == "issue"
    assert documents[0].metadata["repository"] == "owner/repo"
    assert documents[0].metadata["source_type"] == "code"
    assert documents[0].metadata["document_id"] == "owner/repo:issue:2"
    assert "Second issue" in documents[0].content
    assert "(no description provided)" in documents[1].content


def test_load_github_issues_caps_at_max_items(monkeypatch):
    page = [
        {"number": i, "title": f"Issue {i}", "body": "b", "state": "open", "user": {}, "html_url": "u"}
        for i in range(1, 6)
    ]
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", Mock(return_value=_mock_json_response(page))
    )

    documents = load_github_issues("owner", "repo", max_items=3)

    assert len(documents) == 3


def test_load_github_pull_requests_basic(monkeypatch):
    page = [
        {"number": 5, "title": "Add feature", "body": "details", "state": "open", "user": {"login": "c"}, "html_url": "u5"},
    ]

    def fake_get(url, headers=None, params=None, timeout=None):
        assert url == "https://api.github.com/repos/owner/repo/pulls"
        if params["page"] == 1:
            return _mock_json_response(page)
        return _mock_json_response([])

    monkeypatch.setattr("app.ingestion.loader.requests.get", Mock(side_effect=fake_get))

    documents = load_github_pull_requests("owner", "repo")

    assert len(documents) == 1
    assert documents[0].metadata["content_type"] == "pull_request"
    assert documents[0].metadata["document_id"] == "owner/repo:pr:5"


def test_load_github_discussions_requires_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        load_github_discussions("owner", "repo")


def test_load_github_discussions_paginates_via_graphql(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url == "https://api.github.com/graphql"
        after = json["variables"]["after"]
        if after is None:
            data = {
                "repository": {
                    "discussions": {
                        "pageInfo": {"hasNextPage": True, "endCursor": "cursor1"},
                        "nodes": [
                            {"number": 1, "title": "Discussion one", "body": "d1", "url": "du1",
                             "author": {"login": "x"}, "category": {"name": "Q&A"}},
                        ],
                    }
                }
            }
        else:
            assert after == "cursor1"
            data = {
                "repository": {
                    "discussions": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [
                            {"number": 2, "title": "Discussion two", "body": None, "url": "du2",
                             "author": None, "category": None},
                        ],
                    }
                }
            }
        return _mock_json_response({"data": data})

    monkeypatch.setattr("app.ingestion.loader.requests.post", Mock(side_effect=fake_post))

    documents = load_github_discussions("owner", "repo")

    assert [doc.metadata["number"] for doc in documents] == [1, 2]
    assert documents[0].metadata["content_type"] == "discussion"
    assert documents[0].metadata["document_id"] == "owner/repo:discussion:1"
    assert documents[1].metadata["author"] is None


def test_load_github_discussions_raises_on_graphql_errors(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    monkeypatch.setattr(
        "app.ingestion.loader.requests.post",
        Mock(return_value=_mock_json_response({"errors": [{"message": "Discussions not enabled"}]})),
    )
    with pytest.raises(ValueError, match="GraphQL error"):
        load_github_discussions("owner", "repo")


def test_ibm_docs_product_path_strips_prefix_and_query():
    url = "https://www.ibm.com/docs/en/targetprocess/tp-dev-hub/saas?topic=targetprocess"
    assert _ibm_docs_product_path(url) == "targetprocess/tp-dev-hub/saas"


def test_flatten_ibm_docs_topics_depth_first_includes_only_topics_with_ids():
    tree = {
        "topicId": "root",
        "label": "Root",
        "topics": [
            {"topicId": "a", "label": "A"},
            {
                "label": "Group (no content of its own)",
                "topics": [
                    {"topicId": "b", "label": "B"},
                    {"topicId": "c", "label": "C"},
                ],
            },
        ],
    }

    flat = _flatten_ibm_docs_topics(tree)

    assert flat == [
        {"topicId": "root", "label": "Root"},
        {"topicId": "a", "label": "A"},
        {"topicId": "b", "label": "B"},
        {"topicId": "c", "label": "C"},
    ]


def _mock_ibm_docs_api(toc: dict, content_by_topic: dict[str, str]):
    """
    Build a fake `requests.get` that serves _fetch_ibm_docs_toc() from
    an /api/v1/toc/ URL (as JSON) and _fetch_ibm_docs_topic_html() from
    an /api/v1/content/ URL (as an HTML fragment), keyed by the
    ?topic= query param - used by crawl_ibm_docs() tests.
    """

    def fake_get(url, **kwargs):
        response = Mock()
        if "/api/v1/toc/" in url:
            response.raise_for_status = Mock()
            response.json = Mock(return_value=toc)
            return response
        topic_id = url.split("topic=")[1].split("&")[0]
        if topic_id not in content_by_topic:
            response.raise_for_status = Mock(side_effect=requests.exceptions.HTTPError("404"))
            return response
        response.raise_for_status = Mock()
        response.text = content_by_topic[topic_id]
        return response

    return Mock(side_effect=fake_get)


def test_crawl_ibm_docs_fetches_toc_and_each_topics_content(monkeypatch):
    toc = {
        "toc": {
            "topicId": "hub",
            "label": "Hub",
            "topics": [
                {"topicId": "getting-started", "label": "Getting Started"},
                {"topicId": "authentication", "label": "Authentication"},
            ],
        }
    }
    content_by_topic = {
        "hub": '<html><body><main><article><h1>Hub</h1><p>Hub content.</p></article></main></body></html>',
        "getting-started": (
            '<html><body><main><article><h1>Getting Started</h1>'
            "<p>Getting started content.</p></article></main></body></html>"
        ),
        "authentication": (
            '<html><body><main><article><h1>Authentication</h1>'
            "<p>Authentication content.</p></article></main></body></html>"
        ),
    }
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", _mock_ibm_docs_api(toc, content_by_topic)
    )

    documents = crawl_ibm_docs("https://www.ibm.com/docs/en/targetprocess/tp-dev-hub/saas")

    assert len(documents) == 3
    contents = {doc.metadata["file_name"]: doc.content for doc in documents}
    assert "Getting started content." in contents["Getting Started"]
    assert "Authentication content." in contents["Authentication"]
    assert "Hub content." in contents["Hub"]
    # Every topic's content must be genuinely distinct (the original bug
    # this API-based crawler fixes was every topic returning the same
    # generic hub content).
    assert len({doc.content for doc in documents}) == 3


def test_crawl_ibm_docs_skips_topics_that_fail_to_fetch(monkeypatch):
    toc = {
        "toc": {
            "topicId": "hub",
            "label": "Hub",
            "topics": [{"topicId": "group-header-with-no-content", "label": "Group"}],
        }
    }
    content_by_topic = {
        "hub": '<html><body><main><article><h1>Hub</h1><p>Hub content.</p></article></main></body></html>',
    }
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", _mock_ibm_docs_api(toc, content_by_topic)
    )

    documents = crawl_ibm_docs("https://www.ibm.com/docs/en/targetprocess/tp-dev-hub/saas")

    assert len(documents) == 1
    assert documents[0].metadata["file_name"] == "Hub"


def test_crawl_website_delegates_to_crawl_ibm_docs_for_ibm_docs_urls(monkeypatch):
    toc = {"toc": {"topicId": "hub", "label": "Hub", "topics": []}}
    content_by_topic = {
        "hub": '<html><body><main><article><h1>Hub</h1><p>Hub content.</p></article></main></body></html>',
    }
    monkeypatch.setattr(
        "app.ingestion.loader.requests.get", _mock_ibm_docs_api(toc, content_by_topic)
    )

    documents = crawl_website("https://www.ibm.com/docs/en/targetprocess/tp-dev-hub/saas")

    assert len(documents) == 1
    assert documents[0].metadata["file_name"] == "Hub"


"use strict";

// ---------- Icons (same path data as ui/icons.py, kept in sync manually) ----------
const ICON_PATHS = {
  markdown: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M9 15v-2l1.5 2L12 13v2"/><path d="M15 15v-2l1.5 2"/>',
  pdf: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M9.5 17v-4h1a1.5 1.5 0 0 1 0 3h-1"/><path d="M13.5 17v-4h1.5"/><path d="M13.5 15h1.2"/>',
  docx: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8.5 13l1 4 1.2-4 1.2 4 1-4"/>',
  web: '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
  github: '<path d="M15 22v-4.5c0-.63-.25-1.17-.66-1.58C17.14 15.44 19 13.4 19 10.5 19 6.36 15.87 3 12 3S5 6.36 5 10.5c0 2.9 1.86 4.94 4.66 5.42-.41.41-.66.95-.66 1.58V22"/><path d="M9 19c-1.5.5-2.7-.5-3-1.5"/>',
  audio: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/>',
  video: '<path d="m22 8-6 4 6 4V8Z"/><rect x="2" y="6" width="14" height="12" rx="2" ry="2"/>',
  brand: '<rect x="8" y="2.5" width="12" height="15" rx="1.6" stroke-width="1.3" opacity="0.35"/><rect x="5.5" y="4.5" width="12" height="15" rx="1.6" stroke-width="1.5" opacity="0.65"/><rect x="3" y="6.5" width="12" height="15" rx="1.8" stroke-width="2"/>',
  user: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
  search: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
  close: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
};

function svgIcon(name, { size = 18, stroke = "currentColor" } = {}) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICON_PATHS[name] || ""}</svg>`;
}

const ROW_ICON_STYLE = {
  markdown: { bg: "var(--surface-sunken)", fg: "var(--ink-soft)" },
  pdf: { bg: "var(--bad-wash)", fg: "var(--bad)" },
  docx: { bg: "var(--accent-wash)", fg: "var(--accent-strong)" },
  web: { bg: "var(--accent-wash)", fg: "var(--accent-strong)" },
  github: { bg: "var(--surface-sunken)", fg: "var(--ink)" },
  audio: { bg: "var(--good-wash)", fg: "var(--good)" },
  video: { bg: "var(--warn-wash)", fg: "var(--warn)" },
};

const TYPE_LABELS = { markdown: "Markdown", pdf: "PDF", docx: "DOCX", web: "Web", github: "GitHub", audio: "Audio", video: "Video" };

// ---------- Utilities ----------
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// GFM table detection/rendering for mdLite() below - a line of cells
// followed by a "|---|---|" separator line is the one piece of
// markdown table syntax the model reliably produces, and (unlike
// bold/code/fences) there was no handling for it at all, so a table
// in the answer rendered as literal "| a | b |" text lines instead of
// an actual <table>. Operates on lines so multi-row tables of any
// width are supported, not just a fixed column count.
function _isTableRow(line) {
  const t = line.trim();
  return t.startsWith("|") && t.endsWith("|") && t.length > 1;
}

function _isTableSeparatorRow(line) {
  if (!_isTableRow(line)) return false;
  const cells = line.trim().slice(1, -1).split("|");
  return cells.length > 0 && cells.every((c) => /^\s*:?-+:?\s*$/.test(c));
}

function _splitTableRow(line) {
  return line.trim().slice(1, -1).split("|").map((c) => c.trim());
}

function _renderMarkdownTables(html) {
  const lines = html.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    if (_isTableRow(lines[i]) && lines[i + 1] !== undefined && _isTableSeparatorRow(lines[i + 1])) {
      const headerCells = _splitTableRow(lines[i]);
      i += 2;
      const bodyRows = [];
      while (i < lines.length && _isTableRow(lines[i])) {
        bodyRows.push(_splitTableRow(lines[i]));
        i++;
      }
      const thead = "<tr>" + headerCells.map((c) => `<th>${c}</th>`).join("") + "</tr>";
      const tbody = bodyRows.map((row) => "<tr>" + row.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("");
      // Built as a single line (no embedded "\n") so the later
      // "\n -> <br>" pass in mdLite() can't inject stray <br>s inside
      // the table. Wrapped in a scrollable div so a wide table can't
      // force the whole message bubble to overflow horizontally.
      out.push(`<div style="overflow-x:auto;"><table class="md-table">${thead}${tbody}</table></div>`);
    } else {
      out.push(lines[i]);
      i++;
    }
  }
  return out.join("\n");
}

// Lightweight, safe token colorizer for fenced code blocks - JSON is
// the one structured format the model reliably emits verbatim (see
// app/prompting/prompt_builder.py's code-block instructions), so this
// recognizes JSON-shaped tokens (strings, keys, numbers, booleans/
// null, punctuation) and leaves everything else as plain escaped text
// rather than attempting full language-aware highlighting for every
// possible fenced language. Every character is escaped exactly once
// here (never trusts the model's output as HTML), so this is safe
// even though it's applied before mdLite's own top-level escapeHtml
// pass (see _renderCodeBlock/mdLite below for why code fences are
// pulled out and escaped separately from the rest of the message).
function _highlightCode(rawCode) {
  const tokenRe = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+\.?\d*(?:[eE][+-]?\d+)?)|([{}[\],:])/g;
  let out = "";
  let lastIndex = 0;
  let match;
  while ((match = tokenRe.exec(rawCode)) !== null) {
    out += escapeHtml(rawCode.slice(lastIndex, match.index));
    const [, str, colon, keyword, num, punct] = match;
    if (str !== undefined) {
      out += `<span class="${colon ? "tok-key" : "tok-string"}">${escapeHtml(str)}</span>`;
      if (colon) out += escapeHtml(colon);
    } else if (keyword !== undefined) {
      out += `<span class="tok-bool">${escapeHtml(keyword)}</span>`;
    } else if (num !== undefined) {
      out += `<span class="tok-number">${escapeHtml(num)}</span>`;
    } else if (punct !== undefined) {
      out += `<span class="tok-punct">${escapeHtml(punct)}</span>`;
    }
    lastIndex = tokenRe.lastIndex;
  }
  out += escapeHtml(rawCode.slice(lastIndex));
  return out;
}

// Builds a scrollable code block with a language label + copy button.
// See the document-level "click" listener below (wireCodeCopyButtons)
// for what actually copies the code - kept as event delegation since
// this HTML gets re-inserted via innerHTML on every streamed token
// (see sendMessageTo), so a directly-bound onclick would be discarded
// each time.
function _renderCodeBlock(lang, rawCode) {
  const trimmed = rawCode.replace(/^\n/, "").replace(/\n$/, "");
  const label = lang ? escapeHtml(lang) : "code";
  return (
    `<div class="code-block">` +
      `<div class="code-block-head">` +
        `<span class="code-lang">${label}</span>` +
        `<button type="button" class="code-copy-btn">Copy</button>` +
      `</div>` +
      `<pre><code>${_highlightCode(trimmed)}</code></pre>` +
    `</div>`
  );
}

// Minimal, SAFE markdown-lite: escapes HTML first, then adds a small
// set of formatting tags around the already-escaped text (no CDN
// markdown library - this app stays fully local/offline-capable).
function mdLite(text) {
  // Fenced code blocks are pulled out of the RAW (unescaped) text
  // first, so the language tag can be read and the code content
  // escaped/highlighted exactly once (see _highlightCode). Escaping
  // the WHOLE message first, like every other step below does, would
  // mean matching against already-HTML-escaped text (quotes turned
  // into "&quot;", etc.) - workable but needlessly fragile, so code
  // fences get pulled out via a placeholder token instead and spliced
  // back in as the very last step, after every other substitution.
  const codeBlocks = [];
  const withPlaceholders = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_m, lang, code) => {
    const token = `\u0000CODEBLOCK${codeBlocks.length}\u0000`;
    codeBlocks.push(_renderCodeBlock(lang, code));
    return token;
  });

  let html = escapeHtml(withPlaceholders);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = _renderMarkdownTables(html);
  html = html.replace(/\n/g, "<br>");
  html = html.replace(/\u0000CODEBLOCK(\d+)\u0000/g, (_m, idx) => codeBlocks[Number(idx)]);
  return html;
}

// Copy-to-clipboard for fenced code blocks, wired once via event
// delegation on `document` (not a per-button onclick) since
// mdLite()'s output is re-inserted via innerHTML on every streamed
// token - a directly-bound handler would be thrown away each time,
// but a listener on document survives any number of innerHTML swaps.
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".code-copy-btn");
  if (!btn) return;
  const codeEl = btn.closest(".code-block")?.querySelector("code");
  if (!codeEl) return;
  const restore = () => { btn.textContent = "Copy"; };
  navigator.clipboard.writeText(codeEl.textContent).then(
    () => { btn.textContent = "Copied!"; setTimeout(restore, 1500); },
    () => { btn.textContent = "Failed"; setTimeout(restore, 1500); }
  );
});

function rowsTable(rows) {
  if (!rows || !rows.length) return "";
  const headers = Object.keys(rows[0]);
  return (
    "<table><tr>" + headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("") + "</tr>" +
    rows.map((row) => "<tr>" + headers.map((h) => `<td>${escapeHtml(String(row[h]))}</td>`).join("") + "</tr>").join("") +
    "</table>"
  );
}

function pollJob(jobId, { onProgress, onDone, onError, interval = 1200 } = {}) {
  const tick = async () => {
    let data;
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      data = await res.json();
    } catch (err) {
      onError && onError(String(err));
      return;
    }
    if (data.status === "running") {
      onProgress && onProgress(data.progress || {});
      setTimeout(tick, interval);
    } else if (data.status === "done") {
      onDone && onDone(data.result);
    } else if (data.status === "error") {
      onError && onError(data.error || "Job failed.");
    } else {
      onError && onError("Job not found.");
    }
  };
  tick();
}

// ---------- Navigation ----------
const state = { activeType: "markdown" };
let repoOptionsLoaded = false;
let evalDatasetsLoaded = false;

const pageLoaders = {
  sources: async () => {
    renderIngestControls();
    await Promise.all([loadSourceSummary(), loadSourceList()]);
  },
  chat: loadChatPage,
  evaluation: loadEvalPage,
  settings: loadSettingsPage,
};

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", async () => {
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    item.classList.add("active");
    const page = item.dataset.page;
    document.getElementById("page-" + page).classList.add("active");
    await pageLoaders[page]();
  });
});

document.querySelectorAll(".type-card").forEach((card) => {
  card.addEventListener("click", async () => {
    sourceChatSnapshots[state.activeType] = sourceChatThreadEl.innerHTML;

    document.querySelectorAll(".type-card").forEach((c) => c.classList.remove("active"));
    card.classList.add("active");
    state.activeType = card.dataset.type;
    document.getElementById("ingest-status").textContent = "";
    renderIngestControls();
    showSourceChatForType(state.activeType);
    if (state.activeType === "github") await ensureRepoOptionsLoaded();
    await loadSourceList();
  });
});

// ---------- Sources page ----------
async function loadSourceSummary() {
  const data = await (await fetch("/api/sources")).json();
  document.getElementById("sources-meta").textContent =
    `${data.total_sources} source${data.total_sources !== 1 ? "s" : ""} · ${data.total_chunks} chunks`;
}

async function loadSourceList() {
  const type = state.activeType;
  const listEl = document.getElementById("sources-list");
  listEl.innerHTML = '<div class="empty-state">Loading…</div>';
  const rows = await (await fetch(`/api/sources/${type}`)).json();
  if (!rows.length) {
    listEl.innerHTML = `<div class="empty-state">No ${TYPE_LABELS[type]} sources indexed yet.</div>`;
    return;
  }
  listEl.innerHTML = rows.map((row) => sourceRowHtml(type, row)).join("");
  listEl.onclick = async (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const id = btn.dataset.id;
    if (btn.dataset.action === "save") {
      await fetch(`/api/sources/${type}/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: id }),
      });
    } else if (btn.dataset.action === "delete") {
      await fetch(`/api/sources/${type}?doc_id=${encodeURIComponent(id)}`, { method: "DELETE" });
    }
    await Promise.all([loadSourceList(), loadSourceSummary()]);
  };
}

function sourceRowHtml(type, row) {
  const style = ROW_ICON_STYLE[type];
  const pillHtml = row.status === "pending"
    ? '<span class="pill pill-warn">Pending</span>'
    : '<span class="pill pill-good">Indexed</span>';
  const saveBtn = row.status === "pending"
    ? `<button class="btn btn-ghost" data-action="save" data-id="${escapeHtml(row.id)}" style="font-size:12.5px; padding:6px 10px;">Save</button>`
    : "";
  return `
    <div class="src-row">
      <div class="src-icon" style="background:${style.bg};">${svgIcon(type, { size: 19, stroke: style.fg })}</div>
      <div class="src-body">
        <div class="src-name">${escapeHtml(row.id)}</div>
        <div class="src-sub">${TYPE_LABELS[type]} · ${row.chunk_count} chunks</div>
      </div>
      <div class="src-actions">
        ${pillHtml}
        ${saveBtn}
        <button class="btn btn-ghost btn-icon" data-action="delete" data-id="${escapeHtml(row.id)}">${svgIcon("close", { size: 15 })}</button>
      </div>
    </div>`;
}

function renderIngestControls() {
  const container = document.getElementById("ingest-controls");
  const type = state.activeType;
  if (type === "markdown" || type === "pdf" || type === "docx" || type === "audio" || type === "video") {
    const accept = {
      markdown: ".md",
      pdf: ".pdf",
      docx: ".docx",
      audio: ".mp3,.wav,.m4a,.flac,.ogg",
      video: ".mp4,.mov,.mkv,.avi,.webm",
    }[type];
    container.innerHTML = `
      <div class="ingest-row">
        <input type="file" id="ingest-file-input" accept="${accept}" style="display:none" />
        <button class="btn btn-primary" id="ingest-file-btn">${svgIcon("upload", { size: 15, stroke: "#fff" })} Add ${TYPE_LABELS[type]} file</button>
      </div>`;
    document.getElementById("ingest-file-btn").onclick = () => document.getElementById("ingest-file-input").click();
    document.getElementById("ingest-file-input").onchange = onFileSelected;
  } else if (type === "web") {
    container.innerHTML = `
      <div class="ingest-row">
        <input class="text-input" id="ingest-web-url" placeholder="https://example.com/docs/page" />
        <label class="check-item"><input type="checkbox" id="ingest-web-crawl" /> Crawl whole site</label>
        <button class="btn btn-primary" id="ingest-web-btn">Ingest URL</button>
      </div>`;
    document.getElementById("ingest-web-btn").onclick = onWebIngest;
  } else if (type === "github") {
    container.innerHTML = `
      <div class="ingest-row">
        <input class="text-input" id="ingest-gh-repo" placeholder="owner/repo or full URL" />
        <input class="text-input" id="ingest-gh-branch" placeholder="branch (optional)" style="max-width:160px;" />
        <button class="btn btn-primary" id="ingest-gh-btn">Ingest repo</button>
      </div>
      <div class="ingest-row" style="margin-top:8px;">
        <input class="text-input" id="ingest-gh-activity-repo" placeholder="owner/repo (already-ingested or new)" />
        <label class="check-item"><input type="checkbox" id="ingest-gh-issues" checked /> Issues</label>
        <label class="check-item"><input type="checkbox" id="ingest-gh-prs" checked /> Pull requests</label>
        <label class="check-item"><input type="checkbox" id="ingest-gh-discussions" /> Discussions (needs GITHUB_TOKEN)</label>
        <button class="btn btn-primary" id="ingest-gh-activity-btn">Ingest activity</button>
      </div>`;
    document.getElementById("ingest-gh-btn").onclick = onGithubIngest;
    document.getElementById("ingest-gh-activity-btn").onclick = onGithubActivityIngest;
  }
}

async function onFileSelected(e) {
  const file = e.target.files[0];
  if (!file) return;
  const type = state.activeType;
  const statusEl = document.getElementById("ingest-status");
  statusEl.textContent = `Ingesting ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`/api/sources/${type}/upload`, { method: "POST", body: form });
  const { job_id } = await res.json();
  pollJob(job_id, {
    onProgress: (p) => { statusEl.textContent = p.message || "Ingesting…"; },
    onDone: async () => {
      statusEl.textContent = `Ingested ${file.name}. Marked as pending — save to keep.`;
      await Promise.all([loadSourceList(), loadSourceSummary()]);
    },
    onError: (err) => { statusEl.textContent = `Failed to ingest: ${err}`; },
  });
  e.target.value = "";
}

async function onWebIngest() {
  const url = document.getElementById("ingest-web-url").value.trim();
  if (!url) return;
  const crawl = document.getElementById("ingest-web-crawl").checked;
  const statusEl = document.getElementById("ingest-status");
  statusEl.textContent = crawl ? `Crawling ${url}…` : `Fetching ${url}…`;
  const res = await fetch("/api/sources/web/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, crawl }),
  });
  const { job_id } = await res.json();
  pollJob(job_id, {
    onProgress: (p) => { statusEl.textContent = p.message || "Working…"; },
    onDone: async (result) => {
      statusEl.textContent = result.saved
        ? `Crawled, ingested, and saved ${result.pages_ingested} page(s).`
        : `Ingested '${result.document_id}'. Marked as pending — save to keep.`;
      await Promise.all([loadSourceList(), loadSourceSummary()]);
    },
    onError: (err) => { statusEl.textContent = `Failed: ${err}`; },
  });
}

async function onGithubIngest() {
  const repoUrl = document.getElementById("ingest-gh-repo").value.trim();
  if (!repoUrl) return;
  const branch = document.getElementById("ingest-gh-branch").value.trim() || null;
  const statusEl = document.getElementById("ingest-status");
  statusEl.textContent = `Ingesting ${repoUrl}…`;
  const res = await fetch("/api/sources/github/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_url: repoUrl, branch }),
  });
  const { job_id } = await res.json();
  pollJob(job_id, {
    onProgress: (p) => { statusEl.textContent = p.message || "Working…"; },
    onDone: async (result) => {
      statusEl.textContent = `Ingested '${result.repository}'. Marked as pending — save to keep.`;
      await Promise.all([loadSourceList(), loadSourceSummary()]);
    },
    onError: (err) => { statusEl.textContent = `Failed: ${err}`; },
  });
}

async function onGithubActivityIngest() {
  const repoUrl = document.getElementById("ingest-gh-activity-repo").value.trim();
  if (!repoUrl) return;
  const includeIssues = document.getElementById("ingest-gh-issues").checked;
  const includePrs = document.getElementById("ingest-gh-prs").checked;
  const includeDiscussions = document.getElementById("ingest-gh-discussions").checked;
  const statusEl = document.getElementById("ingest-status");
  statusEl.textContent = `Ingesting activity for ${repoUrl}…`;
  const res = await fetch("/api/sources/github/activity", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_url: repoUrl,
      include_issues: includeIssues,
      include_prs: includePrs,
      include_discussions: includeDiscussions,
    }),
  });
  const { job_id } = await res.json();
  pollJob(job_id, {
    onProgress: (p) => { statusEl.textContent = p.message || "Working…"; },
    onDone: async (result) => {
      statusEl.textContent =
        `Ingested ${result.issues} issue(s), ${result.pull_requests} PR(s), ${result.discussions} ` +
        `discussion(s) for '${result.repository}'. Marked as pending — save to keep.`;
      await Promise.all([loadSourceList(), loadSourceSummary()]);
    },
    onError: (err) => { statusEl.textContent = `Failed: ${err}`; },
  });
}

// ---------- Chat page ----------
// Populates both the cross-KB Chat page's scope select AND the
// Sources page's per-github-type repo select, since both list the
// same set of ingested repos.
async function ensureRepoOptionsLoaded() {
  if (repoOptionsLoaded) return;
  const repos = await (await fetch("/api/sources/github/repositories")).json();
  for (const selectId of ["chat-scope-select", "source-chat-repo-select"]) {
    const select = document.getElementById(selectId);
    repos.forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r;
      opt.textContent = r;
      select.appendChild(opt);
    });
  }
  repoOptionsLoaded = true;
}

async function loadChatPage() {
  const summary = await (await fetch("/api/sources")).json();
  for (const kb of Object.keys(summary.kbs)) {
    const el = document.getElementById(`kb-count-${kb}`);
    if (el) el.textContent = summary.kbs[kb].toLocaleString();
  }
  await ensureRepoOptionsLoaded();
}

function userMessageHtml(question) {
  return `
    <div class="msg msg-user">
      <div class="msg-avatar user">${svgIcon("user", { size: 15, stroke: "#fff" })}</div>
      <div class="msg-bubble">${escapeHtml(question)}</div>
    </div>`;
}

const THINKING_HTML = '<span class="thinking">Thinking <span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span></span>';

function assistantShellHtml(groupId) {
  return `
    <div class="msg-group" id="${groupId}">
      <div class="msg msg-assistant">
        <div class="msg-avatar assistant">${svgIcon("brand", { size: 15, stroke: "#fff" })}</div>
        <div class="msg-bubble" id="${groupId}-bubble">${THINKING_HTML}</div>
      </div>
    </div>`;
}

function formatTimestamp(seconds) {
  const total = Math.floor(seconds || 0);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function renderChatImages(groupId, images) {
  if (!images || !images.length) return;
  const group = document.getElementById(groupId);
  const html = images.map((img) => {
    // A video-frame hit (see app.ingestion.ingest._ingest_video_frames)
    // has no page_url - show which video/timestamp it's from instead.
    const isVideoFrame = img.origin === "video_frame";
    const captionHtml = isVideoFrame
      ? `<div style="font-size:12px; color:var(--muted); margin-top:4px;">From video: ${escapeHtml(img.video_document_id)} at ${formatTimestamp(img.timestamp_seconds)}</div>`
      : img.page_url
        ? `<div style="font-size:12px; color:var(--muted); margin-top:4px;">From: ${escapeHtml(img.page_url)}</div>`
        : "";
    return `
      <div style="margin-left:42px; margin-top:8px; max-width:400px;">
        <img src="${escapeHtml(img.image_url)}" style="max-width:100%; border-radius:8px;" alt="${escapeHtml(img.alt_text || "")}" />
        ${captionHtml}
      </div>`;
  }).join("");
  group.insertAdjacentHTML("beforeend", html);
}

function renderChatAudioClips(groupId, audioClips) {
  if (!audioClips || !audioClips.length) return;
  const group = document.getElementById(groupId);
  const html = audioClips.map((clip) => `
      <div style="margin-left:42px; margin-top:8px; max-width:400px;">
        <audio controls src="${escapeHtml(clip.audio_url)}#t=${clip.start_seconds}" style="width:100%;"></audio>
        <div style="font-size:12px; color:var(--muted); margin-top:4px;">From: ${escapeHtml(clip.document_id)} at ${formatTimestamp(clip.start_seconds)}</div>
      </div>`).join("");
  group.insertAdjacentHTML("beforeend", html);
}

function techDetailsHtml(details) {
  if (!details) return "";
  let inner = "";
  const r = details.routing;
  if (r) {
    inner += `<p><strong>Method:</strong> ${escapeHtml(r.method)} · <strong>Explicit KB signal:</strong> ${r.explicit_kb_signal ? escapeHtml(r.explicit_kb_signal.join(", ")) : "none"}</p>`;
    if (r.kb_scores && r.kb_scores.length) inner += rowsTable(r.kb_scores);
  }
  if (details.github_intent) {
    const gi = details.github_intent;
    inner += `<p><strong>GitHub intent:</strong> ${escapeHtml(gi.label)} (${escapeHtml(gi.strategy)}) · method: ${escapeHtml(gi.method)}</p>`;
  }
  if (details.contextualization) {
    const cx = details.contextualization;
    inner += `<p><strong>Conversation memory:</strong> original "${escapeHtml(cx.original)}"${cx.changed ? ` → resolved "${escapeHtml(cx.resolved)}"` : " (already standalone)"}${cx.fell_back ? " · fell back" : ""}</p>`;
  }
  if (details.query_transform) {
    const qt = details.query_transform;
    let lines = "";
    if (qt.rewritten) lines += `<li>Rewritten: ${escapeHtml(qt.rewritten)}</li>`;
    qt.sub_queries.forEach((s) => (lines += `<li>Sub-question: ${escapeHtml(s)}</li>`));
    qt.paraphrases.forEach((s) => (lines += `<li>Paraphrase: ${escapeHtml(s)}</li>`));
    if (qt.hyde_answer) lines += `<li>HyDE: ${escapeHtml(qt.hyde_answer)}</li>`;
    if (lines) inner += `<p><strong>Query variants:</strong></p><ul>${lines}</ul>`;
  }
  if (details.crag) {
    const c = details.crag;
    inner += `<p><strong>Corrective RAG:</strong> ${escapeHtml(c.overall)} (correct ${c.counts.correct}, partial ${c.counts.partial}, incorrect ${c.counts.incorrect})</p>`;
    inner += rowsTable(c.verdicts);
  }
  if (details.self_rag && details.self_rag.dimensions) {
    inner += "<p><strong>Self-RAG scores:</strong></p>" + rowsTable(details.self_rag.dimensions);
  }
  if (!inner) return "";
  return `<details class="tech-details"><summary>Technical details</summary>${inner}</details>`;
}

function renderInsight(groupId, insight) {
  if (!insight) return;
  const group = document.getElementById(groupId);
  const pillClass = { good: "pill-good", warn: "pill-warn", muted: "pill-muted" }[insight.pill_kind] || "pill-muted";
  let body = "";
  if (insight.contextualized_as) body += `<p>${escapeHtml(insight.contextualized_as)}</p>`;
  if (insight.served_from_cache) body += `<p><strong>${escapeHtml(insight.served_from_cache)}</strong></p>`;
  body += `<p>${escapeHtml(insight.searched)}</p>`;
  if (insight.before_answering) body += `<p>${escapeHtml(insight.before_answering)}</p>`;
  if (insight.after_answering) body += `<p>${escapeHtml(insight.after_answering)}</p>`;
  (insight.hallucinations || []).forEach((s) => (body += `<p style="font-style:italic;">${escapeHtml(s)}</p>`));
  body += techDetailsHtml(insight.technical_details);
  const html = `
    <div class="insight">
      <div class="insight-head">
        <div class="insight-head-left">${svgIcon("search", { size: 14 })} How this answer was checked</div>
        <span class="pill ${pillClass}">${escapeHtml(insight.pill_label)}</span>
      </div>
      <div class="insight-body">${body}</div>
    </div>`;
  group.insertAdjacentHTML("beforeend", html);
}

// Event delegation (not a per-element .onclick) so the toggle keeps
// working even after a thread's innerHTML is snapshotted/restored
// (see the Sources-page per-type chat below) - inline event-listener
// bindings don't survive an innerHTML round-trip, but a listener on
// the thread container itself does.
function wireInsightToggle(threadEl) {
  threadEl.addEventListener("click", (e) => {
    const head = e.target.closest(".insight-head");
    if (head) head.parentElement.classList.toggle("open");
  });
}

/**
 * Send a question to /api/chat/stream and render the streamed answer
 * + insight panel into `threadEl`. Shared by the cross-KB Chat page
 * and each Sources-page per-type chat - kb=null means cross-KB.
 */
async function sendMessageTo(threadEl, question, { kb = null, repository = null } = {}) {
  threadEl.insertAdjacentHTML("beforeend", userMessageHtml(question));
  const groupId = "msg-group-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8);
  threadEl.insertAdjacentHTML("beforeend", assistantShellHtml(groupId));
  threadEl.scrollTop = threadEl.scrollHeight;

  const bubble = document.getElementById(groupId + "-bubble");

  let res;
  try {
    res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, kb, repository }),
    });
  } catch (err) {
    bubble.textContent = "Could not reach the server: " + err;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answerText = "";
  // Bubble keeps showing the "Thinking…" indicator (set by
  // assistantShellHtml) until the FIRST token actually arrives -
  // retrieval + LLM prompt-processing on CPU can take 15-60+ s before
  // any token shows up, and clearing the bubble to blank right away
  // (instead of leaving the indicator up) makes the app look frozen
  // for that whole stretch.
  let gotFirstToken = false;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const line = rawEvent.replace(/^data: /, "");
        if (!line) continue;
        let payload;
        try {
          payload = JSON.parse(line);
        } catch {
          continue;
        }
        if (payload.token !== undefined) {
          gotFirstToken = true;
          answerText += payload.token;
          bubble.innerHTML = mdLite(answerText);
          threadEl.scrollTop = threadEl.scrollHeight;
        } else if (payload.event === "done") {
          renderChatImages(groupId, payload.images);
          renderChatAudioClips(groupId, payload.audio_clips);
          renderInsight(groupId, payload.insight);
          threadEl.scrollTop = threadEl.scrollHeight;
        }
      }
    }
  } catch (err) {
    bubble.textContent = "Connection to the server was interrupted: " + err;
    return;
  }
  if (!gotFirstToken) bubble.textContent = "The model returned an empty response. Check the server log for errors.";
}

const chatThreadEl = document.getElementById("chat-thread");
wireInsightToggle(chatThreadEl);

async function sendChatMessage() {
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  const repoScope = document.getElementById("chat-scope-select").value || null;
  await sendMessageTo(chatThreadEl, question, { kb: null, repository: repoScope });
}

document.getElementById("chat-send").onclick = sendChatMessage;
document.getElementById("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendChatMessage();
});

// ---------- Per-source-type chat (Sources page) ----------
// Each type keeps its own conversation, mirroring the previous
// Streamlit build's separate messages_by_kb histories - since there's
// one shared thread/composer DOM (not five), the currently-inactive
// types' HTML is snapshotted into memory on switch and restored when
// the user switches back.
const sourceChatSnapshots = { markdown: "", pdf: "", docx: "", web: "", github: "", audio: "", video: "" };
const sourceChatThreadEl = document.getElementById("source-chat-thread");
wireInsightToggle(sourceChatThreadEl);

function showSourceChatForType(type) {
  document.getElementById("source-chat-label").textContent = TYPE_LABELS[type];
  sourceChatThreadEl.innerHTML = sourceChatSnapshots[type] || "";
  sourceChatThreadEl.scrollTop = sourceChatThreadEl.scrollHeight;
  document.getElementById("source-repo-scope-row").style.display = type === "github" ? "flex" : "none";
}

async function sendSourceChatMessage() {
  const input = document.getElementById("source-chat-input");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  const type = state.activeType;
  const repoScope = type === "github" ? (document.getElementById("source-chat-repo-select").value || null) : null;
  await sendMessageTo(sourceChatThreadEl, question, { kb: type, repository: repoScope });
  sourceChatSnapshots[type] = sourceChatThreadEl.innerHTML;
}

document.getElementById("source-chat-send").onclick = sendSourceChatMessage;
document.getElementById("source-chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendSourceChatMessage();
});

// ---------- Evaluation page ----------
async function loadEvalPage() {
  if (!evalDatasetsLoaded) {
    const datasets = await (await fetch("/api/evaluation/datasets")).json();
    const select = document.getElementById("eval-dataset-select");
    select.innerHTML = datasets.length
      ? datasets.map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join("")
      : '<option value="">No datasets found in data/eval/</option>';
    evalDatasetsLoaded = true;
  }
  const last = await (await fetch("/api/evaluation/last")).json();
  if (last) renderEvalResults(last);
}

function renderEvalResults(report) {
  const el = document.getElementById("eval-results");
  const metricCards = Object.entries(report.aggregate)
    .map(([name, value]) => {
      const isFraction = value !== null && value <= 1;
      const valStr = value === null ? "—" : isFraction ? value.toFixed(2) : value.toFixed(1);
      const pct = value === null ? 0 : Math.max(0, Math.min(100, value * 100));
      return `
        <div class="metric-card">
          <div class="metric-name">${escapeHtml(name.replace(/_/g, " "))}</div>
          <div class="metric-val">${valStr}</div>
          <div class="metric-bar"><span style="width:${pct}%;"></span></div>
        </div>`;
    })
    .join("");

  const rows = report.items
    .map(
      (it) => `
      <tr>
        <td>${escapeHtml(it.question)}</td>
        <td>${escapeHtml(it.kb_used || "-")}</td>
        <td class="num">${it.faithfulness ?? "—"}</td>
        <td class="num">${it.answer_relevance ?? "—"}</td>
        <td class="num">${it.doc_precision ?? "—"}</td>
        <td class="num">${it.doc_recall ?? "—"}</td>
        <td class="num">${it.latency_total_s ? it.latency_total_s.toFixed(1) + "s" : "—"}</td>
      </tr>`
    )
    .join("");

  el.innerHTML = `
    <div class="metric-grid">${metricCards}</div>
    <div style="height:20px;"></div>
    <div class="panel" style="padding:0;">
      <div style="padding:18px 22px 4px;"><h3 style="margin-bottom:0;">Per-question results</h3></div>
      <div class="table-wrap">
        <table class="eval-table">
          <thead><tr><th>Question</th><th>KB</th><th class="num">Faithfulness</th><th class="num">Relevance</th><th class="num">Doc precision</th><th class="num">Doc recall</th><th class="num">Latency</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>`;
}

document.getElementById("eval-run-btn").onclick = async () => {
  const datasetPath = document.getElementById("eval-dataset-select").value;
  if (!datasetPath) return;
  const body = {
    dataset_path: datasetPath,
    default_kb: document.getElementById("eval-default-kb").value,
    top_k: parseInt(document.getElementById("eval-top-k").value, 10),
    use_crag: document.getElementById("eval-use-crag").checked,
    use_query_transform: document.getElementById("eval-use-qt").checked,
    judge_faithfulness: document.getElementById("eval-judge-faith").checked,
    judge_answer_relevance: document.getElementById("eval-judge-ans").checked,
    judge_context_relevance: document.getElementById("eval-judge-ctx").checked,
    report_name: document.getElementById("eval-report-name").value || "eval",
  };
  const statusEl = document.getElementById("eval-status");
  statusEl.textContent = "Starting…";
  const res = await fetch("/api/evaluation/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const { job_id } = await res.json();
  pollJob(job_id, {
    interval: 1500,
    onProgress: (p) => { statusEl.textContent = p.total ? `[${p.index + 1}/${p.total}] ${p.question}` : "Running…"; },
    onDone: (result) => { statusEl.textContent = "Done."; renderEvalResults(result); },
    onError: (err) => { statusEl.textContent = `Failed: ${err}`; },
  });
};

// ---------- Settings page ----------
async function loadSettingsPage() {
  const data = await (await fetch("/api/settings")).json();
  document.getElementById("setting-topk").value = data.top_k;
  document.getElementById("setting-topk-val").textContent = data.top_k;
  document.getElementById("setting-use-vision").checked = data.use_vision;
  document.getElementById("setting-query-transform").checked = data.query_transform_enabled;
  document.getElementById("setting-crag").checked = data.crag_enabled;
  document.getElementById("setting-self-rag").checked = data.self_rag_enabled;
  document.getElementById("setting-conversation-memory").checked = data.conversation_memory_enabled;
}

document.getElementById("setting-topk").addEventListener("input", (e) => {
  document.getElementById("setting-topk-val").textContent = e.target.value;
});
document.getElementById("setting-topk").addEventListener("change", (e) => {
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: parseInt(e.target.value, 10) }),
  });
});
document.getElementById("setting-use-vision").addEventListener("change", (e) => {
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ use_vision: e.target.checked }),
  });
});
document.getElementById("setting-query-transform").addEventListener("change", (e) => {
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query_transform_enabled: e.target.checked }),
  });
});
document.getElementById("setting-crag").addEventListener("change", (e) => {
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ crag_enabled: e.target.checked }),
  });
});
document.getElementById("setting-self-rag").addEventListener("change", (e) => {
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ self_rag_enabled: e.target.checked }),
  });
});
document.getElementById("setting-conversation-memory").addEventListener("change", (e) => {
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_memory_enabled: e.target.checked }),
  });
});

// ---------- Clear conversation ----------
// Clears server-side remembered history for one thread (see
// api/chat.py's POST /api/chat/clear) and empties the visible DOM -
// server-side persistence means a page reload no longer silently
// resets memory the way client-only history used to, so this button
// is the deliberate way to start fresh.
async function clearChatThread(threadEl, kb) {
  await fetch("/api/chat/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kb }),
  });
  threadEl.innerHTML = "";
  if (kb) {
    // Also wipe the Sources-page snapshot for this KB, otherwise
    // switching type-cards away and back would restore the
    // just-cleared HTML from sourceChatSnapshots.
    sourceChatSnapshots[kb] = "";
  }
}

document.getElementById("chat-clear-btn").addEventListener("click", () => {
  clearChatThread(chatThreadEl, null);
});
document.getElementById("source-chat-clear-btn").addEventListener("click", () => {
  clearChatThread(sourceChatThreadEl, state.activeType);
});

// ---------- Current user ----------
async function loadCurrentUser() {
  try {
    const res = await fetch("/api/auth/me");
    const data = await res.json();
    const username = data.username || "?";
    document.getElementById("rail-foot-avatar").textContent = username.charAt(0).toUpperCase();
    document.getElementById("rail-foot-name").textContent = username;
  } catch (err) {
    document.getElementById("rail-foot-name").textContent = "Unknown";
  }
}

document.getElementById("rail-foot-logout").addEventListener("click", async () => {
  await fetch("/logout", { method: "POST" });
  window.location.href = "/login";
});

// ---------- Initial load ----------
document.querySelector('.type-card[data-type="markdown"]').classList.add("active");
showSourceChatForType("markdown");
pageLoaders.sources();
loadCurrentUser();

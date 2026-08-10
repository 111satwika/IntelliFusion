"""
Git-history retriever for the GitHub KB's "history" adaptive intent.

Fetches recent commits from the GitHub REST API on demand rather than
cloning the repo locally - matches app.ingestion.loader's existing
API-driven pattern (see load_github_repository) so no new dependencies
(git binary, on-disk clones) are needed. Optional GITHUB_TOKEN raises
the anonymous rate limit; without it, low-volume interactive use still
works.

Returns hit dicts in the same shape as
app.retrieval.retriever.retrieve() so the prompt builder
(app.prompting.prompt_builder.build_prompt) and the LLM stream both
consume them without any special-casing.

When to call: only from app.retrieval.github_adaptive when the GitHub
intent classifier picks "history". Never mix into generic retrieve():
commit metadata surfaced alongside a normal "how does X work" answer
would just be noise.
"""

import logging
import os
import re

import requests

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"

# GitHub's commits API caps per_page at 100. This retriever's callers
# ask for small top_k (~5-10); the cap only matters if we ever raise
# top_k, but keeping it explicit avoids a silent surprise later.
_COMMITS_API_MAX_PER_PAGE = 100
_DEFAULT_TIMEOUT = 30


def _github_headers() -> dict[str, str]:
    """Same header shape as app.ingestion.loader._github_headers -
    duplicated (rather than imported) so git-history retrieval doesn't
    depend on loader internals and could in principle be pulled out
    into its own package."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# Same file-path recognizer as app.routing.github_intent's, kept local
# so this module can be called even without going through the intent
# classifier (e.g. from a test).
_PATH_RE = re.compile(
    r"([A-Za-z0-9_./\-]+\.(?:py|md|txt|js|ts|tsx|jsx|json|yaml|yml|toml|rst|go|rs|java|c|cpp|h|hpp|cs))\b"
)


def _extract_path(query_text: str) -> str | None:
    match = _PATH_RE.search(query_text)
    return match.group(1) if match else None


def _format_commit_content(commit: dict) -> str:
    """
    Turn a GitHub REST commit object into a readable chunk body.

    Deliberately compact: the LLM needs sha / author / date / subject
    to answer "who / when / what changed", not the full patch (which
    would blow up prompt size for many-commit history queries and
    also isn't returned by the commits list endpoint anyway - would
    require a second per-commit API call).
    """
    sha = commit.get("sha", "")[:7]
    commit_data = commit.get("commit", {}) or {}
    author_data = commit_data.get("author") or {}
    author = author_data.get("name", "unknown")
    date = author_data.get("date", "unknown")
    message = (commit_data.get("message") or "").strip()
    lines = [
        f"Commit {sha} by {author} on {date}",
        message,
    ]
    return "\n".join(lines)


def retrieve_git_history(
    query_text: str,
    repository: str,
    top_k: int = 5,
    timeout: int = _DEFAULT_TIMEOUT,
) -> list[dict]:
    """
    Fetch recent commits from `repository`, optionally scoped to a
    file path parsed out of `query_text`.

    Args:
        query_text: The user's raw question (used only to extract an
            optional file path via _extract_path).
        repository: "owner/repo" - which repo to query. Required
            because git history is per-repo, and there's no defensible
            way to pick a repo if the caller doesn't supply one.
        top_k: How many commits to return (capped by the GitHub API's
            per-call maximum).
        timeout: Per-request HTTP timeout in seconds.

    Returns:
        A list of hit dicts (same shape as retrieve()'s output)
        representing commits, ordered most-recent-first. Empty list on
        any API failure - "no history available" is a valid answer,
        not an error the pipeline should propagate up (the caller in
        app.retrieval.github_adaptive falls back to hybrid retrieval
        when this returns []).
    """
    if not repository or "/" not in repository:
        logger.warning(
            "retrieve_git_history called without a valid 'owner/repo': %r", repository
        )
        return []

    owner, repo = repository.split("/", 1)
    per_page = max(1, min(top_k, _COMMITS_API_MAX_PER_PAGE))
    params: dict[str, str | int] = {"per_page": per_page}
    path = _extract_path(query_text)
    if path:
        params["path"] = path

    url = f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
    try:
        response = requests.get(
            url, params=params, headers=_github_headers(), timeout=timeout
        )
        response.raise_for_status()
        commits = response.json()
    except requests.RequestException as exc:
        logger.warning(
            "Git history fetch failed for %s (path=%r): %s", repository, path, exc
        )
        return []

    if not isinstance(commits, list):
        # Defensive: /commits should return a list, but a rate-limit or
        # error response is a dict. Don't blow up the answer path.
        logger.warning(
            "Git history: unexpected response shape for %s: %r", repository, commits
        )
        return []

    hits: list[dict] = []
    for commit in commits[:top_k]:
        content = _format_commit_content(commit)
        sha = commit.get("sha", "")
        hits.append({
            "content": content,
            "metadata": {
                "source_type": "commit",
                "content_type": "commit",
                "repository": repository,
                "commit_sha": sha,
                "commit_url": commit.get("html_url", ""),
                "file_path": path,
                # Stable id per commit so a downstream deduplicator
                # (if any is ever added) treats two references to the
                # same commit as duplicates.
                "document_id": f"{repository}:commit:{sha}",
                "chunk_index": 0,
            },
            # Distance/similarity are dense-retriever concepts; a
            # commit hit doesn't have them. Kept as None so downstream
            # code reading hit.get("similarity") still works.
            "distance": None,
            "similarity": None,
        })

    logger.info(
        "Git history: fetched %d commit(s) from %s (path=%r)",
        len(hits), repository, path,
    )
    return hits

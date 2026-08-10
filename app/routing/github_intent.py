"""
GitHub-specific intent classifier for adaptive RAG.

Complements app.routing.router (which decides WHICH KBs/routes to
search for any query) with a second, GitHub-KB-only decision: for
queries scoped to the GitHub codebase, which retrieval STRATEGY best
answers them?

Seven intents:
- explanation:  "how does X work", "why is Y used" - dense semantic
                search over code chunks. BM25 fusion often adds noise
                here because natural-language paraphrases don't share
                literal tokens with the code that implements them.
- exact_code:   "find function foo", "where is class Bar defined" -
                BM25-first. Exact identifier names are lexical signals
                BM25 handles cleanly and dense embedding smears
                together.
- navigational: "in file X.py", "in the tests directory" - metadata
                filter on file_name, then dense retrieval within that
                filter. Skips scoring against unrelated files.
- history:      "who changed X", "recent commits to Y" - a completely
                separate retriever (app.retrieval.github_history) that
                hits the GitHub REST commits API, since git history
                isn't in the vector store at all.
- activity:     "what are the open issues", "latest pull requests" -
                metadata-filtered retrieval scoped to content_type in
                {issue, pull_request, discussion} (see
                app.ingestion.ingest.ingest_github_activity). Without
                this intent, a generic "what are the latest PRs and
                issues" question falls through to plain hybrid search,
                where a handful of issue/PR chunks routinely lose the
                similarity contest against hundreds of code chunks
                even when they're the only chunks that actually answer
                the question - a metadata filter sidesteps ranking
                entirely instead of trying to out-embed the rest of
                the repo.
- visual:       "show me the diagram", "screenshot" - runs the
                standard hybrid pipeline; the caller (cli.py) already
                layers image retrieval on top of any GitHub answer.
- structural:   "what calls X", "who imports Y", "subclasses of Z" -
                traverses the per-repo call/import/inherit graph
                built at ingest time (app.graph.code_graph), then
                materializes touched nodes back into chunks. Answers
                the class of question dense/BM25 alone can't - "give
                me every caller of retrieve_hybrid" needs a graph,
                not a similarity search.

Design mirrors app.routing.router: hybrid keyword + semantic
classification. Unlike router.py, we pick a SINGLE winning intent
(with a "general" fallback that the dispatcher treats as "run the
existing hybrid pipeline unchanged") because each intent dispatches to
a different retrieval pipeline - blending them would cost latency for
marginal recall.

Called only from app.retrieval.github_adaptive; every other retrieval
path in this repo is unchanged.
"""

import logging
import math
import re

from app.embeddings.embedder import embed_texts

logger = logging.getLogger(__name__)


# ---- Rule-based / keyword intent hints -------------------------------

# Deliberately short, high-precision keyword lists (same convention as
# app.routing.router). Each entry is checked as a plain substring of
# the (lowercased, space-padded) query.
_EXPLANATION_KEYWORDS = [
    "how does", "how do", "why does", "why is", "what does", "explain",
    "walk me through", "describe", "overview of", "purpose of",
]
_EXACT_CODE_KEYWORDS = [
    "find function", "find the function", "find class", "find the class",
    "find method", "grep", "locate", "definition of",
    "signature of", "call site", "declared in", "defined in",
]
_NAVIGATIONAL_KEYWORDS = [
    "in file", "in the file", "in module", "in the module",
    "in directory", "in folder", "tests directory", "under app/",
]
_HISTORY_KEYWORDS = [
    "recent commit", "recent change", "who changed", "who wrote",
    "when was", "history of", "changelog", "last commit",
    "who added", "blame", "commit log", "commit history",
]
_ACTIVITY_KEYWORDS = [
    "pull request", "pull requests", "merge request", "merge requests",
    "open issue", "open issues", "closed issue", "closed issues",
    "github issue", "github issues", "recent issue", "recent issues",
    "recent pull request", "recent pull requests", "latest issue",
    "latest issues", "latest pull request", "latest pull requests",
    "bug report", "bug reports", "feature request", "feature requests",
    "open pr", "open prs", "discussion thread", "discussion threads",
    "what issues", "which issues", "any issues", "list issues", "show issues",
    "what prs", "which prs", "any prs", "list prs", "list pull requests",
    "show pull requests",
]
_VISUAL_KEYWORDS = [
    "diagram", "screenshot", "picture of", "image of",
    "architecture diagram", "flowchart", "visualize",
]
_STRUCTURAL_KEYWORDS = [
    "who calls", "what calls", "callers of", "used by", "who uses",
    "calls to", "calls from", "callees of",
    "dependencies of", "depends on", "depend on",
    "who imports", "importers of", "imports of",
    "subclasses of", "who extends", "who inherits", "inherits from",
    "extends", "base classes of", "parent class of", "override",
]

_KEYWORDS_BY_INTENT = {
    "explanation": _EXPLANATION_KEYWORDS,
    "exact_code": _EXACT_CODE_KEYWORDS,
    "navigational": _NAVIGATIONAL_KEYWORDS,
    "history": _HISTORY_KEYWORDS,
    "activity": _ACTIVITY_KEYWORDS,
    "visual": _VISUAL_KEYWORDS,
    "structural": _STRUCTURAL_KEYWORDS,
}


def _rule_based_intents(query_text: str) -> set[str]:
    """Intents whose keyword list has a literal substring match."""
    query_lower = f" {query_text.lower()} "
    return {
        intent
        for intent, keywords in _KEYWORDS_BY_INTENT.items()
        if any(keyword in query_lower for keyword in keywords)
    }


# ---- Semantic intent exemplars ---------------------------------------

# Small, hand-written example questions per intent. Embedded once (see
# _get_intent_exemplar_embeddings) and compared to the query via
# cosine similarity - the highest-scoring exemplar in each intent
# stands in for "how semantically close is this query to a typical
# <intent> question", same idea as app.routing.router.
_INTENT_EXEMPLARS: dict[str, list[str]] = {
    "explanation": [
        "how does the retrieval pipeline work in this repository",
        "explain the purpose of the router module",
        "walk me through how authentication is implemented",
        "what does the ingest step do",
        "why is the parent child retriever used",
    ],
    "exact_code": [
        "find the function classify_route",
        "where is BM25Okapi imported from",
        "definition of retrieve_hybrid",
        "signature of load_github_repository",
        "find the class BM25Index",
    ],
    "navigational": [
        "what is in the tests directory",
        "list files in app/retrieval",
        "which module contains the chunker",
        "what does retriever.py contain",
    ],
    "history": [
        "who added the parent child retriever",
        "recent commits to the router module",
        "when was the ingest module last changed",
        "git log for the retrieval package",
        "what did the last commit change",
        "show me the commit that added streaming",
    ],
    "activity": [
        "what are the open issues in this repository",
        "show me the latest pull requests",
        "what pull requests are open right now",
        "list the recent issues reported",
        "are there any open bugs reported",
        "what discussions are happening in this repo",
        "has anyone reported this as an issue",
    ],
    "visual": [
        "show me the architecture diagram",
        "picture of the retrieval pipeline",
        "diagram of the ingestion flow",
    ],
    # structural has NO semantic exemplars: its rule keywords
    # ("who calls", "who imports", "subclasses of", ...) are dense
    # and unambiguous, and adding exemplars caused every question
    # that MENTIONS a graphed symbol (e.g. "how does retrieve_hybrid
    # work") to score high on structural via cosine similarity to the
    # symbol name, incorrectly stealing dispatch away from
    # explanation. Rule-only is a strictly higher-precision signal
    # for this intent.
}

# Same threshold as app.routing.router's _SEMANTIC_ROUTE_THRESHOLD -
# intent exemplars are similarly short, topic-focused sentences, so
# the cutoff that works for route/KB classification applies here too.
_SEMANTIC_INTENT_THRESHOLD = 0.4

_intent_exemplar_embeddings_cache: dict[str, list[list[float]]] | None = None


def _get_intent_exemplar_embeddings() -> dict[str, list[list[float]]]:
    """Lazily embed every intent's exemplars once, then cache in memory."""
    global _intent_exemplar_embeddings_cache
    if _intent_exemplar_embeddings_cache is None:
        _intent_exemplar_embeddings_cache = {
            intent: embed_texts(exemplars)
            for intent, exemplars in _INTENT_EXEMPLARS.items()
        }
    return _intent_exemplar_embeddings_cache


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _semantic_intents(query_vector: list[float]) -> dict[str, float]:
    """Each intent's score = max cosine similarity between the query
    and any single exemplar in that intent (same aggregation as
    app.routing.router._semantic_routes)."""
    exemplar_embeddings = _get_intent_exemplar_embeddings()
    return {
        intent: max(_cosine_similarity(query_vector, exemplar) for exemplar in exemplars)
        for intent, exemplars in exemplar_embeddings.items()
    }


# ---- Metadata hints for GitHub-KB filtering ------------------------------

# Every hint below narrows retrieval to a specific slice of the GitHub
# KB via a Chroma metadata $eq filter. app.retrieval.github_adaptive
# combines every extracted hint into an $and filter at query time.
# The extractors are deliberately conservative: better to miss a hint
# and fall back to broader retrieval than to filter down to the wrong
# subset and lose the correct chunk entirely.

# Plausible file path token: <name>.<ext>, allowing optional directory
# prefix like "app/retrieval/retriever.py". Extension list mirrors
# what app.ingestion.loader treats as ingestible - the goal is
# EXTRACTING a hint, not validating every possible extension.
_PATH_EXTENSION_RE = re.compile(
    r"([A-Za-z0-9_./\-]+\.(?:py|md|txt|js|ts|tsx|jsx|json|yaml|yml|toml|rst|go|rs|java|c|cpp|h|hpp|cs))\b"
)

# Directory-shaped token: one or more path segments ending in a slash
# (e.g. "app/retrieval/", "tests/"). Bare words like "retrieval" are
# NOT enough - too easy to false-positive on a natural-language
# description of what the code does.
_DIR_RE = re.compile(r"([A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-]+)*)/(?=\s|$)")

# Programming languages this repo's loader recognizes (see
# app.ingestion.loader._infer_language). Keys are the substring to
# match in the query; values are the canonical language string stored
# in metadata (so a "javascript" hint filters against language="js"
# etc. - matches what _infer_language actually writes).
_LANGUAGE_KEYWORDS = {
    "python file": "python",
    "python code": "python",
    ".py file": "python",
    "javascript file": "javascript",
    "typescript file": "typescript",
    "markdown file": "markdown",
    "yaml file": "yaml",
    "json file": "json",
    "rust file": "rust",
    "go file": "go",
    "java file": "java",
}

# snake_case or CamelCase identifier surrounded by word boundaries.
# Used for exact_code hint extraction so a query like "find
# classify_route" can narrow BM25 to chunks whose method_name or
# class_name equals that identifier. Kept restrictive: min 4 chars +
# must contain an underscore OR two capitals to weed out common English
# words ("code", "class") that would false-positive as identifiers.
_IDENTIFIER_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]{3,})\b")


def _looks_like_identifier(word: str) -> bool:
    """Heuristic: does this word look like a code identifier the user
    is naming (as opposed to an English word)? A snake_case name has
    an underscore; a CamelCase name has 2+ capitals. Everything else
    is treated as English and ignored."""
    if "_" in word:
        return True
    return sum(1 for c in word if c.isupper()) >= 2


# Common English words that pass _looks_like_identifier's structural
# check (2+ capitals) but aren't identifiers users are asking about.
# Deny-listed here so a query like "how does GitHub REST work" doesn't
# extract "GitHub" as a symbol filter.
_IDENTIFIER_DENYLIST = {
    "GitHub", "OpenAI", "PDF", "URL", "URI", "JSON", "YAML", "HTML",
    "HTTP", "HTTPS", "API", "SDK", "CLI", "UI", "OS",
}


def _extract_file_hint(query_text: str) -> str | None:
    """Pull the first plausible file path out of the query, if any.
    Used by the navigational branch to build a metadata filter."""
    match = _PATH_EXTENSION_RE.search(query_text)
    return match.group(1) if match else None


def _extract_dir_hint(query_text: str) -> str | None:
    """Pull a directory-shaped token out of the query (e.g.
    "app/retrieval/", "tests/"). Trailing slash is required to
    disambiguate from a plain English word. Returned without the
    trailing slash to match what app.ingestion.loader stores in
    file_dir metadata."""
    match = _DIR_RE.search(query_text)
    if not match:
        return None
    return match.group(1)


def _extract_language_hint(query_text: str) -> str | None:
    """Pull a programming-language filter out of the query, if any.
    Matches on multi-word phrases (e.g. "python file") rather than
    bare language names, because "python" alone often appears in
    descriptions of what the code does ("python's requests library")
    rather than as a filter the user wants applied."""
    query_lower = query_text.lower()
    for keyword, language in _LANGUAGE_KEYWORDS.items():
        if keyword in query_lower:
            return language
    return None


def _extract_symbol_hint(query_text: str) -> str | None:
    """Pull the first plausible code-identifier out of the query, if
    any. Used by the exact_code branch to add a class_name /
    method_name / function_name filter on top of the BM25 search,
    when the query names a specific symbol (as opposed to a general
    theme like "authentication logic")."""
    for match in _IDENTIFIER_RE.finditer(query_text):
        word = match.group(1)
        if word in _IDENTIFIER_DENYLIST:
            continue
        if _looks_like_identifier(word):
            return word
    return None


def _extract_metadata_hints(query_text: str) -> dict[str, str]:
    """Extract every filterable metadata hint from the query at once.

    Returns a dict keyed by the Chroma metadata field that the
    dispatcher (app.retrieval.github_adaptive) will $eq-filter on:
    "file_name", "file_dir", "language", and "symbol_name" (a virtual
    key the dispatcher expands into an $or across the three real
    identifier fields class_name/method_name/function_name).

    Only fields with an extracted value are included, so the result
    can be dropped directly into _build_where without any None checks
    at the call site. An empty dict means "no metadata narrowing
    signals in this query" - fall back to unfiltered retrieval.
    """
    hints: dict[str, str] = {}
    file_name = _extract_file_hint(query_text)
    if file_name is not None:
        hints["file_name"] = file_name.rsplit("/", 1)[-1]
    dir_hint = _extract_dir_hint(query_text)
    if dir_hint is not None:
        hints["file_dir"] = dir_hint
    language = _extract_language_hint(query_text)
    if language is not None:
        hints["language"] = language
    symbol = _extract_symbol_hint(query_text)
    if symbol is not None:
        hints["symbol_name"] = symbol
    return hints


# ---- Decision --------------------------------------------------------------


class GitHubIntent:
    """
    Result of classify_github_intent().

    Attributes:
        intent:         one of "explanation", "exact_code",
                        "navigational", "history", "activity", "visual",
                        "structural", or "general" (fallback - the
                        dispatcher runs the standard hybrid pipeline
                        unchanged).
        method:         how the intent was chosen - "rule", "semantic",
                        "rule+semantic", or "default". Informational.
        scores:         each intent's semantic similarity to the query,
                        for observability/debugging.
        metadata_hints: extracted per-field filters (file_name,
                        file_dir, language, symbol_name) that
                        app.retrieval.github_adaptive combines into a
                        Chroma $eq metadata filter. Empty dict when
                        the query names no concrete file, directory,
                        language, or identifier - in that case the
                        dispatcher retrieves without metadata
                        narrowing. Populated for EVERY intent
                        (including non-navigational) so exact_code can
                        also narrow by identifier and explanation can
                        stay within a specific file when named.
        file_hint:      backward-compat convenience surfacing
                        metadata_hints.get("file_name"); the badge UI
                        still reads this to render the small file
                        chip next to the intent label.
    """

    def __init__(
        self,
        intent: str,
        method: str,
        scores: dict[str, float],
        metadata_hints: dict[str, str] | None = None,
    ):
        self.intent = intent
        self.method = method
        self.scores = scores
        self.metadata_hints = metadata_hints if metadata_hints is not None else {}

    @property
    def file_hint(self) -> str | None:
        """Backward-compat: same value app_ui.py's badge reads."""
        return self.metadata_hints.get("file_name")

    def __repr__(self) -> str:
        return (
            f"GitHubIntent(intent={self.intent!r}, method={self.method!r}, "
            f"scores={self.scores!r}, metadata_hints={self.metadata_hints!r})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, GitHubIntent):
            return NotImplemented
        return (
            self.intent == other.intent
            and self.method == other.method
            and self.scores == other.scores
            and self.metadata_hints == other.metadata_hints
        )


# When two matched intents have nearly identical semantic scores
# (within _TIE_MARGIN), this order breaks the tie: explicit
# history/visual/navigational signals are strong and rare, so they
# beat the more common explanation/exact_code split when it's a
# genuine coin-flip. When the top score is clearly ahead of the
# runner-up (delta > _TIE_MARGIN), the higher score wins regardless -
# so a paraphrase like "how does the retrieval pipeline work" that
# happens to share tokens with a history exemplar still lands on
# explanation as long as explanation scores highest.
# structural sits above exact_code because "callers of foo" is a
# structural question, but its symbol_name extraction alone would
# also fire exact_code semantic exemplars ('find function foo'). The
# structural intent handles both the graph traversal and (via
# fallback in app.retrieval.github_adaptive) the exact_code path if
# no graph node matches, so preferring it doesn't lose exact_code
# functionality when the query genuinely warrants both.
_INTENT_PRIORITY = ["history", "activity", "visual", "navigational", "structural", "exact_code", "explanation"]
_TIE_MARGIN = 0.03

# Bonus added to an intent's semantic score when its rule/keyword list
# also fired. Rule matches are designed to be high-precision (see the
# _*_KEYWORDS lists' short/unambiguous entries), so a rule hit is worth
# roughly as much as a strong semantic match on its own - without this
# bonus, a semantic-only intent scoring just above the threshold
# (~0.45) could beat a rule-hit intent whose semantic score is a
# genuine but weak paraphrase match (~0.37), which is backwards.
_RULE_MATCH_BONUS = 0.15


def classify_github_intent(query_text: str) -> GitHubIntent:
    """
    Decide which GitHub-KB retrieval strategy best fits this query.

    Combines rule-based keywords and semantic exemplar similarity.
    Unlike app.routing.router.classify_route (which lets multiple
    routes fire because each only ADDS candidates to the merge), this
    picks a SINGLE winning intent because each intent dispatches to a
    DIFFERENT retrieval pipeline in app.retrieval.github_adaptive -
    blending them would double retrieval latency without meaningful
    recall gain.

    Tie-break policy: among the intents that fired (via rule or
    semantic threshold), the one with the highest effective score
    wins, where effective_score = semantic_score + _RULE_MATCH_BONUS
    if the intent's rule/keyword list also fired. When the top score
    is within _TIE_MARGIN of the runner-up, the intent with a rule
    match wins (higher-precision signal); if that's still ambiguous,
    _INTENT_PRIORITY breaks the tie deterministically.

    Metadata hints (file_name, file_dir, language, symbol_name) are
    extracted UNCONDITIONALLY of the winning intent, so:
      - A concrete file path in the query still forces navigational
        to win (via a synthetic rule-match on navigational).
      - Non-navigational intents can still narrow retrieval by any
        hints the user did name - e.g. exact_code narrows BM25 by
        symbol_name, explanation stays within a file when named.
    Hints that don't apply to the chosen intent are dropped by the
    dispatcher in app.retrieval.github_adaptive.
    """
    query_vector = embed_texts([query_text])[0]

    rule_intents = _rule_based_intents(query_text)
    semantic_scores = _semantic_intents(query_vector)
    semantic_intents = {
        intent for intent, score in semantic_scores.items()
        if score >= _SEMANTIC_INTENT_THRESHOLD
    }
    metadata_hints = _extract_metadata_hints(query_text)

    # A concrete file path or directory in the query is a strong
    # navigational signal on its own - stronger than any semantic
    # exemplar match for another intent - so treat it as a rule-based
    # navigational hit even when no navigational keyword fired.
    if "file_name" in metadata_hints or "file_dir" in metadata_hints:
        rule_intents = rule_intents | {"navigational"}

    # Rule-hit intents win over semantic-only intents when either
    # fires. Rationale: the rule keyword lists are short, hand-picked,
    # high-precision phrases ("what calls", "how does", "who added");
    # a semantic exemplar that just happens to share a symbol name
    # with the query (e.g. exact_code's "definition of retrieve_hybrid"
    # scoring 0.92 against "what calls retrieve_hybrid") is a much
    # weaker signal than a matched verb. If MULTIPLE rules fire, the
    # additive semantic score below still tie-breaks by which one
    # also has the stronger paraphrase match. If NO rules fire, we
    # fall back to semantic-only classification.
    matched = rule_intents if rule_intents else semantic_intents

    if not matched:
        chosen = "general"
        method = "default"
    else:
        # Additive bonus (not a floor) so a rule hit strengthens an
        # intent's score without capping it - a rule + strong semantic
        # correctly beats a rule + weak semantic.
        effective_scores = {
            intent: semantic_scores.get(intent, 0.0)
            + (_RULE_MATCH_BONUS if intent in rule_intents else 0.0)
            for intent in matched
        }
        top_intent = max(effective_scores, key=effective_scores.get)
        top_score = effective_scores[top_intent]
        near_top = {
            intent for intent, score in effective_scores.items()
            if top_score - score <= _TIE_MARGIN
        }
        if len(near_top) == 1:
            chosen = top_intent
        else:
            # Multi-way near-tie: prefer intents whose keyword list also
            # fired (rule + semantic beats semantic-only, because the
            # keyword hit is a much higher-precision signal than a
            # semantic near-match on unrelated exemplar tokens). If
            # every near-top intent has the same rule/no-rule status,
            # fall back to _INTENT_PRIORITY for a deterministic pick.
            rule_matched_in_tie = near_top & rule_intents
            if rule_matched_in_tie:
                near_top = rule_matched_in_tie
            chosen = next(intent for intent in _INTENT_PRIORITY if intent in near_top)
        if rule_intents and semantic_intents:
            method = "rule+semantic"
        elif rule_intents:
            method = "rule"
        else:
            method = "semantic"

    # Which hints are ACTUALLY applied at retrieval time depends on
    # the chosen intent - see app.retrieval.github_adaptive:
    #   navigational: file_name, file_dir, language
    #   exact_code:   language, symbol_name
    #   explanation:  file_name, file_dir, language (all optional)
    #   history:      none (git log has its own path filter)
    #   visual/general: none (uses standard hybrid pipeline)
    # We STORE the full extracted dict on the decision anyway so the
    # UI badge can show what was detected even for intents that don't
    # use it (a small transparency win for debugging).
    decision = GitHubIntent(
        intent=chosen, method=method, scores=semantic_scores,
        metadata_hints=metadata_hints,
    )
    logger.info(
        "GitHub intent for %r: intent=%s method=%s metadata_hints=%r scores=%s",
        query_text, decision.intent, decision.method,
        decision.metadata_hints, decision.scores,
    )
    return decision

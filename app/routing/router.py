"""
Query routing for the RAG pipeline.

Responsibility: given a user's natural-language question, decide WHICH
retriever(s) should be asked, before any actual retrieval happens. No
retrieval or generation happens here - this module only classifies a
query into one or more named "routes" that app.retrieval.retriever
(and cli.py, for the separate image retriever) use to decide what to
search.

Three complementary signals are combined ("hybrid" routing):

1. Rule-based / keyword routing (_rule_based_routes): fast, cheap,
   deterministic substring matching against a fixed keyword list per
   route. High precision when it fires (e.g. the word "screenshot" is
   an unambiguous signal to also check the image retriever), but low
   recall on its own - most real questions don't use these exact
   words, so it can't be the only mechanism.
2. Semantic routing (_semantic_routes): embeds the query with the SAME
   text embedding model used everywhere else in this pipeline
   (app.embeddings.embedder) and compares it, via cosine similarity,
   against a small set of hand-written example questions per route -
   the same mechanism retrieval itself uses (embed + compare), just
   against route *exemplars* instead of stored chunks. Catches
   paraphrases the keyword list would miss (e.g. "how does the login
   function work" has no literal "code" keyword but is semantically a
   code question).
3. Multi-retriever combination (classify_route): rather than forcing a
   single winning route, every route that EITHER mechanism is
   confident about is included - a single question can legitimately
   need more than one retriever at once (e.g. "show me the code and a
   picture of the login screen" needs both "code" and "image"). The
   "general" route (today's existing unscoped/unfiltered behavior) is
   always included alongside any specifically-matched route, so a
   wrong or low-confidence routing decision can only ADD extra,
   more-targeted candidates for the caller to merge in - it can never
   narrow away/exclude the baseline that already worked before routing
   existed.

Available routes:
- "code":    Python class/method/function chunks (see
             app.chunking.code_chunker) - questions about how something
             is implemented, a specific function/method, etc.
- "table":   Markdown/PDF table chunks (whole-table or per-row, see
             app.chunking.chunker's table handling) - questions asking
             about specific values, rows, or comparisons.
- "image":   Not a text content_type at all - a signal to also query
             the separate CLIP-based image retriever
             (app.retrieval.retriever.retrieve_images), which lives in
             a different vector space/collection entirely (see
             app.vectorstore.store's IMAGE_COLLECTION_NAME) and so
             can't be expressed as a text-collection metadata filter.
- "general": No content_type restriction at all - today's default,
             unscoped text search. Always included (see point 3 above).

A SECOND, INDEPENDENT routing dimension: which KB(s) to search.

app.vectorstore.store splits text chunks across separate
source-specific KB collections (KB_NAMES: "markdown", "pdf", "docx",
"web", "github") instead of one shared collection, specifically so a
query can search only the KB(s) it's actually about. classify_route()
applies the SAME two mechanisms (rule-based keywords + semantic
exemplar similarity) to decide `.kbs` - which KB(s) a query explicitly
signals (e.g. "in the PDF", "on the website", "in the codebase").
Unlike `.routes`, `.kbs` has NO always-included default here: an empty
list means "no explicit KB signal" and it's left to the caller
(app.retrieval.retriever) to decide the fallback - searching every KB
that actually has data (app.vectorstore.store.list_populated_kbs()) -
since only the caller knows which KBs are populated. This keeps this
module's classification pure (no store/disk access) and keeps the same
safety-net philosophy: an empty/wrong `.kbs` guess never loses recall,
it only fails to narrow the search.
"""

import logging
import math

from app.embeddings.embedder import embed_texts

logger = logging.getLogger(__name__)

# ---- Rule-based / keyword routing --------------------------------------

# Deliberately short, high-precision keyword lists: each entry is
# checked as a plain substring of the (lowercased, space-padded) query,
# so multi-word phrases like "return value" work too, not just single
# tokens. These are meant to catch the OBVIOUS, unambiguous cases
# cheaply - semantic routing below covers everything else.
_CODE_KEYWORDS = [
    "function", "method", " def ", "class ", "code", "implement",
    "algorithm", "variable", "parameter", "argument", "import ",
    "module", "exception", "return value", "refactor", "bug in",
]
_TABLE_KEYWORDS = [
    "table", " row ", "column", "spreadsheet", "which row",
]
_IMAGE_KEYWORDS = [
    "image", "screenshot", "picture", "diagram", "figure", "photo",
    "graphic", "chart", "visual",
]

_KEYWORDS_BY_ROUTE = {
    "code": _CODE_KEYWORDS,
    "table": _TABLE_KEYWORDS,
    "image": _IMAGE_KEYWORDS,
}


def _rule_based_routes(query_text: str) -> set[str]:
    """Routes whose keyword list has a literal substring match."""
    # Padded with spaces so keywords like " def " can match at the
    # very start/end of the query too, not just mid-string.
    query_lower = f" {query_text.lower()} "
    return {
        route
        for route, keywords in _KEYWORDS_BY_ROUTE.items()
        if any(keyword in query_lower for keyword in keywords)
    }


# ---- Semantic routing ---------------------------------------------------

# Small, hand-written example questions per route. These are embedded
# once (see _get_exemplar_embeddings) and compared against the actual
# query via cosine similarity - the highest-scoring exemplar in each
# route stands in for "how semantically close is this query to a
# typical <route> question".
_ROUTE_EXEMPLARS: dict[str, list[str]] = {
    "code": [
        "show me the function that handles user authentication",
        "how is this class implemented",
        "what does this method return",
        "find the code that parses the config file",
        "explain how this algorithm works",
        "where is this function used",
    ],
    "table": [
        "what values are shown in this table",
        "which row has the highest score",
        "compare these options in the comparison table",
        "what is listed in the column for status",
    ],
    "image": [
        "show me a screenshot of the login screen",
        "what does this diagram look like",
        "is there a picture of the dashboard",
        "display the architecture diagram",
    ],
}

# Cosine similarity is on roughly the same scale as elsewhere in this
# pipeline (e.g. cli.py's IMAGE_SIMILARITY_THRESHOLD = 0.2) but route
# exemplars are short, topic-focused sentences rather than long
# document chunks, so genuine matches score meaningfully higher than
# the loose threshold used for image relevance - 0.4 was chosen by
# checking that on-topic paraphrases (e.g. "how does the login
# function work") clear it while generic/unrelated questions don't.
_SEMANTIC_ROUTE_THRESHOLD = 0.4

_exemplar_embeddings_cache: dict[str, list[list[float]]] | None = None


def _get_exemplar_embeddings() -> dict[str, list[list[float]]]:
    """Lazily embed every route's exemplars once, then cache in memory."""
    global _exemplar_embeddings_cache
    if _exemplar_embeddings_cache is None:
        _exemplar_embeddings_cache = {
            route: embed_texts(exemplars) for route, exemplars in _ROUTE_EXEMPLARS.items()
        }
    return _exemplar_embeddings_cache


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _semantic_routes(query_vector: list[float]) -> dict[str, float]:
    """
    Each route's score is the highest cosine similarity between the
    query and any single exemplar in that route (max, not average) -
    a query only needs to closely match ONE example question to count
    as "about" that route, it doesn't need to be close to all of them.
    """
    exemplar_embeddings = _get_exemplar_embeddings()
    return {
        route: max(_cosine_similarity(query_vector, exemplar) for exemplar in exemplars)
        for route, exemplars in exemplar_embeddings.items()
    }


# ---- KB (source) routing -------------------------------------------------

# Which KB (see app.vectorstore.store.KB_NAMES) a query explicitly
# names. Distinct from the content-type keywords above: those ask
# "what KIND of chunk" (code/table/image), these ask "which SOURCE"
# (a PDF file vs. a website vs. a GitHub repo). A query can match both
# dimensions at once (e.g. "show me the function in the PDF").
_MARKDOWN_KB_KEYWORDS = ["markdown", ".md file", "readme file"]
_PDF_KB_KEYWORDS = ["pdf", ".pdf"]
_DOCX_KB_KEYWORDS = ["docx", "word doc", "word document", ".docx"]
_WEB_KB_KEYWORDS = ["website", "web page", "webpage", "docs site", "documentation site", "online docs", "the url"]
_GITHUB_KB_KEYWORDS = ["github", "repository", "repo ", " repo", "codebase", "source code", "the code"]

_KEYWORDS_BY_KB = {
    "markdown": _MARKDOWN_KB_KEYWORDS,
    "pdf": _PDF_KB_KEYWORDS,
    "docx": _DOCX_KB_KEYWORDS,
    "web": _WEB_KB_KEYWORDS,
    "github": _GITHUB_KB_KEYWORDS,
}


def _rule_based_kbs(query_text: str) -> set[str]:
    """KBs whose keyword list has a literal substring match."""
    query_lower = f" {query_text.lower()} "
    return {kb for kb, keywords in _KEYWORDS_BY_KB.items() if any(keyword in query_lower for keyword in keywords)}


# Small, hand-written example questions per KB, same idea as
# _ROUTE_EXEMPLARS above but for "which source is this query about"
# instead of "what kind of chunk".
_KB_EXEMPLARS: dict[str, list[str]] = {
    "markdown": [
        "what does the markdown file say",
        "summarize the readme file",
    ],
    "pdf": [
        "what does the pdf document say about this",
        "find this in the pdf report",
    ],
    "docx": [
        "what does the word document say",
        "find this in the docx file",
    ],
    "web": [
        "what does the website say about this",
        "find this on the documentation site",
    ],
    "github": [
        "what does the codebase do",
        "how is this implemented in the github repository",
    ],
}

# Same threshold as _SEMANTIC_ROUTE_THRESHOLD (see that constant's
# comment) - KB exemplars are similarly short, topic-focused sentences,
# so the same cutoff that works for content-type routing applies here
# too.
_SEMANTIC_KB_THRESHOLD = 0.4

_kb_exemplar_embeddings_cache: dict[str, list[list[float]]] | None = None


def _get_kb_exemplar_embeddings() -> dict[str, list[list[float]]]:
    """Lazily embed every KB's exemplars once, then cache in memory."""
    global _kb_exemplar_embeddings_cache
    if _kb_exemplar_embeddings_cache is None:
        _kb_exemplar_embeddings_cache = {kb: embed_texts(exemplars) for kb, exemplars in _KB_EXEMPLARS.items()}
    return _kb_exemplar_embeddings_cache


def _semantic_kbs(query_vector: list[float]) -> dict[str, float]:
    """Same idea as _semantic_routes(), scored against KB exemplars instead of route exemplars."""
    exemplar_embeddings = _get_kb_exemplar_embeddings()
    return {
        kb: max(_cosine_similarity(query_vector, exemplar) for exemplar in exemplars)
        for kb, exemplars in exemplar_embeddings.items()
    }


# ---- Hybrid combination / multi-retriever routing -----------------------


class RouteDecision:
    """
    The result of classify_route().

    Attributes:
        routes: every route the query should be sent to, always
            including "general" (see module docstring, point 3).
        method: which mechanism(s) produced this decision - "rule",
            "semantic", "rule+semantic", or "default" (no confident
            match from either mechanism, fell back to "general" only).
            Purely informational/for logging, callers don't need to
            branch on it.
        scores: each route's semantic similarity score (0.0 for routes
            that only matched by keyword), for observability/debugging.
        kbs: which KB(s) (see app.vectorstore.store.KB_NAMES) the query
            explicitly signaled, if any - unlike `routes`, this has NO
            always-included default (see module docstring): an empty
            list means no explicit KB signal was found, and it's up to
            the caller to decide what to search in that case (see
            app.retrieval.retriever, which falls back to every KB that
            actually has data). Defaults to an empty list so existing
            positional-arg call sites (RouteDecision(routes, method,
            scores)) that predate KB routing keep working unchanged.
    """

    def __init__(self, routes: list[str], method: str, scores: dict[str, float], kbs: list[str] | None = None):
        self.routes = routes
        self.method = method
        self.scores = scores
        self.kbs = kbs if kbs is not None else []

    def __repr__(self) -> str:
        return (
            f"RouteDecision(routes={self.routes!r}, method={self.method!r}, "
            f"scores={self.scores!r}, kbs={self.kbs!r})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, RouteDecision):
            return NotImplemented
        return (
            self.routes == other.routes
            and self.method == other.method
            and self.scores == other.scores
            and self.kbs == other.kbs
        )


def classify_route(query_text: str) -> RouteDecision:
    """
    Decide which retriever route(s) - and, independently, which KB(s) -
    a query should be sent to.

    Combines rule-based keyword matching and semantic exemplar
    similarity (see module docstring) for BOTH dimensions. Every route
    matched by EITHER mechanism is included (multi-retriever routing) -
    this is deliberately inclusive, not a single best-route classifier,
    since a wrong/low-confidence extra route can only add more
    candidates for the caller to merge and rerank, never remove good
    ones. KBs work the same way, except with no always-included
    default (see RouteDecision.kbs).
    """
    query_vector = embed_texts([query_text])[0]

    rule_routes = _rule_based_routes(query_text)
    semantic_scores = _semantic_routes(query_vector)
    semantic_routes = {route for route, score in semantic_scores.items() if score >= _SEMANTIC_ROUTE_THRESHOLD}
    matched_routes = rule_routes | semantic_routes

    rule_kbs = _rule_based_kbs(query_text)
    kb_scores = _semantic_kbs(query_vector)
    semantic_kbs = {kb for kb, score in kb_scores.items() if score >= _SEMANTIC_KB_THRESHOLD}
    kbs = sorted(rule_kbs | semantic_kbs)

    if not matched_routes:
        decision = RouteDecision(routes=["general"], method="default", scores=semantic_scores, kbs=kbs)
    else:
        routes = sorted(matched_routes | {"general"})
        if rule_routes and semantic_routes:
            method = "rule+semantic"
        elif rule_routes:
            method = "rule"
        else:
            method = "semantic"
        decision = RouteDecision(routes=routes, method=method, scores=semantic_scores, kbs=kbs)

    logger.info(
        "Routed query %r to %s (method=%s, scores=%s, kbs=%s)",
        query_text, decision.routes, decision.method, decision.scores, decision.kbs,
    )
    return decision

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
applies the same two-mechanism idea (rule-based keywords + semantic
similarity) to decide `.kbs` - which KB(s) a query explicitly signals
(e.g. "in the PDF", "on the website", "in the codebase"). Unlike
`.routes`, `.kbs` has NO always-included default here: an empty list
means "no explicit KB signal" and it's left to the caller
(app.retrieval.retriever) to decide the fallback - searching every KB
that actually has data (app.vectorstore.store.list_populated_kbs()) -
since only the caller knows which KBs are populated. An empty/wrong
`.kbs` guess never loses recall, it only fails to narrow the search.

KB semantic scoring is CORPUS-DERIVED, not hand-written exemplars (a
deliberate difference from the content-type routes above, which stay
hand-written - see _ROUTE_EXEMPLARS): each KB's score is the query's
best cosine similarity against a random SAMPLE of that KB's own
already-stored chunk embeddings (app.vectorstore.store.
sample_kb_embeddings), not against a fixed set of example questions
someone had to guess in advance. Hand-written exemplars like "what
does the website say about this" only work when a user's real
questions happen to resemble that generic meta-phrasing - for a KB
whose actual content is, say, Targetprocess automation-rule docs, a
real question ("close a User Story when all its Tasks are closed")
reads nothing like that exemplar even though it's exactly the kind of
content that KB holds, so it scored near zero and the router silently
fell back to searching every KB. Sampling the KB's own real content
sidesteps needing anyone to anticipate every deployment's domain
vocabulary - the corpus IS the exemplar set. This does mean this
module now touches the vector store (previously documented as
deliberately "pure"); the sample is small (default 40 chunks) and
cached per KB, invalidated by app.vectorstore.store whenever that KB's
chunks change (see invalidate_kb_routing_cache below), so the cost is
one-time per KB per ingest/delete, not per query.

A KB only makes it into `.kbs` if it clears BOTH an absolute floor
(_SEMANTIC_KB_THRESHOLD) AND a relative margin against whichever KB
scored highest (_SEMANTIC_KB_RELATIVE_MARGIN) - the floor alone let
through KBs that happened to cluster just above it even when a
different KB was the obvious, much stronger match (see
_SEMANTIC_KB_RELATIVE_MARGIN's own comment for the real example that
motivated this). `.kb_scores` always reports the raw, unfiltered score
for every KB regardless of which ones made it into `.kbs` - so the
observability data always shows the full picture even when the actual
search narrowed to one KB.
"""

import logging
import math

from app.embeddings.embedder import embed_texts
from app.vectorstore.store import KB_NAMES, sample_kb_embeddings

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
# Deliberately distinct from KB-routing's own _AUDIO_KB_KEYWORDS
# ("podcast", "recording", ...) - that dimension picks WHICH KB to
# search; this one picks WHICH RETRIEVER KIND (native acoustic
# similarity via CLAP, see app.retrieval.retriever.retrieve_audio_clips,
# vs. the ordinary text/transcript path). A query can fire both
# dimensions at once with no conflict.
_SOUND_KEYWORDS = [
    "sounds like", "similar sound", "sound similar", "similar-sounding",
    "audio clip", "background noise", "sound effect",
]

_KEYWORDS_BY_ROUTE = {
    "code": _CODE_KEYWORDS,
    "table": _TABLE_KEYWORDS,
    "image": _IMAGE_KEYWORDS,
    "sound": _SOUND_KEYWORDS,
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
    "sound": [
        "find audio that sounds like a car engine",
        "is there a recording with laughter in it",
        "play me something that sounds like rain",
        "find a clip with similar background music",
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
_AUDIO_KB_KEYWORDS = ["audio", "podcast", "recording", "the transcript"]
_VIDEO_KB_KEYWORDS = ["video", "the footage", "the clip", "screen recording"]

_KEYWORDS_BY_KB = {
    "markdown": _MARKDOWN_KB_KEYWORDS,
    "pdf": _PDF_KB_KEYWORDS,
    "docx": _DOCX_KB_KEYWORDS,
    "web": _WEB_KB_KEYWORDS,
    "github": _GITHUB_KB_KEYWORDS,
    "audio": _AUDIO_KB_KEYWORDS,
    "video": _VIDEO_KB_KEYWORDS,
}


def _rule_based_kbs(query_text: str) -> set[str]:
    """KBs whose keyword list has a literal substring match."""
    query_lower = f" {query_text.lower()} "
    return {kb for kb, keywords in _KEYWORDS_BY_KB.items() if any(keyword in query_lower for keyword in keywords)}


# How many of a KB's own stored chunk embeddings to sample as its
# semantic "exemplar set" (see module docstring). 40 mirrors this
# codebase's other retrieval pool sizes (e.g.
# app.retrieval.github_adaptive._INTENT_POOL_SIZE) - enough to capture
# real topical variety within a KB without making the sample fetch
# itself expensive.
_KB_SAMPLE_SIZE = 40

# Real stored chunks are longer and noisier than the short, clean,
# question-shaped sentences _ROUTE_EXEMPLARS uses, so genuine on-topic
# matches score lower on raw cosine similarity than they would against
# a hand-written exemplar - this threshold is deliberately looser than
# _SEMANTIC_ROUTE_THRESHOLD's 0.4 for exactly that reason (verified
# empirically: a real on-topic query scored well below 0.4 but clearly
# above generic/unrelated queries when compared against real chunk
# samples).
_SEMANTIC_KB_THRESHOLD = 0.2

# A flat absolute floor alone isn't enough: real usage showed a query
# ("what is automation rule") scoring web=0.46 (clearly the right KB)
# while markdown/docx/github all happened to cluster at 0.20-0.22 -
# comfortably clearing the same 0.2 floor despite being nowhere near as
# confident a match, so all four KBs got searched instead of just the
# one that actually mattered. This margin adds a SECOND, relative cut:
# once at least one KB clears the absolute floor, any other KB also
# needs to score within this fraction of the best score to stay in the
# running. A close race (e.g. 0.46 vs 0.40) still keeps both KBs - this
# only drops KBs that cleared the floor by coincidence, not genuine
# multi-KB contenders.
_SEMANTIC_KB_RELATIVE_MARGIN = 0.7

_kb_sample_embeddings_cache: dict[str, list[list[float]]] | None = None


def _get_kb_sample_embeddings() -> dict[str, list[list[float]]]:
    """Lazily sample + cache each KB's real stored chunk embeddings.
    See invalidate_kb_routing_cache() for how this stays fresh as
    content is ingested/deleted."""
    global _kb_sample_embeddings_cache
    if _kb_sample_embeddings_cache is None:
        _kb_sample_embeddings_cache = {
            kb: sample_kb_embeddings(kb, limit=_KB_SAMPLE_SIZE) for kb in KB_NAMES
        }
    return _kb_sample_embeddings_cache


def invalidate_kb_routing_cache(kb: str | None = None) -> None:
    """
    Drop the cached corpus-derived KB samples so the next routing
    decision re-samples fresh content. Called by
    app.vectorstore.store._invalidate_hybrid_cache whenever any KB's
    stored chunks change (ingest/delete) - same trigger points as the
    BM25 cache's own invalidation.

    Always clears every KB's sample (ignores `kb`, kept only so the
    call shape matches invalidate_bm25_cache's) rather than dropping
    just one - re-sampling all 5 KBs is cheap (a handful of small
    Chroma lookups), and this avoids a whole class of partial-
    staleness bugs a per-KB clear would need to get right.
    """
    global _kb_sample_embeddings_cache
    _kb_sample_embeddings_cache = None


def _semantic_kbs(query_vector: list[float]) -> dict[str, float]:
    """Score each KB by how close the query is to a random sample of
    that KB's OWN real stored content (see module docstring) - a KB
    with no chunks yet (or that sampled to empty) scores 0.0 for every
    query, same as having no semantic signal to offer."""
    samples = _get_kb_sample_embeddings()
    scores: dict[str, float] = {}
    for kb, embeddings in samples.items():
        if not embeddings:
            scores[kb] = 0.0
            continue
        scores[kb] = max(_cosine_similarity(query_vector, embedding) for embedding in embeddings)
    return scores


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
        kb_scores: each KB's semantic similarity score (0.0-1.0), for
            observability/debugging - the UI uses this to show "which
            KB is this query about, and how confidently". Defaults to
            an empty dict for the same backward-compat reason as `kbs`.
    """

    def __init__(
        self,
        routes: list[str],
        method: str,
        scores: dict[str, float],
        kbs: list[str] | None = None,
        kb_scores: dict[str, float] | None = None,
    ):
        self.routes = routes
        self.method = method
        self.scores = scores
        self.kbs = kbs if kbs is not None else []
        self.kb_scores = kb_scores if kb_scores is not None else {}

    def __repr__(self) -> str:
        return (
            f"RouteDecision(routes={self.routes!r}, method={self.method!r}, "
            f"scores={self.scores!r}, kbs={self.kbs!r}, kb_scores={self.kb_scores!r})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, RouteDecision):
            return NotImplemented
        return (
            self.routes == other.routes
            and self.method == other.method
            and self.scores == other.scores
            and self.kbs == other.kbs
            and self.kb_scores == other.kb_scores
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
    kbs_above_floor = {kb: score for kb, score in kb_scores.items() if score >= _SEMANTIC_KB_THRESHOLD}
    if kbs_above_floor:
        best_kb_score = max(kbs_above_floor.values())
        semantic_kbs = {
            kb for kb, score in kbs_above_floor.items() if score >= best_kb_score * _SEMANTIC_KB_RELATIVE_MARGIN
        }
    else:
        semantic_kbs = set()
    kbs = sorted(rule_kbs | semantic_kbs)

    if not matched_routes:
        decision = RouteDecision(
            routes=["general"], method="default", scores=semantic_scores, kbs=kbs, kb_scores=kb_scores,
        )
    else:
        routes = sorted(matched_routes | {"general"})
        if rule_routes and semantic_routes:
            method = "rule+semantic"
        elif rule_routes:
            method = "rule"
        else:
            method = "semantic"
        decision = RouteDecision(
            routes=routes, method=method, scores=semantic_scores, kbs=kbs, kb_scores=kb_scores,
        )

    logger.info(
        "Routed query %r to routes=%s kbs=%s (method=%s, route_scores=%s, kb_scores=%s)",
        query_text, decision.routes, decision.kbs, decision.method, decision.scores, decision.kb_scores,
    )
    return decision

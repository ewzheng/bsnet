"""Multi-backend search layer for evidence retrieval.

Sits between the claim extractor and the NLI scorer. Accepts a
search query string and returns raw text snippets aggregated from
multiple free search backends for use as evidence in the scoring
stage. Backends run in parallel; exceptions on any single backend
are swallowed so one flaky engine cannot starve the pipeline.
Snippets are additionally passed through a lightweight semantic
relevance filter (MiniLM sentence embeddings + cosine similarity)
to drop tangential hits before they reach the NLI scorer — smaller
NLI models in particular are easily fooled by query-term-overlap
snippets that full-text search surfaces but aren't actually about
the claim's topic.
"""

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ddgs import DDGS

from bsnet.src.model.embedder import embed
from bsnet.src.utils._common import (
    SENTENCE_SPLIT_RE,
    content_tokens,
    strip_markdown,
    truncate_to_word_boundary,
)
from bsnet.src.utils.outputs import EvidenceSnippet

# Backend whitelist for the per-backend parallel search.
#
# - google (via DDGS): primary source — breadth, recency, best
#   general ranking. Does most of the heavy lifting.
# - duckduckgo (via DDGS): Bing-backed redundancy. Kept so a
#   transient Google rate-limit does not zero out the whole search
#   stage. Overlap with Google is acceptable — dedup cleans it up.
# - wikipedia (via MediaWiki REST, *not* DDGS): DDGS's Wikipedia
#   adapter uses OpenSearch, which is title-prefix matching only —
#   it returns zero for every full-sentence claim. The MediaWiki
#   REST ``search/page`` endpoint does real full-text search and
#   returns page-level excerpts suitable for NLI. No API key, free.
#
# Brave is dropped: DDGS's Brave adapter returned "No results found"
# for every query we tested (likely anti-scrape blocking), adding
# only wasted thread-pool time. Bing and Yahoo are noisy; Mojeek's
# index is too small; Yandex and Grokipedia return low-quality
# results for fact-check queries.
DEFAULT_BACKENDS = "google,duckduckgo,wikipedia"

# Subtractive query-string operators appended to every DDGS query.
# Tells the backend's query parser to exclude pages that match these
# content terms — observed in the wild when satirical articles
# (defconnews-style "Joe Biden claims to be gay") rank highly enough
# to dominate a sparse evidence pool and lexically entail the claim.
# Subtractive only: any positive directive ("news", "factual",
# "according to") reshapes the ranker the way the dropped
# ``"fact check"`` prefix did, dragging results toward Snopes-style
# rebuttals instead of primary sources. Wikipedia REST does not
# honor Google query operators (the terms would be treated as
# additional keywords, hurting recall) so this string is appended
# only on the DDGS path.
_SUBTRACTIVE_QUERY_TERMS = "-satire -parody -humor -joke"

# Hostnames whose pages have no useful text body for the NLI scorer.
# Video platforms surface in DDGS text-search results because their
# pages have indexed metadata, but the snippet body is the video's
# description / channel byline — not factual prose. NLI on those
# strings is effectively pattern-matching the title against the claim
# (whoever uploaded "Trump is gay - YMCA reaction" gets entailment
# signal from the title alone). Drop them at the snippet aggregation
# stage so the relevance filter and downstream scorer never see them.
# Match is suffix-based so subdomains (``m.youtube.com``,
# ``www.youtube.com``) are caught by their registrable parent.
_BLOCKED_HOSTS = frozenset({
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "vimeo.com",
    "twitch.tv",
    "dailymotion.com",
})


def _is_blocked_host(url: str) -> bool:
    """Decide whether ``url`` belongs to a host on the no-body-text list.

    Parses the URL, normalizes the hostname (lowercase, strip a
    leading ``www.``), and matches by suffix against
    ``_BLOCKED_HOSTS`` so any subdomain (``m.youtube.com``,
    ``music.youtube.com``) is caught by its registrable parent.

    Args:
        url: The candidate snippet URL. May be empty (Wikipedia REST
            results sometimes have no canonical URL).

    Returns:
        ``True`` when the URL parses to a hostname that exactly
        matches or ends in ``.<blocked>`` for some entry of
        ``_BLOCKED_HOSTS``.

    Preconditions:
        - ``url`` is a string.

    Postconditions:
        - Returns ``False`` for empty / unparseable URLs (the
          aggregation layer should not drop snippets that lack a
          URL just because we can't classify them).
        - Match is case-insensitive on the hostname component.
        - Does not mutate the input.
    """
    if not url:
        return False
    try:
        host = urllib.parse.urlsplit(url).hostname or ""
    except ValueError:
        return False
    host = host.lower().lstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return False
    for blocked in _BLOCKED_HOSTS:
        if host == blocked or host.endswith("." + blocked):
            return True
    return False


# MediaWiki REST search endpoint. Full-text search returning page
# title, short description, and a contextual excerpt. See
# https://www.mediawiki.org/wiki/API:REST_API/Reference
_WIKIPEDIA_REST_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"

# Wikipedia's API policy requires a descriptive User-Agent that
# identifies the project. A default urllib UA gets a 403.
_WIKIPEDIA_UA = (
    "bsnet-factcheck/0.1 "
    "(https://github.com/ewzheng/bsnet; student fact-check project)"
)

# Strip the <span class="searchmatch"> tags MediaWiki wraps around
# highlighted excerpt terms — NLI doesn't need the markup.
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Per-snippet character cap applied after title+body concat. The NLI
# scorer pads every snippet in a batch up to the longest one, so a
# single 400-char outlier inflates the batched forward pass 3–4×.
# Capping preserves the title and the first sentence-or-two of the
# body — where entailment signal is densest — while keeping the
# padded-batch cost in check. 250 (up from 200) leaves enough room
# for snippets like Britannica's "In a vacuum, the speed of light is
# 299,792,458 metres per second" to keep the actual numerical value
# inside the cap; at 200 the number would be cut off and DeBERTa
# would lose the entailment signal.
_MAX_SNIPPET_CHARS = 250

# Stripped from the start of snippet bodies before composition. DDGS
# prepends date stamps to many results — relative ("5 days ago - ",
# "1 week ago · ") and absolute ("Oct 29, 2024 · ", "October 29,
# 2024 - ") — that eat 10–20 chars of the per-snippet budget for
# zero NLI value. Both `-` and `·` separators are observed.
_DATE_PREFIX_RE = re.compile(
    r"^(?:"
    r"\d+\s+(?:second|minute|hour|day|week|month|year)s?\s+ago"
    r"|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"\s+\d{1,2},?\s+\d{4}"
    r")\s*[\-·]\s*",
    re.IGNORECASE,
)

# Markdown stripping, word-boundary truncation, content-token
# extraction, sentence boundary splitting, and the MiniLM sentence
# embedder live in ``bsnet.src.utils._common`` — they're not
# search-specific and several other modules use them. See that
# module for the regex definitions and the embedder singleton.

# Fact-check editorial framing patterns at the start of a snippet
# title. When present, the title is the fact-checker's restatement
# of the claim being investigated rather than the conclusion — the
# body carries the actual answer, but DeBERTa often entails the
# question form because it lexically restates the claim. Dropping
# such titles forces NLI to read the body's verdict text instead.
# Patterns target explicit editorial markers, not generic
# question-shaped titles, to avoid over-eager dropping of legitimate
# titles that happen to phrase a topic.
#
# Two pattern families:
#
#   1. Generic verdict markers — leading colon-separated labels
#      used across most fact-check publications (``FACT CHECK:``,
#      ``REALITY CHECK:``, ``VERDICT:``, ``CLAIM:``, ``DEBUNKED:``,
#      ``HOAX ALERT:``, etc.).
#
#   2. Publication house-style framings — phrases that signal a
#      refutation article without an explicit colon-prefix marker.
#      Observed in eval traces and confirmed as principled
#      editorial intent (not generic phrasing):
#        - FactCheck.org: ``Misleading Posts Target X``
#        - Reuters / AP / others:
#            ``Posts Falsely Claim X`` /
#            ``Videos Wrongly Suggest X``
_FACT_CHECK_FRAMING_RE = re.compile(
    r"""
    ^
    (?:
        # ── Generic verdict markers ────────────────────────────
        FACT[\s\-]?CHECK(?:ING)?[:.]?\s+
      | REALITY[\s\-]?CHECK[:.]?\s+
      | VERDICT[:.]?\s+
      | CLAIM[:.]?\s+
      | VIRAL[:.]?\s+
      | DEBUNK(?:ED|ING)?[:.]?\s+
      | FALSE[:.]\s+
      | TRUE[:.]\s+
      | HOAX[\s\-]?ALERT[:.]?\s+
      | MISINFO(?:RMATION)?[:.]?\s+
      | RUMOR[\s\-]?CHECK[:.]?\s+

        # ── Publication house-style framings ──────────────────
        # "Misleading Posts Target X" (FactCheck.org)
      | MISLEADING\s+
        (?:POSTS?|CLAIMS?|VIDEOS?|PHOTOS?|IMAGES?|MEMES?|TWEETS?
            |HEADLINES?|ADS?|REPORTS?)
        \s+

        # "Posts Falsely Claim X" / "Videos Wrongly Suggest X"
      | (?:POSTS?|VIDEOS?|TWEETS?|PHOTOS?|IMAGES?|MEMES?|RUMORS?
            |REPORTS?|HEADLINES?|ARTICLES?)
        \s+
        (?:FALSELY|WRONGLY|MISLEADINGLY|INCORRECTLY)
        \s+
        (?:CLAIM(?:S|ED)?|SAY(?:S)?|SUGGEST(?:S|ED)?
            |STATE(?:S|D)?|REPORT(?:S|ED)?|SHOW(?:S|ED)?)
        \s+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Jaccard threshold over content tokens above which a snippet title
# is considered a near-restatement of the claim and dropped before
# concatenation with the body. DeBERTa-v3-base short-circuits when
# the premise begins with the literal hypothesis, scoring entailment
# ~1.0 even when the body refutes the claim — observed on the
# "The Earth is flat" calibration case where Wikipedia titles like
# "Flat Earth - Wikipedia" produced 0.78–0.99 support for a claim
# the body explicitly denies. Stripping the redundant title forces
# the scorer to read the actual evidence.
_TITLE_CLAIM_OVERLAP_DROP = 0.6

# Adaptive threshold: keep snippets whose cosine similarity to the
# claim is within ``_RELEVANCE_BAND`` of the top-scoring snippet.
# Using a band rather than an absolute threshold makes the filter
# robust across different claim types — a highly-covered claim
# where all snippets score ~0.7 keeps them all, while a sparse
# claim where only one snippet scores 0.5 and the rest 0.15 drops
# the tangential hits.
_RELEVANCE_BAND = 0.20

# Never drop below this fraction of input snippets even if the
# band filter would. Prevents catastrophic pruning on claims the
# embedder has low confidence on.
_RELEVANCE_MIN_KEEP_FRAC = 0.5


def _clean_body(body: str) -> str:
    """Strip search-result noise from a snippet body before composition.

    Decodes HTML entities (DDGS frequently surfaces ``&quot;`` /
    ``&#039;`` / ``&amp;`` verbatim, which both wastes characters of
    the per-snippet budget and confuses NLI/embedder tokenizers),
    removes DDGS-prepended date stamps, and drops markdown
    formatting. All three are pure noise from the scorer's
    perspective — every char dropped here is a char of real content
    that survives the cap downstream.

    Args:
        body: Raw snippet body returned by a search backend.

    Returns:
        The body with HTML entities decoded, leading date stamps
        removed, markdown stripped via ``strip_markdown``, and
        leading/trailing whitespace stripped.

    Preconditions:
        - ``body`` is a string.

    Postconditions:
        - Returned text contains no HTML named or numeric entity
          references (``&quot;``, ``&#039;``, etc.).
        - Returned text does not start with a recognized DDGS date
          prefix.
        - Returned text contains no markdown formatting characters.
        - Does not mutate the input.
    """
    cleaned = html.unescape(body)
    cleaned = _DATE_PREFIX_RE.sub("", cleaned, count=1)
    cleaned = strip_markdown(cleaned)
    return cleaned.strip()


def _smart_window(claim: str, text: str, max_chars: int) -> str:
    """Slide a ``max_chars`` window over ``text`` anchored on relevance.

    Splits ``text`` into sentences, embeds each sentence and the claim
    via the shared MiniLM, picks the highest-similarity sentence as
    the anchor, and greedily extends the window forward (then
    backward when forward is exhausted) by adjacent sentences while
    the running length stays within ``max_chars``. Forward-first
    matches the structure of fact-check articles where the
    conclusion follows the topic sentence rather than preceding it.
    Falls back to ``truncate_to_word_boundary`` when ``text`` has no
    detectable sentence structure (single-sentence body, URL-only
    body, etc.) so a degenerate input still gets bounded text out.

    The relevance anchor lets us keep the most claim-aligned content
    inside the cap when the search-engine snippet is longer than the
    cap — a common pattern where the search engine's curated body is
    300–400 chars and our default top-N truncation cut the
    conclusion. Hardware constraint: ``max_chars`` is held fixed at
    the lowest-common-denominator runtime budget so per-NLI-batch
    padding cost stays bounded; this routine moves the window, never
    grows it.

    Args:
        claim: The claim text used as the relevance query for the
            sentence ranker. Must be non-empty.
        text: Snippet body to window. Already cleaned of date
            prefixes / markdown by the caller.
        max_chars: Inclusive upper bound on the returned window's
            length. Must be a positive integer.

    Returns:
        ``text`` unchanged when it already fits within ``max_chars``;
        otherwise the joined run of sentences anchored on the most
        claim-relevant sentence and extended forward / backward to
        fill the budget. When ``text`` has only one sentence the
        return value is ``truncate_to_word_boundary(text, max_chars)``.

    Preconditions:
        - ``claim`` is a non-empty string.
        - ``text`` is a string.
        - ``max_chars`` is a positive integer.

    Postconditions:
        - ``len(result) <= max_chars``.
        - The returned text is a contiguous run of sentences from
          ``text`` (no reordering, no synthetic insertions).
        - Does not mutate the input.
    """
    text = text.strip()
    if not text:
        return text
    if len(text) <= max_chars:
        return text

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text)]
    sentences = [s for s in sentences if s]
    if len(sentences) <= 1:
        return truncate_to_word_boundary(text, max_chars)

    embeds = embed([claim] + sentences)
    claim_emb = embeds[0:1]
    sentence_embs = embeds[1:]
    sims = (sentence_embs @ claim_emb.T).squeeze(-1).tolist()

    anchor = max(range(len(sentences)), key=lambda i: sims[i])

    # Anchor sentence may itself exceed the budget — fall back to
    # word-boundary truncation on it directly rather than emitting a
    # window with negative remaining capacity.
    if len(sentences[anchor]) >= max_chars:
        return truncate_to_word_boundary(sentences[anchor], max_chars)

    left = right = anchor
    used = len(sentences[anchor])
    extended = True
    while used < max_chars and extended:
        extended = False
        if right + 1 < len(sentences):
            cost = 1 + len(sentences[right + 1])
            if used + cost <= max_chars:
                right += 1
                used += cost
                extended = True
                continue  # Forward-first: always try forward before backward.
        if left > 0:
            cost = 1 + len(sentences[left - 1])
            if used + cost <= max_chars:
                left -= 1
                used += cost
                extended = True

    return " ".join(sentences[left:right + 1])


def _title_is_fact_check_framing(title: str) -> bool:
    """Decide whether a title carries explicit fact-check framing.

    Fact-checkers commonly title an article with the claim phrased
    as a question (``FACT CHECK: Did X happen?``). The title is the
    *question being investigated*, not the answer — and DeBERTa
    routinely entails on the question form because the question
    lexically restates the claim. Dropping such titles before
    composition forces NLI to read the body's verdict text instead.

    The detector is anchored on explicit editorial markers
    (``FACT CHECK``, ``REALITY CHECK``, ``VERDICT``, ``CLAIM``,
    ``DEBUNKED``, ``FALSE:`` / ``TRUE:``) rather than generic
    question-shaped titles to avoid over-eager dropping of
    legitimate titles that phrase a topic as a question.

    Args:
        title: Candidate snippet title.

    Returns:
        ``True`` when the title starts with a fact-check editorial
        marker and should be dropped before composition.

    Preconditions:
        - ``title`` is a string.

    Postconditions:
        - Returns ``False`` for empty or whitespace-only titles.
        - Does not mutate the input.
    """
    if not title.strip():
        return False
    return bool(_FACT_CHECK_FRAMING_RE.match(title.strip()))


def _title_restates_claim(claim_tokens: set[str], title: str) -> bool:
    """Decide whether a snippet title is a near-restatement of the claim.

    Uses Jaccard similarity over content tokens. When a title's
    content tokens overlap with the claim's at or above
    ``_TITLE_CLAIM_OVERLAP_DROP``, the title carries no new evidence
    — it just restates the hypothesis with extra boilerplate (e.g.
    "- Wikipedia"). Stripping such titles before scoring prevents
    DeBERTa-base from short-circuiting on lexical overlap.

    Args:
        claim_tokens: Pre-computed content tokens of the claim,
            shared across all titles in a single search call.
        title: The candidate snippet title.

    Returns:
        ``True`` when the title should be dropped before scoring.

    Preconditions:
        - ``claim_tokens`` was produced by ``content_tokens``.
        - ``title`` is a string.

    Postconditions:
        - Returns ``False`` when ``title`` is empty after stripping.
        - Returns ``False`` when either token set is empty.
    """
    title = title.strip()
    if not title or not claim_tokens:
        return False
    title_tokens = content_tokens(title)
    if not title_tokens:
        return False
    intersection = claim_tokens & title_tokens
    union = claim_tokens | title_tokens
    return (len(intersection) / len(union)) >= _TITLE_CLAIM_OVERLAP_DROP


def _relevance_filter(
    query: str, snippets: list[EvidenceSnippet],
) -> list[EvidenceSnippet]:
    """Drop snippets that look tangential to the query.

    Uses MiniLM sentence embeddings to compute cosine similarity
    between the claim and each snippet's text body, keeping only
    those within ``_RELEVANCE_BAND`` of the top-scoring snippet. The
    filter is conservative: it never drops below
    ``_RELEVANCE_MIN_KEEP_FRAC`` of the input, and when ≤2 snippets
    are provided it passes them through unchanged (too little signal
    to filter reliably).

    Args:
        query: The claim text.
        snippets: Post-dedup snippets from the aggregator.

    Returns:
        A filtered list of snippets, in the same order as the input.
        Each kept snippet retains its source URL.

    Preconditions:
        - ``query`` is a non-empty string.
        - Each snippet has non-empty ``text``.

    Postconditions:
        - Returned list preserves input order.
        - Returned list is a subset of ``snippets``.
        - ``len(result) >= max(floor(len(snippets) * MIN_KEEP_FRAC), 1)``
          when ``len(snippets) > 2``; passes the input through
          unchanged otherwise.
    """
    if len(snippets) <= 2:
        return snippets
    texts = [s.text for s in snippets]
    embeds = embed([query] + texts)
    claim_emb = embeds[0:1]
    snippet_embs = embeds[1:]
    sims = (snippet_embs @ claim_emb.T).squeeze(-1).tolist()
    top = max(sims)
    cutoff = top - _RELEVANCE_BAND
    kept = [s for s, sim in zip(snippets, sims) if sim >= cutoff]
    min_keep = max(int(len(snippets) * _RELEVANCE_MIN_KEEP_FRAC), 1)
    if len(kept) < min_keep:
        ranked = sorted(zip(snippets, sims), key=lambda x: -x[1])
        kept_texts = {s.text for s, _ in ranked[:min_keep]}
        kept = [s for s in snippets if s.text in kept_texts]
    return kept


def get_search_snippets(
    query: str,
    num_results: int = 3,
    timeout: int = 5,
    backend: str = DEFAULT_BACKENDS,
) -> list[EvidenceSnippet]:
    """Retrieve evidence snippets across every configured backend.

    Spins up one thread per backend, each running an independent
    search, and aggregates whatever comes back. A backend that
    times out, rate-limits, or raises silently contributes zero
    snippets — the others still return their results, so transient
    per-engine failures degrade gracefully instead of starving the
    whole search. Each snippet is formed by concatenating the
    result's title and body so the NLI scorer downstream sees the
    headline signal alongside the body — titles on Wikipedia and
    news sources routinely name the entities in the claim, which
    nudges entailment probabilities up. Snippets are deduplicated
    by exact content across backends so identical hits don't
    double-count downstream.

    DDGS backends use the normal web scraper path; ``"wikipedia"``
    is special-cased to call the MediaWiki REST search endpoint
    directly because DDGS's Wikipedia adapter is title-prefix-only
    and returns empty for most full-sentence claims.

    On the DDGS path the query is suffixed with the subtractive
    content operators in ``_SUBTRACTIVE_QUERY_TERMS`` (``-satire``,
    ``-parody``, ``-humor``, ``-joke``) so the backend's query parser
    drops obvious satire/parody pages before ranking. No positive
    directive is added — a ``"fact check"`` prefix biased Google
    toward Snopes-style rebuttals rather than primary sources, and a
    date suffix hurt evergreen claims by dragging rankings toward
    "what's happening today" content. Wikipedia REST does not honor
    query operators, so the original ``query`` is sent verbatim there.

    Args:
        query: A search query string — typically a claim sentence
            from the extractor.
        num_results: Maximum snippets requested *per backend*. Total
            returned can be up to ``num_results × len(backends)``
            before deduplication. Defaults to 3 — with three
            backends this yields roughly 6–8 unique snippets after
            dedup, a balance between aggregation robustness (enough
            snippets that one noisy hit doesn't swing the label)
            and saturated-pipeline throughput (NLI forward-pass cost
            grows with padded batch size). Each snippet is also
            truncated to ``_MAX_SNIPPET_CHARS`` so a single long
            outlier can't inflate the padded-batch scoring cost.
        timeout: Per-request HTTP timeout in seconds for each backend
            call. Defaults to 5. Kept low so a stuck backend does not
            block its orchestrator stage slot for long.
        backend: Comma-delimited list of backends to query in
            parallel. Defaults to ``DEFAULT_BACKENDS``. Unknown
            names are routed through DDGS.

    Returns:
        A list of unique non-empty ``EvidenceSnippet`` objects
        aggregated across every backend that returned successfully.
        Each carries the composed text body and the source URL the
        snippet came from (empty string when the backend did not
        provide one). Empty list when no backend produced any
        results.

    Preconditions:
        - ``query`` is a non-empty string.
        - ``num_results`` and ``timeout`` are positive integers.
        - A network connection is available.

    Postconditions:
        - Returned snippet texts are stripped of leading/trailing
          whitespace.
        - Returned snippets are deduplicated by exact text content.
        - Returned snippets exclude any URL whose hostname matches
          ``_BLOCKED_HOSTS`` (video platforms whose page bodies are
          metadata, not factual prose).
        - Returns ``[]`` only when every backend failed or returned
          no usable bodies.
        - Does not mutate any external state.
        - Per-backend exceptions are swallowed.
    """
    backends = [b.strip() for b in backend.split(",") if b.strip()]
    if not backends:
        return []

    claim_tokens = content_tokens(query)

    def _compose(title: str, body: str) -> str:
        """Compose title + relevance-anchored body window into one snippet.

        Strips search-result noise (HTML entities, date prefixes,
        markdown markers) from both fields, drops the title when it
        near-restates the claim (see ``_TITLE_CLAIM_OVERLAP_DROP``),
        and reserves the title's char budget at the front of the cap.
        The body itself gets smart-windowed via ``_smart_window`` so
        the most claim-relevant sentences land inside the cap rather
        than always the first ``_MAX_SNIPPET_CHARS`` chars —
        search-engine bodies regularly run 300–400 chars and the
        conclusion of fact-check articles often sits past the cap
        when truncation is top-down.

        When the title alone would already exceed the cap (degenerate
        case), falls back to word-boundary truncation on the title.
        """
        title = html.unescape(title).strip()
        body = _clean_body(body)
        title_keep = (
            title
            and not _title_restates_claim(claim_tokens, title)
            and not _title_is_fact_check_framing(title)
        )
        if not title_keep:
            return _smart_window(query, body, _MAX_SNIPPET_CHARS)
        prefix = f"{title}. "
        body_budget = _MAX_SNIPPET_CHARS - len(prefix)
        if body_budget <= 0:
            return truncate_to_word_boundary(title, _MAX_SNIPPET_CHARS)
        windowed_body = _smart_window(query, body, body_budget)
        return prefix + windowed_body

    def _search_ddgs(single_backend: str) -> list[EvidenceSnippet]:
        """Run a DDGS search against one backend and extract title+body+url.

        Appends ``_SUBTRACTIVE_QUERY_TERMS`` so the backend's query
        parser drops pages tagged as satire / parody / humor / joke
        before ranking — observed live where defconnews-style satire
        outranked legitimate sources on sparse-evidence claims and
        lexically entailed the claim into a false positive.
        """
        try:
            results = DDGS(timeout=timeout).text(
                f"{query} {_SUBTRACTIVE_QUERY_TERMS}",
                max_results=num_results,
                backend=single_backend,
            )
        except Exception:
            return []
        snippets: list[EvidenceSnippet] = []
        for result in results[:num_results]:
            title = result.get("title", "").strip()
            body = result.get("body", "").strip()
            href = (result.get("href") or result.get("link") or "").strip()
            if not body:
                continue
            snippets.append(EvidenceSnippet(text=_compose(title, body), url=href))
        return snippets

    def _search_wikipedia() -> list[EvidenceSnippet]:
        """Query MediaWiki REST search/page and extract title+excerpt+url.

        Falls back to the page ``description`` when ``excerpt`` is
        missing so we still contribute a snippet for pages where the
        query-term highlight produced no excerpt. The page URL is
        synthesized from the page key — MediaWiki REST does not
        return canonical URLs in search responses.
        """
        params = urllib.parse.urlencode({"q": query, "limit": num_results})
        url = f"{_WIKIPEDIA_REST_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": _WIKIPEDIA_UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return []
        except Exception:
            return []
        snippets: list[EvidenceSnippet] = []
        for page in data.get("pages", [])[:num_results]:
            title = (page.get("title") or "").strip()
            excerpt = _HTML_TAG_RE.sub("", (page.get("excerpt") or "")).strip()
            description = (page.get("description") or "").strip()
            body = excerpt or description
            if not body:
                continue
            key = (page.get("key") or "").strip()
            page_url = f"https://en.wikipedia.org/wiki/{key}" if key else ""
            snippets.append(
                EvidenceSnippet(text=_compose(title, body), url=page_url),
            )
        return snippets

    def _search_one(b: str) -> list[EvidenceSnippet]:
        if b == "wikipedia":
            return _search_wikipedia()
        return _search_ddgs(b)

    snippets: list[EvidenceSnippet] = []
    seen: set[str] = set()
    with ThreadPoolExecutor(max_workers=len(backends)) as pool:
        for per_backend_snippets in pool.map(_search_one, backends):
            for snippet in per_backend_snippets:
                if _is_blocked_host(snippet.url):
                    continue
                if snippet.text not in seen:
                    seen.add(snippet.text)
                    snippets.append(snippet)

    return _relevance_filter(query, snippets)

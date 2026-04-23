"""Multi-backend search layer for evidence retrieval.

Sits between the claim extractor and the NLI scorer. Accepts a
search query string and returns raw text snippets aggregated from
multiple free search backends for use as evidence in the scoring
stage. Backends run in parallel; exceptions on any single backend
are swallowed so one flaky engine cannot starve the pipeline.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ddgs import DDGS

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
# Capping at 200 chars preserves the title and typically the first
# sentence of the body — where entailment signal is densest — while
# keeping padded-batch cost roughly flat across snippets.
_MAX_SNIPPET_CHARS = 200


def get_search_snippets(
    query: str,
    num_results: int = 3,
    timeout: int = 5,
    backend: str = DEFAULT_BACKENDS,
) -> list[str]:
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

    The query is passed through verbatim — no ``"fact check"``
    prefix (biased Google toward Snopes-style rebuttals rather than
    primary sources) and no date suffix (hurt evergreen claims by
    dragging rankings toward "what's happening today" content).

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
        A list of unique non-empty snippet strings aggregated across
        every backend that returned successfully. Empty list when no
        backend produced any results.

    Preconditions:
        - ``query`` is a non-empty string.
        - ``num_results`` and ``timeout`` are positive integers.
        - A network connection is available.

    Postconditions:
        - Returned snippets are stripped of leading/trailing whitespace.
        - Returned snippets are deduplicated by exact content.
        - Returns ``[]`` only when every backend failed or returned
          no usable bodies.
        - Does not mutate any external state.
        - Per-backend exceptions are swallowed.
    """
    backends = [b.strip() for b in backend.split(",") if b.strip()]
    if not backends:
        return []

    def _compose(title: str, body: str) -> str:
        """Concat title + body and truncate to the per-snippet char cap."""
        text = f"{title}. {body}" if title else body
        if len(text) > _MAX_SNIPPET_CHARS:
            text = text[:_MAX_SNIPPET_CHARS].rstrip()
        return text

    def _search_ddgs(single_backend: str) -> list[str]:
        """Run a DDGS search against one backend and extract title+body."""
        try:
            results = DDGS(timeout=timeout).text(
                query,
                max_results=num_results,
                backend=single_backend,
            )
        except Exception:
            return []
        snippets: list[str] = []
        for result in results[:num_results]:
            title = result.get("title", "").strip()
            body = result.get("body", "").strip()
            if not body:
                continue
            snippets.append(_compose(title, body))
        return snippets

    def _search_wikipedia() -> list[str]:
        """Query MediaWiki REST search/page and extract title+excerpt.

        Falls back to the page ``description`` when ``excerpt`` is
        missing so we still contribute a snippet for pages where the
        query-term highlight produced no excerpt.
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
        snippets: list[str] = []
        for page in data.get("pages", [])[:num_results]:
            title = (page.get("title") or "").strip()
            excerpt = _HTML_TAG_RE.sub("", (page.get("excerpt") or "")).strip()
            description = (page.get("description") or "").strip()
            body = excerpt or description
            if not body:
                continue
            snippets.append(_compose(title, body))
        return snippets

    def _search_one(b: str) -> list[str]:
        if b == "wikipedia":
            return _search_wikipedia()
        return _search_ddgs(b)

    snippets: list[str] = []
    seen: set[str] = set()
    with ThreadPoolExecutor(max_workers=len(backends)) as pool:
        for per_backend_snippets in pool.map(_search_one, backends):
            for body in per_backend_snippets:
                if body not in seen:
                    seen.add(body)
                    snippets.append(body)

    return snippets

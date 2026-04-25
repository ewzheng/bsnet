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

import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from ddgs import DDGS
from transformers import AutoModel, AutoTokenizer

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

# Sentence-embedding model used by the post-retrieval relevance
# filter. MiniLM-L6-v2 is ~22M params, ~80MB on disk, and hits
# sub-100ms batched inference for 10 short snippets on CPU — small
# enough to sit in the search stage without touching the pipeline
# bottleneck. Any drop-in replacement must produce L2-normalized
# 384-d embeddings of (query, snippets) so cosine comes out as dot
# product.
_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

# Lazy module-level singleton. Loading the MiniLM weights takes
# ~1s; we amortize it across every search call.
_embedder_lock = threading.Lock()
_embedder_tokenizer: AutoTokenizer | None = None
_embedder_model: AutoModel | None = None


def _get_embedder() -> tuple[AutoTokenizer, AutoModel]:
    """Return the shared embedder tokenizer + model, loading lazily.

    Postconditions:
        - Both objects are in eval mode after the first call.
        - Subsequent calls return the cached instances.
    """
    global _embedder_tokenizer, _embedder_model
    if _embedder_model is not None:
        return _embedder_tokenizer, _embedder_model
    with _embedder_lock:
        if _embedder_model is None:
            tok = AutoTokenizer.from_pretrained(_EMBEDDER_MODEL)
            mdl = AutoModel.from_pretrained(_EMBEDDER_MODEL)
            mdl.eval()
            _embedder_tokenizer = tok
            _embedder_model = mdl
    return _embedder_tokenizer, _embedder_model


def _embed(texts: list[str]) -> torch.Tensor:
    """Encode texts into L2-normalized mean-pooled sentence embeddings.

    Args:
        texts: Non-empty list of strings to encode.

    Returns:
        A ``(N, 384)`` float tensor where each row is the unit-norm
        embedding of the corresponding input.

    Preconditions:
        - ``texts`` is non-empty.

    Postconditions:
        - Each returned row has L2 norm ≈ 1.
        - Does not mutate the loaded model.
    """
    tokenizer, model = _get_embedder()
    batch = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**batch)
    token_embeds = out.last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (token_embeds * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


def _relevance_filter(query: str, snippets: list[str]) -> list[str]:
    """Drop snippets that look tangential to the query.

    Uses MiniLM sentence embeddings to compute cosine similarity
    between the claim and each snippet, keeping only those within
    ``_RELEVANCE_BAND`` of the top-scoring snippet. The filter is
    conservative: it never drops below ``_RELEVANCE_MIN_KEEP_FRAC``
    of the input, and when ≤2 snippets are provided it passes them
    through unchanged (too little signal to filter reliably).

    Args:
        query: The claim text.
        snippets: Post-dedup snippets from the aggregator.

    Returns:
        A filtered list of snippets, in the same order as the input.

    Preconditions:
        - ``query`` is a non-empty string.
        - Each snippet is a non-empty string.

    Postconditions:
        - Returned list preserves input order.
        - Returned list is a subset of ``snippets``.
        - ``len(result) >= max(floor(len(snippets) * MIN_KEEP_FRAC), 1)``
          when ``len(snippets) > 2``; passes the input through
          unchanged otherwise.
    """
    if len(snippets) <= 2:
        return snippets
    embeds = _embed([query] + snippets)
    claim_emb = embeds[0:1]
    snippet_embs = embeds[1:]
    sims = (snippet_embs @ claim_emb.T).squeeze(-1).tolist()
    top = max(sims)
    cutoff = top - _RELEVANCE_BAND
    kept = [s for s, sim in zip(snippets, sims) if sim >= cutoff]
    min_keep = max(int(len(snippets) * _RELEVANCE_MIN_KEEP_FRAC), 1)
    if len(kept) < min_keep:
        ranked = sorted(zip(snippets, sims), key=lambda x: -x[1])
        kept_set = {s for s, _ in ranked[:min_keep]}
        kept = [s for s in snippets if s in kept_set]
    return kept


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

    return _relevance_filter(query, snippets)

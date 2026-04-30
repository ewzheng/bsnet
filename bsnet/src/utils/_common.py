"""Shared text-processing utilities for the utils package.

Houses helpers that started life inside ``search.py`` but aren't
search-specific — markdown stripping, word-boundary truncation,
content tokenization, and the sentence-boundary regex. Search
keeps its snippet-domain logic (date-prefix stripping, fact-check
framing detection, Wikipedia REST plumbing, backend orchestration)
and consumes these primitives from here so other modules can share
the same implementations.

The MiniLM sentence embedder used to live here too but graduated
to its own module at ``bsnet.src.model.embedder`` once it picked up
its second consumer (smart-window snippet selection). Import from
there directly when you need ``embed`` / ``get_embedder``.
"""

import re

# ── Markdown stripping ──────────────────────────────────────────────────────
# Some pages (wikis, scraped docs) leak markdown delimiters into
# search-result bodies. None of those characters carry semantic
# content for downstream consumers (NLI scorer, sentence embedder)
# and they waste characters of any per-snippet character budget;
# small NLI models can also be confused by the markup. The regex
# order matters: image syntax (``![alt](url)``) must run before
# link syntax (``[text](url)``) since images are a superset of the
# link form.

_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MARKDOWN_HR_RE = re.compile(r"^[\-*_=]{3,}\s*$", re.MULTILINE)
_MARKDOWN_HEADING_RE = re.compile(r"^#+\s*", re.MULTILINE)
_MARKDOWN_BLOCKQUOTE_RE = re.compile(r"^>+\s*", re.MULTILINE)
_MARKDOWN_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_MARKDOWN_CODE_FENCE_RE = re.compile(r"`+")
_MARKDOWN_EMPHASIS_RE = re.compile(r"\*+")
_MARKDOWN_STRIKE_RE = re.compile(r"~+")


def strip_markdown(text: str) -> str:
    """Remove markdown formatting from ``text``.

    Walks the regexes in dependency order: image syntax before link
    syntax (images are a superset of link syntax); HTML comments;
    line-prefix markers (HR / heading / blockquote / list); then
    character delimiters (code fence / emphasis / strikethrough).
    The result is plain prose with all formatting characters
    removed and link / list / heading text preserved.

    Args:
        text: Possibly-markdown text.

    Returns:
        ``text`` with markdown delimiters removed and link / list /
        heading content unwrapped.

    Preconditions:
        - ``text`` is a string.

    Postconditions:
        - Returned text contains no markdown emphasis (``*``,
          ``~``), code fence (`` ` ``), heading-hash, blockquote-
          caret, or list-marker characters in their markdown
          contexts.
        - Image and HTML-comment content is removed entirely.
        - Link text is preserved without the URL.
        - Does not mutate the input.
    """
    text = _MARKDOWN_IMAGE_RE.sub("", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _MARKDOWN_HR_RE.sub("", text)
    text = _MARKDOWN_HEADING_RE.sub("", text)
    text = _MARKDOWN_BLOCKQUOTE_RE.sub("", text)
    text = _MARKDOWN_LIST_RE.sub("", text)
    text = _MARKDOWN_CODE_FENCE_RE.sub("", text)
    text = _MARKDOWN_EMPHASIS_RE.sub("", text)
    text = _MARKDOWN_STRIKE_RE.sub("", text)
    return text


# ── Word-boundary truncation ────────────────────────────────────────────────
# When truncating to a fixed character budget, back off to the most
# recent whitespace boundary to avoid cutting mid-word — but only
# when the backoff distance is small. A pathological string with no
# whitespace in the last ``TRUNCATE_BACKOFF_CHARS`` chars (rare,
# mostly URL-only bodies) falls back to the original hard cut so we
# don't lose half the content to backoff.
TRUNCATE_BACKOFF_CHARS = 30


def truncate_to_word_boundary(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars``, backing off to whitespace.

    A hard cut at ``max_chars`` often lands mid-word ("approx"
    instead of "approximately") — small NLI models will hallucinate
    on the fragment. Backing off to the most recent whitespace
    preserves the last full word. The backoff is capped at
    ``TRUNCATE_BACKOFF_CHARS`` so a pathological no-whitespace
    string doesn't lose half its content to backoff.

    Args:
        text: Text to truncate.
        max_chars: Inclusive upper bound on returned length.

    Returns:
        ``text`` if it already fits; otherwise the largest prefix
        not exceeding ``max_chars`` and ending at a whitespace
        boundary (or hard-cut at ``max_chars`` when no whitespace is
        within ``TRUNCATE_BACKOFF_CHARS`` of the cut).

    Preconditions:
        - ``max_chars`` is a positive integer.

    Postconditions:
        - ``len(result) <= max_chars``.
        - Returned text has no trailing whitespace.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if 0 < max_chars - last_space <= TRUNCATE_BACKOFF_CHARS:
        truncated = truncated[:last_space]
    return truncated.rstrip()


# ── Sentence + token tokenization ───────────────────────────────────────────
# Sentence boundary regex used by relevance-anchored windowing.
# Matches a sentence terminator (``.`` / ``!`` / ``?``) followed by
# whitespace.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Tokens shorter than this are treated as stopwords and ignored when
# computing content-token sets. Catches ``the``, ``is``, ``of``,
# ``to``, ``in``, ``on``, ``a``, ``an`` without maintaining an
# explicit list.
CONTENT_TOKEN_MIN_LEN = 3

# Strip everything but lowercase letters, digits, and whitespace
# before tokenizing. Numbers survive (so ``299792`` tokenizes
# cleanly after the comma is stripped from ``299,792``).
_TOKEN_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]")


def content_tokens(text: str) -> set[str]:
    """Return lowercase content-word tokens from ``text``.

    Strips non-alphanumeric characters, lowercases, splits on
    whitespace, and drops tokens shorter than
    ``CONTENT_TOKEN_MIN_LEN`` so common stopwords are filtered
    without maintaining an explicit list.

    Args:
        text: Arbitrary text.

    Returns:
        A set of normalized content tokens.

    Preconditions:
        - ``text`` is a string.

    Postconditions:
        - Every returned token is lowercase alphanumeric.
        - Every returned token has length
          ``>= CONTENT_TOKEN_MIN_LEN``.
    """
    cleaned = _TOKEN_NORMALIZE_RE.sub(" ", text.lower())
    return {t for t in cleaned.split() if len(t) >= CONTENT_TOKEN_MIN_LEN}



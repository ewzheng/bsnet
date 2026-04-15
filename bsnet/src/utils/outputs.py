"""Shared data contracts passed between pipeline stages.

Defines the structured output types that flow through the
extractor, scorer, and renderer stages of the fact-checking pipeline.
"""

from dataclasses import dataclass


@dataclass
class Claim:
    """A single factual claim extracted from a transcript sentence.

    Holds the normalized claim text and a set of search queries
    intended for the downstream search API.
    """

    text: str
    queries: list[str]


@dataclass
class EvidenceScore:
    """NLI scores for a single (claim, snippet) pair."""

    snippet: str
    support: float
    contradict: float
    neutral: float


@dataclass
class ScoredClaim:
    """A claim with NLI evidence scores for all retrieved snippets.

    Produced by the scorer after comparing a claim against every
    retrieved evidence snippet. Carries the original claim and the
    full list of scored evidence.
    """

    claim: str
    scores: list[EvidenceScore]

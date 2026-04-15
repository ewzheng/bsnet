"""Tests for the NLI evidence scorer.

Loads the cross-encoder NLI model and runs score() with varied claim-evidence
pairs to verify scoring accuracy across domains, edge cases, and invariants.
These are integration tests that require the model weights to be downloaded.
"""

import pytest

from bsnet.src.model.scorer import Scorer


@pytest.fixture(scope="module")
def scorer() -> Scorer:
    """Load the scorer model once for all tests in this module."""
    return Scorer(device="cpu")


# ── Basic smoke tests ────────────────────────────────────────────────────────


def test_supporting_evidence_scores_high_support(scorer: Scorer) -> None:
    """Matching evidence should have support as the dominant score."""
    result = scorer.score(
        claim="The unemployment rate dropped to 3.4% in January 2023.",
        snippets=[
            "The Bureau of Labor Statistics reported that unemployment "
            "fell to 3.4% in January 2023, the lowest level since 1969.",
        ],
    )
    print(f"\n--- supporting evidence ---")
    s = result.scores[0]
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    assert result is not None
    assert s.support > s.contradict
    assert s.support > s.neutral


def test_contradicting_evidence_scores_high_contradict(scorer: Scorer) -> None:
    """Contradicting evidence should have contradiction as the dominant score."""
    result = scorer.score(
        claim="The unemployment rate dropped to 3.4% in January 2023.",
        snippets=[
            "According to official data, the unemployment rate in "
            "January 2023 was 5.7%, significantly higher than expected.",
        ],
    )
    print(f"\n--- contradicting evidence ---")
    s = result.scores[0]
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    assert result is not None
    assert s.contradict > s.support


def test_empty_snippets_returns_none(scorer: Scorer) -> None:
    """No evidence should return None."""
    result = scorer.score(
        claim="The unemployment rate dropped to 3.4%.",
        snippets=[],
    )
    print(f"\n--- empty snippets ---")
    print(f"  result: {result}")

    assert result is None


def test_all_snippets_scored(scorer: Scorer) -> None:
    """Every input snippet should receive its own score entry."""
    result = scorer.score(
        claim="GDP grew 2.1% last quarter.",
        snippets=[
            "The weather was sunny in most of the country last week.",
            "The economy expanded by 2.1% in the most recent quarter, "
            "according to the Commerce Department.",
            "Several analysts predicted slower growth next year.",
        ],
    )
    print(f"\n--- all snippets scored ---")
    for s in result.scores:
        print(f"  [{s.snippet[:50]:50s}]  s={s.support:.3f} c={s.contradict:.3f} n={s.neutral:.3f}")

    assert result is not None
    assert len(result.scores) == 3


# ── Probability invariants ───────────────────────────────────────────────────


def test_probabilities_sum_to_one(scorer: Scorer) -> None:
    """Each score's three probabilities should sum to approximately 1.0."""
    result = scorer.score(
        claim="The Eiffel Tower is 330 meters tall.",
        snippets=[
            "The Eiffel Tower stands at 330 metres including its antenna.",
            "The Tower of London was built in 1066.",
            "The Eiffel Tower is only 250 meters tall according to some sources.",
        ],
    )
    print(f"\n--- probability sums ---")
    for s in result.scores:
        total = s.support + s.contradict + s.neutral
        print(f"  [{s.snippet[:40]:40s}]  sum={total:.4f}")
        assert abs(total - 1.0) < 0.01


def test_scores_are_non_negative(scorer: Scorer) -> None:
    """All probability values should be non-negative."""
    result = scorer.score(
        claim="Mars is the fourth planet from the Sun.",
        snippets=[
            "Mars orbits the Sun at a mean distance of 228 million km, "
            "making it the fourth planet in our solar system.",
        ],
    )
    s = result.scores[0]
    assert s.support >= 0.0
    assert s.contradict >= 0.0
    assert s.neutral >= 0.0


# ── Neutral / unrelated evidence ─────────────────────────────────────────────


def test_unrelated_evidence_does_not_contradict(scorer: Scorer) -> None:
    """Unrelated evidence should not trigger strong contradiction."""
    result = scorer.score(
        claim="The Great Wall of China is over 13,000 miles long.",
        snippets=[
            "Penguins are found primarily in the Southern Hemisphere.",
        ],
    )
    s = result.scores[0]
    print(f"\n--- unrelated evidence ---")
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    # NLI models may treat two independent true statements as entailment,
    # but unrelated evidence should not register as contradiction
    assert s.contradict < 0.5


def test_tangentially_related_evidence(scorer: Scorer) -> None:
    """Evidence on the same topic but not addressing the claim should lean neutral."""
    result = scorer.score(
        claim="Tesla delivered 1.8 million vehicles in 2023.",
        snippets=[
            "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning "
            "and is headquartered in Austin, Texas.",
        ],
    )
    s = result.scores[0]
    print(f"\n--- tangentially related ---")
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    # Should not score high contradiction -- it doesn't deny the claim
    assert s.contradict < 0.5


# ── Paraphrased evidence ────────────────────────────────────────────────────


def test_paraphrased_support_scores_high(scorer: Scorer) -> None:
    """Semantically equivalent but differently worded evidence should support."""
    result = scorer.score(
        claim="The population of Tokyo is approximately 14 million.",
        snippets=[
            "About 14 million people reside in the Japanese capital city "
            "of Tokyo, making it one of the most populous cities in the world.",
        ],
    )
    s = result.scores[0]
    print(f"\n--- paraphrased support ---")
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    assert s.support > s.contradict


# ── Near-miss numerical claims ───────────────────────────────────────────────


def test_near_miss_number_contradiction(scorer: Scorer) -> None:
    """Evidence with a close but different number should lean toward contradiction."""
    result = scorer.score(
        claim="The speed of sound is 343 meters per second.",
        snippets=[
            "The speed of sound in dry air at 20 degrees Celsius is "
            "approximately 300 meters per second.",
        ],
    )
    s = result.scores[0]
    print(f"\n--- near-miss number ---")
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    # The numbers disagree -- should not score high support
    assert s.support < s.contradict or s.neutral > s.support


def test_exact_number_match_supports(scorer: Scorer) -> None:
    """Evidence with the exact same number should strongly support."""
    result = scorer.score(
        claim="Mount Everest is 8,849 meters tall.",
        snippets=[
            "A joint Chinese-Nepalese survey in 2020 established the "
            "height of Mount Everest at 8,849 meters above sea level.",
        ],
    )
    s = result.scores[0]
    print(f"\n--- exact number match ---")
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    assert s.support > s.contradict
    assert s.support > 0.5


# ── Mixed-signal multi-snippet scoring ───────────────────────────────────────


def test_mixed_snippets_produce_varied_scores(scorer: Scorer) -> None:
    """A mix of supporting, contradicting, and neutral snippets should
    produce correspondingly varied score distributions."""
    result = scorer.score(
        claim="The Amazon rainforest produces 20% of the world's oxygen.",
        snippets=[
            # Supporting
            "The Amazon is often called the lungs of the planet, "
            "producing roughly 20% of the world's oxygen supply.",
            # Contradicting
            "Scientists note the Amazon consumes nearly as much oxygen "
            "as it produces, contributing closer to 6% of net oxygen.",
            # Neutral / unrelated
            "The Amazon River is the largest river by discharge volume.",
        ],
    )
    print(f"\n--- mixed snippets ---")
    for s in result.scores:
        print(f"  [{s.snippet[:50]:50s}]  s={s.support:.3f} c={s.contradict:.3f} n={s.neutral:.3f}")

    supporting = result.scores[0]
    contradicting = result.scores[1]

    # The supporting snippet should lean support
    assert supporting.support > supporting.contradict
    # The contradicting snippet should lean contradiction
    assert contradicting.contradict > contradicting.support


def test_five_snippets_all_scored(scorer: Scorer) -> None:
    """Scoring with five snippets should return exactly five score entries."""
    snippets = [
        "The Sun is a G-type main-sequence star.",
        "The Sun's surface temperature is about 5,500 degrees Celsius.",
        "Mars has two small moons called Phobos and Deimos.",
        "Solar energy accounts for about 4% of global electricity.",
        "The Sun is approximately 4.6 billion years old.",
    ]
    result = scorer.score(
        claim="The Sun's surface temperature is approximately 5,500 degrees Celsius.",
        snippets=snippets,
    )
    assert result is not None
    assert len(result.scores) == 5
    # The second snippet directly addresses the claim
    assert result.scores[1].support > result.scores[2].support


# ── Domain-specific scoring ──────────────────────────────────────────────────


def test_scientific_claim_scoring(scorer: Scorer) -> None:
    """Scientific fact-checking with precise terminology."""
    result = scorer.score(
        claim="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        snippets=[
            "At 1 atmosphere of pressure, pure water transitions from "
            "liquid to gas at 100 degrees Celsius (212 degrees Fahrenheit).",
        ],
    )
    s = result.scores[0]
    print(f"\n--- scientific scoring ---")
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    assert s.support > s.contradict


def test_historical_claim_scoring(scorer: Scorer) -> None:
    """Historical claims with specific dates."""
    result = scorer.score(
        claim="World War II ended in 1945.",
        snippets=[
            "The Second World War concluded in 1945 with the surrender "
            "of Japan on September 2, following the atomic bombings of "
            "Hiroshima and Nagasaki.",
        ],
    )
    s = result.scores[0]
    print(f"\n--- historical scoring ---")
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    assert s.support > s.contradict
    assert s.support > 0.5


def test_sports_claim_scoring(scorer: Scorer) -> None:
    """Sports statistics with specific numbers."""
    result = scorer.score(
        claim="Usain Bolt holds the 100m world record at 9.58 seconds.",
        snippets=[
            "Usain Bolt set the world record for the 100 meters at "
            "9.58 seconds during the 2009 World Championships in Berlin.",
        ],
    )
    s = result.scores[0]
    print(f"\n--- sports scoring ---")
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    assert s.support > s.contradict


def test_financial_claim_with_wrong_date(scorer: Scorer) -> None:
    """A claim with the right event but wrong date should lean toward contradiction."""
    result = scorer.score(
        claim="Lehman Brothers filed for bankruptcy in 2010.",
        snippets=[
            "Lehman Brothers Holdings Inc. filed for Chapter 11 "
            "bankruptcy protection on September 15, 2008.",
        ],
    )
    s = result.scores[0]
    print(f"\n--- wrong date ---")
    print(f"  support:    {s.support:.3f}")
    print(f"  contradict: {s.contradict:.3f}")
    print(f"  neutral:    {s.neutral:.3f}")

    assert s.contradict > s.support


# ── Claim text preserved ─────────────────────────────────────────────────────


def test_scored_claim_preserves_claim_text(scorer: Scorer) -> None:
    """The returned ScoredClaim should carry the original claim text."""
    claim_text = "The Nile is the longest river in the world."
    result = scorer.score(
        claim=claim_text,
        snippets=["The Nile stretches approximately 6,650 km."],
    )
    assert result is not None
    assert result.claim == claim_text


def test_scored_claim_preserves_snippet_text(scorer: Scorer) -> None:
    """Each EvidenceScore should carry the original snippet text."""
    snippet = "Paris is the capital of France."
    result = scorer.score(
        claim="The capital of France is Paris.",
        snippets=[snippet],
    )
    assert result.scores[0].snippet == snippet

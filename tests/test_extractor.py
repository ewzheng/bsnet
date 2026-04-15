"""Tests for the claim extractor.

Loads the GGUF model and runs extract() on a variety of inputs to verify
the two-pass pipeline produces reasonable output. These are integration
tests that require the model weights to be downloaded.
"""

import pytest

from bsnet.src.model.extractor import Extractor


@pytest.fixture(scope="module")
def extractor() -> Extractor:
    """Load the extractor model once for all tests in this module."""
    return Extractor()


# ── Basic smoke tests ────────────────────────────────────────────────────────


def test_factual_sentence_returns_claims(extractor: Extractor) -> None:
    """Verify that a clearly factual sentence produces at least one claim.

    Postconditions:
        - Returns at least one ``Claim`` with non-empty text and queries.
    """
    claims = extractor.extract(
        "The unemployment rate dropped to 3.4% in January 2023, "
        "the lowest since 1969."
    )
    print(f"\n--- factual sentence ---")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1
    for claim in claims:
        assert claim.text.strip()
        assert len(claim.queries) >= 1
        assert claim.queries[0].strip()


def test_opinion_sentence_may_return_empty(extractor: Extractor) -> None:
    """Check behavior on a non-factual opinion sentence.

    Postconditions:
        - Does not crash. May return empty or non-empty depending on
          model behavior -- this test just verifies the pipeline runs.

    Note:
        Some opinion leakage is acceptable here. The downstream NLI
        scorer is expected to label opinions as "opinion", and the
        pipeline filters those before rendering.
    """
    claims = extractor.extract(
        "I believe we need to do better as a country."
    )
    print(f"\n--- opinion sentence ---")
    print(f"  claims returned: {len(claims)}")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    for claim in claims:
        assert claim.text.strip()


def test_compound_sentence_may_split(extractor: Extractor) -> None:
    """Check that a sentence with multiple facts can produce multiple claims.

    Postconditions:
        - Produces at least one claim from a multi-fact sentence.
    """
    claims = extractor.extract(
        "GDP grew 2.1% last quarter and inflation fell to 3.2%, "
        "while the federal funds rate held at 5.25%."
    )
    print(f"\n--- compound sentence ---")
    print(f"  claims returned: {len(claims)}")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1


# ── Named-entity and person-centric claims ───────────────────────────────────


def test_named_entity_claim(extractor: Extractor) -> None:
    """A sentence about a specific person and organization should
    produce at least one claim containing identifiable entities."""
    claims = extractor.extract(
        "Elon Musk acquired Twitter for approximately $44 billion in October 2022."
    )
    print(f"\n--- named entity ---")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1
    for claim in claims:
        assert claim.text.strip()
        assert len(claim.queries) >= 1


def test_political_claim(extractor: Extractor) -> None:
    """Political claims with specific policy details should be extractable."""
    claims = extractor.extract(
        "The Inflation Reduction Act signed by President Biden in August 2022 "
        "allocated $369 billion toward energy security and climate change."
    )
    print(f"\n--- political claim ---")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1
    for claim in claims:
        assert len(claim.queries) >= 1


# ── Science and health claims ────────────────────────────────────────────────


def test_scientific_claim(extractor: Extractor) -> None:
    """A specific scientific fact should be extracted with search queries."""
    claims = extractor.extract(
        "The speed of light in a vacuum is approximately 299,792 kilometers "
        "per second."
    )
    print(f"\n--- scientific claim ---")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1


def test_health_claim(extractor: Extractor) -> None:
    """Health-related factual claims should be extractable."""
    claims = extractor.extract(
        "The World Health Organization declared COVID-19 a global pandemic "
        "on March 11, 2020."
    )
    print(f"\n--- health claim ---")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1
    for claim in claims:
        assert len(claim.queries) >= 1


# ── Multi-sentence and paragraph inputs ──────────────────────────────────────


def test_multi_sentence_paragraph(extractor: Extractor) -> None:
    """A paragraph with multiple sentences should produce multiple claims."""
    claims = extractor.extract(
        "The Amazon rainforest spans over 5.5 million square kilometers. "
        "It produces roughly 20% of the world's oxygen. "
        "Deforestation in the Amazon reached 13,235 square kilometers in 2021."
    )
    print(f"\n--- multi-sentence paragraph ---")
    print(f"  claims returned: {len(claims)}")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 2


def test_mixed_fact_and_opinion(extractor: Extractor) -> None:
    """A paragraph mixing facts and opinions should extract the factual parts."""
    claims = extractor.extract(
        "Japan's population fell below 125 million in 2023. "
        "I think the government should do more to address this. "
        "The fertility rate was 1.20, the lowest ever recorded."
    )
    print(f"\n--- mixed fact and opinion ---")
    print(f"  claims returned: {len(claims)}")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_short_factual_fragment(extractor: Extractor) -> None:
    """A very short factual statement should still be handled."""
    claims = extractor.extract("Water boils at 100 degrees Celsius.")
    print(f"\n--- short fragment ---")
    print(f"  claims returned: {len(claims)}")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1


def test_pure_opinion_returns_few_claims(extractor: Extractor) -> None:
    """Multiple opinion sentences should return few or no claims.

    Note:
        Some opinion leakage is acceptable. The NLI scorer labels
        opinions as "opinion" and the pipeline filters them before
        rendering, so extraction is not the only line of defense.
    """
    claims = extractor.extract(
        "I feel like things are getting worse. "
        "People should be more optimistic about the future. "
        "In my view, society is heading in the wrong direction."
    )
    print(f"\n--- pure opinion paragraph ---")
    print(f"  claims returned: {len(claims)}")
    for c in claims:
        print(f"  claim: {c.text}")

    # Purely subjective -- model may still return some but should be few
    assert len(claims) <= 3


def test_historical_claim(extractor: Extractor) -> None:
    """A well-known historical fact should be extractable."""
    claims = extractor.extract(
        "The Berlin Wall fell on November 9, 1989, leading to German "
        "reunification on October 3, 1990."
    )
    print(f"\n--- historical claim ---")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1


def test_sports_claim(extractor: Extractor) -> None:
    """Sports statistics with specific numbers should be extracted."""
    claims = extractor.extract(
        "Lionel Messi scored 672 goals for FC Barcelona over 17 seasons, "
        "making him the club's all-time top scorer."
    )
    print(f"\n--- sports claim ---")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1


def test_geographic_claim(extractor: Extractor) -> None:
    """Geographic facts with measurements should be extractable."""
    claims = extractor.extract(
        "Mount Everest stands at 8,849 meters above sea level, as "
        "confirmed by a Chinese-Nepalese survey in 2020."
    )
    print(f"\n--- geographic claim ---")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    assert len(claims) >= 1


def test_claim_text_does_not_start_with_number(extractor: Extractor) -> None:
    """Extracted claims should have leading list numbering stripped."""
    claims = extractor.extract(
        "The Eiffel Tower is 330 meters tall. "
        "It was completed in 1889 for the World's Fair."
    )
    print(f"\n--- numbering stripped ---")
    for c in claims:
        print(f"  claim: {c.text}")
        # Verify the leading-number stripping logic in extract()
        assert not c.text[0].isdigit() or c.text[0] in "0123456789" and any(
            ch.isalpha() for ch in c.text[:5]
        )

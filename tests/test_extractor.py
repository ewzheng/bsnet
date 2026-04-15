"""Smoke tests for the claim extractor.

Loads the GGUF model and runs extract() on a few sentences to verify
the two-pass pipeline produces reasonable output. These are integration
tests that require the model weights to be downloaded.
"""

from bsnet.src.model.extractor import Extractor


def test_factual_sentence_returns_claims() -> None:
    """Verify that a clearly factual sentence produces at least one claim.

    Preconditions:
        - The GGUF model is available (downloaded or cached).

    Postconditions:
        - Returns at least one ``Claim`` with non-empty text and queries.
    """
    extractor = Extractor()
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


def test_opinion_sentence_may_return_empty() -> None:
    """Check behavior on a non-factual opinion sentence.

    Preconditions:
        - The GGUF model is available (downloaded or cached).

    Postconditions:
        - Does not crash. May return empty or non-empty depending on
          model behavior — this test just verifies the pipeline runs.
    """
    extractor = Extractor()
    claims = extractor.extract(
        "I believe we need to do better as a country."
    )
    print(f"\n--- opinion sentence ---")
    print(f"  claims returned: {len(claims)}")
    for c in claims:
        print(f"  claim: {c.text}")
        print(f"  queries: {c.queries}")

    # No assertion on length — just verify it doesn't blow up
    for claim in claims:
        assert claim.text.strip()


def test_compound_sentence_may_split() -> None:
    """Check that a sentence with multiple facts can produce multiple claims.

    Preconditions:
        - The GGUF model is available (downloaded or cached).

    Postconditions:
        - Does not crash. Prints output for manual inspection of how
          T5 handles compound factual statements.
    """
    extractor = Extractor()
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

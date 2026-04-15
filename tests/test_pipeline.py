"""Integration tests for the fact-checking pipeline.

Loads all models and runs the full extract -> check -> render flow.
These tests require all model weights to be downloaded.
"""

from bsnet.src.runtime.pipeline import Pipeline


def test_factual_claim_produces_verdict() -> None:
    """Verify that a factual claim with supporting evidence produces a verdict.

    Preconditions:
        - All pipeline models are available (downloaded or cached).

    Postconditions:
        - Returns a ``Verdict`` with a non-empty explanation.
    """
    pipe = Pipeline()
    result = pipe.check(
        claim="The unemployment rate dropped to 3.4% in January 2023.",
        snippets=[
            "The Bureau of Labor Statistics reported that unemployment "
            "fell to 3.4% in January 2023, the lowest level since 1969.",
        ],
    )
    print(f"\n--- factual claim with support ---")
    print(f"  label: {result.label}")
    print(f"  evidence: {result.evidence[:60]}")

    assert result is not None
    assert result.label == "true"

    verdict = pipe.render(result)
    print(f"  explanation: {verdict.explanation}")
    assert verdict.explanation.strip()


def test_contradicted_claim_produces_false() -> None:
    """Verify that a contradicted claim is labeled false.

    Preconditions:
        - All pipeline models are available (downloaded or cached).

    Postconditions:
        - Returns a ``Verdict`` labeled ``"false"``.
    """
    pipe = Pipeline()
    result = pipe.check(
        claim="The unemployment rate dropped to 3.4% in January 2023.",
        snippets=[
            "According to official data, the unemployment rate in "
            "January 2023 was 5.7%, significantly higher than claimed.",
        ],
    )
    print(f"\n--- contradicted claim ---")
    print(f"  label: {result.label}")

    assert result is not None
    assert result.label == "false"

    verdict = pipe.render(result)
    print(f"  explanation: {verdict.explanation}")
    assert verdict.explanation.strip()


def test_opinion_returns_none() -> None:
    """Verify that an opinion claim is filtered out.

    Preconditions:
        - All pipeline models are available (downloaded or cached).

    Postconditions:
        - Returns ``None`` for opinion claims.
    """
    pipe = Pipeline()
    result = pipe.check(
        claim="I believe we need to do better as a country.",
        snippets=[
            "The US economy grew 2.4% in 2023 according to the "
            "Bureau of Economic Analysis.",
        ],
    )
    print(f"\n--- opinion claim ---")
    print(f"  result: {result}")

    assert result is None


def test_empty_snippets_returns_none() -> None:
    """Verify that no evidence returns None.

    Preconditions:
        - All pipeline models are available (downloaded or cached).

    Postconditions:
        - Returns ``None`` when no snippets are provided.
    """
    pipe = Pipeline()
    result = pipe.check(
        claim="GDP grew 2.1% last quarter.",
        snippets=[],
    )
    print(f"\n--- no evidence ---")
    print(f"  result: {result}")

    assert result is None


def test_extract_then_check_flow() -> None:
    """Verify the full extract -> check flow works end to end.

    Preconditions:
        - All pipeline models are available (downloaded or cached).

    Postconditions:
        - Extraction produces at least one claim.
        - Checking that claim with evidence produces a verdict.
    """
    pipe = Pipeline()

    claims = pipe.extract(
        "Global temperatures have risen by 1.2 degrees "
        "since pre-industrial times."
    )
    print(f"\n--- full flow ---")
    print(f"  claims: {len(claims)}")
    for c in claims:
        print(f"    {c.text} | queries: {c.queries}")

    assert len(claims) >= 1

    result = pipe.check(
        claim=claims[0].text,
        snippets=[
            "Global temperatures rose by 1.2 degrees Celsius above "
            "pre-industrial levels in 2023.",
        ],
    )
    assert result is not None
    print(f"  label: {result.label}")

    verdict = pipe.render(result)
    print(f"  explanation: {verdict.explanation}")
    assert verdict.explanation.strip()

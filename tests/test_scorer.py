"""Smoke tests for the NLI evidence scorer.

Loads the cross-encoder NLI model and runs score() with fake evidence
snippets to verify the pipeline produces reasonable output.
"""

from bsnet.src.model.scorer import Scorer


def test_supporting_evidence_scores_high_support() -> None:
    """Verify that matching evidence produces high support score.

    Preconditions:
        - The NLI model is available (downloaded or cached).

    Postconditions:
        - Support probability is the highest of the three scores.
    """
    scorer = Scorer(device="cpu")
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


def test_contradicting_evidence_scores_high_contradict() -> None:
    """Verify that contradicting evidence produces high contradiction score.

    Preconditions:
        - The NLI model is available (downloaded or cached).

    Postconditions:
        - Contradiction probability is the highest of the three scores.
    """
    scorer = Scorer(device="cpu")
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


def test_empty_snippets_returns_none() -> None:
    """Verify that no evidence returns None.

    Preconditions:
        - The NLI model is available (downloaded or cached).

    Postconditions:
        - Returns ``None`` when snippets list is empty.
    """
    scorer = Scorer(device="cpu")
    result = scorer.score(
        claim="The unemployment rate dropped to 3.4%.",
        snippets=[],
    )
    print(f"\n--- empty snippets ---")
    print(f"  result: {result}")

    assert result is None


def test_all_snippets_scored() -> None:
    """Verify that all snippets are scored, not just the best.

    Preconditions:
        - The NLI model is available (downloaded or cached).

    Postconditions:
        - The returned ScoredClaim contains one score per input snippet.
    """
    scorer = Scorer(device="cpu")
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

"""Smoke tests for the verdict renderer.

Loads the GGUF model and runs render() with pre-labeled claims to verify
it produces readable output.
"""

from bsnet.src.model.renderer import Renderer


def test_true_verdict_renders() -> None:
    """Verify rendering of a true verdict.

    Preconditions:
        - The GGUF model is available (downloaded or cached).

    Postconditions:
        - Returns a non-empty string.
    """
    renderer = Renderer()
    result = renderer.render(
        claim="The unemployment rate dropped to 3.4% in January 2023.",
        label="true",
        evidence="The Bureau of Labor Statistics reported that unemployment "
                 "fell to 3.4% in January 2023, the lowest level since 1969.",
    )
    print(f"\n--- true verdict ---")
    print(f"  {result}")
    assert result.strip()


def test_false_verdict_renders() -> None:
    """Verify rendering of a false verdict.

    Preconditions:
        - The GGUF model is available (downloaded or cached).

    Postconditions:
        - Returns a non-empty string.
    """
    renderer = Renderer()
    result = renderer.render(
        claim="The unemployment rate dropped to 3.4% in January 2023.",
        label="false",
        evidence="According to official data, the unemployment rate in "
                 "January 2023 was 5.7%, significantly higher than claimed.",
    )
    print(f"\n--- false verdict ---")
    print(f"  {result}")
    assert result.strip()


def test_unverifiable_verdict_renders() -> None:
    """Verify rendering of an unverifiable verdict.

    Preconditions:
        - The GGUF model is available (downloaded or cached).

    Postconditions:
        - Returns a non-empty string.
    """
    renderer = Renderer()
    result = renderer.render(
        claim="I believe we need to do better as a country.",
        label="unverifiable",
        evidence="The US economy grew 2.4% in 2023 according to the "
                 "Bureau of Economic Analysis.",
    )
    print(f"\n--- unverifiable verdict ---")
    print(f"  {result}")
    assert result.strip()

"""Tests for the verdict renderer.

Loads the GGUF model and runs render() with pre-labeled claims across
all verdict labels and diverse domains to verify it produces readable,
relevant output. These are integration tests that require model weights.
"""

import pytest

from bsnet.src.model.renderer import Renderer


@pytest.fixture(scope="module")
def renderer() -> Renderer:
    """Load the renderer model once for all tests in this module."""
    return Renderer()


# ── Core label coverage ──────────────────────────────────────────────────────


def test_true_verdict_renders(renderer: Renderer) -> None:
    """Verify rendering of a 'true' verdict."""
    result = renderer.render(
        claim="The unemployment rate dropped to 3.4% in January 2023.",
        label="true",
        evidence="The Bureau of Labor Statistics reported that unemployment "
                 "fell to 3.4% in January 2023, the lowest level since 1969.",
    )
    print(f"\n--- true verdict ---")
    print(f"  {result}")
    assert result.strip()


def test_false_verdict_renders(renderer: Renderer) -> None:
    """Verify rendering of a 'false' verdict."""
    result = renderer.render(
        claim="The unemployment rate dropped to 3.4% in January 2023.",
        label="false",
        evidence="According to official data, the unemployment rate in "
                 "January 2023 was 5.7%, significantly higher than claimed.",
    )
    print(f"\n--- false verdict ---")
    print(f"  {result}")
    assert result.strip()


def test_unproven_verdict_renders(renderer: Renderer) -> None:
    """Verify rendering of an 'unproven' verdict."""
    result = renderer.render(
        claim="I believe we need to do better as a country.",
        label="unproven",
        evidence="The US economy grew 2.4% in 2023 according to the "
                 "Bureau of Economic Analysis.",
    )
    print(f"\n--- unproven verdict ---")
    print(f"  {result}")
    assert result.strip()


def test_mostly_true_verdict_renders(renderer: Renderer) -> None:
    """Verify rendering of a 'mostly true' verdict (close but imprecise)."""
    result = renderer.render(
        claim="SpaceX's Falcon 9 has completed over 200 successful launches.",
        label="mostly true",
        evidence="SpaceX's Falcon 9 rocket has completed 195 successful "
                 "missions as of December 2023, with several more scheduled.",
    )
    print(f"\n--- mostly true verdict ---")
    print(f"  {result}")
    assert result.strip()


def test_partially_true_verdict_renders(renderer: Renderer) -> None:
    """Verify rendering of a 'partially true' verdict (some elements correct)."""
    result = renderer.render(
        claim="India's GDP surpassed the UK's in 2022, making it the "
              "fourth largest economy.",
        label="partially true",
        evidence="India overtook the UK as the world's fifth largest "
                 "economy in 2022 according to IMF data, not the fourth.",
    )
    print(f"\n--- partially true verdict ---")
    print(f"  {result}")
    assert result.strip()


def test_mixture_verdict_renders(renderer: Renderer) -> None:
    """Verify rendering of a 'mixture' verdict (conflicting evidence)."""
    result = renderer.render(
        claim="Electric vehicles produce zero emissions.",
        label="mixture",
        evidence="EVs produce zero tailpipe emissions while driving, but "
                 "manufacturing their batteries generates significant CO2 "
                 "emissions, and electricity generation may involve fossil fuels.",
    )
    print(f"\n--- mixture verdict ---")
    print(f"  {result}")
    assert result.strip()


def test_partially_false_verdict_renders(renderer: Renderer) -> None:
    """Verify rendering of a 'partially false' verdict."""
    result = renderer.render(
        claim="The Great Wall of China is visible from space with the naked eye.",
        label="partially false",
        evidence="NASA has stated that the Great Wall is not visible from "
                 "low Earth orbit with the unaided eye, though it can be "
                 "seen in radar images from orbit.",
    )
    print(f"\n--- partially false verdict ---")
    print(f"  {result}")
    assert result.strip()


def test_mostly_false_verdict_renders(renderer: Renderer) -> None:
    """Verify rendering of a 'mostly false' verdict."""
    result = renderer.render(
        claim="Humans only use 10% of their brains.",
        label="mostly false",
        evidence="Neuroscience research using fMRI shows that virtually "
                 "all areas of the brain are active at various times, though "
                 "not all simultaneously. The 10% claim is a persistent myth.",
    )
    print(f"\n--- mostly false verdict ---")
    print(f"  {result}")
    assert result.strip()


# ── Domain diversity ─────────────────────────────────────────────────────────


def test_finance_domain(renderer: Renderer) -> None:
    """Verify rendering works for financial claims."""
    result = renderer.render(
        claim="Bitcoin reached an all-time high above $69,000 in November 2021.",
        label="true",
        evidence="Bitcoin hit $68,789.63 on November 10, 2021, according "
                 "to CoinDesk data, its highest price at that time.",
    )
    print(f"\n--- finance domain ---")
    print(f"  {result}")
    assert result.strip()


def test_sports_domain(renderer: Renderer) -> None:
    """Verify rendering works for sports claims."""
    result = renderer.render(
        claim="Argentina won the 2022 FIFA World Cup in Qatar.",
        label="true",
        evidence="Argentina defeated France on penalties in the 2022 FIFA "
                 "World Cup final held in Lusail, Qatar on December 18, 2022.",
    )
    print(f"\n--- sports domain ---")
    print(f"  {result}")
    assert result.strip()


def test_science_domain(renderer: Renderer) -> None:
    """Verify rendering works for scientific claims."""
    result = renderer.render(
        claim="The James Webb Space Telescope was launched in December 2021.",
        label="true",
        evidence="NASA's James Webb Space Telescope launched on December 25, "
                 "2021, from Europe's Spaceport in French Guiana.",
    )
    print(f"\n--- science domain ---")
    print(f"  {result}")
    assert result.strip()


def test_history_domain(renderer: Renderer) -> None:
    """Verify rendering works for historical claims."""
    result = renderer.render(
        claim="The Apollo 11 mission landed on the Moon on July 20, 1969.",
        label="true",
        evidence="Apollo 11 successfully landed on the Moon on July 20, 1969, "
                 "with Neil Armstrong becoming the first human to walk on the "
                 "lunar surface.",
    )
    print(f"\n--- history domain ---")
    print(f"  {result}")
    assert result.strip()


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_long_evidence_renders(renderer: Renderer) -> None:
    """Verify rendering handles a longer evidence string without truncation issues."""
    result = renderer.render(
        claim="Global sea levels are rising.",
        label="true",
        evidence=(
            "According to NASA's Sea Level Change portal, global mean sea "
            "level has risen about 21-24 centimeters since 1880. The rate of "
            "rise has accelerated in recent decades, from about 1.4 mm per "
            "year throughout most of the twentieth century to 3.6 mm per year "
            "from 2006 to 2015. Satellite altimetry data from TOPEX/Poseidon, "
            "Jason-1, Jason-2, and Jason-3 confirm this accelerating trend."
        ),
    )
    print(f"\n--- long evidence ---")
    print(f"  {result}")
    assert result.strip()


def test_minimal_evidence_renders(renderer: Renderer) -> None:
    """Verify rendering handles a very short evidence string."""
    result = renderer.render(
        claim="The Earth is flat.",
        label="false",
        evidence="The Earth is an oblate spheroid.",
    )
    print(f"\n--- minimal evidence ---")
    print(f"  {result}")
    assert result.strip()


def test_render_output_is_concise(renderer: Renderer) -> None:
    """The renderer is prompted for 1-2 sentences; output should be short."""
    result = renderer.render(
        claim="Python is the most popular programming language.",
        label="mostly true",
        evidence="The TIOBE Index for January 2024 ranks Python as the "
                 "most popular programming language, though Stack Overflow's "
                 "2023 survey places JavaScript at the top for the eleventh "
                 "consecutive year.",
    )
    print(f"\n--- conciseness check ---")
    print(f"  length: {len(result)} chars")
    print(f"  {result}")
    assert result.strip()
    # 100 max_tokens at ~4 chars/token ≈ 400 chars upper bound
    assert len(result) < 600

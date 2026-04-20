"""Integration tests for the fact-checking pipeline.

Loads all models once and runs the full extract -> check -> render flow.
Includes per-stage latency measurements for benchmarking inference speed.
These tests require all model weights to be downloaded.
"""

import time

import pytest

from bsnet.src.runtime.orchestrator import Orchestrator
from bsnet.src.runtime.pipeline import Pipeline
from bsnet.src.utils.outputs import CheckResult


@pytest.fixture(scope="module")
def pipe() -> Pipeline:
    """Load all pipeline models once for the module."""
    return Pipeline()


# ── Functional tests ─────────────────────────────────────────────────────────


def test_factual_claim_produces_verdict(pipe: Pipeline) -> None:
    """Supporting evidence should produce a 'true' verdict with explanation."""
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


def test_contradicted_claim_produces_false(pipe: Pipeline) -> None:
    """Contradicting evidence should produce a 'false' verdict."""
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


def test_opinion_returns_none(pipe: Pipeline) -> None:
    """Opinion claims should be filtered out by the scorer/labeler."""
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


def test_empty_snippets_returns_none(pipe: Pipeline) -> None:
    """No evidence should return None."""
    result = pipe.check(
        claim="GDP grew 2.1% last quarter.",
        snippets=[],
    )
    print(f"\n--- no evidence ---")
    print(f"  result: {result}")

    assert result is None


def test_extract_then_check_flow(pipe: Pipeline) -> None:
    """Full extract -> check -> render flow end to end."""
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


# ── Orchestrator end-to-end with the real pipeline ───────────────────────────


def test_orchestrator_end_to_end_with_real_pipeline(pipe: Pipeline) -> None:
    """Drive the real pipeline through the orchestrator with dummy inputs.

    Uses a plain string iterator in place of the transcription stream
    and canned search snippets in place of the search backend. Validates
    that all stages wire up correctly and at least one verdict lands
    with the expected content.

    Preconditions:
        - The ``pipe`` fixture is loaded.

    Postconditions:
        - At least one verdict is produced with non-empty fields.
    """

    def canned_search(queries: list[str]) -> list[str]:
        """Return a single corroborating snippet regardless of queries."""
        del queries
        return [
            "The Bureau of Labor Statistics reported that unemployment "
            "fell to 3.4% in January 2023, the lowest level since 1969.",
        ]

    def pass_validate(result: CheckResult) -> bool:
        """Accept every result through the validate stage."""
        del result
        return True

    orch = Orchestrator(
        pipeline=pipe,
        search_fn=canned_search,
        validate_fn=pass_validate,
    )

    chunks = ["The unemployment rate dropped to 3.4% in January 2023."]

    t0 = time.perf_counter()
    verdicts = list(orch.run(iter(chunks)))
    elapsed = time.perf_counter() - t0

    print(f"\n--- orchestrator end-to-end (real pipeline) ---")
    print(f"  time:     {elapsed:.2f}s")
    print(f"  verdicts: {len(verdicts)}")
    for v in verdicts:
        print(f"    [{v.label}] {v.claim}")
        print(f"      evidence:    {v.evidence[:80]}")
        print(f"      explanation: {v.explanation}")

    assert len(verdicts) >= 1
    for v in verdicts:
        assert v.claim.strip()
        assert v.label.strip()
        assert v.evidence.strip()
        assert v.explanation.strip()


# ── Latency benchmarks ───────────────────────────────────────────────────────


def test_extraction_latency(pipe: Pipeline) -> None:
    """Measure extraction latency for a single factual sentence."""
    text = (
        "The James Webb Space Telescope was launched on "
        "December 25, 2021, from French Guiana."
    )
    t0 = time.perf_counter()
    claims = pipe.extract(text)
    elapsed = time.perf_counter() - t0

    print(f"\n--- extraction latency ---")
    print(f"  time:   {elapsed:.2f}s")
    print(f"  claims: {len(claims)}")
    for c in claims:
        print(f"    {c.text}")

    assert len(claims) >= 1


def test_scoring_latency(pipe: Pipeline) -> None:
    """Measure scoring latency for a claim against 3 snippets."""
    claim = "The speed of light is approximately 299,792 km/s."
    snippets = [
        "Light travels at about 299,792 kilometers per second in a vacuum.",
        "The speed of sound in air is roughly 343 meters per second.",
        "Einstein's theory of special relativity was published in 1905.",
    ]
    t0 = time.perf_counter()
    result = pipe.check(claim, snippets)
    elapsed = time.perf_counter() - t0

    print(f"\n--- scoring latency (3 snippets) ---")
    print(f"  time:  {elapsed:.2f}s")
    print(f"  label: {result.label}")

    assert result is not None


def test_rendering_latency(pipe: Pipeline) -> None:
    """Measure rendering latency for a single verdict."""
    result = pipe.check(
        claim="Argentina won the 2022 FIFA World Cup.",
        snippets=[
            "Argentina defeated France on penalties in the 2022 FIFA "
            "World Cup final held in Lusail, Qatar on December 18, 2022.",
        ],
    )
    assert result is not None

    t0 = time.perf_counter()
    verdict = pipe.render(result)
    elapsed = time.perf_counter() - t0

    print(f"\n--- rendering latency ---")
    print(f"  time:        {elapsed:.2f}s")
    print(f"  explanation: {verdict.explanation}")

    assert verdict.explanation.strip()


def test_end_to_end_latency(pipe: Pipeline) -> None:
    """Measure full pipeline latency: extract -> check -> render."""
    text = "Mount Everest is 8,849 meters above sea level."
    snippets = [
        "A joint Chinese-Nepalese survey in 2020 established the "
        "height of Mount Everest at 8,849 meters above sea level.",
    ]

    t0 = time.perf_counter()
    claims = pipe.extract(text)
    t_extract = time.perf_counter() - t0

    assert len(claims) >= 1

    t1 = time.perf_counter()
    result = pipe.check(claims[0].text, snippets)
    t_check = time.perf_counter() - t1

    assert result is not None

    t2 = time.perf_counter()
    verdict = pipe.render(result)
    t_render = time.perf_counter() - t2

    total = time.perf_counter() - t0

    print(f"\n--- end-to-end latency ---")
    print(f"  extract: {t_extract:.2f}s")
    print(f"  check:   {t_check:.2f}s")
    print(f"  render:  {t_render:.2f}s")
    print(f"  total:   {total:.2f}s")
    print(f"  label:   {verdict.label}")
    print(f"  explain: {verdict.explanation}")

    assert verdict.explanation.strip()

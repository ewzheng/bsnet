"""
test_validator.py

Unit tests for the Validator class.

This module verifies the behavior of Validator.evaluate_check_result()
using a variety of scenarios, including:

- Missing or invalid input data
- Strong supporting or contradicting evidence
- Aggregate scoring behavior
- Edge and borderline cases

Dependencies:
- pytest
- Project dataclasses: CheckResult, ScoredClaim, EvidenceScore
"""

import pytest

from bsnet.src.validation.validator import Validator
from bsnet.src.utils.outputs import CheckResult, ScoredClaim, EvidenceScore


@pytest.fixture(scope="module")
def validator() -> Validator:
    """
    Pytest fixture that provides a shared Validator instance.

    Scope:
        module — instantiated once per test module

    Returns:
        Validator: initialized validator instance
    """
    return Validator()


def test_returns_fail_when_scored_is_missing(validator: Validator) -> None:
    """
    Verify that the validator returns "fail" when the scored field is None.

    This represents a case where the pipeline did not produce any
    scoring output for the claim.
    """
    result = CheckResult(
        claim="Sample claim",
        label="supported",
        evidence="",
        scored=None,
    )

    assert validator.evaluate_check_result(result) == "fail"


def test_returns_fail_when_scores_list_is_empty(validator: Validator) -> None:
    """
    Verify that the validator returns "fail" when the scores list is empty.

    Even if a ScoredClaim exists, the absence of EvidenceScore entries
    should be treated as insufficient evidence.
    """
    result = CheckResult(
        claim="Sample claim",
        label="supported",
        evidence="",
        scored=ScoredClaim(
            claim="Sample claim",
            scores=[],
        ),
    )

    assert validator.evaluate_check_result(result) == "fail"


def test_returns_fail_on_very_strong_contradiction(validator: Validator) -> None:
    """
    Verify that the validator returns "fail" when a snippet strongly contradicts the claim.

    A contradiction score >= 0.90 should trigger an immediate fail condition.
    """
    result = CheckResult(
        claim="The sky is green.",
        label="supported",
        evidence="Snippet text",
        scored=ScoredClaim(
            claim="The sky is green.",
            scores=[
                EvidenceScore(
                    snippet="The sky is blue under normal daylight conditions.",
                    support=0.05,
                    contradict=0.95,
                    neutral=0.00,
                )
            ],
        ),
    )

    assert validator.evaluate_check_result(result) == "fail"


def test_returns_pass_on_one_very_strong_supporting_snippet(validator: Validator) -> None:
    """
    Verify that the validator returns "pass" when there is one very strong supporting snippet.

    High support with low contradiction should satisfy strong-pass conditions.
    """
    result = CheckResult(
        claim="Water freezes at 0 degrees Celsius.",
        label="supported",
        evidence="Science reference",
        scored=ScoredClaim(
            claim="Water freezes at 0 degrees Celsius.",
            scores=[
                EvidenceScore(
                    snippet="Pure water freezes at 0°C at standard pressure.",
                    support=0.91,
                    contradict=0.03,
                    neutral=0.06,
                )
            ],
        ),
    )

    assert validator.evaluate_check_result(result) == "pass"


def test_returns_pass_on_multiple_strong_supporting_snippets(validator: Validator) -> None:
    """
    Verify that the validator returns "pass" when multiple snippets strongly support the claim.

    This tests aggregate support and repetition-based scoring logic.
    """
    result = CheckResult(
        claim="The policy began in 2024.",
        label="supported",
        evidence="Several reports",
        scored=ScoredClaim(
            claim="The policy began in 2024.",
            scores=[
                EvidenceScore(
                    snippet="The policy took effect in January 2024.",
                    support=0.80,
                    contradict=0.08,
                    neutral=0.12,
                ),
                EvidenceScore(
                    snippet="Implementation started during 2024.",
                    support=0.78,
                    contradict=0.10,
                    neutral=0.12,
                ),
                EvidenceScore(
                    snippet="The rollout continued after its 2024 launch.",
                    support=0.67,
                    contradict=0.14,
                    neutral=0.19,
                ),
            ],
        ),
    )

    assert validator.evaluate_check_result(result) == "pass"


def test_returns_fail_when_multiple_contradictions_outweigh_support(validator: Validator) -> None:
    """
    Verify that the validator returns "fail" when contradictions dominate.

    This tests the rule:
        - multiple strong contradictions
        - total contradiction outweighs total support
    """
    result = CheckResult(
        claim="The event happened in 2025.",
        label="supported",
        evidence="Mixed reports",
        scored=ScoredClaim(
            claim="The event happened in 2025.",
            scores=[
                EvidenceScore(
                    snippet="The event took place in 2024.",
                    support=0.10,
                    contradict=0.72,
                    neutral=0.18,
                ),
                EvidenceScore(
                    snippet="Records show the event occurred in December 2024.",
                    support=0.08,
                    contradict=0.75,
                    neutral=0.17,
                ),
                EvidenceScore(
                    snippet="One summary mentions later discussion in 2025.",
                    support=0.40,
                    contradict=0.22,
                    neutral=0.38,
                ),
            ],
        ),
    )

    assert validator.evaluate_check_result(result) == "fail"


def test_label_only_does_not_force_pass(validator: Validator) -> None:
    """
    Verify that a positive label alone does not force a "pass".

    Weak or inconclusive evidence should still result in "fail".
    """
    result = CheckResult(
        claim="A false claim.",
        label="supported",
        evidence="Weak evidence",
        scored=ScoredClaim(
            claim="A false claim.",
            scores=[
                EvidenceScore(
                    snippet="This snippet is mostly unrelated.",
                    support=0.20,
                    contradict=0.23,
                    neutral=0.57,
                ),
                EvidenceScore(
                    snippet="This snippet also does not clearly support the claim.",
                    support=0.25,
                    contradict=0.21,
                    neutral=0.54,
                ),
            ],
        ),
    )

    assert validator.evaluate_check_result(result) == "fail"


def test_contradictory_label_does_not_force_fail_when_evidence_is_strong(validator: Validator) -> None:
    """
    Verify that a negative label does not override strong supporting evidence.

    The validator should rely more heavily on evidence than on the label.
    """
    result = CheckResult(
        claim="Earth orbits the Sun.",
        label="false",
        evidence="Astronomy references",
        scored=ScoredClaim(
            claim="Earth orbits the Sun.",
            scores=[
                EvidenceScore(
                    snippet="Earth revolves around the Sun once every year.",
                    support=0.90,
                    contradict=0.03,
                    neutral=0.07,
                ),
                EvidenceScore(
                    snippet="The Earth is in orbit around the Sun.",
                    support=0.81,
                    contradict=0.06,
                    neutral=0.13,
                ),
            ],
        ),
    )

    assert validator.evaluate_check_result(result) == "pass"


def test_borderline_case_returns_fail(validator: Validator) -> None:
    """
    Verify that weak, borderline evidence results in "fail".

    This ensures that the validator does not pass claims without
    sufficiently strong supporting signals.
    """
    result = CheckResult(
        claim="The company was founded in 1999.",
        label="supported",
        evidence="Unclear references",
        scored=ScoredClaim(
            claim="The company was founded in 1999.",
            scores=[
                EvidenceScore(
                    snippet="The company was active by the early 2000s.",
                    support=0.33,
                    contradict=0.33,
                    neutral=0.34,
                ),
                EvidenceScore(
                    snippet="A later profile mentions the company in 2001.",
                    support=0.30,
                    contradict=0.35,
                    neutral=0.45,
                ),
            ],
        ),
    )

    assert validator.evaluate_check_result(result) == "fail"


def test_invalid_score_raises_value_error(validator: Validator) -> None:
    """
    Verify that invalid probability values raise a ValueError.

    Any EvidenceScore value outside the [0, 1] range should be rejected.
    """
    result = CheckResult(
        claim="Invalid score claim",
        label="supported",
        evidence="Bad data",
        scored=ScoredClaim(
            claim="Invalid score claim",
            scores=[
                EvidenceScore(
                    snippet="Invalid score snippet",
                    support=1.10,
                    contradict=0.00,
                    neutral=0.00,
                )
            ],
        ),
    )

    with pytest.raises(ValueError):
        validator.evaluate_check_result(result)
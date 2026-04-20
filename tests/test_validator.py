"""
test_validator.py

Unit tests for validator.py.
-Imports project dataclasses CheckResult, ScoredClaim, 
 and EvidenceScore from models.py 
-Invokes validator's evaluate_check_result function with test data
-Tests edge cases like missing CheckResult object, missing scores, and invalid scores. 

"""

import pytest

from validator import evaluate_check_result
from models import CheckResult, ScoredClaim, EvidenceScore


def test_returns_fail_when_scored_is_missing() -> None:
    """Validator should fail when no scored claim is present."""
    result = CheckResult(
        claim="Sample claim",
        label="supported",
        evidence="",
        scored=None,
    )
    assert evaluate_check_result(result) == "fail"


def test_returns_fail_when_scores_list_is_empty() -> None:
    """Validator should fail when there are no evidence scores."""
    result = CheckResult(
        claim="Sample claim",
        label="supported",
        evidence="",
        scored=ScoredClaim(
            claim="Sample claim",
            scores=[],
        ),
    )
    assert evaluate_check_result(result) == "fail"


def test_returns_fail_on_very_strong_contradiction() -> None:
    """Validator should fail when any snippet strongly contradicts the claim."""
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
    assert evaluate_check_result(result) == "fail"


def test_returns_pass_on_one_very_strong_supporting_snippet() -> None:
    """Validator should pass when there is one very strong supporting snippet."""
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
    assert evaluate_check_result(result) == "pass"


def test_returns_pass_on_multiple_strong_supporting_snippets() -> None:
    """Validator should pass when multiple snippets support the claim strongly."""
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
    assert evaluate_check_result(result) == "pass"


def test_returns_fail_when_multiple_contradictions_outweigh_support() -> None:
    """Validator should fail when contradiction appears repeatedly and outweighs support."""
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
    assert evaluate_check_result(result) == "fail"


def test_label_only_does_not_force_pass() -> None:
    """Validator should not pass based only on a supportive label."""
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
                    contradict=0.18,
                    neutral=0.62,
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
    assert evaluate_check_result(result) == "fail"


def test_contradictory_label_does_not_force_fail_when_evidence_is_strong() -> None:
    """Validator should still pass when evidence is strong even if label is negative."""
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
    assert evaluate_check_result(result) == "pass"


def test_borderline_case_returns_fail() -> None:
    """Validator should fail a weak, borderline case with no strong support."""
    result = CheckResult(
        claim="The company was founded in 1999.",
        label="supported",
        evidence="Unclear references",
        scored=ScoredClaim(
            claim="The company was founded in 1999.",
            scores=[
                EvidenceScore(
                    snippet="The company was active by the early 2000s.",
                    support=0.46,
                    contradict=0.18,
                    neutral=0.36,
                ),
                EvidenceScore(
                    snippet="A later profile mentions the company in 2001.",
                    support=0.39,
                    contradict=0.22,
                    neutral=0.39,
                ),
            ],
        ),
    )
    assert evaluate_check_result(result) == "fail"


def test_invalid_score_raises_value_error() -> None:
    """Validator should reject probabilities outside the range [0, 1]."""
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
        evaluate_check_result(result)
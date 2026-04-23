"""
test_validator.py

Integration-style pytest test for the real Pipeline + Validator flow.
"""
import pytest

from bsnet.src.runtime.pipeline import Pipeline
from bsnet.src.validation.validator import Validator
from bsnet.src.utils.outputs import CheckResult


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def pipeline() -> Pipeline:
    """Create a reusable Pipeline instance."""
    return Pipeline()


@pytest.fixture
def validator() -> Validator:
    """Create a reusable Validator instance."""
    return Validator()


# -----------------------
# Tests
# -----------------------

def test_pipeline_check_result_passes_validator_obviously_true(pipeline, validator) -> None:
    """A clearly supported claim should produce a passing validator result."""
    claim = "Water freezes at 0 degrees Celsius at standard atmospheric pressure."
    snippets = [
        "Pure water freezes at 0 degrees Celsius at standard atmospheric pressure.",
        "At standard pressure, the freezing point of water is 0°C.",
        "Science references state that water reaches its freezing point at 0 degrees Celsius under normal atmospheric conditions.",
    ]

    check_result = pipeline.check(claim, snippets)

    assert check_result is not None
    assert isinstance(check_result, CheckResult)
    assert check_result.scored is not None
    assert check_result.scored.scores is not None
    assert len(check_result.scored.scores) > 0

    result = validator.evaluate_check_result(check_result)

    assert result is True


def test_pipeline_check_result_fails_validator_obviously_false(pipeline, validator) -> None:
    """A clearly false claim should fail validation."""
    claim = "The Earth is flat."
    snippets = [
        "Scientific consensus confirms that the Earth is an oblate spheroid.",
        "Satellite imagery shows the Earth is round.",
        "Physics and astronomy demonstrate Earth's curvature.",
    ]

    check_result = pipeline.check(claim, snippets)
    result = validator.evaluate_check_result(check_result)

    assert result is False


def test_pipeline_check_result_passes_strong_consensus(pipeline, validator) -> None:
    """A widely accepted scientific fact should pass validation."""
    claim = "The speed of light in vacuum is approximately 299,792 kilometers per second."
    snippets = [
        "The speed of light in a vacuum is about 299,792 km/s.",
        "Physics constants define the speed of light as roughly 3.0 × 10^8 meters per second.",
        "The universally accepted value of the speed of light is approximately 299,792 km/s.",
    ]

    check_result = pipeline.check(claim, snippets)
    result = validator.evaluate_check_result(check_result)

    assert result is True


def test_pipeline_check_result_fails_with_strong_contradiction(pipeline, validator) -> None:
    """Direct contradiction in evidence should fail validation."""
    claim = "Humans can breathe unaided in outer space."
    snippets = [
        "Humans cannot survive in the vacuum of space without protective equipment.",
        "Exposure to space without a suit leads to rapid unconsciousness.",
        "Astronauts require pressurized suits to breathe in space.",
    ]

    check_result = pipeline.check(claim, snippets)
    result = validator.evaluate_check_result(check_result)

    assert result is False


def test_pipeline_check_result_mixed_evidence_should_fail(pipeline, validator) -> None:
    """Mixed but contradictory evidence should typically fail."""
    claim = "Coffee causes dehydration."
    snippets = [
        "Moderate coffee consumption does not lead to dehydration.",
        "Caffeine has mild diuretic effects.",
        "Studies show coffee contributes to daily fluid intake.",
    ]

    check_result = pipeline.check(claim, snippets)
    result = validator.evaluate_check_result(check_result)

    assert result is False


def test_pipeline_check_result_weak_support_should_fail(pipeline, validator) -> None:
    """Weak or vague support should fail validation."""
    claim = "Eating chocolate significantly improves intelligence."
    snippets = [
        "Some studies explore correlations between diet and cognition.",
        "Chocolate contains flavonoids that may benefit brain function.",
        "There is limited evidence linking chocolate consumption to cognitive improvements.",
    ]

    check_result = pipeline.check(claim, snippets)
    result = validator.evaluate_check_result(check_result)

    assert result is False


def test_pipeline_check_result_partial_support_should_pass(pipeline, validator) -> None:
    """Moderately supported claim should pass."""
    claim = "Exercise improves cardiovascular health."
    snippets = [
        "Regular physical activity strengthens the heart and improves circulation.",
        "Exercise reduces the risk of cardiovascular disease.",
        "Studies consistently show exercise benefits heart health.",
    ]

    check_result = pipeline.check(claim, snippets)
    result = validator.evaluate_check_result(check_result)

    assert result is True


def test_pipeline_check_result_ambiguous_claim_edge_case(pipeline, validator) -> None:
    """Ambiguous claims with unclear support should fail."""
    claim = "Technology is harmful to society."
    snippets = [
        "Technology has both positive and negative effects on society.",
        "Some studies highlight risks like addiction and privacy concerns.",
        "Other research shows technology improves productivity and communication.",
    ]

    check_result = pipeline.check(claim, snippets)
    result = validator.evaluate_check_result(check_result)

    assert result is False
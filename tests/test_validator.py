"""Tests for the Validator.

Two layers:

1. Fast contract-level unit tests against synthetic ``EvidenceScore``
   fixtures — no model loads. Cover the ladder the validator walks
   per call: input-validity checks, uncertain-label drop, hard-fail
   on a near-certain opposing peak, redundancy gate on the aligned
   axis, and directional-aggregate threshold for both confident-true
   and confident-false verdicts.
2. One integration test that runs a hand-picked unambiguous claim
   through the real search backend and the real scorer, then feeds
   the resulting ``CheckResult`` into the validator. Exercises the
   validator against the score distribution it will see in
   production so threshold tuning has something realistic to lean on.

Note on labels: the validator is label-aware — it dispatches off
``CheckResult.label``, which always comes from
``bsnet.src.model._common.label_claim``. Tests therefore use that
function's actual outputs (``"true"``, ``"mostly true"``, ``"false"``,
``"mostly false"``, ``"mixture"``, ``"unproven"``,
``"partially true"``, ``"partially false"``, ``"no-evidence"``).
"""

import pytest

from bsnet.src.runtime.pipeline import Pipeline
from bsnet.src.utils.outputs import CheckResult, EvidenceScore, ScoredClaim
from bsnet.src.utils.search import get_search_snippets
from bsnet.src.validation.validator import Validator


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def validator() -> Validator:
    """Construct a fresh Validator for each unit test.

    Returns:
        A stateless ``Validator`` instance.

    Preconditions:
        - None.

    Postconditions:
        - Returned validator is independent of any other test.
    """
    return Validator()


@pytest.fixture(scope="module")
def pipe() -> Pipeline:
    """Load the pipeline once for the module.

    Returns:
        A ready-to-use ``Pipeline`` instance.

    Preconditions:
        - Model weights are downloaded or downloadable.

    Postconditions:
        - All three pipeline models are loaded into memory.
    """
    return Pipeline()


def _result(
    label: str,
    scores: list[EvidenceScore],
    claim: str = "Sample claim",
) -> CheckResult:
    """Build a ``CheckResult`` with the given label and scores.

    Args:
        label: One of the labels emitted by ``label_claim``.
        scores: Per-snippet score fixtures.
        claim: Claim text — defaults to a stub since most tests only
            exercise score-driven logic.

    Returns:
        A populated ``CheckResult``.

    Preconditions:
        - ``scores`` is a list (possibly empty) of ``EvidenceScore``.

    Postconditions:
        - Returned object is independent of any other fixture.
    """
    return CheckResult(
        claim=claim,
        label=label,
        evidence=scores[0].snippet if scores else "",
        scored=ScoredClaim(claim=claim, scores=scores),
    )


# ── Input-validity checks ────────────────────────────────────────────────────


def test_returns_false_when_scored_is_missing(validator: Validator) -> None:
    """Validator returns ``False`` when ``scored`` is ``None``.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator rejects the result.
    """
    result = CheckResult(
        claim="Sample claim",
        label="true",
        evidence="",
        scored=None,
    )
    assert validator.evaluate_check_result(result) is False


def test_returns_false_when_scores_list_is_empty(validator: Validator) -> None:
    """Validator returns ``False`` when there are no evidence scores.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator rejects the result.
    """
    assert validator.evaluate_check_result(_result("true", [])) is False


def test_invalid_score_raises_value_error(validator: Validator) -> None:
    """Validator rejects probabilities outside the range [0, 1].

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts ``ValueError`` is raised on out-of-range input.
    """
    result = _result(
        "true",
        [
            EvidenceScore(
                snippet="Invalid score snippet",
                support=1.10,
                contradict=0.00,
                neutral=0.00,
            ),
        ],
    )
    with pytest.raises(ValueError):
        validator.evaluate_check_result(result)


# ── Uncertain-label pass-through ────────────────────────────────────────────


@pytest.mark.parametrize(
    "label",
    ["mixture", "partially true", "partially false"],
)
def test_borderline_directional_labels_pass_through(
    validator: Validator, label: str,
) -> None:
    """Borderline-but-directional labels reach the renderer.

    Mixture / partially X carry real directional NLI signal — the
    labeler's way of saying "the verdict is real but borderline." The
    renderer carries dedicated emojis (〰️ / 🟡 / 🟠) for each so the
    user sees the labeler's nuance directly. The validator's job is to
    catch hallucination on definitive labels, not to drop every label
    below "definitive."

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator accepts the result regardless of the
          underlying score distribution.
    """
    # Arbitrary score distribution — pass-through happens before any
    # score-based gate.
    scores = [
        EvidenceScore(snippet="A", support=0.50, contradict=0.30, neutral=0.20),
        EvidenceScore(snippet="B", support=0.40, contradict=0.40, neutral=0.20),
    ]
    assert validator.evaluate_check_result(_result(label, scores)) is True


def test_unproven_label_drops(validator: Validator) -> None:
    """``"unproven"`` drops at the validator.

    By definition no snippet has meaningful directional NLI signal
    when the labeler returns ``"unproven"`` — both ``max_support`` and
    ``max_contradict`` are below ``WEAK_SIGNAL``. The renderer would
    still paraphrase the strongest-of-weak snippet under a forced
    ``"rated unproven because"`` template, producing a confident
    explanation grounded in noise. Dropping at the validator skips
    render entirely.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator drops the result regardless of the
          underlying score distribution.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.20, contradict=0.10, neutral=0.70),
        EvidenceScore(snippet="B", support=0.10, contradict=0.20, neutral=0.70),
    ]
    assert validator.evaluate_check_result(_result("unproven", scores)) is False


def test_no_evidence_label_drops(validator: Validator) -> None:
    """The ``no-evidence`` sentinel drops at the validator.

    The orchestrator bypasses the validator for ``no-evidence``
    results so ``Pipeline.render`` can produce its static notice,
    but the validator stays defensive in case a caller routes one
    through directly.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator drops the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.05, contradict=0.05, neutral=0.90),
    ]
    assert validator.evaluate_check_result(_result("no-evidence", scores)) is False


def test_unknown_label_drops(validator: Validator) -> None:
    """An unrecognized label drops defensively.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts an unknown label produces ``False``.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.85, contradict=0.05, neutral=0.10),
        EvidenceScore(snippet="B", support=0.80, contradict=0.05, neutral=0.15),
    ]
    assert validator.evaluate_check_result(_result("supported", scores)) is False


# ── Hard fail on opposing peak ──────────────────────────────────────────────


def test_definitive_true_hard_fails_on_strong_opposing_peak(
    validator: Validator,
) -> None:
    """A definitive ``"true"`` drops when any single snippet contradicts hard.

    A 0.95 contradict snippet is enough to distrust the labeler's
    ``"true"`` verdict no matter what the rest of the distribution
    looks like — a definitive label means the labeler claimed
    *no* significant opposing signal exists, so a strong opposing
    peak indicates the labeler missed something major.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator rejects the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.85, contradict=0.10, neutral=0.05),
        EvidenceScore(snippet="B", support=0.80, contradict=0.10, neutral=0.10),
        EvidenceScore(
            snippet="Counter", support=0.02, contradict=0.95, neutral=0.03,
        ),
    ]
    assert validator.evaluate_check_result(_result("true", scores)) is False


def test_definitive_false_hard_fails_on_strong_opposing_peak(
    validator: Validator,
) -> None:
    """A definitive ``"false"`` drops when any single snippet supports hard.

    Mirror of the ``"true"`` case: a 0.95 support snippet against a
    ``"false"`` label is enough to distrust the labeler.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator rejects the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.05, contradict=0.85, neutral=0.10),
        EvidenceScore(snippet="B", support=0.10, contradict=0.80, neutral=0.10),
        EvidenceScore(
            snippet="Counter", support=0.95, contradict=0.02, neutral=0.03,
        ),
    ]
    assert validator.evaluate_check_result(_result("false", scores)) is False


def test_mostly_true_passes_with_strong_opposing_peak(
    validator: Validator,
) -> None:
    """``"mostly true"`` survives a strong opposing peak.

    The ``mostly *`` qualifier explicitly encodes the labeler's
    acknowledgment that opposing signal exists. Re-checking for an
    opposing peak there double-counts the labeler's own concession;
    AVeriTeC eval showed this was costing real verdicts where a
    fact-check article quoting the claim verbatim drove
    ``max_contradict`` above the hard-fail threshold even though the
    labeler had already weighted contradict by count.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator passes the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.85, contradict=0.10, neutral=0.05),
        EvidenceScore(snippet="B", support=0.80, contradict=0.10, neutral=0.10),
        EvidenceScore(
            snippet="Counter", support=0.02, contradict=0.95, neutral=0.03,
        ),
    ]
    assert validator.evaluate_check_result(_result("mostly true", scores)) is True


def test_mostly_false_passes_with_strong_opposing_peak(
    validator: Validator,
) -> None:
    """``"mostly false"`` survives a strong opposing peak.

    Mirror of the ``"mostly true"`` case. The validator trusts the
    labeler's mixture-aware verdict rather than dropping on a single
    strong support snippet.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator passes the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.05, contradict=0.85, neutral=0.10),
        EvidenceScore(snippet="B", support=0.10, contradict=0.80, neutral=0.10),
        EvidenceScore(
            snippet="Counter", support=0.95, contradict=0.02, neutral=0.03,
        ),
    ]
    assert validator.evaluate_check_result(_result("mostly false", scores)) is True


# ── Redundancy gate ─────────────────────────────────────────────────────────


def test_single_moderate_aligned_snippet_passes(
    validator: Validator,
) -> None:
    """One moderate supporting snippet now surfaces "true".

    With ``MIN_ALIGNED_REDUNDANCY = 1`` (lowered after AVeriTeC eval
    showed the previous 2-snippet floor over-pruned real refutations),
    a single snippet above ``MODERATE_SIGNAL`` on the aligned axis is
    sufficient redundancy. The hard-fail opposing-peak check still
    catches the worst cherry-pick case (strong contrary signal that
    the labeler missed); without that, the validator now trusts the
    labeler's directional verdict.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator passes the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.78, contradict=0.03, neutral=0.19),
        EvidenceScore(snippet="B", support=0.10, contradict=0.05, neutral=0.85),
        EvidenceScore(snippet="C", support=0.15, contradict=0.04, neutral=0.81),
    ]
    assert validator.evaluate_check_result(_result("true", scores)) is True


def test_single_moderate_aligned_contradict_passes(
    validator: Validator,
) -> None:
    """One moderate contradicting snippet now surfaces "false".

    Mirror of the support-axis redundancy relaxation: a single
    snippet above ``MODERATE_SIGNAL`` on the contradict axis is
    sufficient under ``MIN_ALIGNED_REDUNDANCY = 1``.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator passes the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.03, contradict=0.78, neutral=0.19),
        EvidenceScore(snippet="B", support=0.10, contradict=0.10, neutral=0.80),
        EvidenceScore(snippet="C", support=0.05, contradict=0.20, neutral=0.75),
    ]
    assert validator.evaluate_check_result(_result("false", scores)) is True


def test_single_near_certain_snippet_passes_via_alt_path(
    validator: Validator,
) -> None:
    """A single very-confident snippet substitutes for count redundancy.

    Distinguishes a genuinely-supported niche claim (one Wikipedia
    page at sup=0.99 plus neutrals from related pages) from the
    cherry-picked subjective pattern. Both have only one moderate
    aligned snippet, but the near-certain peak is corroboration
    enough on its own.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator passes the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.99, contradict=0.01, neutral=0.00),
        EvidenceScore(snippet="B", support=0.05, contradict=0.05, neutral=0.90),
        EvidenceScore(snippet="C", support=0.10, contradict=0.05, neutral=0.85),
    ]
    assert validator.evaluate_check_result(_result("true", scores)) is True


# ── Confident-true verdicts ─────────────────────────────────────────────────


def test_multiple_strong_supporting_snippets_pass(
    validator: Validator,
) -> None:
    """Validator passes when multiple snippets support a "true" claim.

    Three moderate-or-stronger supports + low contradiction →
    redundancy and aggregate both clear, verdict surfaces.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator accepts the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.80, contradict=0.08, neutral=0.12),
        EvidenceScore(snippet="B", support=0.78, contradict=0.10, neutral=0.12),
        EvidenceScore(snippet="C", support=0.67, contradict=0.14, neutral=0.19),
    ]
    assert validator.evaluate_check_result(_result("true", scores)) is True


# ── Confident-false verdicts (the new contract) ─────────────────────────────


def test_multiple_strong_contradicting_snippets_pass_as_false(
    validator: Validator,
) -> None:
    """Validator passes a confidently-false claim so the user sees ❌.

    Mirror of the confident-true case on the contradict axis. Without
    this branch the previous validator silently dropped definitively
    falsifiable claims like "The Earth is flat" before render.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator accepts the result so the renderer
          can produce a "false" verdict.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.05, contradict=0.82, neutral=0.13),
        EvidenceScore(snippet="B", support=0.07, contradict=0.75, neutral=0.18),
        EvidenceScore(snippet="C", support=0.20, contradict=0.65, neutral=0.15),
    ]
    assert validator.evaluate_check_result(_result("false", scores)) is True


# ── Label-vs-evidence disagreement drops ────────────────────────────────────


def test_label_evidence_disagreement_drops(validator: Validator) -> None:
    """When the label and the evidence disagree, drop instead of overriding.

    A "false" label paired with strongly supporting evidence means
    something upstream (the labeler or the search) is confused.
    Surfacing either direction is worse than dropping — the validator
    trusts the labeler's verdict only when the score distribution
    actually corroborates it on the matching axis.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator drops the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.90, contradict=0.03, neutral=0.07),
        EvidenceScore(snippet="B", support=0.81, contradict=0.06, neutral=0.13),
    ]
    assert validator.evaluate_check_result(_result("false", scores)) is False


# ── Borderline / weak-evidence cases ────────────────────────────────────────


def test_weak_aligned_evidence_drops(validator: Validator) -> None:
    """Weak support across multiple snippets does not surface a "true" verdict.

    All snippets sit below ``MODERATE_SIGNAL`` on the support axis,
    so the redundancy gate fires on count alone.

    Preconditions:
        - ``validator`` fixture is loaded.

    Postconditions:
        - Asserts the validator drops the result.
    """
    scores = [
        EvidenceScore(snippet="A", support=0.46, contradict=0.18, neutral=0.36),
        EvidenceScore(snippet="B", support=0.39, contradict=0.22, neutral=0.39),
    ]
    assert validator.evaluate_check_result(_result("true", scores)) is False


# ── Integration test on real pipeline output ─────────────────────────────────


def test_validator_passes_well_supported_claim(
    pipe: Pipeline, validator: Validator,
) -> None:
    """Validator returns ``True`` for a claim with strong web consensus.

    Pulls real snippets from the search backend, scores them with the
    real NLI model, and asserts that the validator accepts the
    resulting ``CheckResult``. Prints the per-snippet score
    distribution so threshold tuning has visibility into what the
    validator actually saw.

    Preconditions:
        - ``pipe`` and ``validator`` fixtures are loaded.
        - Network access to the configured search backends is available.

    Postconditions:
        - Asserts the validator accepts the result.
        - Prints scores for each snippet.
    """
    claim = "The speed of light in a vacuum is approximately 299,792 kilometers per second."

    snippets = get_search_snippets(claim)
    assert snippets, "search returned no snippets — check network / backends"

    result = pipe.check(claim, snippets)
    assert isinstance(result, CheckResult)
    assert result.scored is not None
    assert result.scored.scores, "scorer produced no scores"

    print(f"\n--- validator integration: '{claim}' ---")
    print(f"  label: {result.label}")
    for s in result.scored.scores:
        print(
            f"    sup={s.support:.3f}  con={s.contradict:.3f}  "
            f"neu={s.neutral:.3f}  | {s.snippet[:80]}"
        )

    passed = validator.evaluate_check_result(result)
    print(f"  validator: {passed}")

    assert passed is True

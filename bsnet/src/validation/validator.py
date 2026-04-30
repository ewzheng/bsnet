"""Heuristic evaluator for ``CheckResult`` objects.

The validator is a *hallucination guard*, not a label filter — it
gates which scored claims reach the renderer by catching cases
where the labeler's verdict disagrees with the underlying NLI
score distribution. Verdicts the labeler itself flagged as
uncertain (``"partially true"``, ``"partially false"``,
``"mixture"``, ``"unproven"``) are passed straight through; the
renderer carries dedicated emojis for each so the user sees the
labeler's nuance verbatim. Only definitive labels
(``"true"``/``"mostly true"``/``"false"``/``"mostly false"``) get
the redundancy + opposing-peak hard-fail checks, and they're only
dropped when the score distribution actively contradicts the
labeled direction.
"""

from typing import List

from bsnet.src.model._common import MODERATE_SIGNAL
from bsnet.src.utils.outputs import CheckResult, EvidenceScore


class Validator:
    """Hallucination-guard gate for fact-check results.

    Walks a small ladder per call:

    1. **Pass-through uncertain labels.** ``"partially true"``,
       ``"partially false"``, ``"mixture"``, and ``"unproven"`` are
       the labeler's way of saying "this is the verdict, but it's
       borderline / no-signal." The renderer carries 🟡 / 🟠 / 〰️ /
       ❓ emojis for those, so the user sees the labeler's nuance
       directly. Validator does no further filtering on them.
    2. **Pick the aligned axis for definitive labels.** Positive
       labels (``"true"``/``"mostly true"``) align with the
       ``support`` axis; negative labels (``"false"``/
       ``"mostly false"``) align with the ``contradict`` axis. Both
       directions then run the same hallucination checks.
    3. **Hard fail on a near-certain opposing peak.** A single
       snippet at ≥ ``HARD_FAIL_OPPOSING_PEAK`` on the wrong axis
       suggests the labeler missed something major. Drop.
    4. **Redundancy check.** Require either at least
       ``MIN_ALIGNED_REDUNDANCY`` snippets above ``MODERATE_SIGNAL``
       on the aligned axis, OR a single near-certain peak at ≥
       ``SINGLE_ALIGNED_PASS``. Catches the cherry-picked one-sided
       pattern (one strongly-supportive blog post + neutrals → a
       definitive verdict from the labeler that doesn't actually
       have corroboration).

    Anything else (unknown labels, ``"no-evidence"``) drops
    defensively. ``"no-evidence"`` is also bypassed at the
    orchestrator before it reaches the validator.
    """

    # Labels the labeler used to signal nuance / uncertainty. Pass
    # through to the renderer so the user sees the appropriate
    # emoji rather than a silent drop.
    PASS_THROUGH_LABELS: frozenset[str] = frozenset({
        "partially true",
        "partially false",
        "mixture",
        "unproven",
    })

    # Labels that mean "the claim is supported by the evidence."
    POSITIVE_LABELS: frozenset[str] = frozenset({"true", "mostly true"})

    # Labels that mean "the claim is contradicted by the evidence."
    NEGATIVE_LABELS: frozenset[str] = frozenset({"false", "mostly false"})

    # An opposing-axis peak at or above this triggers a hard fail —
    # one near-certain counter-snippet is enough to distrust the
    # verdict regardless of what dominates by count.
    HARD_FAIL_OPPOSING_PEAK: float = 0.90

    # Minimum number of moderate-or-higher snippets that must agree
    # with the verdict's direction. Two corroborating snippets are
    # cheap defense in depth against extractor leakage of subjective
    # claims that happen to surface one strongly-supportive source.
    MIN_ALIGNED_REDUNDANCY: int = 2

    # Alternative redundancy path: a single near-certain aligned
    # snippet is also sufficient corroboration. Distinguishes
    # genuinely-supported niche claims (Argentina-style — one
    # Wikipedia snippet at 0.99 plus neutrals from related pages)
    # from cherry-picked subjective claims (tech-harmful-style —
    # one supportive blog at 0.77 plus unrelated neutrals).
    SINGLE_ALIGNED_PASS: float = 0.90

    def evaluate_check_result(self, result: CheckResult) -> bool:
        """Decide whether a ``CheckResult`` should reach the renderer.

        Args:
            result: The scored, labeled check result coming out of
                ``Pipeline.check``.

        Returns:
            ``True`` when the verdict should surface to the user,
            ``False`` when the result should be dropped (either
            structurally invalid or the labeler/evidence disagree
            badly enough to suggest hallucination).

        Preconditions:
            - ``result`` is either ``None`` or a ``CheckResult``
              produced by ``Pipeline.check``.

        Postconditions:
            - Does not mutate ``result``.
            - Raises ``ValueError`` only when individual score
              probabilities fall outside ``[0, 1]``.
        """
        if not result or not result.scored:
            return False

        scores: List[EvidenceScore] = result.scored.scores or []
        if not scores:
            return False

        self._validate_scores(scores)

        label = (result.label or "").strip().lower()
        if label in self.PASS_THROUGH_LABELS:
            return True

        if label in self.POSITIVE_LABELS:
            aligned, opposing = "support", "contradict"
        elif label in self.NEGATIVE_LABELS:
            aligned, opposing = "contradict", "support"
        else:
            # ``no-evidence`` and any unknown label fall through to a
            # drop. ``no-evidence`` is also bypassed in the orchestrator
            # so it never actually reaches us, but we stay defensive.
            return False

        aligned_values = [getattr(s, aligned) for s in scores]
        opposing_values = [getattr(s, opposing) for s in scores]

        if max(opposing_values) >= self.HARD_FAIL_OPPOSING_PEAK:
            return False

        n_moderate_aligned = sum(
            1 for v in aligned_values if v > MODERATE_SIGNAL
        )
        if (
            n_moderate_aligned < self.MIN_ALIGNED_REDUNDANCY
            and max(aligned_values) < self.SINGLE_ALIGNED_PASS
        ):
            return False

        return True

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _validate_scores(self, scores: List[EvidenceScore]) -> None:
        """Reject probabilities that fall outside ``[0, 1]``.

        Args:
            scores: Per-snippet scores to validate.

        Raises:
            ValueError: When any ``support``, ``contradict``, or
                ``neutral`` probability is outside ``[0, 1]``.

        Preconditions:
            - ``scores`` is iterable of ``EvidenceScore``.

        Postconditions:
            - Returns ``None`` only when every value is in range.
            - Does not mutate ``scores``.
        """
        for s in scores:
            for val in (s.support, s.contradict, s.neutral):
                if val < 0.0 or val > 1.0:
                    raise ValueError(
                        "EvidenceScore values must be between 0 and 1"
                    )

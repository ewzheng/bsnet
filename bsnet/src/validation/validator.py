"""Heuristic evaluator for ``CheckResult`` objects.

The validator is a *hallucination guard*, not a label filter — it
gates which scored claims reach the renderer by catching cases
where the labeler's verdict disagrees with the underlying NLI
score distribution. Borderline-but-directional labels
(``"partially true"``, ``"partially false"``, ``"mixture"``) are
passed straight through; the renderer carries dedicated emojis for
each so the user sees the labeler's nuance verbatim. ``"unproven"``
is dropped because by definition no snippet has meaningful
directional NLI signal, and the renderer — which paraphrases the
strongest-of-weak snippet under a forced ``"rated unproven
because"`` template — would otherwise produce a confident
explanation grounded in noise. Only definitive labels
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

    1. **Pass-through borderline-but-directional labels.**
       ``"partially true"``, ``"partially false"``, and ``"mixture"``
       still carry directional NLI signal — the labeler's way of
       saying "the verdict is real but borderline." The renderer
       carries 🟡 / 🟠 / 〰️ emojis for those, so the user sees the
       labeler's nuance directly. Validator does no further
       filtering on them.
    2. **Pick the aligned axis for definitive labels.** Positive
       labels (``"true"``/``"mostly true"``) align with the
       ``support`` axis; negative labels (``"false"``/
       ``"mostly false"``) align with the ``contradict`` axis. Both
       directions then run the same hallucination checks.
    3. **Hard fail on a near-certain opposing peak.** A single
       snippet at ≥ ``HARD_FAIL_OPPOSING_PEAK`` on the wrong axis
       suggests the labeler missed something major. Drop. Applies
       only to definitive labels (``true``/``false``); the
       ``mostly *`` family already encodes the labeler's
       acknowledgment of opposing signal so we don't double-count.
    4. **Redundancy check.** Require either at least
       ``MIN_ALIGNED_REDUNDANCY`` snippets above ``MODERATE_SIGNAL``
       on the aligned axis, OR a single near-certain peak at ≥
       ``SINGLE_ALIGNED_PASS``. With ``MIN_ALIGNED_REDUNDANCY = 1``
       this effectively trusts any single moderate-aligned snippet
       once the hard-fail opposing-peak check has passed; the
       count-based floor was lowered after AVeriTeC eval showed the
       previous 2-snippet floor over-pruned real refutations.

    Anything else (``"unproven"``, unknown labels, ``"no-evidence"``)
    drops defensively. ``"unproven"`` is dropped because no snippet
    has meaningful directional NLI signal — the renderer would still
    paraphrase the strongest-of-weak snippet, producing a confident
    explanation grounded in noise (observed on AVeriTeC eval where
    a clear refutation snippet scored below ``WEAK_SIGNAL`` on both
    axes and the renderer restated the original false claim's
    specifics under a "rated unproven because" template).
    ``"no-evidence"`` is also bypassed at the orchestrator before it
    reaches the validator.
    """

    # Labels the labeler used to signal a real-but-borderline
    # verdict. Pass through to the renderer so the user sees the
    # appropriate emoji rather than a silent drop. ``"unproven"`` is
    # deliberately excluded — see the class docstring for rationale.
    PASS_THROUGH_LABELS: frozenset[str] = frozenset({
        "partially true",
        "partially false",
        "mixture",
    })

    # Labels that mean "the claim is supported by the evidence."
    POSITIVE_LABELS: frozenset[str] = frozenset({"true", "mostly true"})

    # Labels that mean "the claim is contradicted by the evidence."
    NEGATIVE_LABELS: frozenset[str] = frozenset({"false", "mostly false"})

    # An opposing-axis peak at or above this triggers a hard fail
    # for **definitive** labels (``true``/``false`` only). For the
    # ``mostly *`` variants the labeler has already explicitly
    # acknowledged some opposing signal — that's what the "mostly"
    # qualifier means — so re-checking for an opposing peak there
    # double-counts the labeler's own concession. AVeriTeC eval
    # showed the symmetric hard-fail was costing real refutations:
    # ``mostly false`` verdicts with a strong support snippet (often
    # a fact-check article quoting the false claim) were dropped
    # despite the labeler correctly weighting contradict by count.
    HARD_FAIL_OPPOSING_PEAK: float = 0.90

    # Labels that get the hard-fail opposing-peak check. The
    # ``mostly *`` family is excluded by design — see
    # ``HARD_FAIL_OPPOSING_PEAK`` for rationale.
    DEFINITIVE_LABELS: frozenset[str] = frozenset({"true", "false"})

    # Minimum number of moderate-or-higher snippets that must agree
    # with the verdict's direction. Originally 2 to defend against
    # cherry-picked one-sided evidence, but AVeriTeC eval showed the
    # 2-snippet floor was over-pruning real refutations: 22 of 116
    # gold-Refuted-but-NEI cases were ``mostly false`` verdicts whose
    # labeler-direction matched gold and only failed redundancy by
    # one snippet. Lowered to 1 to trust the labeler's single-
    # moderate-aligned signal; the hard-fail opposing-peak rule
    # (``HARD_FAIL_OPPOSING_PEAK``) still catches the worst cherry-
    # pick case where strong contrary signal exists but the labeler
    # missed it.
    MIN_ALIGNED_REDUNDANCY: int = 1

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

        if (
            label in self.DEFINITIVE_LABELS
            and max(opposing_values) >= self.HARD_FAIL_OPPOSING_PEAK
        ):
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

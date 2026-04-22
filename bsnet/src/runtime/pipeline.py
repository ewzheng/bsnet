"""Fact-checking pipeline that strings together extraction, scoring, and rendering.

Provides a single entry point for running text through the
fact-checking pipeline. The three stages (extract, check, render)
are exposed separately so the orchestrator can insert validation
or other logic between them.
"""

from bsnet.src.model._common import STRONG_SIGNAL, WEAK_SIGNAL, label_claim
from bsnet.src.model.extractor import Extractor
from bsnet.src.model.renderer import Renderer
from bsnet.src.model.scorer import Scorer
from bsnet.src.utils.outputs import CheckResult, Claim, ScoredClaim, Verdict


class Pipeline:
    """Wires together the extractor, scorer, labeler, and renderer.

    Loads all models once at construction. Exposes three methods
    that map to the pipeline stages:

    1. ``extract()`` — pull claims from transcript text
    2. ``check()`` — score a claim against search results and label it
    3. ``render()`` — generate a human-readable explanation

    The orchestrator can insert validation between ``check()`` and
    ``render()``.
    """

    def __init__(self) -> None:
        """Load all pipeline models.

        Preconditions:
            - Model weights are available (downloaded or cached).

        Postconditions:
            - Extractor, scorer, and renderer are loaded and ready.
        """
        self._extractor = Extractor()
        self._scorer = Scorer()
        self._renderer = Renderer()

    def extract(self, sentence: str) -> list[Claim]:
        """Extract checkable claims from a transcript sentence.

        Args:
            sentence: The current transcript sentence to extract
                claims from.

        Returns:
            A list of ``Claim`` objects. Empty when no checkable
            claims are found.

        Preconditions:
            - ``sentence`` is a non-empty string.

        Postconditions:
            - Each returned ``Claim`` has non-empty text.
        """
        return self._extractor.extract(sentence)

    def check(self, claim: str, snippets: list[str]) -> CheckResult:
        """Score a claim against search results and assign a label.

        Always returns a ``CheckResult`` so downstream stages can
        surface dropped cases (no evidence, opinion) instead of
        silently discarding them. The distinguishing signal lives on
        ``label``:

        - ``"no-evidence"`` — the search returned nothing usable; the
          scorer never ran. ``evidence`` is empty and ``scored.scores``
          is empty.
        - ``"opinion"`` — the scorer ran but produced no support or
          contradiction signal. ``render()`` will short-circuit and
          emit a minimal verdict for both of these labels.
        - One of the remaining eight factual labels — normal path.

        Does not render an explanation — call ``render()`` after
        validation to produce the final verdict.

        Args:
            claim: The factual claim text to verify.
            snippets: Evidence snippets from the search API.

        Returns:
            A ``CheckResult`` whose ``label`` reflects either a
            normal verdict or a dropped-case marker.

        Preconditions:
            - ``claim`` is a non-empty string.

        Postconditions:
            - ``label`` is one of the factual verdict strings,
              ``"opinion"``, or ``"no-evidence"``.
            - For ``"no-evidence"`` labels, ``scored.scores`` is empty.
        """
        scored = self._scorer.score(claim, snippets)
        if scored is None:
            return CheckResult(
                claim=claim,
                label="no-evidence",
                evidence="",
                scored=ScoredClaim(claim=claim, scores=[]),
            )

        label, best_snippet = label_claim(scored.scores)

        return CheckResult(
            claim=claim,
            label=label,
            evidence=best_snippet,
            scored=scored,
        )

    def render(self, result: CheckResult) -> Verdict:
        """Generate a human-readable explanation for a check result.

        For factual results the returned ``explanation`` concatenates
        the LLM-generated natural-language verdict with a short
        aggregation summary of the evidence that drove the label,
        separated by a ``-----`` divider. For dropped cases
        (``"no-evidence"`` or ``"opinion"``) the LLM is bypassed and
        the explanation is a short static notice — the claim is still
        surfaced to the user, but no expensive rendering is spent on
        a claim that could not be fact-checked.

        Call this after ``check()`` and any validation steps.

        Args:
            result: A ``CheckResult`` from ``check()``.

        Returns:
            A ``Verdict`` with a non-empty explanation.

        Preconditions:
            - ``result`` is a valid ``CheckResult`` from ``check()``.

        Postconditions:
            - The ``Verdict`` has a non-empty explanation.
            - Factual verdicts include a ``-----`` divider followed
              by an evidence-aggregation summary line.
            - Dropped verdicts (``"no-evidence"`` / ``"opinion"``)
              carry a short static notice instead of an LLM output.
        """
        if result.label == "no-evidence":
            return Verdict(
                claim=result.claim,
                label=result.label,
                evidence=result.evidence,
                explanation=(
                    "Dropped — the search returned no usable evidence "
                    "snippets for this claim."
                ),
            )
        if result.label == "opinion":
            return Verdict(
                claim=result.claim,
                label=result.label,
                evidence=result.evidence,
                explanation=(
                    "Dropped — this claim was classified as opinion "
                    "and not fact-checked."
                ),
            )

        explanation = self._renderer.render(
            result.claim, result.label, result.evidence,
        )
        summary = self._summarize_evidence(result.scored)

        return Verdict(
            claim=result.claim,
            label=result.label,
            evidence=result.evidence,
            explanation=f"{explanation}\n-----\n{summary}",
        )

    def _summarize_evidence(self, scored: ScoredClaim) -> str:
        """Count how each evidence snippet contributed to the label.

        Bucketizes every snippet into exactly one category — strong
        support, strong contradict, weak support, weak contradict,
        or neutral — using the same ``0.9`` / ``0.3`` thresholds
        ``label_claim`` uses so the summary and the final label
        describe the same evidence the same way.

        Args:
            scored: The per-snippet NLI scores carried on the
                ``CheckResult``.

        Returns:
            A one-line summary such as
            ``"based on 5 snippets: 1 strong support, 4 neutral"``.

        Preconditions:
            - ``scored.scores`` is a list of ``EvidenceScore`` values.

        Postconditions:
            - Returned string is non-empty.
            - Does not mutate ``scored``.
        """
        scores = scored.scores
        n = len(scores)
        if n == 0:
            return "based on 0 snippets"

        strong_support = 0
        strong_contradict = 0
        weak_support = 0
        weak_contradict = 0
        neutral = 0
        for s in scores:
            if s.support > STRONG_SIGNAL:
                strong_support += 1
            elif s.contradict > STRONG_SIGNAL:
                strong_contradict += 1
            elif s.support > WEAK_SIGNAL and s.support >= s.contradict:
                weak_support += 1
            elif s.contradict > WEAK_SIGNAL:
                weak_contradict += 1
            else:
                neutral += 1

        parts: list[str] = []
        if strong_support:
            parts.append(f"{strong_support} strong support")
        if strong_contradict:
            parts.append(f"{strong_contradict} strong contradict")
        if weak_support:
            parts.append(f"{weak_support} weak support")
        if weak_contradict:
            parts.append(f"{weak_contradict} weak contradict")
        if neutral:
            parts.append(f"{neutral} neutral")

        snippet_word = "snippet" if n == 1 else "snippets"
        return f"based on {n} {snippet_word}: " + ", ".join(parts)

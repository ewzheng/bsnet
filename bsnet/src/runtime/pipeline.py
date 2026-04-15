"""Fact-checking pipeline that strings together extraction, scoring, and rendering.

Provides a single entry point for running text through the
fact-checking pipeline. The three stages (extract, check, render)
are exposed separately so the orchestrator can insert validation
or other logic between them.
"""

from dataclasses import dataclass

from bsnet.src.model._common import label_claim
from bsnet.src.model.extractor import Extractor
from bsnet.src.model.renderer import Renderer
from bsnet.src.model.scorer import Scorer
from bsnet.src.utils.outputs import Claim, ScoredClaim


@dataclass
class CheckResult:
    """Output of the scoring and labeling stage.

    Produced by ``Pipeline.check()``. Contains the label and
    evidence but no explanation yet — that comes from ``render()``.
    """

    claim: str
    label: str
    evidence: str
    scored: ScoredClaim


@dataclass
class Verdict:
    """Final output of the fact-checking pipeline for a single claim.

    Produced by ``Pipeline.render()`` after validation has passed.
    Contains everything the UI needs to display a fact-check result.
    """

    claim: str
    label: str
    evidence: str
    explanation: str


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

    def extract(self, text: str) -> list[Claim]:
        """Extract checkable claims from transcript text.

        The orchestrator calls this first, then sends the queries
        to the search API.

        Args:
            text: Raw transcript text, optionally with context
                prepended by the orchestrator.

        Returns:
            A list of ``Claim`` objects with text and search queries.

        Preconditions:
            - ``text`` is a non-empty string.

        Postconditions:
            - Each returned ``Claim`` has at least one query string.
        """
        return self._extractor.extract(text)

    def check(self, claim: str, snippets: list[str]) -> CheckResult | None:
        """Score a claim against search results and assign a label.

        Does not render an explanation — call ``render()`` after
        validation to produce the final verdict.

        Args:
            claim: The factual claim text to verify.
            snippets: Evidence snippets from the search API.

        Returns:
            A ``CheckResult`` with the label and best evidence, or
            ``None`` if no snippets were provided or the claim was
            classified as opinion.

        Preconditions:
            - ``claim`` is a non-empty string.

        Postconditions:
            - If returned, ``label`` is one of the nine verdict strings.
            - Returns ``None`` for opinion claims or empty snippets.
        """
        scored = self._scorer.score(claim, snippets)
        if scored is None:
            return None

        label, best_snippet = label_claim(scored.scores)

        if label == "opinion":
            return None

        return CheckResult(
            claim=claim,
            label=label,
            evidence=best_snippet,
            scored=scored,
        )

    def render(self, result: CheckResult) -> Verdict:
        """Generate a human-readable explanation for a check result.

        Call this after ``check()`` and any validation steps.

        Args:
            result: A ``CheckResult`` from ``check()``.

        Returns:
            A ``Verdict`` with the full explanation.

        Preconditions:
            - ``result`` is a valid ``CheckResult`` from ``check()``.

        Postconditions:
            - The ``Verdict`` has a non-empty explanation.
        """
        explanation = self._renderer.render(
            result.claim, result.label, result.evidence,
        )

        return Verdict(
            claim=result.claim,
            label=result.label,
            evidence=result.evidence,
            explanation=explanation,
        )

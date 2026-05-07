"""Fact-checking pipeline that strings together extraction, scoring, and rendering.

Provides a single entry point for running text through the
fact-checking pipeline. The three stages (extract, check, render)
are exposed separately so the orchestrator can insert validation
or other logic between them.
"""

from bsnet.src.model._common import STRONG_SIGNAL, WEAK_SIGNAL, label_claim
from bsnet.src.model.embedder import get_embedder
from bsnet.src.model.extractor import Extractor
from bsnet.src.model.renderer import Renderer
from bsnet.src.model.scorer import Scorer
from bsnet.src.utils.outputs import (
    CheckResult,
    Claim,
    EvidenceSnippet,
    ScoredClaim,
    Verdict,
)


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

        Eagerly loads the search-side MiniLM embedder too — it
        normally lazy-loads on the first claim's relevance-filter
        call, which slips a ~1s weight-load into the first verdict's
        latency. Pulling it forward here matches the rest of the
        pipeline's load-once-at-startup posture.

        Preconditions:
            - Model weights are available (downloaded or cached).

        Postconditions:
            - Extractor, scorer, renderer, and embedder are loaded
              and ready.
        """
        print("Loading extractor (Qwen3.5-0.8B GGUF) …")
        self._extractor = Extractor()
        print("Loading scorer (DeBERTa-v3-base-mnli-fever-anli) …")
        self._scorer = Scorer()
        print("Loading renderer (Qwen3.5-0.8B GGUF) …")
        self._renderer = Renderer()
        print("Loading embedder (MiniLM-L6-v2) …")
        get_embedder()

    def extract(self, sentence: str, context: str = "") -> list[Claim]:
        """Extract checkable claims from a transcript sentence.

        Args:
            sentence: The current transcript sentence to extract
                claims from.
            context: Concatenated prior sentences from the transcript
                buffer. Used only to resolve pronouns and implicit
                references in ``sentence``; facts that appear only
                in the context are not re-extracted. Empty string
                (the default) means no context is available — first
                sentence in a stream, or tests that don't care.

        Returns:
            A list of ``Claim`` objects. Empty when no checkable
            claims are found.

        Preconditions:
            - ``sentence`` is a non-empty string.

        Postconditions:
            - Each returned ``Claim`` has non-empty text.
        """
        return self._extractor.extract(sentence, context=context)

    def check(
        self,
        claim: str,
        snippets: list[str] | list[EvidenceSnippet],
    ) -> CheckResult:
        """Score a claim against search results and assign a label.

        Accepts either bare strings (legacy / test-fixture path) or
        ``EvidenceSnippet`` objects carrying the source URL alongside
        the body. URLs ride through to ``EvidenceScore.url`` so the
        renderer can surface citations; string inputs leave the URL
        empty.

        Always returns a ``CheckResult`` so downstream stages can
        surface dropped cases instead of silently discarding them.
        The distinguishing signal lives on ``label``:

        - ``"no-evidence"`` — the search returned nothing usable; the
          scorer never ran. ``evidence`` is empty and ``scored.scores``
          is empty. ``render()`` short-circuits with a static notice.
        - One of the eight factual labels from ``label_claim`` —
          normal path.

        Does not render an explanation — call ``render()`` after
        validation to produce the final verdict.

        Args:
            claim: The factual claim text to verify.
            snippets: Evidence snippets from the search API. Either a
                list of plain text strings or a list of
                ``EvidenceSnippet`` (text + url) — the two are not
                mixed.

        Returns:
            A ``CheckResult`` whose ``label`` reflects either a
            normal verdict or ``"no-evidence"``.

        Preconditions:
            - ``claim`` is a non-empty string.

        Postconditions:
            - ``label`` is one of the factual verdict strings or
              ``"no-evidence"``.
            - For ``"no-evidence"`` labels, ``scored.scores`` is empty.
            - When inputs are ``EvidenceSnippet``, each
              ``EvidenceScore.url`` matches its source snippet's URL.
        """
        if snippets and isinstance(snippets[0], EvidenceSnippet):
            texts = [s.text for s in snippets]
            urls = [s.url for s in snippets]
        else:
            texts = list(snippets)
            urls = [""] * len(snippets)

        scored = self._scorer.score(claim, texts)
        if scored is None:
            return CheckResult(
                claim=claim,
                label="no-evidence",
                evidence="",
                scored=ScoredClaim(claim=claim, scores=[]),
            )

        for es, url in zip(scored.scores, urls):
            es.url = url

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
        separated by a ``-----`` divider. For the ``"no-evidence"``
        case the LLM is bypassed and the explanation is a short static
        notice — the claim is still surfaced to the user, but no
        expensive rendering is spent on a claim that could not be
        fact-checked.

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
            - ``"no-evidence"`` verdicts carry a short static notice
              instead of an LLM output.
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

        explanation = self._renderer.render(
            result.claim, result.label, result.evidence,
        )
        summary = self._summarize_evidence(result.scored)
        citations = self._format_citations(result.scored)

        body = f"{explanation}\n-----\n{summary}"
        if citations:
            body = f"{body}\n{citations}"

        return Verdict(
            claim=result.claim,
            label=result.label,
            evidence=result.evidence,
            explanation=body,
        )

    def _format_citations(self, scored: ScoredClaim) -> str:
        """Build a numbered citations block from snippet URLs.

        Walks ``scored.scores`` in order and emits one ``[i] url``
        line per snippet that has a non-empty URL. Returns an empty
        string when no snippet carries a URL — typical for the legacy
        string-input path or the placeholder default search — so the
        verdict body stays unchanged for those callers.

        Args:
            scored: The per-snippet NLI scores carried on the
                ``CheckResult``.

        Returns:
            A multi-line string like ``"[1] https://...\\n[2] ..."``
            or an empty string when no URLs are present.

        Preconditions:
            - ``scored.scores`` is a list of ``EvidenceScore`` values.

        Postconditions:
            - Returned string has no trailing newline.
            - Does not mutate ``scored``.
        """
        lines: list[str] = []
        for i, s in enumerate(scored.scores, 1):
            if s.url:
                lines.append(f"[{i}] {s.url}")
        return "\n".join(lines)

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

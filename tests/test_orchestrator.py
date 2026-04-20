"""Unit tests for the streaming orchestrator's stage wiring.

Exercise the full extract → search → check → validate → render flow
against a fake ``Pipeline`` and injectable dummy search and validate
callables. No model weights or audio hardware are required; each test
runs in well under a second.
"""

import time
from collections.abc import Callable

import pytest

from bsnet.src.runtime.orchestrator import Orchestrator
from bsnet.src.utils.outputs import (
    CheckResult,
    Claim,
    EvidenceScore,
    ScoredClaim,
    Verdict,
)


# ── Test doubles ─────────────────────────────────────────────────────────────


class FakePipeline:
    """In-memory stand-in for ``Pipeline`` used to avoid model loads.

    Records every call to each stage method so tests can assert on
    what the orchestrator actually fed into the pipeline. Default
    behavior for each method is deterministic and covers the common
    cases; callers may override any stage by passing a callable to
    the constructor.
    """

    def __init__(
        self,
        extract_fn: Callable[[str], list[Claim]] | None = None,
        check_fn: Callable[[str, list[str]], CheckResult | None] | None = None,
        render_fn: Callable[[CheckResult], Verdict] | None = None,
    ) -> None:
        """Construct a fake pipeline with optional stage overrides.

        Args:
            extract_fn: Custom extract implementation. Defaults to a
                one-claim-per-sentence stub.
            check_fn: Custom check implementation. Defaults to a
                ``true``-label result when snippets exist, ``None``
                otherwise.
            render_fn: Custom render implementation. Defaults to a
                verdict whose explanation echoes the claim text.

        Preconditions:
            - Overrides, if supplied, match the pipeline method
              signatures.

        Postconditions:
            - Recording lists (``extract_calls`` etc.) are empty.
        """
        self.extract_calls: list[str] = []
        self.check_calls: list[tuple[str, list[str]]] = []
        self.render_calls: list[CheckResult] = []
        self._extract_fn = extract_fn
        self._check_fn = check_fn
        self._render_fn = render_fn

    def extract(self, sentence: str) -> list[Claim]:
        """Record the call and return claims per the override or default.

        Preconditions:
            - ``sentence`` is a string.

        Postconditions:
            - The sentence has been appended to ``extract_calls``.
        """
        self.extract_calls.append(sentence)
        if self._extract_fn is not None:
            return self._extract_fn(sentence)
        return [Claim(text=f"claim:{sentence}", queries=[sentence])]

    def check(
        self, claim: str, snippets: list[str],
    ) -> CheckResult | None:
        """Record the call and return a scored result or ``None``.

        Preconditions:
            - ``claim`` is a string.

        Postconditions:
            - The call has been appended to ``check_calls``.
        """
        self.check_calls.append((claim, list(snippets)))
        if self._check_fn is not None:
            return self._check_fn(claim, snippets)
        if not snippets:
            return None
        evidence = snippets[0]
        return CheckResult(
            claim=claim,
            label="true",
            evidence=evidence,
            scored=ScoredClaim(
                claim=claim,
                scores=[EvidenceScore(
                    snippet=evidence, support=1.0, contradict=0.0, neutral=0.0,
                )],
            ),
        )

    def render(self, result: CheckResult) -> Verdict:
        """Record the call and return a verdict per override or default.

        Preconditions:
            - ``result`` is a valid ``CheckResult``.

        Postconditions:
            - The call has been appended to ``render_calls``.
        """
        self.render_calls.append(result)
        if self._render_fn is not None:
            return self._render_fn(result)
        return Verdict(
            claim=result.claim,
            label=result.label,
            evidence=result.evidence,
            explanation=f"explained:{result.claim}",
        )


def _echo_search(queries: list[str]) -> list[str]:
    """Return one snippet per non-empty query, echoing the query text.

    Preconditions:
        - ``queries`` is a list of strings.

    Postconditions:
        - Does not perform any I/O.
    """
    return [f"snippet for {q}" for q in queries if q]


def _pass_validate(_: CheckResult) -> bool:
    """Validate stub that accepts every result.

    Preconditions:
        - Called with a ``CheckResult``.

    Postconditions:
        - Always returns ``True``.
    """
    return True


# ── Smoke tests ──────────────────────────────────────────────────────────────


def test_end_to_end_produces_verdict() -> None:
    """A complete sentence should flow through every stage to a verdict.

    Preconditions:
        - The orchestrator is freshly constructed with a FakePipeline.

    Postconditions:
        - Exactly one verdict is produced.
        - Its label, evidence, and explanation reflect the fake
          pipeline's default behavior.
    """
    fake = FakePipeline()
    orch = Orchestrator(
        pipeline=fake,
        search_fn=_echo_search,
        validate_fn=_pass_validate,
    )

    verdicts = list(orch.run(iter(["The sky is blue."])))

    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.claim == "claim:The sky is blue."
    assert v.label == "true"
    assert v.evidence == "snippet for The sky is blue."
    assert v.explanation == "explained:claim:The sky is blue."


def test_sentences_flow_to_extract_in_order() -> None:
    """Every sentence emitted by the buffer should reach extract in order.

    Preconditions:
        - Chunk contains three terminated sentences.

    Postconditions:
        - The fake pipeline's extract recorded each sentence exactly
          once in the order the buffer produced them.
    """
    fake = FakePipeline()
    orch = Orchestrator(pipeline=fake, search_fn=_echo_search)

    list(orch.run(iter(["First. Second. Third."])))

    assert fake.extract_calls == ["First.", "Second.", "Third."]


def test_empty_chunk_source_yields_no_verdicts() -> None:
    """An empty iterable should drain cleanly and produce no output.

    Preconditions:
        - The chunk source is an empty iterator.

    Postconditions:
        - The verdict iterator exits without raising or yielding.
    """
    fake = FakePipeline()
    orch = Orchestrator(pipeline=fake, search_fn=_echo_search)

    verdicts = list(orch.run(iter([])))

    assert verdicts == []


def test_validate_false_drops_result_before_render() -> None:
    """Results for which validate_fn returns False should not reach render.

    Preconditions:
        - ``validate_fn`` rejects claims whose text contains ``bad``.

    Postconditions:
        - Only the accepted claim produces a verdict.
        - The rejected claim was scored but never rendered.
    """
    fake = FakePipeline()

    def reject_bad(result: CheckResult) -> bool:
        return "bad" not in result.claim.lower()

    orch = Orchestrator(
        pipeline=fake,
        search_fn=_echo_search,
        validate_fn=reject_bad,
    )

    verdicts = list(orch.run(iter(["Good sentence. Bad one."])))

    rendered_claims = [v.claim for v in verdicts]
    assert "claim:Good sentence." in rendered_claims
    assert "claim:Bad one." not in rendered_claims
    assert len(verdicts) == 1

    # Both sentences reached check; only one reached render.
    checked_claims = [claim for (claim, _snips) in fake.check_calls]
    rendered = [r.claim for r in fake.render_calls]
    assert "claim:Bad one." in checked_claims
    assert "claim:Bad one." not in rendered


def test_search_stage_runs_in_parallel() -> None:
    """Slow search calls should overlap via the thread pool.

    Preconditions:
        - Four claims flow into the search stage.
        - ``search_fn`` sleeps 100ms per call.
        - ``search_workers`` is at least 4.

    Postconditions:
        - Total wall time is closer to one slot than four (loose
          tolerance to absorb thread-startup jitter).
    """
    delay = 0.1

    def slow_search(queries: list[str]) -> list[str]:
        time.sleep(delay)
        return [f"snippet for {q}" for q in queries if q]

    fake = FakePipeline()
    orch = Orchestrator(
        pipeline=fake,
        search_fn=slow_search,
        search_workers=4,
    )

    t0 = time.perf_counter()
    verdicts = list(orch.run(iter(["A. B. C. D."])))
    elapsed = time.perf_counter() - t0

    assert len(verdicts) == 4
    assert elapsed < delay * 3, (
        f"expected parallel search (~{delay:.2f}s); took {elapsed:.2f}s"
    )


def test_exception_in_extract_propagates_after_drain() -> None:
    """An exception raised in extract should surface on the consumer.

    Preconditions:
        - ``extract_fn`` raises unconditionally.

    Postconditions:
        - Iterating the verdict stream raises the original exception.
    """

    def bad_extract(sentence: str) -> list[Claim]:
        del sentence
        raise RuntimeError("extract kaboom")

    fake = FakePipeline(extract_fn=bad_extract)
    orch = Orchestrator(pipeline=fake, search_fn=_echo_search)

    with pytest.raises(RuntimeError, match="extract kaboom"):
        list(orch.run(iter(["One. Two."])))


def test_exception_in_source_propagates_after_prior_verdicts() -> None:
    """Exceptions from the chunk source surface after buffered verdicts.

    Preconditions:
        - The source yields a terminated sentence, then raises.

    Postconditions:
        - The verdict for the first sentence is yielded first.
        - Iterating further raises the original exception.
    """

    def bad_source():
        """Yield one chunk then raise."""
        yield "Hello."
        raise RuntimeError("source kaboom")

    fake = FakePipeline()
    orch = Orchestrator(pipeline=fake, search_fn=_echo_search)

    it = orch.run(bad_source())
    first = next(it)
    assert first.claim == "claim:Hello."

    with pytest.raises(RuntimeError, match="source kaboom"):
        next(it)

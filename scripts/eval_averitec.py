"""Open-retrieval AVeriTeC evaluation for the bsnet pipeline.

Streams the AVeriTeC dev split via HuggingFace ``datasets`` (Arrow-
mapped, no RAM blowup) and pushes each claim through the bsnet
pipeline stages directly (``extract → search → check → validate →
render``) using the production search backend and validator. The
heavy ``Pipeline`` is constructed once and shared across rows so
model weights load exactly once per eval run.

Direct stage calls instead of ``Orchestrator`` give us per-stage
visibility — the trace dump records extracted sub-claims, retrieved
snippets, per-snippet NLI scores, the validator's decision, and the
rendered explanation. The orchestrator only adds streaming-throughput
threading on top of the same stages; for sequential per-row eval
that threading is not load-bearing, and bypassing it lets the eval
emit a full pipeline trace alongside the headline numbers.

Two output files are written:

- ``--samples-out``: a small curated qualitative-review subset
  (default 10 rows, miss-first interleaved with correct rows).
- ``--trace-out``: one record per evaluated row capturing the full
  per-stage pipeline state, suitable for offline scorer / labeler
  / renderer analysis.

Predicted bsnet verdicts are folded into AVeriTeC's 4-class
taxonomy (``Supported`` / ``Refuted`` / ``Conflicting Evidence/
Cherrypicking`` / ``Not Enough Evidence``). Rows that produce no
verdict (extractor yielded nothing or validator dropped every
claim) collapse to ``Not Enough Evidence``, matching what the live
system would surface to the user.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from bsnet.src.runtime.pipeline import Pipeline
from bsnet.src.utils.outputs import Verdict
from bsnet.src.utils.search import get_search_snippets
from bsnet.src.validation.validator import Validator

# Map bsnet's nine emitted labels (eight verdicts plus
# ``"no-evidence"``) onto AVeriTeC's four gold classes. The folding
# loses bsnet's "mostly" / "partially" nuance, but those bins are an
# internal confidence signal — AVeriTeC reports a single coarse
# class per claim, so the eval has to as well.
_LABEL_MAP: dict[str, str] = {
    "true": "Supported",
    "mostly true": "Supported",
    # ``partially true`` / ``partially false`` come from bsnet's
    # max-pool fallback when neither side reaches ``MODERATE_SIGNAL``
    # — weak-but-directional signal, not "evidence on both sides."
    # AVeriTeC's ``Conflicting Evidence/Cherrypicking`` requires
    # actual evidence in both directions, so map the partials to the
    # directional class their max-pool side indicates.
    "partially true": "Supported",
    "partially false": "Refuted",
    "false": "Refuted",
    "mostly false": "Refuted",
    "mixture": "Conflicting Evidence/Cherrypicking",
    "unproven": "Not Enough Evidence",
    "no-evidence": "Not Enough Evidence",
}

# Fixed ordering for the printed confusion breakdown so the diagonal
# is in the same place across runs.
_AVERITEC_CLASSES: tuple[str, ...] = (
    "Supported",
    "Refuted",
    "Conflicting Evidence/Cherrypicking",
    "Not Enough Evidence",
)


def _terminate(claim: str) -> str:
    """Ensure a claim ends with sentence-terminating punctuation.

    The transcript buffer splits on ``[.!?]\\s+`` boundaries, so a
    claim missing terminal punctuation would sit in the buffer until
    flush rather than being emitted alongside its peers. AVeriTeC
    claims usually already terminate, but we defend against the rare
    exception.

    Args:
        claim: Raw claim text from the AVeriTeC ``claim`` column.

    Returns:
        The same text with a trailing ``.`` appended when no
        sentence-ender is already present.

    Preconditions:
        - ``claim`` is a non-empty string.

    Postconditions:
        - Returned string ends with one of ``.``, ``!``, or ``?``.
        - Does not mutate the input.
    """
    s = claim.strip()
    if s[-1] not in ".!?":
        s += "."
    return s


def _evaluate_row(
    pipeline: Pipeline,
    validator: Validator,
    claim: str,
) -> tuple[list[Verdict], list[dict[str, object]]]:
    """Run one AVeriTeC claim through the full bsnet pipeline.

    Calls the pipeline stages directly (``extract → search → check →
    validate → render``) instead of routing through ``Orchestrator``
    so each stage's intermediate state is visible for the trace
    dump. The orchestrator only adds streaming-throughput threading
    on top of the same stages; for sequential eval that threading is
    not load-bearing, and bypassing it lets us capture per-snippet
    NLI scores, the validator's decision, and the rendered
    explanation alongside the verdict. Bypassing the
    ``TranscriptBuffer`` is a small behavioral diff: the extractor
    sees the full input rather than buffer-pre-split sentences,
    which is arguably a better fit for AVeriTeC's already-atomic
    claims than the live transcript-streaming path.

    Args:
        pipeline: Loaded ``Pipeline`` instance shared across rows.
        validator: ``Validator`` whose ``evaluate_check_result`` is
            applied at the validate stage. ``"no-evidence"`` results
            bypass the validator and short-circuit at render — same
            rule the orchestrator's validate-loop applies.
        claim: AVeriTeC claim text to evaluate.

    Returns:
        ``(verdicts, sub_traces)`` where ``verdicts`` are the
        rendered ``Verdict`` objects (one per extracted sub-claim
        that survived the validator) and ``sub_traces`` is a per-
        sub-claim list of dicts capturing the full pipeline state
        (extracted text, snippets, per-snippet NLI scores, label,
        best evidence, validator decision, rendered explanation).

    Preconditions:
        - ``claim`` is a non-empty string.
        - A network connection is available for the search stage.

    Postconditions:
        - Does not mutate ``pipeline`` or ``validator``.
        - Performs live HTTP requests via ``get_search_snippets``.
        - ``len(sub_traces) == len(pipeline.extract(claim))``.
        - ``len(verdicts) <= len(sub_traces)``.
    """
    extracted = pipeline.extract(_terminate(claim))

    sub_traces: list[dict[str, object]] = []
    verdicts: list[Verdict] = []

    for sub in extracted:
        snippets = get_search_snippets(sub.text)
        check_result = pipeline.check(sub.text, snippets)

        # Mirror the orchestrator's validate-loop rule: ``no-evidence``
        # bypasses the validator and short-circuits at render.
        if check_result.label == "no-evidence":
            validator_passed = True
        else:
            validator_passed = validator.evaluate_check_result(check_result)

        verdict: Verdict | None = None
        if validator_passed:
            verdict = pipeline.render(check_result)
            verdicts.append(verdict)

        scores = (
            check_result.scored.scores
            if check_result.scored is not None
            else []
        )
        sub_traces.append({
            "extracted_claim": sub.text,
            "snippet_count": len(snippets),
            "snippets": list(snippets),
            "scores": [
                {
                    "snippet": s.snippet,
                    "support": float(s.support),
                    "contradict": float(s.contradict),
                    "neutral": float(s.neutral),
                }
                for s in scores
            ],
            "label": check_result.label,
            "best_evidence": check_result.evidence,
            "validator_passed": validator_passed,
            "rendered_explanation": (
                verdict.explanation if verdict is not None else None
            ),
        })

    return verdicts, sub_traces


def _coarse_label(verdict: Verdict | None) -> str:
    """Fold a bsnet ``Verdict`` into AVeriTeC's 4-class taxonomy.

    Args:
        verdict: The verdict the orchestrator emitted for this row,
            or ``None`` when no verdict was emitted (extractor
            produced no claims or validator dropped every claim).

    Returns:
        One of the four AVeriTeC class strings.

    Preconditions:
        - When ``verdict`` is not ``None``, ``verdict.label`` is a
          key of ``_LABEL_MAP``.

    Postconditions:
        - Returns ``"Not Enough Evidence"`` when ``verdict`` is
          ``None``.
        - Does not mutate ``verdict``.
    """
    if verdict is None:
        return "Not Enough Evidence"
    return _LABEL_MAP[verdict.label]


def _select_samples(
    records: list[dict[str, object]],
    n: int,
) -> list[dict[str, object]]:
    """Pick a balanced subset of records for qualitative review.

    Interleaves misses and correct predictions so the dump is
    diverse: the most informative cases (misses, including dropped
    rows) come first, padded with correct predictions to fill the
    quota. Falls through to whatever's left when one bucket is
    smaller than expected.

    Args:
        records: Per-row eval records, in dataset order.
        n: Maximum number of samples to return.

    Returns:
        Up to ``n`` records, miss-first interleaved.

    Preconditions:
        - Every record has a boolean ``correct`` field.
        - ``n`` is non-negative.

    Postconditions:
        - ``len(result) <= n``.
        - ``len(result) <= len(records)``.
        - Returned records are a subset of the input.
    """
    wrong = [r for r in records if not r["correct"]]
    correct = [r for r in records if r["correct"]]
    out: list[dict[str, object]] = []
    i = 0
    while len(out) < n and (i < len(wrong) or i < len(correct)):
        if i < len(wrong) and len(out) < n:
            out.append(wrong[i])
        if i < len(correct) and len(out) < n:
            out.append(correct[i])
        i += 1
    return out


def main() -> None:
    """Run open-retrieval evaluation on a subset of AVeriTeC dev.

    Preconditions:
        - The HuggingFace ``datasets`` cache is reachable, or the
          target repo has already been downloaded.
        - A network connection is available for the search stage.
        - Pipeline model weights are available (downloaded or cached).

    Postconditions:
        - Per-row predictions, accuracy, and a confusion breakdown
          are printed to stdout.
        - A JSON dump of qualitative-review samples is written to
          ``--samples-out``.
    """
    parser = argparse.ArgumentParser(
        description="Open-retrieval AVeriTeC evaluation for bsnet.",
    )
    parser.add_argument(
        "--repo",
        default="pminervini/averitec",
        help="HuggingFace dataset repo. Defaults to a public mirror.",
    )
    parser.add_argument(
        "--split",
        default="dev",
        help="Dataset split. Test labels are held out by AVeriTeC.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum rows to evaluate. Defaults to the full split "
            "(~500 for AVeriTeC dev). Pass an integer to subset."
        ),
    )
    parser.add_argument(
        "--samples-out",
        type=Path,
        default=Path("scripts/averitec_samples.json"),
        help="Where to write the qualitative-review sample dump.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="How many samples to dump for LLM qualitative review.",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=Path("scripts/averitec_trace.json"),
        help=(
            "Where to write the per-row pipeline trace dump for "
            "offline analysis (one record per evaluated row, full "
            "per-stage state)."
        ),
    )
    args = parser.parse_args()

    ds = load_dataset(args.repo, split=args.split)
    if args.limit is not None and args.limit < len(ds):
        ds = ds.select(range(args.limit))

    pipeline = Pipeline()
    validator = Validator()

    confusion: Counter[tuple[str, str]] = Counter()
    n_correct = 0
    n_dropped = 0
    records: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []

    for i, row in enumerate(ds):
        claim = row["claim"].strip()
        gold = row["label"]
        justification = row.get("justification") or ""

        verdicts, sub_traces = _evaluate_row(pipeline, validator, claim)
        # Pick the first verdict when the extractor split a compound
        # claim into sub-claims. Aggregating across sub-claims is
        # tempting but bsnet's labeler already runs per-claim, so the
        # first is the most representative single-row signal.
        verdict = verdicts[0] if verdicts else None
        if verdict is None:
            n_dropped += 1

        predicted = _coarse_label(verdict)
        confusion[(gold, predicted)] += 1
        is_correct = predicted == gold
        if is_correct:
            n_correct += 1

        marker = "ok" if is_correct else "MISS"
        print(
            f"[{i + 1}/{len(ds)}] [{marker}] gold={gold!r} "
            f"pred={predicted!r} verdicts={len(verdicts)}"
        )

        records.append({
            "claim": claim,
            "gold_label": gold,
            "gold_justification": justification,
            "predicted_label": predicted,
            "bsnet_label": verdict.label if verdict is not None else None,
            "evidence": verdict.evidence if verdict is not None else None,
            "explanation": verdict.explanation if verdict is not None else None,
            "verdict_emitted": verdict is not None,
            "correct": is_correct,
        })

        traces.append({
            "row_index": i,
            "input_claim": claim,
            "gold_label": gold,
            "gold_justification": justification,
            "predicted_label": predicted,
            "correct": is_correct,
            "extracted_claims": sub_traces,
        })

    samples = _select_samples(records, args.num_samples)
    args.samples_out.parent.mkdir(parents=True, exist_ok=True)
    with args.samples_out.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    args.trace_out.parent.mkdir(parents=True, exist_ok=True)
    with args.trace_out.open("w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)

    n_total = len(records)
    accuracy = n_correct / n_total if n_total else 0.0
    print()
    print(f"accuracy: {n_correct}/{n_total} = {accuracy:.4f}")
    print(f"dropped (no verdict): {n_dropped}/{n_total}")
    print(f"qualitative samples dumped to: {args.samples_out}")
    print(f"full pipeline trace dumped to: {args.trace_out}")
    print()
    print("confusion (gold -> predicted):")
    for gold in _AVERITEC_CLASSES:
        for predicted in _AVERITEC_CLASSES:
            count = confusion.get((gold, predicted), 0)
            if count:
                print(f"  {gold!r} -> {predicted!r}: {count}")


if __name__ == "__main__":
    main()

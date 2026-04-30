"""Open-retrieval AVeriTeC evaluation for the bsnet pipeline.

Streams the AVeriTeC dev split via HuggingFace ``datasets`` (Arrow-
mapped, no RAM blowup) and pushes the claims through one
``Orchestrator`` instance — same usage pattern as the live CLI in
``bsnet.src.__main__`` — wired up with the production search and
validator. Emitted verdicts are mapped back to AVeriTeC rows by
exact ``claim`` text and folded into AVeriTeC's 4-class taxonomy
(``Supported`` / ``Refuted`` / ``Conflicting Evidence/Cherrypicking``
/ ``Not Enough Evidence``). Rows that produce no verdict (extractor
yielded nothing or validator dropped every claim) collapse to
``Not Enough Evidence``, matching what the live system would
surface for the user.

Open-retrieval search is the bottleneck stage, so the default
``--limit`` keeps a single run short — bump it once a small subset
looks healthy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from bsnet.src.runtime.orchestrator import Orchestrator
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
    "false": "Refuted",
    "mostly false": "Refuted",
    "mixture": "Conflicting Evidence/Cherrypicking",
    "partially true": "Conflicting Evidence/Cherrypicking",
    "partially false": "Conflicting Evidence/Cherrypicking",
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
        default=50,
        help="Maximum rows to evaluate. Open-retrieval is slow.",
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
    args = parser.parse_args()

    ds = load_dataset(args.repo, split=args.split)
    if args.limit is not None and args.limit < len(ds):
        ds = ds.select(range(args.limit))

    # One pass over the Arrow table: builds the verdict→row lookup,
    # caches the human justifications for the qualitative dump, and
    # produces the chunk list for the orchestrator. Arrow keeps RAM
    # bounded across the dev split.
    gold_by_claim: dict[str, str] = {}
    justification_by_claim: dict[str, str] = {}
    chunks: list[str] = []
    for row in ds:
        claim = row["claim"].strip()
        gold_by_claim[claim] = row["label"]
        justification_by_claim[claim] = (row.get("justification") or "")
        chunks.append(_terminate(claim))

    orch = Orchestrator(
        search_fn=get_search_snippets,
        validate_fn=Validator().evaluate_check_result,
    )

    verdicts_by_claim: dict[str, Verdict] = {}
    n_unmatched = 0
    for verdict in orch.run(iter(chunks)):
        key = verdict.claim.strip()
        if key not in gold_by_claim:
            # Extractor reformulated the claim text — can't attribute
            # back to a row. Logged separately so the headline
            # accuracy reflects only attributable verdicts.
            n_unmatched += 1
            print(
                f"unmatched verdict: claim={verdict.claim!r} "
                f"label={verdict.label!r}"
            )
            continue
        verdicts_by_claim[key] = verdict
        predicted = _LABEL_MAP[verdict.label]
        gold = gold_by_claim[key]
        marker = "ok" if predicted == gold else "MISS"
        print(f"[{marker}] gold={gold!r} pred={predicted!r}")

    confusion: Counter[tuple[str, str]] = Counter()
    n_correct = 0
    n_dropped = 0
    records: list[dict[str, object]] = []
    for claim, gold in gold_by_claim.items():
        verdict = verdicts_by_claim.get(claim)
        if verdict is None:
            n_dropped += 1
        predicted = _coarse_label(verdict)
        confusion[(gold, predicted)] += 1
        is_correct = predicted == gold
        if is_correct:
            n_correct += 1
        records.append({
            "claim": claim,
            "gold_label": gold,
            "gold_justification": justification_by_claim.get(claim, ""),
            "predicted_label": predicted,
            "bsnet_label": verdict.label if verdict is not None else None,
            "evidence": verdict.evidence if verdict is not None else None,
            "explanation": verdict.explanation if verdict is not None else None,
            "verdict_emitted": verdict is not None,
            "correct": is_correct,
        })

    samples = _select_samples(records, args.num_samples)
    args.samples_out.parent.mkdir(parents=True, exist_ok=True)
    with args.samples_out.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    n_total = len(gold_by_claim)
    accuracy = n_correct / n_total if n_total else 0.0
    print()
    print(f"accuracy: {n_correct}/{n_total} = {accuracy:.4f}")
    print(f"dropped (no verdict): {n_dropped}/{n_total}")
    print(f"unmatched (extractor reformulation): {n_unmatched}")
    print(f"qualitative samples dumped to: {args.samples_out}")
    print()
    print("confusion (gold -> predicted):")
    for gold in _AVERITEC_CLASSES:
        for predicted in _AVERITEC_CLASSES:
            count = confusion.get((gold, predicted), 0)
            if count:
                print(f"  {gold!r} -> {predicted!r}: {count}")


if __name__ == "__main__":
    main()

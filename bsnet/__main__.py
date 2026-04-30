"""Command-line entry point for the bsnet package.

When invoked via ``python -m bsnet`` the module opens a live
microphone transcription stream and drives the orchestrator from it,
printing each rendered ``Verdict`` as the fact-checking pipeline
produces it. Importing and calling ``main()`` without arguments is a
no-op and exists to keep the smoke test independent of audio hardware
and model weights.
"""

import sys
from collections.abc import Iterable

from bsnet.src.runtime.orchestrator import Orchestrator
from bsnet.src.utils.outputs import Verdict
from bsnet.src.utils.search import get_search_snippets as search_fn
from bsnet.src.validation.validator import Validator

_LABEL_EMOJI: dict[str, str] = {
    "true": "✔️",
    "mostly true": "✔️",
    "partially true": "🟡",
    "mixture": "〰️",
    "partially false": "🟠",
    "mostly false": "❌",
    "false": "❌",
    "unproven": "❓",
    "no-evidence": "⛔",
}


def _format_verdict(verdict: Verdict) -> str:
    """Render a ``Verdict`` as a terminal-friendly block.

    Builds a multi-line string with an emoji + uppercase-label header,
    the claim on its own line, an optional evidence line when an
    evidence quote is present, a blank line, and finally the
    ``Pipeline.render`` output (which already carries the ``-----``
    divider and aggregation summary for factual verdicts).

    Args:
        verdict: The verdict to render.

    Returns:
        A formatted multi-line string ready for ``print``.

    Preconditions:
        - ``verdict.label`` is a non-empty string.
        - ``verdict.explanation`` is a non-empty string.

    Postconditions:
        - Does not mutate ``verdict``.
        - Returned string contains no trailing newline.
    """
    emoji = _LABEL_EMOJI.get(verdict.label, "📄")
    lines = [
        f"{emoji} {verdict.label.upper()}",
        f"Claim: {verdict.claim}",
    ]
    if verdict.evidence:
        lines.append(f"Evidence: {verdict.evidence}")
    lines.append("")
    lines.append(verdict.explanation)
    return "\n".join(lines)


def main(chunk_source: Iterable[str] | None = None) -> int:
    """Run the default bsnet command-line entry point.

    Args:
        chunk_source: Optional iterable of transcript chunks. When
            provided, each chunk is pushed through the orchestrator
            and the resulting verdicts are printed to stdout. When
            ``None``, the function returns immediately so importing
            and invoking ``main()`` from tests does not require audio
            hardware or model weights.

    Returns:
        Process exit code for the CLI invocation. ``0`` on normal
        completion.

    Preconditions:
        - If ``chunk_source`` is provided, it yields ``str`` chunks.

    Postconditions:
        - When ``chunk_source`` is ``None``, returns ``0`` without
          loading any models or performing I/O beyond imports.
        - When ``chunk_source`` is provided, every rendered verdict
          has been printed before the function returns.
    """
    if chunk_source is None:
        return 0

    # Windows terminals default to cp1252 which cannot encode emoji.
    # Force UTF-8 on stdout and degrade unknown glyphs to "?" instead
    # of letting the first emoji printed crash the whole session.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    orch = Orchestrator(
        search_fn=search_fn,
        validate_fn=Validator().evaluate_check_result,
    )
    for verdict in orch.run(chunk_source):
        print(_format_verdict(verdict))
        print()
    return 0


def run() -> int:
    """Console-script entry point: open the mic and run the pipeline."""
    from bsnet.src.utils.transcription import listen

    return main(listen())


if __name__ == "__main__":
    raise SystemExit(run())

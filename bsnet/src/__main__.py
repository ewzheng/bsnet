"""Command-line entry point for the bsnet package.

When invoked via ``python -m bsnet.src`` the module opens a live
microphone transcription stream and drives the orchestrator from it,
printing each rendered ``Verdict`` as the fact-checking pipeline
produces it. Importing and calling ``main()`` without arguments is a
no-op and exists to keep the smoke test independent of audio hardware
and model weights.
"""

from collections.abc import Iterable

from bsnet.src.runtime.orchestrator import Orchestrator


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

    orch = Orchestrator()
    for verdict in orch.run(chunk_source):
        print(f"[{verdict.label}] {verdict.claim}")
        print(f"  evidence: {verdict.evidence}")
        print(f"  explanation: {verdict.explanation}")
    return 0


if __name__ == "__main__":
    from bsnet.src.utils.transcription import listen

    raise SystemExit(main(listen()))

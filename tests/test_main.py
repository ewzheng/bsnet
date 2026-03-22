"""Smoke tests for the bsnet command-line entry point."""

from bsnet.src.__main__ import main


def test_main_returns_success_exit_code() -> None:
    """Verify the default CLI entry point returns a success exit code.

    Preconditions:
        - The `bsnet` package is importable from the repository root.

    Postconditions:
        - Confirms `main()` returns `0`.
    """
    assert main() == 0

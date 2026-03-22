"""Command-line entry point for the bsnet package."""


def main() -> int:
    """Run the default bsnet command-line entry point.

    Returns:
        Process exit code for the CLI invocation.

    Preconditions:
        - The module is executed in a valid Python runtime.

    Postconditions:
        - Returns `0` to indicate the default CLI path completed successfully.
        - Does not mutate package state.
    """
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

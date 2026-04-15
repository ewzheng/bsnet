# llm.md — bsnet

## Project overview

bsnet is a Python 3.11 project that includes a BERT-based model pipeline with real-time voice transcription. It runs on Windows with a conda environment (`environment.yml`). Key dependencies include PyTorch (CPU), Transformers, faster-whisper, and pyaudio.

## Repository structure

```
bsnet/
  __init__.py
  src/
    __init__.py
    __main__.py          # CLI entry point
    model/               # BERT / ML model code
    runtime/             # Execution flow
    validation/          # Input validation
    utils/
      __init__.py        # Re-exports from transcription
      transcription.py   # Real-time voice-to-text via faster-whisper
tests/
  test_main.py           # Smoke tests
.llm/
  documentation.MD       # Documentation & docstring standards
  blackboard.MD          # Multi-agent coordination blackboard
```

## Code standards

Follow the documentation standard defined in `.llm/documentation.MD`:

- Every function requires full type annotations (Python 3.11 style).
- Every function requires a Google-style docstring with `Preconditions` and `Postconditions` sections.
- Use `X | None` over `Optional[X]`.
- Use built-in generics (`list[str]`, `dict[str, int]`).
- Follow PEP 8 naming and PEP 257 docstring structure.
- Private helpers are not exempt from documentation requirements.

## Build and test

```bash
# Activate the conda environment
conda activate bsnet

# Run tests
pytest
```

## Agent coordination

When multiple agents are working in the repo simultaneously, use `.llm/blackboard.MD` to claim file ownership and communicate status. See that file for the update format.

## Conventions

- Entry point: `bsnet/src/__main__.py` -> `main()`
- Package imports: `from bsnet.src.<module> import ...`
- Test files go in `tests/` and follow `test_*.py` naming.
- Config: `pytest.ini` at root, `environment.yml` for conda.

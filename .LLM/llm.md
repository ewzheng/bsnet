# llm.md — bsnet

FOR AGENTS: Link your AGENTS.MD (i.e. Claude.MD, Agents.MD, etc.) to this directory.

## Project overview

bsnet is a Python 3.11 real-time fact-checking pipeline. A live microphone
stream is transcribed, claims are extracted, each claim is searched for
evidence, an NLI model scores the evidence, and a natural-language verdict
is rendered per claim. The runtime is a six-stage streaming pipeline with
per-stage threading so different claims occupy different stages concurrently.

Runs on Windows via a conda environment (`environment.yml`). Models:

- **Extractor + renderer**: `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf` via
  `llama-cpp-python`.
- **Scorer**: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` via
  `transformers`, loaded with `bitsandbytes` 8-bit quantization on CPU.
- **Relevance reranker**: `sentence-transformers/all-MiniLM-L6-v2` via
  `transformers` (mean-pool + L2-norm).
- **Transcription**: `faster-whisper` (base, int8 on CPU by default).

Key dependencies: `torch` (CPU), `transformers`, `accelerate`,
`bitsandbytes`, `llama-cpp-python`, `faster-whisper`, `webrtcvad-wheels`,
`ddgs`, `pytest`.

## Repository structure

```
bsnet/
  __init__.py
  src/
    __init__.py
    __main__.py            CLI entry point
    model/
      _common.py           shared loaders, device resolution, label_claim
      extractor.py         Qwen-based claim extractor
      scorer.py            DeBERTa NLI scorer with bnb int8
      renderer.py          Qwen-based verdict explanation
    runtime/
      pipeline.py          extract / check / render wrapper
      orchestrator.py      six-stage threaded pipeline driver
    utils/
      __init__.py          re-exports from transcription
      buffer.py            transcript buffer with rolling context window
      transcription.py     faster-whisper + VAD microphone loop
      search.py            multi-backend search (Google, DuckDuckGo,
                           Wikipedia REST) with MiniLM rerank
      outputs.py           Claim, EvidenceScore, ScoredClaim,
                           CheckResult, Verdict dataclasses
    validation/
      __init__.py          downstream CheckResult validation hook
tests/
  test_main.py             smoke tests (no model loads)
  test_orchestrator.py     stage-wiring tests with a FakePipeline
  test_extractor.py        extractor integration tests
  test_scorer.py           scorer integration tests
  test_renderer.py         renderer integration tests
  test_pipeline.py         end-to-end with real search, requires network
.llm/
  llm.md                   this file — project overview and standards
  documentation.MD         docstring standard
  blackboard.MD            multi-agent coordination blackboard
```

## Runtime architecture

Six stages wired together in `bsnet/src/runtime/orchestrator.py`:

```
intake → extract → search → check → validate → render
```

Each stage runs on its own daemon thread with a bounded `queue.Queue`
between adjacent stages. The intake stage pulls transcript chunks from
the source (typically `listen()`), pushes them through a
`TranscriptBuffer`, and emits `(context, sentence)` tuples. The extractor
receives both so it can resolve pronouns against the rolling context.
Search is the only fan-out stage (thread pool, I/O-bound); model stages
are single-worker because the underlying model instances aren't safe for
concurrent inference.

On shutdown a sentinel cascades down the pipeline so every worker
terminates cleanly. Stage exceptions are captured and re-raised to the
consumer after downstream queues drain.

## Code standards

Follow the documentation standard in [`.llm/documentation.MD`](documentation.MD):

- Every function requires full type annotations (Python 3.11 style).
- Every function requires a Google-style docstring with `Preconditions`
  and `Postconditions` sections.
- Use `X | None` over `Optional[X]`.
- Use built-in generics (`list[str]`, `dict[str, int]`).
- Follow PEP 8 naming and PEP 257 docstring structure.
- Private helpers are not exempt from documentation requirements.

## Build and test

```bash
conda activate bsnet
pytest
```

`test_orchestrator.py` and `test_main.py` run without model loads or
network. The other test files are integration tests and download model
weights on first run; `test_pipeline.py::test_orchestrator_end_to_end_with_real_search`
additionally requires network access.

On Windows, set `KMP_DUPLICATE_LIB_OK=TRUE` if the OpenMP runtime
conflicts between torch and llama-cpp at import time.

## Agent coordination

When multiple agents work in the repo simultaneously, use
[`.llm/blackboard.MD`](blackboard.MD) to claim file ownership and
communicate status. See that file for the update format.

## Conventions

- CLI entry: `python -m bsnet.src` → `bsnet/src/__main__.py::main()`.
- Package imports: `from bsnet.src.<module> import ...`.
- Tests live in `tests/` and follow `test_*.py` naming.
- Config: `pytest.ini` at root, `environment.yml` for conda.
- Environment variables:
  - `BSNET_GPU_LAYERS` — GGUF layers to offload to GPU (default `0`, CPU only).
  - `BSNET_QUANTIZE_SCORER` — set to `0` / `false` / `no` to load the
    NLI scorer in fp32 instead of bnb int8. Use when `bitsandbytes`
    backend autodetect picks a binary that doesn't ship for the host
    (e.g. ROCm-flavored torch on Windows tries to load
    `libbitsandbytes_rocm*.dll`, which bnb doesn't publish for Windows).
  - `KMP_DUPLICATE_LIB_OK` — Windows OpenMP conflict workaround.

# bsnet

Real-time fact-checking for live speech. Transcribes microphone audio, extracts
factual claims as they're spoken, searches the web for evidence, and prints a
verdict for each claim as the pipeline produces it.

## What it does

Speech goes in, verdicts come out. The runtime is a six-stage streaming
pipeline — transcription chunks get buffered into sentences, claims are pulled
out with a small LLM, each claim is searched across multiple free backends,
the evidence is scored with an NLI model, and a final natural-language
explanation is rendered per verdict. Stages run concurrently so different
claims occupy different stages at the same time.

Labels produced: `true`, `mostly true`, `partially true`, `mixture`,
`partially false`, `mostly false`, `false`, `unproven`, `no-evidence`.

## Install

Requires a working conda install.

```bash
conda env create -f environment.yml
conda activate bsnet
```

The first run downloads model weights on demand:

- Extractor + renderer: `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf` via `llama-cpp-python`
- Scorer: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` via `transformers`
- Reranker: `sentence-transformers/all-MiniLM-L6-v2`
- Transcription: faster-whisper base model

A microphone is required for the default CLI path. Tests do not require
audio hardware.

## Usage

Launch the live CLI:

```bash
python -m bsnet.src
```

The process starts listening, transcribes speech, and prints each verdict
as the pipeline finishes it. Output looks like:

```
TRUE
Claim: The unemployment rate dropped to 3.4% in January 2023.
Evidence: Jobs report January 2023: Payrolls increased by 517,000 ...

This claim is rated true because the January 2023 jobs report ...
-----
based on 6 snippets: 2 strong support, 4 neutral
```

To drive the pipeline from a non-audio source (tests, scripted inputs),
import `main` and pass an iterable of transcript-chunk strings:

```python
from bsnet.src.__main__ import main
main(iter(["The unemployment rate dropped to 3.4% in January 2023."]))
```

## Configuration

Environment variables read at import time:

| Variable | Effect |
|---|---|
| `BSNET_GPU_LAYERS` | Number of GGUF layers to offload to GPU. `0` (default) runs the LLM stages fully on CPU. `-1` offloads everything. Requires a CUDA / Metal / ROCm build of `llama-cpp-python`. |
| `KMP_DUPLICATE_LIB_OK` | Set to `TRUE` on Windows if multiple OpenMP runtimes collide at startup (common with torch + llama-cpp). |

Transcription defaults (`bsnet/src/utils/transcription.py`): whisper size
`base`, device `cpu`, compute type `int8`, VAD aggressiveness `2`. Edit
those constants to swap the model or target a GPU.

## Tests

```bash
pytest
```

Test layout:

- `tests/test_orchestrator.py` — stage-wiring tests with a fake pipeline, no
  model loads.
- `tests/test_extractor.py`, `tests/test_scorer.py`, `tests/test_renderer.py`
  — integration tests that download and run the real models.
- `tests/test_pipeline.py` — end-to-end with live search; requires network.

## Repository layout

```
bsnet/src/
  __main__.py          CLI entry point
  model/               extractor, scorer, renderer, shared model helpers
  runtime/             orchestrator and pipeline
  utils/               transcript buffer, search, transcription, output types
  validation/          downstream validation hook
tests/                 pytest suite
.llm/                  project standards and agent conventions
environment.yml        conda env definition
```

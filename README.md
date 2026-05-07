# BSNet

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

You can create the runtime environment with conda or venv:

```bash
conda env create -f environment.yml
conda activate bsnet
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
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
python -m bsnet
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
from bsnet.__main__ import main
main(iter(["The unemployment rate dropped to 3.4% in January 2023."]))
```

## Configuration

By default the runtime auto-detects available hardware:

- **GGUF models** (extractor, renderer): all layers offload to GPU when
  `llama-cpp-python` was built with a GPU backend [^rocm]; otherwise the
  models run on CPU.
- **Transformers models** (scorer, reranker): land on CUDA when available,
  ROCm when available, CPU otherwise (via
  `torch.cuda.is_available()`).
- **Scorer quantization**: bitsandbytes 8-bit on CUDA / CPU, fp32 on ROCm
  (bnb's 8-bit kernels on AMD GPUs are unreliable and frequently fall back
  to CPU silently — fp32 on GPU is faster than bnb-on-CPU).

Environment variables override the auto-detect defaults:

| Variable | Effect |
|---|---|
| `BSNET_GPU_LAYERS` | Number of GGUF layers to offload to GPU. Unset (default) auto-offloads all layers on a GPU-enabled `llama-cpp-python` build, or stays on CPU otherwise. Set explicitly: `-1` offloads everything, `0` forces CPU, positive integer offloads that many layers (partial offload for low VRAM). |
| `BSNET_QUANTIZE_SCORER` | Scorer DeBERTa quantization. Unset (default) auto-resolves to bnb int8 on CUDA / CPU and fp32 on ROCm. Set explicitly: `0` / `false` / `no` forces fp32, `1` / `true` / `yes` forces int8. |
| `BSNET_DEBUG_TRANSCRIPTION` | Set to `1` / `true` / `yes` / `on` to surface per-utterance running-count and chunk-emission debug lines (`[ NN chars] ...`, `── chunk ──` blocks) from `bsnet/src/utils/transcription.py`. Default off — the live CLI stays clean and only verdicts print. |
| `KMP_DUPLICATE_LIB_OK` | Set to `TRUE` on Windows if multiple OpenMP runtimes collide at startup (common with torch + llama-cpp). |

Transcription defaults (`bsnet/src/utils/transcription.py`): whisper size
`base`, device `cpu`, compute type `int8`, VAD aggressiveness `2`. Edit
those constants to swap the model or target a GPU.

[^rocm]: Default `pip install llama-cpp-python` builds a CPU-only wheel.
    For an **AMD ROCm** build (Linux / WSL with a recent ROCm dev toolkit
    installed), force a clean rebuild against HIPBLAS with the right
    architecture flag — `gfx1100` for RDNA3 (7900 series), `gfx1030` for
    RDNA2 (6000 series), `gfx90a` for MI200, etc.:
    ```bash
    export CMAKE_ARGS="-DGGML_HIP=on -DAMDGPU_TARGETS=gfx1100"
    pip install --no-cache-dir --force-reinstall llama-cpp-python
    ```
    For an **NVIDIA CUDA** build, use `-DGGML_CUDA=on` (no architecture
    flag needed — cmake autodetects from the installed CUDA toolkit):
    ```bash
    export CMAKE_ARGS="-DGGML_CUDA=on"
    pip install --no-cache-dir --force-reinstall llama-cpp-python
    ```
    Verify the rebuilt wheel actually has GPU support before running:
    ```bash
    python -c "from llama_cpp import llama_supports_gpu_offload; print(llama_supports_gpu_offload())"
    # expect: True
    ```
    Two common pitfalls: `set CMAKE_ARGS=...` is bash-interpreted as a
    shell-local var and not exported to `pip` — must be `export`.
    Putting the venv on a Windows-mounted drive (`/mnt/c`, `/mnt/d`) under
    WSL causes flaky uninstalls that leave half-deleted `~`-prefixed
    directories in `site-packages` and a 5–10× slowdown on dependency
    operations; native Linux paths (`~/...`) are strongly recommended.

## Evaluation

Evaluated on the AVeriTeC dev split (500 claims, EMNLP 2024 FEVER
shared task) via `scripts/eval_averitec.py`. Reported numbers use
the deployable `DeBERTa-v3-base` scorer that runs on consumer CPU
at streaming latency. A `DeBERTa-v3-large` row is included as an
ablation to quantify the size-vs-latency tradeoff.

### Headline metrics

| Metric | DeBERTa-base (deployable) | DeBERTa-large (ablation, ~3× slower) |
|---|---|---|
| Accuracy | 0.416 | 0.464 |
| Macro F1 | 0.288 | 0.329 |
| Weighted F1 | 0.447 | 0.497 |
| Drop rate (predicted NEI from validator-drop) | 23.4% | 25.0% |

### Per-class breakdown (DeBERTa-base)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supported | 0.370 | 0.549 | 0.442 | 122 |
| Refuted | 0.729 | 0.423 | 0.535 | 305 |
| Conflicting Evidence/Cherrypicking | 0.045 | 0.026 | 0.033 | 38 |
| Not Enough Evidence | 0.092 | 0.314 | 0.142 | 35 |

### Comparison to FEVER 2024 shared task

The shared task winners use 70B-class language models for the
veracity step plus multi-stage retrieve-decompose-reason pipelines
that run in batch mode. The BSNet pipeline targets real-time
consumer hardware with veracity models two orders of magnitude
smaller, so the comparison is across operating regimes rather than
within the leaderboard.

| System | Score | Notes |
|---|---|---|
| TUDA_MAI (#1) | 0.724 dev acc / 0.630 AVeriTeC | GPT-4o multi-stage |
| HerO / HUMANE (#2) | 0.752 dev acc / 0.570 AVeriTeC | Llama-3.1-70B veracity (standalone acc) |
| Papelo (#5) | 0.754 dev acc / 0.480 AVeriTeC | T5 + GPT-4o multi-hop |
| **BSNet (this work)** | **0.416 dev acc / 0.288 Macro F1** | DeBERTa-v3-base NLI (184M) + Qwen3.5-0.8B, single-pass, real-time |
| Provided baseline | 0.110 AVeriTeC / 0.230 Macro F1 | BLOOM-7B + BM25 + pretrained BERT |
| Uniform Random | 0.250 dev acc / 0.207 Macro F1 | Performance of a model that predicts randomly |
| Majority Class | 0.610 dev acc / 0.190 Macro F1 | Performance of a model that only predicts 'refuted' |

The official AVeriTeC score combines verdict correctness with
retrieved-evidence METEOR similarity to gold question-answer pairs.
Verdict-only accuracy and F1 are reported here since the streaming
pipeline does not generate question-answer pairs and the
leaderboard metric is not directly comparable.

When compared holistically to pipelines running much heavier models and
multi-stage workflows, we believe the performance of BSNet is quite competitive for its size.

### Reproduce

```bash
python scripts/eval_averitec.py
```

Writes `scripts/averitec_samples.json` (curated qualitative
samples) and `scripts/averitec_trace.json` (per-row per-stage
trace) alongside the printed confusion matrix.

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
bsnet/
  __main__.py          CLI entry point
  src/
    model/             extractor, scorer, renderer, shared model helpers
    runtime/           orchestrator and pipeline
    utils/             transcript buffer, search, transcription, output types
    validation/        downstream validation hook
tests/                 pytest suite
.llm/                  project standards and agent conventions
environment.yml        conda env definition
```

"""Shared constants and utilities for model loading.

Centralizes device selection, model loading, and default model
identifiers so that individual model wrappers stay focused on
inference logic. Supports both HuggingFace transformers models
and GGUF models via llama-cpp-python.
"""

import os
import re

import torch
from llama_cpp import Llama, llama_supports_gpu_offload
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ── Default model identifiers ────────────────────────────────────────────────
EXTRACTOR_MODEL = "bartowski/Qwen_Qwen3.5-0.8B-GGUF"
EXTRACTOR_GGUF_FILE = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
SCORER_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
RENDERER_MODEL = "bartowski/Qwen_Qwen3.5-0.8B-GGUF"
RENDERER_GGUF_FILE = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"

# ── Sampling parameters ──────────────────────────────────────────────────────
# Two profiles: thinking mode (lower presence penalty, lets the model
# reason longer) and non-thinking (higher presence penalty to keep
# output concise). Models that don't produce <think> tags simply get
# the non-thinking profile regardless of the flag.
SAMPLING_THINKING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
}

SAMPLING_NON_THINKING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 2.0,
    "repeat_penalty": 1.0,
}

# Regex to strip <think>...</think> blocks from model output
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# ── NLI aggregation thresholds ───────────────────────────────────────────────
# Shared by ``label_claim`` (for labeling) and the pipeline's evidence
# summary (for post-hoc display). Keep these in one place so tuning
# the labeler automatically retunes the summary too.
#
# ``MODERATE_SIGNAL`` is the count-pool threshold the labeler uses
# when no snippet on either side breaks ``STRONG_SIGNAL`` — important
# for adversarial claims where DeBERTa-base lands two or three
# snippets in the 0.6–0.9 band without anything crossing 0.9 (e.g.
# the post-dedup "Earth is flat" case where two contradicts at 0.67
# and 0.75 should outweigh one stubborn 0.79 support snippet).
# ``MODERATE_SIGNAL`` is intentionally not used by the post-hoc
# summary in ``pipeline._summarize_evidence`` so user-facing labels
# like "strong support" / "weak support" stay anchored to the
# original 0.9 / 0.3 bins.
STRONG_SIGNAL = 0.9
MODERATE_SIGNAL = 0.6
WEAK_SIGNAL = 0.3


def label_claim(scores: list) -> tuple[str, str]:
    """Aggregate NLI scores across all evidence snippets into a verdict.

    Walks a layered ladder of pooling signals from strongest to
    weakest:

    1. **Weak gate.** When neither side reaches ``WEAK_SIGNAL`` the
       claim is "unproven" — evidence didn't meaningfully support or
       contradict it.
    2. **Strong-signal count pooling** (``> STRONG_SIGNAL``). With a
       2:1-or-better ratio in one direction the verdict is "mostly
       true" / "mostly false"; otherwise "mixture". Pure-strong on
       one side resolves to "true" or "mostly true" (or the false
       symmetric pair) depending on whether the opposite side broke
       ``WEAK_SIGNAL``.
    3. **Moderate-signal count pooling** (``> MODERATE_SIGNAL``).
       Same 2:1 ratio logic, used when nothing crossed
       ``STRONG_SIGNAL`` on either side. Closes the failure mode
       where multiple 0.6–0.9 snippets accumulate without any
       individually clearing 0.9 — under the old logic the labeler
       fell straight to ``max_support`` vs ``max_contradict`` and
       lost on a single high-support outlier despite the count
       favoring the other direction.
    4. **Max-pool fallback.** Only when no side reaches even
       ``MODERATE_SIGNAL`` — picks "partially true" or "partially
       false" by which side's peak is higher.

    Also returns the most relevant snippet for the renderer (the one
    with the strongest non-neutral signal).

    Args:
        scores: A list of ``EvidenceScore`` objects from the scorer.

    Returns:
        A tuple of ``(label, best_snippet)`` where label is one of
        ``"true"``, ``"mostly true"``, ``"partially true"``,
        ``"mixture"``, ``"partially false"``, ``"mostly false"``,
        ``"false"``, or ``"unproven"``, and ``best_snippet`` is the
        snippet text with the strongest non-neutral signal.

    Preconditions:
        - ``scores`` is a non-empty list of ``EvidenceScore`` objects.
        - Each element has ``snippet`` (str), ``support`` (float),
          ``contradict`` (float), and ``neutral`` (float) attributes.

    Postconditions:
        - Returns exactly one of the eight label strings.
        - ``best_snippet`` is always a non-empty string from the input.
        - ``scores`` is not mutated.
    """
    best_snippet = scores[0].snippet
    best_signal = -1.0
    max_support = 0.0
    max_contradict = 0.0
    n_strong_support = 0
    n_strong_contradict = 0
    n_moderate_support = 0
    n_moderate_contradict = 0

    for s in scores:
        signal = max(s.support, s.contradict)
        if signal > best_signal:
            best_signal = signal
            best_snippet = s.snippet

        if s.support > max_support:
            max_support = s.support
        if s.contradict > max_contradict:
            max_contradict = s.contradict

        if s.support > STRONG_SIGNAL:
            n_strong_support += 1
        if s.contradict > STRONG_SIGNAL:
            n_strong_contradict += 1
        if s.support > MODERATE_SIGNAL:
            n_moderate_support += 1
        if s.contradict > MODERATE_SIGNAL:
            n_moderate_contradict += 1

    if max_support < WEAK_SIGNAL and max_contradict < WEAK_SIGNAL:
        return ("unproven", best_snippet)

    if n_strong_support > 0 and n_strong_contradict > 0:
        if n_strong_support >= 2 * n_strong_contradict:
            return ("mostly true", best_snippet)
        if n_strong_contradict >= 2 * n_strong_support:
            return ("mostly false", best_snippet)
        return ("mixture", best_snippet)

    if n_strong_support > 0 and n_strong_contradict == 0:
        # Demote to "mostly" only when there's a *moderate-or-higher*
        # opposing snippet. Below MODERATE_SIGNAL is NLI noise on the
        # wrong axis — DeBERTa-base routinely produces 0.3–0.5
        # contradiction values for cleanly true claims and we don't
        # want that noise demoting definitive verdicts.
        if max_contradict >= MODERATE_SIGNAL:
            return ("mostly true", best_snippet)
        return ("true", best_snippet)

    if n_strong_contradict > 0 and n_strong_support == 0:
        if max_support >= MODERATE_SIGNAL:
            return ("mostly false", best_snippet)
        return ("false", best_snippet)

    if n_moderate_support > 0 and n_moderate_contradict > 0:
        if n_moderate_support >= 2 * n_moderate_contradict:
            return ("mostly true", best_snippet)
        if n_moderate_contradict >= 2 * n_moderate_support:
            return ("mostly false", best_snippet)
        return ("mixture", best_snippet)

    if n_moderate_support > 0 and n_moderate_contradict == 0:
        # No moderate-or-higher contradiction snippet exists by
        # construction — what's left on the contradict side is NLI
        # noise that wouldn't change a fact-checker's mind. Distinguish
        # *redundant* moderate support (multiple snippets, definitive
        # "true") from *single* moderate support ("mostly true"
        # because there isn't enough redundancy to be definitive).
        if n_moderate_support >= 2:
            return ("true", best_snippet)
        return ("mostly true", best_snippet)

    if n_moderate_contradict > 0 and n_moderate_support == 0:
        if n_moderate_contradict >= 2:
            return ("false", best_snippet)
        return ("mostly false", best_snippet)

    if max_support > max_contradict:
        return ("partially true", best_snippet)
    return ("partially false", best_snippet)


def load_llm(
    repo: str,
    gguf_file: str,
    n_ctx: int = 2048,
) -> Llama:
    """Load a GGUF model via llama-cpp-python.

    Auto-detects GPU offload by default: if the installed
    ``llama-cpp-python`` was built with a GPU backend (CUDA / Metal /
    HIPBLAS) — as reported by ``llama_supports_gpu_offload()`` — every
    layer is offloaded; otherwise the model loads on CPU. Mirrors the
    auto-detect behavior of ``resolve_device`` for the transformers
    side so a ROCm / CUDA box runs on GPU without env-var juggling.

    Override via ``BSNET_GPU_LAYERS``: ``-1`` offloads every layer,
    ``0`` forces CPU, a positive integer offloads that many layers
    (useful for partial offload on small VRAM).

    Args:
        repo: HuggingFace repo ID containing the GGUF file.
        gguf_file: Name of the GGUF file within the repo.
        n_ctx: Context window size in tokens.

    Returns:
        A ``Llama`` instance ready for inference.

    Preconditions:
        - ``repo`` is a valid HuggingFace repo with GGUF files.
        - ``gguf_file`` exists in the repo.
        - ``BSNET_GPU_LAYERS``, if set, is a valid integer.

    Postconditions:
        - The model is loaded and ready for chat completion calls.
        - When ``BSNET_GPU_LAYERS`` is unset, every layer is offloaded
          on GPU-enabled builds and the model stays on CPU otherwise.
        - When ``BSNET_GPU_LAYERS`` is set, that exact value is used
          regardless of build support; an ignored offload on a
          CPU-only build is a no-op handled by llama-cpp itself.
    """
    raw = os.environ.get("BSNET_GPU_LAYERS")
    if raw is not None:
        n_gpu_layers = int(raw)
    else:
        n_gpu_layers = -1 if llama_supports_gpu_offload() else 0
    return Llama.from_pretrained(
        repo,
        filename=gguf_file,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def generate_llm(
    model: Llama,
    prompt: str,
    thinking: bool = False,
    max_tokens: int = 128,
    **kwargs: float,
) -> str:
    """Run a chat completion on a GGUF model.

    Uses generic sampling defaults that work across model families.
    Any additional kwargs override the defaults (e.g. temperature).
    If the model produces ``<think>`` blocks they are stripped
    automatically.

    Args:
        model: A loaded ``Llama`` instance.
        prompt: The user message to send.
        thinking: Hint that the prompt benefits from chain-of-thought.
            Models that support ``<think>`` tags (e.g. Qwen) will use
            them; others will simply reason inline. The flag does not
            change sampling — use kwargs for that.
        max_tokens: Upper bound on generated tokens.
        **kwargs: Sampling overrides (temperature, top_p, etc.).

    Returns:
        The model's response text with any thinking blocks removed.

    Preconditions:
        - ``model`` is a loaded ``Llama`` instance.
        - ``prompt`` is a non-empty string.

    Postconditions:
        - Returned string contains no ``<think>`` tags.
        - Does not mutate the model.
    """
    params = dict(SAMPLING_THINKING if thinking else SAMPLING_NON_THINKING)
    params.update(kwargs)

    messages = [{"role": "user", "content": prompt}]

    result = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        **params,
    )

    text = result["choices"][0]["message"]["content"]
    text = _THINK_RE.sub("", text).strip()
    return text


def load_model(
    model_name: str,
    task: str = "seq2seq",
    device: str = "auto",
    quantization_config: object | None = None,
) -> tuple:
    """Load a HuggingFace transformers model and tokenizer.

    For GGUF models, use ``load_llm`` instead.

    When ``quantization_config`` is supplied, it's forwarded to
    ``from_pretrained`` — the transformers integration handles device
    placement and the explicit ``model.to(device)`` call is skipped
    because quantized weights are already placed by the quantization
    backend (e.g. bitsandbytes dispatches to CUDA, XPU, or CPU based
    on runtime detection and the config's settings).

    Args:
        model_name: HuggingFace model identifier to load.
        task: ``"seq2seq"`` for encoder-decoder models (T5, BART),
            ``"causal"`` for decoder-only models (Gemma, LLaMA), or
            ``"classification"`` for encoder models (NLI, BERT).
        device: ``"auto"`` to detect the best available device,
            or an explicit device string (``"cpu"``, ``"cuda"``).
            Ignored when ``quantization_config`` is provided.
        quantization_config: Optional ``BitsAndBytesConfig`` (or any
            other HF quantization config) to pass through to
            ``from_pretrained``. ``None`` (the default) loads full
            precision.

    Returns:
        A tuple of ``(tokenizer, model, device_str)`` ready for
        inference. ``device_str`` is the resolved device string for
        full-precision loads; for quantized loads it reflects where
        the quantized backend placed the weights.

    Raises:
        ValueError: If ``task`` is not a recognized value.

    Preconditions:
        - ``model_name`` is a valid HuggingFace model identifier.
        - ``task`` is one of ``"seq2seq"``, ``"causal"``, or
          ``"classification"``.

    Postconditions:
        - The model is in eval mode.
        - For full-precision loads, the model is placed on the
          resolved device.
        - For quantized loads, placement is owned by the quantization
          backend.
        - The tokenizer matches the model architecture.
    """
    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs: dict = {}
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    if task == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    elif task == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    elif task == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs,
        )
    else:
        raise ValueError(f"Unknown task: {task!r}. Use 'seq2seq', 'causal', or 'classification'.")

    if quantization_config is None:
        model.to(resolved)
    model.eval()
    return tokenizer, model, resolved


def resolve_device(preferred: str = "auto") -> str:
    """Determine the best available torch device.

    Checks for CUDA availability and falls back to CPU. Passing an
    explicit device string (e.g. ``"cpu"`` or ``"cuda"``) bypasses
    auto-detection and returns that string directly.

    Args:
        preferred: ``"auto"`` to detect the best device, or an explicit
            device string to use as-is.

    Returns:
        A device string accepted by ``torch.device``.

    Preconditions:
        - PyTorch is installed and importable.

    Postconditions:
        - Returns ``"cuda"`` only when ``torch.cuda.is_available()``
          is ``True`` and ``preferred`` is ``"auto"``.
        - Returns the literal ``preferred`` value when it is not
          ``"auto"``.
    """
    if preferred != "auto":
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"

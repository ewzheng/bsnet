"""Shared constants and utilities for model loading.

Centralizes device selection, model loading, and default model
identifiers so that individual model wrappers stay focused on
inference logic. Supports both HuggingFace transformers models
and GGUF models via llama-cpp-python.
"""

import os
import re

import torch
from llama_cpp import Llama
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ── Default model identifiers ────────────────────────────────────────────────
EXTRACTOR_MODEL = "bartowski/Qwen_Qwen3.5-0.8B-GGUF"
EXTRACTOR_GGUF_FILE = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
SCORER_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
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
STRONG_SIGNAL = 0.9
WEAK_SIGNAL = 0.3


def label_claim(scores: list) -> tuple[str, str]:
    """Aggregate NLI scores across all evidence snippets into a verdict.

    Uses three pooling signals in layered order:

    1. **Max pooling** on ``support`` and ``contradict`` gates the
       coarse categories. When the strongest score in either direction
       is below ``WEAK_SIGNAL`` the claim is "unproven" — evidence
       didn't meaningfully support or contradict it.
    2. **Count pooling** of strong-signal snippets (those above
       ``STRONG_SIGNAL``) catches cases with any confident evidence.
    3. **Ratio-aware logic** between the two counts decides between
       "mostly true", "mixture", and "mostly false" when *both*
       directions have strong snippets: a 2:1 ratio or better in either
       direction tilts toward "mostly" rather than "mixture". This
       keeps a single noisy-contradict snippet from flipping a
       confidently supported claim all the way to "mixture".

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

    if max_support < WEAK_SIGNAL and max_contradict < WEAK_SIGNAL:
        return ("unproven", best_snippet)

    if n_strong_support > 0 and n_strong_contradict > 0:
        if n_strong_support >= 2 * n_strong_contradict:
            return ("mostly true", best_snippet)
        if n_strong_contradict >= 2 * n_strong_support:
            return ("mostly false", best_snippet)
        return ("mixture", best_snippet)

    if n_strong_support > 0 and n_strong_contradict == 0:
        if max_contradict > WEAK_SIGNAL:
            return ("mostly true", best_snippet)
        return ("true", best_snippet)

    if n_strong_contradict > 0 and n_strong_support == 0:
        if max_support > WEAK_SIGNAL:
            return ("mostly false", best_snippet)
        return ("false", best_snippet)

    if max_support > max_contradict:
        return ("partially true", best_snippet)
    return ("partially false", best_snippet)


def load_llm(
    repo: str,
    gguf_file: str,
    n_ctx: int = 2048,
) -> Llama:
    """Load a GGUF model via llama-cpp-python.

    Reads ``BSNET_GPU_LAYERS`` from the environment to control GPU
    offload. Set it to ``-1`` to offload every layer to GPU (requires
    a CUDA / Metal / ROCm build of ``llama-cpp-python``), or a positive
    integer to offload that many layers. Defaults to ``0`` (CPU only),
    preserving the original behavior.

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
        - Layers are offloaded per ``BSNET_GPU_LAYERS`` when the
          installed ``llama-cpp-python`` supports the target backend;
          ignored on a CPU-only build.
    """
    n_gpu_layers = int(os.environ.get("BSNET_GPU_LAYERS", "0"))
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
) -> tuple:
    """Load a HuggingFace transformers model and tokenizer.

    For GGUF models, use ``load_llm`` instead.

    Args:
        model_name: HuggingFace model identifier to load.
        task: ``"seq2seq"`` for encoder-decoder models (T5, BART),
            ``"causal"`` for decoder-only models (Gemma, LLaMA), or
            ``"classification"`` for encoder models (NLI, BERT).
        device: ``"auto"`` to detect the best available device,
            or an explicit device string (``"cpu"``, ``"cuda"``).

    Returns:
        A tuple of ``(tokenizer, model, device_str)`` ready for
        inference.

    Raises:
        ValueError: If ``task`` is not a recognized value.

    Preconditions:
        - ``model_name`` is a valid HuggingFace model identifier.
        - ``task`` is one of ``"seq2seq"``, ``"causal"``, or
          ``"classification"``.

    Postconditions:
        - The model is in eval mode and placed on the resolved device.
        - The tokenizer matches the model architecture.
    """
    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if task == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif task == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif task == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        raise ValueError(f"Unknown task: {task!r}. Use 'seq2seq', 'causal', or 'classification'.")

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

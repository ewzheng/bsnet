"""Sentence-embedding helper backed by MiniLM.

Provides a process-wide lazy singleton around
``sentence-transformers/all-MiniLM-L6-v2`` and a thin ``embed``
wrapper that mean-pools the token outputs and L2-normalizes each
row, so a downstream dot product with another row yields cosine
similarity directly. The embedder originally lived inline in
``bsnet.src.utils.search`` for one consumer (the relevance filter)
but is now used by smart-window snippet selection too, and is
likely to grow more callers — splitting it out keeps each call site
free of the singleton plumbing and gives future similarity uses
(claim-claim dedup, evidence-evidence dedup, retrieval reranking)
a single canonical entry point.

The MiniLM weights are ~80MB and load in roughly a second on first
call. Subsequent calls reuse the cached instances. ``get_embedder``
and ``embed`` are thread-safe via an internal lock so the search
stage's worker pool can call ``embed`` concurrently after the first
warm load.
"""

import threading

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from bsnet.src.model._common import resolve_device

# Model identifier for the sentence embedder. Any drop-in replacement
# must produce L2-normalized 384-d embeddings of (query, snippets) so
# cosine comes out as a dot product.
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embedder_lock = threading.Lock()
_embedder_tokenizer: AutoTokenizer | None = None
_embedder_model: AutoModel | None = None
_embedder_device: str | None = None


def get_embedder() -> tuple[AutoTokenizer, AutoModel, str]:
    """Return the shared embedder tokenizer + model + device, loading lazily.

    The first call pays the ~1s weight-load cost and the
    ``.to(device)`` placement; subsequent calls return the cached
    singleton instances. Device follows the same auto-detect path as
    the other transformers models via ``resolve_device("auto")`` —
    CUDA / ROCm if visible, CPU otherwise. Thread-safe via
    ``_embedder_lock`` so concurrent search-stage workers cannot
    double-load the model on first warm-up.

    Returns:
        A ``(tokenizer, model, device)`` tuple ready for inference.
        The model is in eval mode and placed on ``device``.

    Preconditions:
        - ``transformers`` and ``torch`` are importable in the
          current process.

    Postconditions:
        - The model is in eval mode after the first call.
        - The model lives on the resolved device after the first
          call.
        - Subsequent calls return the cached instances.
    """
    global _embedder_tokenizer, _embedder_model, _embedder_device
    if _embedder_model is not None:
        return _embedder_tokenizer, _embedder_model, _embedder_device
    with _embedder_lock:
        if _embedder_model is None:
            device = resolve_device("auto")
            tok = AutoTokenizer.from_pretrained(EMBEDDER_MODEL)
            mdl = AutoModel.from_pretrained(EMBEDDER_MODEL)
            mdl.to(device)
            mdl.eval()
            _embedder_tokenizer = tok
            _embedder_model = mdl
            _embedder_device = device
    return _embedder_tokenizer, _embedder_model, _embedder_device


def embed(texts: list[str]) -> torch.Tensor:
    """Encode ``texts`` into L2-normalized mean-pooled sentence embeddings.

    Loads the embedder lazily on the first call (see
    ``get_embedder``). Mean-pools the token embeddings using the
    attention mask to ignore padding, then L2-normalizes each row so
    a downstream dot product with another row yields cosine
    similarity directly. The returned tensor lives on the same
    device the model was placed on.

    Args:
        texts: Non-empty list of strings to encode.

    Returns:
        A ``(N, 384)`` float tensor where each row is the unit-norm
        embedding of the corresponding input. Lives on the
        embedder's device (CUDA / ROCm if available, else CPU).

    Preconditions:
        - ``texts`` is non-empty.

    Postconditions:
        - Each returned row has L2 norm ≈ 1.
        - Does not mutate the loaded model.
    """
    tokenizer, model, device = get_embedder()
    batch = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**batch)
    token_embeds = out.last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (token_embeds * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1)

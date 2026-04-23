"""NLI-based evidence scorer.

Takes a claim and a list of evidence snippets retrieved by the search
API, scores each (claim, snippet) pair for entailment / contradiction /
neutral, and returns all results for downstream aggregation.
"""

import torch

from bsnet.src.model._common import SCORER_MODEL, load_model
from bsnet.src.utils.outputs import EvidenceScore, ScoredClaim


class Scorer:
    """Wraps an NLI cross-encoder for evidence scoring.

    Loads the model once at construction and reuses it for all
    subsequent inference calls.
    """

    # Label indices for the NLI model output logits. MoritzLaurer's
    # DeBERTa-v3-*-mnli-fever-anli-ling-wanli family orders its labels
    # as entailment/neutral/contradiction; the older cross-encoder
    # family used the reverse. Swapping the model requires swapping
    # these constants.
    _LABEL_ENTAILMENT = 0
    _LABEL_NEUTRAL = 1
    _LABEL_CONTRADICTION = 2

    def __init__(self, model_name: str = SCORER_MODEL, device: str = "auto") -> None:
        """Load the NLI cross-encoder model and tokenizer.

        Args:
            model_name: HuggingFace model identifier to load.
            device: ``"auto"`` to detect the best available device,
                or an explicit device string (``"cpu"``, ``"cuda"``).

        Preconditions:
            - ``model_name`` is a valid HuggingFace NLI model identifier
              with 3-class output (contradiction, entailment, neutral).

        Postconditions:
            - The model and tokenizer are loaded and ready for inference.
            - The model is set to eval mode.
        """
        self._tokenizer, self._model, self._device = load_model(
            model_name, task="classification", device=device,
        )

    def score(self, claim: str, snippets: list[str]) -> ScoredClaim | None:
        """Score a claim against all retrieved evidence snippets.

        Runs a single batched NLI forward pass over every
        ``(snippet, claim)`` pair instead of looping one call per
        snippet. The tokenizer pads the pairs into a single tensor
        with shape ``(N, seq_len)``, the model returns logits of
        shape ``(N, 3)``, and softmax is applied row-wise — so one
        forward pass produces probabilities for every snippet at
        once. On CPU this amortizes BLAS and kernel-launch overhead
        across the batch; on GPU it's free.

        Args:
            claim: The factual claim text to verify.
            snippets: Evidence snippets from the search API. May be
                empty if the search returned no results.

        Returns:
            A ``ScoredClaim`` containing scores for every snippet, or
            ``None`` if ``snippets`` is empty.

        Preconditions:
            - ``claim`` is a non-empty string.
            - The model has been loaded via ``__init__``.

        Postconditions:
            - Does not mutate the model or any input arguments.
            - Each score's probabilities sum to approximately 1.0.
            - Returned ``scores`` preserves input order: ``scores[i]``
              corresponds to ``snippets[i]``.
        """
        if not snippets:
            return None

        inputs = self._tokenizer(
            snippets,
            [claim] * len(snippets),
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=1).tolist()

        scores: list[EvidenceScore] = [
            EvidenceScore(
                snippet=snippet,
                support=row[self._LABEL_ENTAILMENT],
                contradict=row[self._LABEL_CONTRADICTION],
                neutral=row[self._LABEL_NEUTRAL],
            )
            for snippet, row in zip(snippets, probs)
        ]

        return ScoredClaim(claim=claim, scores=scores)

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

    # Label indices for the NLI model output logits
    _LABEL_CONTRADICTION = 0
    _LABEL_ENTAILMENT = 1
    _LABEL_NEUTRAL = 2

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

        Runs NLI inference on each (claim, snippet) pair and returns
        all results so the labeler can aggregate across the full
        evidence set.

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
        """
        if not snippets:
            return None

        scores: list[EvidenceScore] = []
        for snippet in snippets:
            probs = self._classify(snippet, claim)
            scores.append(EvidenceScore(
                snippet=snippet,
                support=probs[1],
                contradict=probs[0],
                neutral=probs[2],
            ))

        return ScoredClaim(claim=claim, scores=scores)

    def _classify(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        """Run NLI classification on a single (premise, hypothesis) pair.

        Args:
            premise: The evidence snippet (NLI premise).
            hypothesis: The claim to verify (NLI hypothesis).

        Returns:
            A tuple of ``(contradiction, entailment, neutral)``
            probabilities.

        Preconditions:
            - Both inputs are non-empty strings.
            - The model is loaded and in eval mode.

        Postconditions:
            - Does not mutate model state.
            - Returned probabilities sum to approximately 1.0.
        """
        inputs = self._tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        return (probs[self._LABEL_CONTRADICTION],
                probs[self._LABEL_ENTAILMENT],
                probs[self._LABEL_NEUTRAL])

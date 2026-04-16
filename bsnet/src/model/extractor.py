"""Claim extractor powered by Qwen3.5 via llama-cpp-python.

Two-pass extraction: the first call pulls factual claims from a
sentence, the second generates search keywords for each claim.
Uses thinking mode for deeper reasoning about claim identification.
"""

from bsnet.src.model._common import (
    EXTRACTOR_MODEL,
    EXTRACTOR_GGUF_FILE,
    generate_llm,
    load_llm,
)
from bsnet.src.utils.outputs import Claim


class Extractor:
    """Wraps a GGUF language model for claim extraction and query generation.

    Loads the model once at construction and reuses it for all
    subsequent inference calls. Inference is two-pass: extract claims,
    then generate search keywords for each.
    """

    def __init__(
        self,
        repo: str = EXTRACTOR_MODEL,
        gguf_file: str = EXTRACTOR_GGUF_FILE,
        n_ctx: int = 2048,
    ) -> None:
        """Load the GGUF model.

        Args:
            repo: HuggingFace repo ID containing the GGUF file.
            gguf_file: Name of the GGUF file within the repo.
            n_ctx: Context window size in tokens.

        Preconditions:
            - ``repo`` is a valid HuggingFace repo with GGUF files.

        Postconditions:
            - The model is loaded and ready for inference.
        """
        self._model = load_llm(repo, gguf_file, n_ctx=n_ctx)

    def extract(self, text: str) -> list[Claim]:
        """Run two-pass extraction on an input string.

        Pass 1 (thinking mode) asks the model to identify factual
        claims in the text. Pass 2 generates search keywords for each
        claim. If pass 1 returns nothing, the sentence has no
        checkworthy claims and an empty list is returned.

        Args:
            text: The fully formatted input string, assembled by the
                orchestrator (may include topic, context, and sentence).

        Returns:
            A list of ``Claim`` objects. Empty if no factual claims
            were found.

        Preconditions:
            - ``text`` is a non-empty string.
            - The model has been loaded via ``__init__``.

        Postconditions:
            - Does not mutate the model or input.
            - Each returned ``Claim`` has at least one query string.
        """
        raw_claims = generate_llm(
            self._model,
            "Extract each checkable fact from this text. Write each as "
            "a full sentence with its subject, one per line. Do not "
            "write sentence fragments or bare numbers. Exclude opinions. "
            f"Say \"none\" if there are no facts.\n\n{text}",
            thinking=True,
            max_tokens=256,
            temperature=0.3,
        )
        if not raw_claims.strip() or raw_claims.strip().lower() == "none":
            return []

        claims: list[Claim] = []
        for claim_text in raw_claims.strip().splitlines():
            claim_text = claim_text.strip().lstrip("0123456789.-) ")
            if not claim_text:
                continue
            query = generate_llm(
                self._model,
                "Topic and keywords for this claim, comma-separated."
                f"\n\n{claim_text}",
                thinking=False,
                max_tokens=64,
                temperature=0.0,
            )
            query = query.strip()
            if not query:
                continue
            claims.append(Claim(text=claim_text, queries=[query]))

        return claims

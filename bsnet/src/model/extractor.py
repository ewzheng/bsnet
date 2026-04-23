"""Claim extractor powered by Qwen3.5 via llama-cpp-python.

Single-pass extraction: pulls factual claims from a sentence and
leaves search-query assembly to the orchestrator (which sends the
claim text directly to the search backend). Uses thinking mode for
deeper reasoning about claim identification.
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
        """Run single-pass claim extraction on an input string.

        Asks the model (in thinking mode) to identify factual claims
        in the text and emit one per line. Returns a ``Claim`` per
        emitted line; the downstream search layer uses the claim text
        itself as the search query, which is both faster (no second
        LLM call) and more reliable (key specifics like numbers and
        dates can't be dropped by a flaky keyword-generation pass).

        Args:
            text: The fully formatted input string.

        Returns:
            A list of ``Claim`` objects. Empty if no factual claims
            were found.

        Preconditions:
            - ``text`` is a non-empty string.
            - The model has been loaded via ``__init__``.

        Postconditions:
            - Does not mutate the model or input.
            - Each returned ``Claim`` has a non-empty ``text``.
        """
        raw_claims = generate_llm(
            self._model,
            "Extract each checkable fact from the text below. Write each "
            "as a full sentence with its subject explicitly named, one "
            "per line. If one sentence joins multiple facts with \"and\", "
            "\"but\", \"while\", or a comma, split them into separate "
            "lines — even when the facts share a subject or a time "
            "phrase. Do not write fragments or bare numbers. Exclude "
            "opinions. Say \"none\" if there are no facts.\n\n"
            f"Text: {text}\n"
            "Facts:\n"
            "{fact1}\n"
            "{fact2}",
            thinking=True,
            max_tokens=256,
            temperature=0.3,
        )
        body = raw_claims.strip()
        if body.lower().startswith("facts:"):
            body = body[len("facts:"):].strip()
        if not body or body.lower() == "none":
            return []

        claims: list[Claim] = []
        for claim_text in body.splitlines():
            claim_text = claim_text.strip().lstrip("0123456789.-) ")
            if not claim_text or claim_text.lower() == "none":
                continue
            if claim_text.lower() in ("facts:", "{fact1}", "{fact2}"):
                continue
            claims.append(Claim(text=claim_text))

        return claims

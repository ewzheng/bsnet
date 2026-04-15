"""Verdict renderer powered by Qwen3.5 via llama-cpp-python.

Takes a labeled claim with its supporting evidence and generates
a concise human-readable explanation of the verdict.
Uses non-thinking mode for fast, direct output.
"""


from bsnet.src.model._common import (
    RENDERER_MODEL,
    RENDERER_GGUF_FILE,
    generate_llm,
    load_llm,
)


class Renderer:
    """Wraps a GGUF language model for verdict summarization.

    Loads the model once at construction and reuses it for all
    subsequent inference calls.

    NOTE: ``"opinion"`` labels should be skipped by the caller,
    as they do not have a clear verdict to explain.
    """

    def __init__(
        self,
        repo: str = RENDERER_MODEL,
        gguf_file: str = RENDERER_GGUF_FILE,
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

    def render(self, claim: str, label: str, evidence: str) -> str:
        """Generate a concise verdict explanation.

        Args:
            claim: The original factual claim text.
            label: The verdict label (e.g. ``"true"``, ``"false"``,
                ``"mostly true"``).
            evidence: The best evidence snippet used for scoring.

        Returns:
            A short human-readable summary explaining the verdict.

        Preconditions:
            - ``claim``, ``label``, and ``evidence`` are non-empty strings.
            - The model has been loaded via ``__init__``.

        Postconditions:
            - Does not mutate the model or any input arguments.
        """
        prompt = (
            f"Claim: \"{claim}\"\n"
            f"Rating: {label}\n"
            f"Evidence: {evidence}\n\n"
            f"Write a 1-2 sentence explanation of this rating. "
            f"Only cite the evidence provided. Do not add outside knowledge."
        )
        return generate_llm(
            self._model,
            prompt,
            thinking=False,
            max_tokens=100,
            temperature=0.3,
        )

"""Claim extractor powered by Qwen3.5 via llama-cpp-python.

Single-pass extraction: pulls factual claims from a sentence and
leaves search-query assembly to the orchestrator (which sends the
claim text directly to the search backend). Uses thinking mode for
deeper reasoning about claim identification.
"""

import re

from bsnet.src.model._common import (
    EXTRACTOR_MODEL,
    EXTRACTOR_GGUF_FILE,
    generate_llm,
    load_llm,
)
from bsnet.src.utils.outputs import Claim

# Normalizer for the context-leak filter. Lowercases, strips
# everything that isn't a word char or whitespace, and collapses
# runs of whitespace. The goal is substring-equivalence — two
# strings normalize to the same form when they differ only in
# punctuation, casing, or spacing.
_NORMALIZE_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Return a punctuation- and case-stripped form of ``text``."""
    return _WHITESPACE_RE.sub(" ", _NORMALIZE_RE.sub(" ", text.lower())).strip()


# First-person intent / meta-utterance prefixes. Live transcription
# routinely picks up speaker stage directions ("I'm going to put it
# on the screen", "let me show you", "I'll explain") that the LLM
# extractor occasionally emits as "facts" since they are surface-form
# declarative sentences. They are not checkable claims about the
# world — verifying them sends the search stage chasing irrelevant
# topics and wastes a slot in the latency-bound pipeline. The list
# is anchored on intent + future / hortative phrasing so present-
# tense factual self-reports ("I am 25 years old") still flow
# through; only the action-about-to-happen forms drop.
_FIRST_PERSON_INTENT_PREFIXES = (
    "i'm going to ",
    "i am going to ",
    "i'm gonna ",
    "i am gonna ",
    "i'll ",
    "i will ",
    "i'd like to ",
    "i would like to ",
    "let me ",
    "let's ",
    "lets ",
    "we're going to ",
    "we are going to ",
    "we'll ",
    "we will ",
    "we're gonna ",
    "we are gonna ",
)


def _is_first_person_intent(text: str) -> bool:
    """Decide whether ``text`` is a first-person intent / stage direction.

    Matches on a small whitelist of leading phrases that signal the
    speaker is announcing what they are about to do rather than
    asserting a checkable fact about the world. The check runs after
    the extractor's per-line strip so leading list markers and
    numbering are already gone.

    Args:
        text: Candidate claim text emitted by the LLM extractor.

    Returns:
        ``True`` when ``text`` starts with a recognized first-person
        intent prefix and should be dropped before search.

    Preconditions:
        - ``text`` is a string.

    Postconditions:
        - Returns ``False`` for empty / whitespace-only input.
        - Comparison is case-insensitive.
        - Does not mutate the input.
    """
    lower = text.lstrip().lower()
    return lower.startswith(_FIRST_PERSON_INTENT_PREFIXES)


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

    def extract(self, text: str, context: str = "") -> list[Claim]:
        """Run single-pass claim extraction on an input string.

        Asks the model (in thinking mode) to identify factual claims
        in the text and emit one per line. Returns a ``Claim`` per
        emitted line; the downstream search layer uses the claim text
        itself as the search query, which is both faster (no second
        LLM call) and more reliable (key specifics like numbers and
        dates can't be dropped by a flaky keyword-generation pass).

        When ``context`` is non-empty, the prompt is reshaped to ask
        the model to extract facts from ``text`` only, using the
        preceding context solely to resolve pronouns and implicit
        references. This keeps ambiguous sentences like
        ``"It was the largest ever recorded"`` from yielding
        ungrounded claims that send the search stage chasing
        irrelevant topics.

        Args:
            text: The fully formatted input string — the current
                sentence to extract claims from.
            context: Concatenated prior sentences from the transcript
                buffer. Empty string means no context is available.

        Returns:
            A list of ``Claim`` objects. Empty if no factual claims
            were found.

        Preconditions:
            - ``text`` is a non-empty string.
            - The model has been loaded via ``__init__``.

        Postconditions:
            - Does not mutate the model or input.
            - Each returned ``Claim`` has a non-empty ``text``.
            - Lines starting with first-person intent / stage-
              direction prefixes (see ``_FIRST_PERSON_INTENT_PREFIXES``)
              are dropped — live transcription routinely surfaces
              them and they are not checkable facts about the world.
        """
        context = context.strip()
        if context:
            prompt = (
                "Extract each checkable fact from the latest sentence. "
                "A checkable fact names specific entities, numbers, "
                "dates, places, or events that can be verified against "
                "an external source. Use the preceding context only to "
                "resolve pronouns and implicit references — do not "
                "extract facts that appear only in the context. Write "
                "each as a full sentence with its subject explicitly "
                "named, one per line. If one sentence joins multiple "
                "facts with \"and\", \"but\", \"while\", or a comma, "
                "split them into separate lines — even when the facts "
                "share a subject or a time phrase.\n\n"
                "Exclude opinions and value judgments, predictions or "
                "future events, hypotheticals (\"if X, then Y\"), vague "
                "generalizations without specifics (\"things are "
                "getting better\"), questions, and first-person stage "
                "directions about the conversation itself (\"I'm going "
                "to X\", \"I'll show you\", \"let me explain\", \"let's "
                "look at\"). Do not write fragments or bare numbers. "
                "Say \"none\" if there are no checkable facts.\n\n"
                f"Context: {context}\n"
                f"Latest: {text}\n"
                "Facts:\n"
                "{fact1}\n"
                "{fact2}"
            )
        else:
            prompt = (
                "Extract each checkable fact from the text below. A "
                "checkable fact names specific entities, numbers, "
                "dates, places, or events that can be verified against "
                "an external source. Write each as a full sentence "
                "with its subject explicitly named, one per line. If "
                "one sentence joins multiple facts with \"and\", "
                "\"but\", \"while\", or a comma, split them into "
                "separate lines — even when the facts share a subject "
                "or a time phrase.\n\n"
                "Exclude opinions and value judgments, predictions or "
                "future events, hypotheticals (\"if X, then Y\"), vague "
                "generalizations without specifics (\"things are "
                "getting better\"), questions, and first-person stage "
                "directions about the conversation itself (\"I'm going "
                "to X\", \"I'll show you\", \"let me explain\", \"let's "
                "look at\"). Do not write fragments or bare numbers. "
                "Say \"none\" if there are no checkable facts.\n\n"
                f"Text: {text}\n"
                "Facts:\n"
                "{fact1}\n"
                "{fact2}"
            )
        raw_claims = generate_llm(
            self._model,
            prompt,
            thinking=True,
            max_tokens=256,
            temperature=0.3,
        )
        body = raw_claims.strip()
        if body.lower().startswith("facts:"):
            body = body[len("facts:"):].strip()
        if not body or body.lower() == "none":
            return []

        # Pre-normalize the context once so the per-claim filter
        # below stays cheap. Qwen3.5-0.8B occasionally regurgitates
        # a sentence from the context as a "new" fact despite the
        # prompt telling it not to — the resulting duplicate verdict
        # is both wasted compute and confusing for the user, so we
        # drop any extracted claim whose normalized text is a
        # substring of the normalized context.
        norm_context = _normalize(context) if context else ""

        claims: list[Claim] = []
        for claim_text in body.splitlines():
            claim_text = claim_text.strip().lstrip("0123456789.-) ")
            if not claim_text or claim_text.lower() == "none":
                continue
            if claim_text.lower() in ("facts:", "{fact1}", "{fact2}"):
                continue
            if _is_first_person_intent(claim_text):
                continue
            if norm_context and _normalize(claim_text) in norm_context:
                continue
            claims.append(Claim(text=claim_text))

        return claims

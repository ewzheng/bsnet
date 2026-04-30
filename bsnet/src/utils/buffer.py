"""Transcript buffer for claim-boundary detection.

Sits between the transcription layer and the claim extractor.
Accumulates raw transcript text, splits on sentence boundaries,
and maintains a sliding context window so the extractor can
resolve coreference across utterances.
"""

import re
from dataclasses import dataclass, field

# Sentence-ending punctuation followed by whitespace or end-of-string
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

DEFAULT_CONTEXT_SENTENCES = 4


@dataclass
class TranscriptBuffer:
    """Rolling buffer that converts raw transcript chunks into sentence-aligned
    claim candidates with surrounding context.

    Maintains an internal accumulator for incomplete sentences and a bounded
    context window of recently completed sentences.
    """

    context_size: int = DEFAULT_CONTEXT_SENTENCES
    _raw: str = ""
    _context: list[str] = field(default_factory=list)

    def push(self, text: str) -> list[tuple[str, str]]:
        """Ingest a transcript chunk and return any completed sentences.

        Args:
            text: Raw transcript text from the STT layer. May contain
                zero, one, or multiple sentence boundaries.

        Returns:
            A list of ``(context, sentence)`` tuples for each completed
            sentence found in the accumulated text. ``context`` is the
            concatenation of up to ``context_size`` prior sentences.
            Returns an empty list when no sentence boundary has been
            reached yet.

        Preconditions:
            - ``text`` is a non-empty string of transcribed speech.

        Postconditions:
            - Completed sentences are removed from the internal accumulator.
            - The context window is updated with each emitted sentence.
            - Incomplete trailing text is retained for the next call.
        """
        self._raw += (" " if self._raw else "") + text.strip()

        parts = _SENT_BOUNDARY.split(self._raw)

        # Last segment is incomplete if _raw doesn't end with punctuation
        if not re.search(r'[.!?]$', self._raw.rstrip()):
            self._raw = parts[-1]
            complete = parts[:-1]
        else:
            complete = parts
            self._raw = ""

        results: list[tuple[str, str]] = []
        for sentence in complete:
            sentence = sentence.strip()
            if not sentence:
                continue
            ctx = self._build_context()
            results.append((ctx, sentence))
            self._context.append(sentence)
            # Keep context list bounded
            if len(self._context) > self.context_size * 2:
                self._context = self._context[-self.context_size:]

        return results

    def flush(self) -> list[tuple[str, str]]:
        """Emit any remaining incomplete text as a final sentence.

        Returns:
            A list containing a single ``(context, sentence)`` tuple if
            there was buffered text, or an empty list otherwise.

        Preconditions:
            - Called when the transcript stream has ended or paused.

        Postconditions:
            - The internal accumulator is cleared.
            - The context window is updated if text was flushed.
        """
        remaining = self._raw.strip()
        self._raw = ""
        if not remaining:
            return []
        ctx = self._build_context()
        self._context.append(remaining)
        return [(ctx, remaining)]

    def _build_context(self) -> str:
        """Assemble the context string from recent sentences.

        Returns:
            The concatenation of up to ``context_size`` recent sentences,
            or an empty string when the context window is empty.

        Preconditions:
            - None.

        Postconditions:
            - Does not mutate any internal state.
        """
        return " ".join(self._context[-self.context_size:])

    def clear(self) -> None:
        """Reset the buffer and context to empty state.

        Preconditions:
            - None.

        Postconditions:
            - The accumulator, context window, and all internal state
              are cleared.
        """
        self._raw = ""
        self._context.clear()

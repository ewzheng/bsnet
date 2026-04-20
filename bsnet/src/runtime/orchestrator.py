"""Streaming orchestrator for the bsnet fact-checking runtime.

Ingests raw transcript chunks from a producer (typically
``bsnet.src.utils.transcription.listen``), pushes them through a
``TranscriptBuffer``, and drives the fact-checking pipeline end-to-end
as a multi-stage concurrent graph:

    intake → extract → search → check → validate → render → sink

Each stage runs on its own thread so different sentences occupy
different stages at the same time (CPU-pipeline-style overlap). Model
stages are single-worker because the underlying ``Llama`` / torch
model instances are not safe for concurrent inference on a single
context; the I/O-bound search stage fans out via a
``ThreadPoolExecutor``.

The orchestrator owns the ``TranscriptBuffer``, the inter-stage queues,
and the worker threads. It does *not* own the transcription model or
the fact-checking models — the caller injects a ``Pipeline`` (or lets
one be constructed lazily) and the external ``search_fn`` and
``validate_fn`` callables. Search and validation are owned by other
branches; this module ships with placeholder callables so the runtime
is exercisable end-to-end while those branches are in flight.
"""

import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

from bsnet.src.runtime.pipeline import Pipeline
from bsnet.src.utils.buffer import DEFAULT_CONTEXT_SENTENCES, TranscriptBuffer
from bsnet.src.utils.outputs import CheckResult, Claim, Verdict

SearchFn = Callable[[list[str]], list[str]]
ValidateFn = Callable[[CheckResult], bool]


def _default_search(queries: list[str]) -> list[str]:
    """Placeholder search callable used until the real branch merges.

    Returns one stub snippet per query so downstream stages have
    something to operate on during end-to-end exercising of the
    runtime. The real implementation will live in ``bsnet.src.runtime``
    or wherever the search branch lands and should be injected into
    ``Orchestrator`` via ``search_fn``.

    Args:
        queries: Search query strings attached to a ``Claim``.

    Returns:
        A list of placeholder snippet strings, one per non-empty query.

    Preconditions:
        - ``queries`` is a list of strings.

    Postconditions:
        - Does not perform any I/O.
        - Returns an empty list only when every query is empty.
    """
    # TODO: replace with real search integration once that branch lands.
    return [f"placeholder evidence for query: {q}" for q in queries if q]


def _default_validate(result: CheckResult) -> bool:
    """Placeholder validation callable used until the real branch merges.

    Passes every ``CheckResult`` through unchanged. The real
    implementation will live under ``bsnet/src/validation/`` and
    should be injected into ``Orchestrator`` via ``validate_fn``.

    Args:
        result: Scored and labeled check result from the scorer stage.

    Returns:
        ``True`` for every input.

    Preconditions:
        - ``result`` is a ``CheckResult`` instance.

    Postconditions:
        - Does not mutate ``result``.
    """
    # TODO: replace with real validation integration once that branch lands.
    del result
    return True


class Orchestrator:
    """Drive the fact-checking pipeline as a staged concurrent graph.

    Starts a daemon thread per stage on ``run()``. Each stage reads
    from its input queue, performs its unit of work, and enqueues the
    result for the next stage. An internal sentinel cascades down the
    pipeline on normal shutdown so each worker terminates cleanly.
    Exceptions raised in any stage are captured on the instance and
    re-raised to the consumer after the downstream queues drain.

    An ``Orchestrator`` is single-use: call ``run()`` exactly once per
    instance. Buffer state, queues, and worker threads are session
    scoped, so reusing an instance after the output iterator is
    exhausted is undefined.
    """

    _SENTINEL: object = object()

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        search_fn: SearchFn | None = None,
        validate_fn: ValidateFn | None = None,
        context_size: int = DEFAULT_CONTEXT_SENTENCES,
        search_workers: int = 4,
        queue_size: int = 64,
    ) -> None:
        """Construct the orchestrator and allocate per-stage queues.

        Args:
            pipeline: Pre-built ``Pipeline`` instance. When ``None``, a
                fresh ``Pipeline()`` is constructed (which loads all
                three models eagerly). Injecting an instance is useful
                for tests and for sharing a warm pipeline across runs.
            search_fn: External search callable taking the queries
                attached to a ``Claim`` and returning evidence snippet
                strings. When ``None``, the module-level placeholder is
                used — sufficient to exercise the rest of the runtime
                but not a real search backend.
            validate_fn: External validation callable gating a scored
                ``CheckResult`` before it reaches the renderer. When
                ``None``, the module-level placeholder is used (passes
                everything through).
            context_size: Number of recent sentences the buffer keeps
                as context for each emitted sentence.
            search_workers: Thread-pool size for the search stage. The
                search stage is the only fan-out stage because search
                is I/O bound; all model stages remain single-worker.
            queue_size: Upper bound on items buffered between adjacent
                stages. Provides backpressure so a slow consumer
                cannot drive unbounded memory growth.

        Preconditions:
            - ``context_size``, ``search_workers``, and ``queue_size``
              are positive integers.

        Postconditions:
            - A fresh ``TranscriptBuffer`` is owned by the instance.
            - All inter-stage queues are empty.
            - No worker threads have been started yet.
        """
        self._pipeline: Pipeline = pipeline if pipeline is not None else Pipeline()
        self._search_fn: SearchFn = search_fn if search_fn is not None else _default_search
        self._validate_fn: ValidateFn = (
            validate_fn if validate_fn is not None else _default_validate
        )
        self._buffer: TranscriptBuffer = TranscriptBuffer(context_size=context_size)
        self._search_workers: int = search_workers

        self._sentence_q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._claim_q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._snippet_q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._check_q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._result_q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._verdict_q: queue.Queue = queue.Queue(maxsize=queue_size)

        self._error_lock: threading.Lock = threading.Lock()
        self._error: BaseException | None = None

    def run(self, chunk_source: Iterable[str]) -> Iterator[Verdict]:
        """Start every stage and return an iterator over verdicts.

        Spawns one daemon thread per stage. The returned iterator
        pulls completed ``Verdict`` objects off the final queue in the
        order they arrive and terminates once every upstream stage has
        drained and emitted its sentinel. If any stage caught an
        exception, the iterator re-raises it after yielding every
        verdict that was produced before the failure.

        Args:
            chunk_source: Any iterable of transcript chunk strings.
                Typically ``listen()`` from the transcription module.

        Returns:
            An iterator of ``Verdict`` objects covering every claim
            that survived the scorer and the validator.

        Preconditions:
            - ``chunk_source`` has not yet been iterated elsewhere.
            - ``run()`` has not been called previously on this
              instance.

        Postconditions:
            - All stage threads are running as daemons.
            - The returned iterator is the sole consumer of the final
              verdict queue.
            - After the iterator is exhausted, every stage thread has
              emitted its sentinel and any captured exception has been
              surfaced to the caller.
        """
        threading.Thread(
            target=self._intake_loop, args=(chunk_source,), daemon=True,
        ).start()
        threading.Thread(target=self._extract_loop, daemon=True).start()
        threading.Thread(target=self._search_loop, daemon=True).start()
        threading.Thread(target=self._check_loop, daemon=True).start()
        threading.Thread(target=self._validate_loop, daemon=True).start()
        threading.Thread(target=self._render_loop, daemon=True).start()

        return self._iter_output()

    def _intake_loop(self, chunk_source: Iterable[str]) -> None:
        """Read chunks from the source and enqueue buffered sentences.

        Runs on the intake daemon thread. Iterating ``chunk_source``
        here — not on the consumer thread — isolates live
        transcription from downstream backpressure.

        Args:
            chunk_source: The iterable supplied to ``run()``.

        Preconditions:
            - Invoked only from the intake thread started by ``run()``.

        Postconditions:
            - All buffered sentences emitted before failure have been
              enqueued onto ``_sentence_q``.
            - Any exception raised by the source or the buffer has
              been captured via ``_record_error``.
            - Exactly one sentinel has been enqueued onto
              ``_sentence_q``.
        """
        try:
            for chunk in chunk_source:
                for _context, sentence in self._buffer.push(chunk):
                    self._sentence_q.put(sentence)
            for _context, sentence in self._buffer.flush():
                self._sentence_q.put(sentence)
        except BaseException as exc:
            self._record_error(exc)
        finally:
            self._sentence_q.put(self._SENTINEL)

    def _extract_loop(self) -> None:
        """Pull sentences, run extraction, enqueue resulting claims.

        Preconditions:
            - Invoked only from the extract stage thread.

        Postconditions:
            - Every claim produced before failure has been enqueued
              onto ``_claim_q``.
            - Any exception has been captured via ``_record_error``.
            - Exactly one sentinel has been enqueued onto ``_claim_q``.
        """
        try:
            while True:
                item = self._sentence_q.get()
                if item is self._SENTINEL:
                    break
                for claim in self._pipeline.extract(item):
                    self._claim_q.put(claim)
        except BaseException as exc:
            self._record_error(exc)
        finally:
            self._claim_q.put(self._SENTINEL)

    def _search_loop(self) -> None:
        """Dispatch claims to the search thread pool and fan results in.

        Submits each claim as an independent task so the I/O-bound
        search backend can service multiple queries concurrently. On
        sentinel, exits the pool context manager — which waits for all
        in-flight tasks to finish before returning — then emits one
        sentinel downstream.

        Preconditions:
            - Invoked only from the search stage thread.

        Postconditions:
            - Every successfully searched claim has produced a
              ``(claim_text, snippets)`` tuple on ``_snippet_q``.
            - Any exception from the dispatcher or a task has been
              captured via ``_record_error``.
            - Exactly one sentinel has been enqueued onto
              ``_snippet_q``.
        """
        try:
            with ThreadPoolExecutor(max_workers=self._search_workers) as pool:
                while True:
                    item = self._claim_q.get()
                    if item is self._SENTINEL:
                        break
                    pool.submit(self._search_task, item)
        except BaseException as exc:
            self._record_error(exc)
        finally:
            self._snippet_q.put(self._SENTINEL)

    def _search_task(self, claim: Claim) -> None:
        """Run a single search call and enqueue snippets for scoring.

        Exceptions are captured rather than propagated so one failing
        search does not poison the pool. The shutdown handshake is
        unaffected; missing snippets simply mean no ``CheckResult`` is
        produced for that claim.

        Args:
            claim: The claim whose queries should be searched.

        Preconditions:
            - Runs on a worker thread inside the search thread pool.

        Postconditions:
            - On success, exactly one ``(claim_text, snippets)`` tuple
              has been enqueued onto ``_snippet_q``.
            - On failure, the exception has been captured and no tuple
              is enqueued.
        """
        try:
            snippets = self._search_fn(claim.queries)
            self._snippet_q.put((claim.text, snippets))
        except BaseException as exc:
            self._record_error(exc)

    def _check_loop(self) -> None:
        """Score claims against retrieved snippets and forward results.

        ``None`` results from the scorer (opinion claims, empty
        snippets) are dropped so the validator only sees labeled
        ``CheckResult`` objects.

        Preconditions:
            - Invoked only from the check stage thread.

        Postconditions:
            - Every non-``None`` scorer output has been enqueued onto
              ``_check_q``.
            - Any exception has been captured via ``_record_error``.
            - Exactly one sentinel has been enqueued onto ``_check_q``.
        """
        try:
            while True:
                item = self._snippet_q.get()
                if item is self._SENTINEL:
                    break
                claim_text, snippets = item
                result = self._pipeline.check(claim_text, snippets)
                if result is not None:
                    self._check_q.put(result)
        except BaseException as exc:
            self._record_error(exc)
        finally:
            self._check_q.put(self._SENTINEL)

    def _validate_loop(self) -> None:
        """Gate scored results through ``validate_fn`` before rendering.

        Results for which ``validate_fn`` returns ``False`` are
        dropped; accepted results are forwarded unchanged.

        Preconditions:
            - Invoked only from the validate stage thread.

        Postconditions:
            - Every accepted result has been enqueued onto
              ``_result_q``.
            - Any exception has been captured via ``_record_error``.
            - Exactly one sentinel has been enqueued onto
              ``_result_q``.
        """
        try:
            while True:
                item = self._check_q.get()
                if item is self._SENTINEL:
                    break
                if self._validate_fn(item):
                    self._result_q.put(item)
        except BaseException as exc:
            self._record_error(exc)
        finally:
            self._result_q.put(self._SENTINEL)

    def _render_loop(self) -> None:
        """Render final verdict explanations for validated results.

        Preconditions:
            - Invoked only from the render stage thread.

        Postconditions:
            - Every validated result has been rendered into a
              ``Verdict`` and enqueued onto ``_verdict_q``.
            - Any exception has been captured via ``_record_error``.
            - Exactly one sentinel has been enqueued onto
              ``_verdict_q``.
        """
        try:
            while True:
                item = self._result_q.get()
                if item is self._SENTINEL:
                    break
                self._verdict_q.put(self._pipeline.render(item))
        except BaseException as exc:
            self._record_error(exc)
        finally:
            self._verdict_q.put(self._SENTINEL)

    def _iter_output(self) -> Iterator[Verdict]:
        """Yield verdicts until the render-stage sentinel arrives.

        Blocks on ``queue.get()`` so slow producers do not spin the
        consumer. When the sentinel is received the generator
        re-raises any captured stage exception or returns cleanly.

        Returns:
            A generator of ``Verdict`` objects.

        Raises:
            BaseException: The first exception captured by any stage
                thread, propagated only after every verdict enqueued
                before the failure has been yielded.

        Preconditions:
            - Every stage thread has been started.
            - Only one consumer iterates this generator.

        Postconditions:
            - ``_verdict_q`` has been drained up to and including the
              sentinel.
            - If ``self._error`` was set, it has been re-raised.
        """
        while True:
            item = self._verdict_q.get()
            if item is self._SENTINEL:
                if self._error is not None:
                    raise self._error
                return
            yield item

    def _record_error(self, exc: BaseException) -> None:
        """Record the first exception observed across any stage thread.

        Subsequent exceptions are ignored so the consumer sees the
        original cause. Uses a lock because several stages may fail
        concurrently during a cascading shutdown.

        Args:
            exc: The exception caught in a stage's ``except`` block.

        Preconditions:
            - Called from a stage thread's exception handler.

        Postconditions:
            - ``self._error`` is non-``None`` after the first call.
            - Repeated calls are no-ops.
        """
        with self._error_lock:
            if self._error is None:
                self._error = exc

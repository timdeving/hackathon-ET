"""Iterate in a background thread, so the CPU decodes ahead while the GPU works."""
from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from queue import Full, Queue
from typing import TypeVar

T = TypeVar("T")

_DONE = object()


class _Failure:
    """Carries an exception from the background thread to the consumer."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


def prefetch(items: Iterable[T], depth: int) -> Iterator[T]:
    """Yield `items` in order while a background thread keeps up to `depth` of them ready.

    Decoding and GPU inference both release Python's GIL, so the next frames decode while the
    detector works on the current one. An exception raised while producing an item is re-raised
    here; closing this iterator early stops the background thread.
    """
    queue: Queue = Queue(maxsize=depth)
    stop = threading.Event()

    def put(item: object) -> bool:
        """Queue an item; give up (False) once the consumer has stopped."""
        while not stop.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def produce() -> None:
        try:
            for item in items:
                if not put(item):
                    return
        except BaseException as error:  # handed to the consumer, which re-raises it
            put(_Failure(error))
            return
        put(_DONE)

    thread = threading.Thread(target=produce, name="prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = queue.get()
            if item is _DONE:
                return
            if isinstance(item, _Failure):
                raise item.error
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)

"""prefetch(): background iteration that keeps order, passes errors on and stops cleanly."""
from __future__ import annotations

import threading
from itertools import count

import pytest

from src.video.prefetch import prefetch


def test_items_arrive_in_order():
    assert list(prefetch(range(100), depth=4)) == list(range(100))


def test_an_error_in_the_background_reaches_the_consumer():
    def broken():
        yield 1
        raise ValueError("decoder failed")

    with pytest.raises(ValueError, match="decoder failed"):
        list(prefetch(broken(), depth=2))


def test_stopping_early_stops_the_background_thread():
    items = prefetch(count(), depth=2)  # an endless producer
    assert [next(items) for _ in range(3)] == [0, 1, 2]
    items.close()
    assert not any(t.name == "prefetch" and t.is_alive() for t in threading.enumerate())

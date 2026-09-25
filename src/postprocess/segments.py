"""Clean raw rule output into a valid event list.

The harness does not repair events: it silently drops same-class overlaps and events outside
the video. So every event list goes through finalize_events() before it is returned.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from src.events import CLASSES, Segment

log = logging.getLogger(__name__)


def merge_segments(segments: Iterable[Segment], max_gap: float) -> list[Segment]:
    """Merge segments of one class that overlap or are less than max_gap seconds apart.

    Overlaps are always merged, whatever max_gap is: the task allows one segment per class
    at a time ("two same-class events at once = one segment covering both").
    """
    merged: list[Segment] = []
    for seg in sorted(segments):
        if merged and seg.start - merged[-1].end < max_gap:
            last = merged[-1]
            merged[-1] = last._replace(end=max(last.end, seg.end))
        else:
            merged.append(seg)
    return merged


def finalize_events(
    segments: Iterable[Segment], duration: float, params: Mapping[str, Any]
) -> list[list]:
    """Clip, merge and filter segments per class; return [[start, end, label], ...] by start.

    params is the `postprocess` section of configs/params.yaml: a `default` block with
    merge_gap_sec and min_duration_sec, optionally overridden per class under `per_class`.
    Fragments are merged before the minimum-duration filter, so an event that flickers is
    kept whole instead of being dropped piece by piece.
    """
    by_class: dict[str, list[Segment]] = defaultdict(list)
    for seg in segments:
        if seg.label not in CLASSES:
            log.warning("dropping segment with unknown label %r", seg.label)
            continue
        start, end = max(0.0, seg.start), min(duration, seg.end)
        if start < end:
            by_class[seg.label].append(Segment(start, end, seg.label))

    events = []
    for label, class_segments in by_class.items():
        p = {**params["default"], **params.get("per_class", {}).get(label, {})}
        for seg in merge_segments(class_segments, p["merge_gap_sec"]):
            start, end = round(seg.start, 3), round(seg.end, 3)
            if end - start >= p["min_duration_sec"] and start < end:
                events.append([start, end, label])
    return sorted(events)

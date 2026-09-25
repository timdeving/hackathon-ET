"""Part A: turn one video into a list of timed events.

The finished pipeline: probe the video, run perception (decode, detect, track), compute
per-track features, apply one rule per class, and pass every segment through
finalize_events(). The perception, feature and rule stages are added in later steps.
"""
from __future__ import annotations

import logging
import time

from src.config import load_params
from src.events import Segment
from src.postprocess.segments import finalize_events
from src.video.probe import probe_video

log = logging.getLogger(__name__)


def detect_events(video_path: str) -> list[list]:
    """Return [[start_sec, end_sec, label], ...] for one video.

    Times are seconds from the first frame, 0 <= start < end <= duration; labels come from
    CLASSES and same-class segments never overlap.
    """
    t0 = time.perf_counter()
    info = probe_video(video_path)
    segments: list[Segment] = []  # no rules yet
    events = finalize_events(segments, info.duration, load_params()["postprocess"])
    log.info("%s: %d events in %.1f s", info.name, len(events), time.perf_counter() - t0)
    return events

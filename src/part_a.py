"""Part A: turn one video into a list of timed events.

Pipeline: perception (decode, detect and track in one pass; src/perception/pipeline.py), then
per-track features and one rule per class (not built yet), then finalize_events(). Perception
stops at a deadline that leaves the harness time for Part B (src/budget.py).
"""
from __future__ import annotations

import logging
import random
import time

import numpy as np

from src.budget import harness_read_seconds, part_a_seconds, part_b_seconds
from src.config import load_params
from src.events import Segment
from src.perception.pipeline import Perception
from src.postprocess.segments import finalize_events
from src.video.probe import probe_video

log = logging.getLogger(__name__)

_perception: Perception | None = None  # loaded once per process


def load_models() -> None:
    """Load the detector now.

    solution.py calls this when the harness imports it, before the harness starts timing, so
    loading the model and starting the GPU don't count against any video. A failure is logged,
    not raised: the harness imports solution.py outside its error handling, so raising here
    would stop it before it writes any predictions. detect_events() then tries again and fails
    per video, which the harness records in its log.
    """
    global _perception
    # Nothing in the pipeline draws random numbers; the seeds are fixed so that anything that
    # ever does stays repeatable.
    random.seed(0)
    np.random.seed(0)
    try:
        _perception = _build_perception()
    except Exception:  # see the docstring: log it and let detect_events() report it per video
        log.exception("could not load the perception models")


def detect_events(video_path: str) -> list[list]:
    """Return [[start_sec, end_sec, label], ...] for one video.

    Times are seconds from the first frame, 0 <= start < end <= duration; labels come from
    CLASSES and same-class segments never overlap.
    """
    start = time.perf_counter()
    perception = _get_perception()
    params = load_params()
    info = probe_video(video_path)

    budget = params["budget"]
    read_seconds = harness_read_seconds(video_path, budget["calibration_frames"])
    part_b = part_b_seconds(info.duration, info.n_frames, read_seconds, budget)
    allowed = part_a_seconds(info.duration, part_b, budget)
    log.info(
        "%s: budget %.0f s; Part B's decoding estimated at %.0f s; Part A may use %.0f s",
        info.name,
        budget["time_factor"] * info.duration,
        part_b,
        allowed,
    )
    if allowed <= 0:
        log.error("%s: Part B's decoding alone may not fit in the time budget", info.name)

    result = perception.run(video_path, deadline=start + max(allowed, 0.0))
    segments: list[Segment] = []  # no rules yet
    events = finalize_events(segments, info.duration, params["postprocess"])
    if not result.complete:
        covered = result.n_analysed * result.stride / info.fps
        log.warning(
            "%s: perception stopped at %.0f s of %.0f s to stay inside the time budget",
            info.name,
            covered,
            info.duration,
        )
    elapsed = time.perf_counter() - start
    log.info(
        "%s: %d of %d frames analysed, %.1f detections per frame, %d tracks, %d events, "
        "%.1f s = %.2fx the video's duration",
        info.name,
        result.n_analysed,
        info.n_frames,
        len(result.detections) / max(1, result.n_analysed),
        len(set(result.tracks["track_id"].tolist())),
        len(events),
        elapsed,
        elapsed / info.duration,
    )
    return events


def _get_perception() -> Perception:
    """The loaded perception models, loading them now if load_models() couldn't."""
    global _perception
    if _perception is None:
        try:
            _perception = _build_perception()
        except Exception as error:
            raise RuntimeError(f"perception models unavailable: {error}") from error
    return _perception


def _build_perception() -> Perception:
    # Imported here, not at the top: the detector needs PyTorch, which laptops don't have.
    from src.perception.detector import load_detector

    params = load_params()
    return Perception(load_detector(params), params)

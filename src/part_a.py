"""Part A: turn one video into a list of timed events.

Pipeline: perception (decode, detect and track in one pass; src/perception/pipeline.py), then
per-track features and one rule per class (not built yet), then finalize_events().
"""
from __future__ import annotations

import logging

from src.config import load_params
from src.events import Segment
from src.perception.pipeline import Perception
from src.postprocess.segments import finalize_events

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
    try:
        _perception = _build_perception()
    except Exception:  # see the docstring: log it and let detect_events() report it per video
        log.exception("could not load the perception models")


def detect_events(video_path: str) -> list[list]:
    """Return [[start_sec, end_sec, label], ...] for one video.

    Times are seconds from the first frame, 0 <= start < end <= duration; labels come from
    CLASSES and same-class segments never overlap.
    """
    global _perception
    if _perception is None:
        try:
            _perception = _build_perception()
        except Exception as error:
            raise RuntimeError(f"perception models unavailable: {error}") from error

    result = _perception.run(video_path)
    segments: list[Segment] = []  # no rules yet
    events = finalize_events(segments, result.info.duration, load_params()["postprocess"])
    n_tracks = len(set(result.tracks["track_id"].tolist()))
    log.info(
        "%s: %d of %d frames analysed, %.1f detections per frame, %d tracks, %d events, "
        "%.1f s = %.2fx the video's duration",
        result.info.name,
        result.n_analysed,
        result.info.n_frames,
        len(result.detections) / max(1, result.n_analysed),
        n_tracks,
        len(events),
        result.seconds,
        result.seconds / result.info.duration,
    )
    return events


def _build_perception() -> Perception:
    # Imported here, not at the top: the detector needs PyTorch, which laptops don't have.
    from src.perception.detector import load_detector

    params = load_params()
    return Perception(load_detector(params), params)

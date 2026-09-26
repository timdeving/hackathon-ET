"""Part A: turn one video into a list of timed events.

Pipeline: perception (decode, detect and track in one pass; src/perception/pipeline.py), then
per-object features and one rule per enabled class (src/rules/), then finalize_events().
Perception stops at a deadline that leaves the harness time for Part B (src/budget.py); the
rules after it take seconds, which the budget's margin covers.
"""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.budget import harness_read_seconds, part_a_seconds, part_b_seconds
from src.config import CONFIG_DIR, load_params
from src.events import Segment
from src.features.tracks import NoAlignment, PointMapper
from src.perception.pipeline import Perception, PerceptionResult
from src.postprocess.segments import finalize_events
from src.rules import find_events
from src.rules.common import PhaseTimeline
from src.scene.scene_map import SceneMap
from src.video.probe import probe_video

log = logging.getLogger(__name__)

SCENE_MAP_PATH = CONFIG_DIR / "scene_map.json"

_perception: Perception | None = None  # loaded once per process
_scene_map: SceneMap | None = None  # likewise, on first use


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
    segments = _find_events(result, params)  # camera alignment and light phases: see there
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


def _find_events(
    result: PerceptionResult,
    params: Mapping[str, Any],
    mapper: PointMapper | None = None,
    phases: Mapping[str, PhaseTimeline] | None = None,
) -> list[Segment]:
    """Raw segments of the enabled rules; none while no class is enabled or the scene map is
    missing. A rule that fails costs only its own class.

    mapper: the video's camera alignment (src/scene/alignment.py, from the GPU PC); until it
    exists, the video is treated as framed like the reference. phases: the signal phase per arm
    (src/scene/signals.py, from the GPU PC); until then red_light makes no call.
    """
    if not params["rules"]["enabled"]:
        return []
    scene = _get_scene_map()
    if scene is None:
        return []
    return find_events(result, mapper or NoAlignment(), scene, params, skip_failures=True,
                       phases=phases)


def _get_scene_map() -> SceneMap | None:
    """The scene map, loaded once per process; None, with an error logged, until it's drawn."""
    global _scene_map
    if _scene_map is None and SCENE_MAP_PATH.exists():
        _scene_map = SceneMap.load(SCENE_MAP_PATH)
    if _scene_map is None:
        log.error("no scene map at %s: the rules can't run, so no events", SCENE_MAP_PATH)
    return _scene_map


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

"""Keep Part A inside the time budget, so the harness never scores a video as empty.

The harness gives Part A and Part B together `time_factor` x the video's duration. After Part A
it decodes every frame again with OpenCV for Part B, and we can't make that faster. So before
starting, Part A measures how long OpenCV takes to read this video's frames on the machine it
runs on, exactly as the harness will, and gives itself a deadline that leaves Part B that much
time, plus a margin. Perception stops at the deadline and returns what it has.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2

WARM_UP_FRAMES = 10  # read but not timed: opening the file and the first keyframe cost extra


def harness_read_seconds(video_path: str | Path, frames: int) -> float:
    """Seconds OpenCV takes to read one frame of this video, the way the harness's Part B does.

    Returns 0.0 if the video is too short to measure.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        for _ in range(WARM_UP_FRAMES):
            if not cap.read()[0]:
                return 0.0
        start = time.perf_counter()
        read = 0
        while read < frames and cap.read()[0]:
            read += 1
        elapsed = time.perf_counter() - start
    finally:
        cap.release()
    return elapsed / read if read else 0.0


def part_b_seconds(
    duration: float, n_frames: int, read_seconds: float, params: Mapping[str, Any]
) -> float:
    """Estimated time of the harness's Part B: reading every frame, plus step()'s own work.

    params is the `budget` section of configs/params.yaml.
    """
    decoding = n_frames * read_seconds * params["part_b_safety"]
    return decoding + params["part_b_extra_ratio"] * duration


def part_a_seconds(duration: float, part_b: float, params: Mapping[str, Any]) -> float:
    """How long Part A may run: the budget, minus a margin and Part B's estimated time.

    Zero or negative when Part B alone won't fit in the budget.
    """
    return params["time_factor"] * duration * (1.0 - params["margin_ratio"]) - part_b

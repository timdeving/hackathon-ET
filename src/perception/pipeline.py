"""Perception for Part A: decode, detect and track one video, in a single pass.

A background thread decodes (every frame; every `stride`-th one becomes a working image) while
the detector runs on the GPU. Detections are mapped to full-resolution pixels, the coordinates
every later stage uses, and passed to a new tracker for each video. The result is two tables
(numpy structured arrays): every detection, and every track position. The tables are also what
the cache stores for rule development on laptops.

This module doesn't import PyTorch: the detector is passed in, so the pipeline can be tested
with a stand-in and the tables loaded anywhere.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.perception.boxes import Detections
from src.perception.tracker import ByteTracker, TrackedBox
from src.video.prefetch import prefetch
from src.video.probe import VideoInfo
from src.video.reader import VideoReader

PREFETCH_FRAMES = 8  # decoded frames waiting for the detector: enough to ride out a slow one

DETECTION_DTYPE = np.dtype(
    [
        ("frame", np.int32),
        ("x0", np.float32),
        ("y0", np.float32),
        ("x1", np.float32),
        ("y1", np.float32),
        ("score", np.float32),
        ("class_id", np.int16),
    ]
)
TRACK_DTYPE = np.dtype(
    [
        ("frame", np.int32),
        ("track_id", np.int32),
        ("x0", np.float32),  # the box after the tracker's Kalman correction
        ("y0", np.float32),
        ("x1", np.float32),
        ("y1", np.float32),
        ("det_x0", np.float32),  # the raw detection the track matched in this frame
        ("det_y0", np.float32),
        ("det_x1", np.float32),
        ("det_y1", np.float32),
        ("score", np.float32),
        ("class_id", np.int16),
        ("confirmed", np.bool_),  # False only in a track's first frame
    ]
)


class DetectorLike(Protocol):
    input_size: tuple[int, int]  # (height, width) of the images it expects

    def __call__(self, image: np.ndarray) -> Detections: ...


@dataclass(frozen=True)
class PerceptionResult:
    """What perception found in one video. Boxes are full-resolution pixels; the time of a row
    is its frame / info.fps."""

    info: VideoInfo
    stride: int
    detections: np.ndarray  # DETECTION_DTYPE rows: every detection the detector kept
    tracks: np.ndarray  # TRACK_DTYPE rows: one per track per analysed frame, confirmed tracks only
    seconds: float  # processing time
    n_analysed: int  # frames the detector saw: 0, stride, 2 * stride, ...
    complete: bool  # False if perception stopped early to stay inside the time budget
    # The video's empty road in grey (per-pixel median of frames spread over it), for camera
    # alignment; None if none was collected. Not stored in the cache.
    background: np.ndarray | None = None


class Perception:
    """Decode, detect and track one video at a time; the detector stays loaded between videos.

    Args:
        detector: turns a BGR working image into Detections (src.perception.detector.Detector,
            or a stand-in in tests).
        params: the whole of configs/params.yaml; the `video`, `tracker` and `alignment`
            sections are used.
    """

    def __init__(self, detector: DetectorLike, params: Mapping[str, Any]) -> None:
        self.detector = detector
        self.video_params = params["video"]
        self.tracker_params = params["tracker"]
        self.alignment_params = params["alignment"]
        if detector.input_size[1] != self.video_params["width"]:
            raise ValueError(
                f"video.width is {self.video_params['width']} px, but the detector expects "
                f"images {detector.input_size[1]} px wide; the two must match"
            )

    def run(self, video_path: str | Path, deadline: float | None = None) -> PerceptionResult:
        """Perceive one video.

        deadline: a time.perf_counter() value. When it passes, perception stops after the frame
        in progress and returns what it has (at least one frame is always analysed).
        """
        start = time.perf_counter()
        reader = VideoReader(
            video_path,
            stride=self.video_params["stride"],
            width=self.video_params["width"],
            threads=self.video_params["decode_threads"],
            grey_samples=self.alignment_params["background_frames"],
            grey_width=self.alignment_params["match_width"],
        )
        tracker = ByteTracker(reader.info.fps / reader.stride, self.tracker_params)
        detection_rows, track_rows, greys = [], [], []
        n_analysed, complete = 0, True
        with closing(prefetch(reader, depth=PREFETCH_FRAMES)) as frames:
            for frame in frames:
                found = self.detector(frame.image)
                full_res = reader.geometry.to_full_res(found.boxes).astype(np.float32)
                found = Detections(full_res, found.scores, found.class_ids)
                detection_rows.append(_detection_rows(frame.index, found))
                track_rows.append(_track_rows(frame.index, tracker.update(found), found))
                if frame.grey is not None:
                    greys.append(frame.grey)
                n_analysed += 1
                if deadline is not None and time.perf_counter() > deadline:
                    complete = frame.index + reader.stride >= reader.info.n_frames
                    break
        return PerceptionResult(
            info=reader.info,
            stride=reader.stride,
            detections=_concatenate(detection_rows, DETECTION_DTYPE),
            tracks=_confirmed_only(_concatenate(track_rows, TRACK_DTYPE)),
            seconds=time.perf_counter() - start,
            n_analysed=n_analysed,
            complete=complete,
            background=_median(greys),
        )


def _detection_rows(frame_index: int, detections: Detections) -> np.ndarray:
    rows = np.zeros(len(detections), dtype=DETECTION_DTYPE)
    rows["frame"] = frame_index
    rows["x0"], rows["y0"], rows["x1"], rows["y1"] = detections.boxes.T
    rows["score"] = detections.scores
    rows["class_id"] = detections.class_ids
    return rows


def _track_rows(frame_index: int, tracked: list[TrackedBox], found: Detections) -> np.ndarray:
    rows = np.zeros(len(tracked), dtype=TRACK_DTYPE)
    if not tracked:
        return rows
    rows["frame"] = frame_index
    rows["track_id"] = [t.track_id for t in tracked]
    rows["x0"], rows["y0"], rows["x1"], rows["y1"] = np.array([t.box for t in tracked]).T
    matched = found.boxes[[t.detection for t in tracked]]
    rows["det_x0"], rows["det_y0"], rows["det_x1"], rows["det_y1"] = matched.T
    rows["score"] = [t.score for t in tracked]
    rows["class_id"] = [t.class_id for t in tracked]
    rows["confirmed"] = [t.confirmed for t in tracked]
    return rows


def _concatenate(parts: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)


def _median(pictures: list[np.ndarray]) -> np.ndarray | None:
    """Per-pixel median of equally sized pictures: moving traffic vanishes, the road stays."""
    if not pictures:
        return None
    return np.median(np.stack(pictures), axis=0).astype(np.uint8)


def _confirmed_only(tracks: np.ndarray) -> np.ndarray:
    """Drop tracks that were never confirmed (one-frame blips). A confirmed track keeps its first
    row, where it was not yet confirmed: that is where the object first appeared."""
    confirmed_ids = np.unique(tracks["track_id"][tracks["confirmed"]])
    return tracks[np.isin(tracks["track_id"], confirmed_ids)]

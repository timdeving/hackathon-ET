"""The live demo's processing: one uploaded clip -> its events, risk curve and annotated video.

It runs the submission's own code, with the settings of the demo profile when the app sets
WIUT_PROFILE=demo (configs/profiles/demo.yaml: the detector on the CPU, no time limit). Part A's
analyse() finds the events and hands over every frame it analysed, to draw on. Part B's
RiskEstimator then steps through every frame; instead of running the detector a second time it
gets the detections Part A made on the same frames (ReplayDetector): the same model on the same
frames, in half the time on a CPU.
"""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src import part_a
from src.config import load_params
from src.perception.boxes import Detections
from src.risk import estimator
from src.rules.common import AMBER, GREEN, RED
from src.video.probe import VideoInfo, probe_video
from src.visualize import H264Writer, by_frame, draw_frame, events_under_way

MAX_SECONDS = 120.0  # the demo accepts clips up to 2 minutes
PHASE_NAMES = {RED: "red", AMBER: "amber", GREEN: "green"}
JPEG_QUALITY = 90  # the analysed frames, kept on disk until they are drawn on

Progress = Callable[[float, str], None]  # (fraction done, what is happening)


@dataclass
class DemoResult:
    info: VideoInfo
    events: list[list]  # [[start_sec, end_sec, label], ...], as detect_events() returns them
    risk: list[list]  # [[t_sec, score], ...], one per frame, as the harness records them
    video: Path  # the annotated clip, H.264
    predictions: Path  # events and risk in the harness's predictions.json format
    seconds: float  # processing time
    lights: list[list]  # the top road's phases as read: [[start, end, "red" | ...], ...]


class ReplayDetector:
    """Stands in for Part B's detector: returns the detections Part A made on the frame being
    analysed, in the working image's pixels, as the real detector would."""

    def __init__(self, detections: np.ndarray, width: int, info: VideoInfo, frame_now) -> None:
        self.scale = width / info.width
        self.input_size = (round(info.height * self.scale), width)  # (h, w), like the detector's
        self.by_frame = by_frame(detections)
        self.frame_now = frame_now  # () -> the frame number the estimator is analysing

    def __call__(self, image: np.ndarray) -> Detections:
        rows = self.by_frame.get(self.frame_now())
        if rows is None:
            return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                              np.zeros(0, np.int64))
        boxes = np.column_stack([rows["x0"], rows["y0"], rows["x1"], rows["y1"]]) * self.scale
        return Detections(boxes.astype(np.float32), rows["score"].astype(np.float32),
                          rows["class_id"].astype(np.int64))


def check(video_path: str | Path, max_seconds: float | None = MAX_SECONDS) -> VideoInfo:
    """The clip's properties; ValueError, with a message for the visitor, if the demo can't take
    it. max_seconds: the longest clip accepted (None: any length, for the website's renders)."""
    try:
        info = probe_video(video_path)
    except Exception as error:
        raise ValueError(f"This file can't be read as a video ({error}).") from error
    if info.n_frames < 2 or info.fps <= 0:
        raise ValueError("This file holds no video frames.")
    if max_seconds is not None and info.duration > max_seconds + 1.0:
        raise ValueError(f"The clip is {info.duration:.0f} s long; the demo takes up to "
                         f"{max_seconds:.0f} s. Please cut it shorter.")
    return info


def process(video_path: str | Path, workdir: str | Path, progress: Progress | None = None,
            max_seconds: float | None = MAX_SECONDS) -> DemoResult:
    """Run the system on one clip; everything it writes goes into workdir."""
    progress = progress or (lambda fraction, text: None)
    start = time.perf_counter()
    info = check(video_path, max_seconds)
    params = load_params()
    if params["risk"]["stride"] != params["video"]["stride"]:
        raise RuntimeError("the replay needs Part A and Part B to analyse the same frames")
    workdir = Path(workdir)
    frames_dir = workdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[int, Path]] = []

    def keep(frame) -> None:
        path = frames_dir / f"{frame.index:06d}.jpg"
        cv2.imwrite(str(path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved.append((frame.index, path))
        progress(0.85 * frame.index / info.n_frames,
                 f"Finding events: {frame.t_sec:.0f} s of {info.duration:.0f} s")

    progress(0.0, "Starting")
    analysis = part_a.analyse(str(video_path), on_frame=keep)
    progress(0.85, "Accident risk")
    risk = risk_curve(analysis, info, params)
    video = workdir / "annotated.mp4"
    render(analysis, info, saved, risk, video,
           lambda share: progress(0.9 + 0.1 * share, "Drawing the annotated video"))
    shutil.rmtree(frames_dir, ignore_errors=True)
    predictions = workdir / "predictions.json"
    predictions.write_text(json.dumps(
        {"team": "wiut-demo", "videos": {info.name: {"events": analysis.events, "risk": risk}}}
    ))
    return DemoResult(info, analysis.events, risk, video, predictions,
                      time.perf_counter() - start, light_spans(analysis, info.fps))


def light_spans(analysis: part_a.Analysis, fps: float) -> list[list]:
    """The phases of the arm whose lights were read (the scene map has one), as time spans."""
    phases = next(iter(analysis.phases.values()), None)
    if phases is None:
        return []
    return [[round(start, 2), round(end, 2), PHASE_NAMES.get(code, "unknown")]
            for start, end, code in phases.spans(fps)]


def risk_curve(analysis: part_a.Analysis, info: VideoInfo, params: dict) -> list[list]:
    """Part B's score on every frame, as the harness would record it, from Part A's detections.

    The estimator only looks at a frame's size (its filters for boxes cut off by the edge and
    for the far road), so it steps through one blank frame of the clip's size."""
    risk = estimator.RiskEstimator()
    replay = ReplayDetector(analysis.result.detections, params["video"]["width"], info,
                            lambda: risk.index - 1)
    blank = np.zeros((info.height, info.width, 3), np.uint8)
    previous, estimator._detector = estimator._detector, replay
    try:
        risk.reset({"video_id": info.name, "fps": info.fps, "width": info.width,
                    "height": info.height, "n_frames": info.n_frames})
        scores = [risk.step(blank, index / info.fps) for index in range(info.n_frames)]
    finally:
        estimator._detector = previous
    return [[round(index / info.fps, 4), round(score, 4)] for index, score in enumerate(scores)]


def render(analysis: part_a.Analysis, info: VideoInfo, saved: list[tuple[int, Path]],
           risk: list[list], path: Path, progress: Callable[[float], None]) -> None:
    """The analysed frames with the tracks, the events under way, the light and the risk drawn
    on them, as an H.264 clip at the analysed frame rate."""
    if not saved:
        raise RuntimeError("no frame was analysed")
    tracks = by_frame(analysis.result.tracks)
    none = analysis.result.tracks[:0]
    lights = next(iter(analysis.phases.values()), None)  # the scene map has one arm with lights
    first = cv2.imread(str(saved[0][1]))
    height, width = first.shape[:2]
    scale = (width / info.width, height / info.height)
    stride = saved[1][0] - saved[0][0] if len(saved) > 1 else 1
    with H264Writer(path, info.fps / stride, (width, height)) as writer:
        for k, (index, jpeg) in enumerate(saved):
            t_sec = index / info.fps
            light = int(lights.at(np.array([index]))[0]) if lights is not None else None
            writer.write(draw_frame(cv2.imread(str(jpeg)), tracks.get(index, none), scale,
                                    t_sec, events_under_way(analysis.events, t_sec),
                                    risk[index][1], light))
            if k % 20 == 0:
                progress(k / len(saved))

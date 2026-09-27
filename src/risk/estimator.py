"""Part B: a risk score per frame that an accident starts within the next 5 seconds.

The harness makes one RiskEstimator per video, calls reset() once, then step() for every frame
in order, and counts that time in the same budget as Part A. The estimator is causal: it sees
only the frames it's given and never Part A's results (tracks, events, the cache).

Every `stride`-th frame it shrinks the frame to the detector's width, detects road users, and
follows them with its own tracker. Each track's velocity is the slope of its last
velocity_sec of ground points (past frames only). Boxes cut off by the frame's edge (their
bottom isn't where they touch the road) and ground points on the far road at the top of the
picture (perspective squeezes distances there: every false alarm on the samples that the pair
filters left came from these two) take no part. The raw risk is the riskiest pair's time to
collision (src/risk/conflicts.py), smoothed with an exponential average of past values, so one
noisy frame can't raise an alarm. On the frames in between, step() returns the last score and
costs nothing.

It never raises (an exception would lose the whole risk curve): a failure keeps the last score.
Its own work is capped at the time Part A leaves for it (budget.part_b_extra_ratio x the
duration): past that it stops analysing, so it can't push the video over the time budget.
"""
from __future__ import annotations

import logging
import time
from collections import deque

import cv2
import numpy as np

from src.config import load_params
from src.features.tracks import PERSON, VEHICLES
from src.perception.boxes import Detections
from src.perception.tracker import ByteTracker
from src.risk.conflicts import OTHER, VEHICLE, VULNERABLE, conflict_risk

log = logging.getLogger(__name__)

RISK_HORIZON_SEC = 5.0  # the metric's anticipation horizon H (evaluate.py)
BICYCLE = 1

_detector = None  # loaded once per process, when solution.py is imported (load_models)


def load_models() -> None:
    """Load Part B's detector now, before the harness times any video. A failure is logged, not
    raised: step() then returns the base score for every frame."""
    global _detector
    try:
        # Imported here, not at the top: the detector needs PyTorch, which laptops don't have.
        from src.perception.detector import load_detector

        _detector = load_detector(load_params())
    except Exception:
        log.exception("Part B: could not load the detector; the risk stays at its base score")


class RiskEstimator:
    """Causal: step() sees frames in order and nothing else; it never opens the video."""

    def reset(self, meta: dict) -> None:
        """Start a new video. meta = {"video_id", "fps", "width", "height", "n_frames"}."""
        params = load_params()
        self.p = params["risk"]
        self.meta = meta
        fps = float(meta.get("fps") or 30.0)
        self.fps = fps
        self.tracker = ByteTracker(fps / self.p["stride"], params["tracker"])
        self.history: dict[int, deque] = {}  # track id -> recent (t, x, y, height)
        self.kind: dict[int, int] = {}
        self.index = 0
        self.last_score = float(self.p["base"])
        self.work_sec = 0.0
        duration = float(meta.get("n_frames") or 0) / fps
        self.work_budget = params["budget"]["part_b_extra_ratio"] * duration
        self.stopped = False
        self.failed = False

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within RISK_HORIZON_SEC) in [0, 1] for this frame.

        frame is BGR uint8 with shape (H, W, 3); t_sec is its timestamp in seconds.
        """
        index, self.index = self.index, self.index + 1
        if index % self.p["stride"] or _detector is None or self.stopped:
            return self.last_score
        start = time.perf_counter()
        try:
            raw = self._raw_risk(frame, t_sec)
            smoothing = self.p["smoothing"]
            self.last_score = float(np.clip(
                smoothing * raw + (1.0 - smoothing) * self.last_score, 0.0, 1.0
            ))
        except Exception:  # see the module docstring: keep the last score
            if not self.failed:  # once per video, not on every frame
                log.exception("Part B: step failed at %.2f s; keeping the last score", t_sec)
            self.failed = True
        self.work_sec += time.perf_counter() - start
        if self.work_sec > self.work_budget:
            self.stopped = True
            log.warning("%s: Part B stopped at %.0f s to stay inside the time budget",
                        self.meta.get("video_id"), t_sec)
        return self.last_score

    def _raw_risk(self, frame: np.ndarray, t_sec: float) -> float:
        """Detect, track, update each track's recent path, and score the riskiest pair."""
        height, width = frame.shape[:2]
        work_width = _detector.input_size[1]
        scale = work_width / width
        image = cv2.resize(frame, (work_width, max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
        found = _detector(image)
        boxes = (found.boxes / scale).astype(np.float32)
        tracked = self.tracker.update(Detections(boxes, found.scores, found.class_ids))
        margin = self.p["edge_margin"] * max(width, height)
        seen = set()
        for box in tracked:
            x0, _, x1, y1 = box.box
            cut_off = x0 < margin or x1 > width - margin or y1 > height - margin
            far = y1 < self.p["far_share"] * height
            if not box.confirmed or cut_off or far:
                continue
            path = self.history.setdefault(box.track_id, deque(maxlen=self.p["history_len"]))
            path.append((t_sec, (x0 + x1) / 2, y1, box.box[3] - box.box[1]))
            self.kind[box.track_id] = _kind(box.class_id)
            seen.add(box.track_id)
        for gone in [track for track in self.history if track not in seen]:
            del self.history[gone], self.kind[gone]
        return self._pair_risk(t_sec)

    def _pair_risk(self, t_sec: float) -> float:
        """The riskiest pair among the tracks with enough recent path to have a velocity."""
        points, velocities, heights, kinds = [], [], [], []
        for track, path in self.history.items():
            recent = [row for row in path if row[0] >= t_sec - self.p["velocity_sec"]]
            if len(recent) < self.p["min_points"]:
                continue
            rows = np.array(recent)
            times = rows[:, 0] - rows[:, 0].mean()
            points.append(rows[-1, 1:3])
            # least-squares slope of x and y against time
            velocities.append(times @ (rows[:, 1:3] - rows[:, 1:3].mean(axis=0)) / (times @ times))
            heights.append(max(float(np.median(rows[:, 3])), 1.0))
            kinds.append(self.kind[track])
        if len(points) < 2:
            return float(self.p["base"])
        return conflict_risk(np.array(points), np.array(velocities), np.array(heights),
                             np.array(kinds), self.p)


def _kind(class_id: int) -> int:
    if class_id in VEHICLES:
        return VEHICLE
    if class_id in (PERSON, BICYCLE):
        return VULNERABLE
    return OTHER

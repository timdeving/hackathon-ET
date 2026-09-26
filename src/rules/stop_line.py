"""stop_line: a vehicle stops past the stop line on red, without entering the junction.

Starts when the vehicle stops and ends when the signal turns green (task conventions). Until the
traffic lights' colours are read (PLAN.md B6), red is inferred from the queue and green from the
vehicle moving off:

- past the line: the front of the vehicle (the bottom point of its box furthest along its
  lane's flow) is beyond its arm's stop line, while its ground point isn't in the junction;
- red: during most of the stop, another vehicle of the same arm waits, stationary, in an
  incoming lane. With nobody else waiting there's no telling, so no call;
- the stop ends when the vehicle moves off, a stand-in for the light turning green.

The arm is the one of the last incoming lane the vehicle drove in before stopping.
"""
from __future__ import annotations

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import (
    FrameIndex,
    RuleContext,
    last_incoming_lane,
    mostly,
    span,
    spread,
    stops,
)
from src.scene.geometry import side_of_polyline
from src.scene.scene_map import Lane, SceneMap

LABEL = "stop_line"


def find(context: RuleContext) -> list[Segment]:
    p, scene, features = context.params, context.scene, context.features
    others = FrameIndex(features.rows_of(VEHICLES))
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        for first, last in stops(rows, features.dt, p["min_sec"], p["gap_sec"]):
            lane = last_incoming_lane(rows[: last + 1], scene)
            if lane is None or scene.lanes[lane].arm not in scene.stop_lines:
                continue
            stop = rows[first : last + 1]
            past = _front_past_line(stop, scene.lanes[lane], scene) & ~stop["junction"]
            if not mostly(past):
                continue
            if not mostly(_others_waiting(stop[spread(len(stop))], scene.lanes[lane].arm,
                                          others, scene)):
                continue
            start, end = span(rows["t"], first, last, features.dt)
            segments.append(Segment(start, end, LABEL, (int(info["track_id"]),)))
    return segments


def _front_past_line(rows: np.ndarray, lane: Lane, scene: SceneMap) -> np.ndarray:
    """For each row: is the front of the vehicle beyond its arm's stop line?"""
    line = scene.stop_lines[lane.arm]
    flow = lane.flow[-1] - lane.flow[-2]  # the lane's direction where it meets the stop line
    flow = flow / max(float(np.linalg.norm(flow)), 1e-9)
    bottom = np.stack(
        [
            np.column_stack([rows["left_x"], rows["left_y"]]),
            np.column_stack([rows["x"], rows["y"]]),
            np.column_stack([rows["right_x"], rows["right_y"]]),
        ],
        axis=1,
    )  # (N, 3, 2)
    front = bottom[np.arange(len(rows)), (bottom @ flow).argmax(axis=1)]
    beyond = line.mean(axis=0) + flow * 10.0  # a point just past the line, to learn which side
    downstream = np.sign(side_of_polyline(beyond[None], line))[0]
    return np.sign(side_of_polyline(front, line)) == downstream


def _others_waiting(rows: np.ndarray, arm: str, others: FrameIndex, scene: SceneMap) -> np.ndarray:
    """For each row: does another vehicle wait, stationary, in an incoming lane of `arm`?"""
    arm_lanes = [i for i, lane in enumerate(scene.lanes) if lane.arm == arm and lane.role == "in"]
    waiting = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        near = others.at(int(row["frame"]))
        waiting[i] = bool(
            (near["stationary"] & np.isin(near["lane"], arm_lanes)
             & (near["track_id"] != row["track_id"])).any()
        )
    return waiting

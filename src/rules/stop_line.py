"""stop_line: a vehicle stops past the stop line on red, without entering the junction.

Starts when the vehicle stops and ends when the signal turns green (task conventions):

- past the line: the front of the vehicle (the bottom point of its box furthest along its
  lane's flow) is beyond its arm's stop line, while its ground point isn't in the junction;
- on red, and the end: where the arm's traffic lights are read (docs/SIGNAL_DESIGN.md), the
  stop must mostly be on red before the light turns green, and it ends at green. Where they
  aren't, red is guessed from another vehicle of the arm waiting too (no call with nobody
  waiting), and the stop ends when the vehicle moves off.

The arm is the one of the last incoming lane the vehicle drove in before stopping; only lanes
under the arm's signal count.
"""
from __future__ import annotations

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import (
    GREEN,
    RED,
    UNKNOWN,
    FrameIndex,
    RuleContext,
    front_points,
    lane_direction,
    last_incoming_lane,
    mostly,
    past_stop_line,
    span,
    spread,
    stops,
)
from src.scene.scene_map import SceneMap

LABEL = "stop_line"


def find(context: RuleContext) -> list[Segment]:
    p, scene, features = context.params, context.scene, context.features
    others = FrameIndex(features.rows_of(VEHICLES))
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        for first, last in stops(rows, features.dt, p["min_sec"], p["gap_sec"]):
            lane = last_incoming_lane(rows[: last + 1], scene)
            if lane is None or not scene.lanes[lane].signal:
                continue
            if scene.lanes[lane].arm not in scene.stop_lines:
                continue
            stop = rows[first : last + 1]
            front = front_points(stop, lane_direction(lane, scene))
            if not mostly(past_stop_line(front, lane, scene) & ~stop["junction"]):
                continue
            timed = _on_red(rows, first, last, scene.lanes[lane].arm, context, others)
            if timed is not None:
                segments.append(Segment(*timed, LABEL, (int(info["track_id"]),)))
    return segments


def _on_red(
    rows: np.ndarray, first: int, last: int, arm: str, context: RuleContext, others: FrameIndex
) -> tuple[float, float] | None:
    """(start, end) if this stop is on red, else None: from the lights where they're read,
    otherwise guessed from the queue."""
    dt = context.features.dt
    start, end = span(rows["t"], first, last, dt)
    timeline = context.phases.get(arm)
    if timeline is not None:
        phase = timeline.at(rows["frame"][first : last + 1])
        if mostly(phase != UNKNOWN):
            green = np.flatnonzero(phase == GREEN)
            before_green = phase[: green[0]] if len(green) else phase
            known = before_green[before_green != UNKNOWN]
            if not mostly(known == RED):
                return None
            if len(green):  # the event ends when the signal turns green
                end = float(rows["t"][first + green[0]] - dt / 2)
            return start, end
    stop = rows[first : last + 1]
    waiting = _others_waiting(stop[spread(len(stop))], arm, others, context.scene)
    return (start, end) if mostly(waiting) else None


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

"""red_light: a vehicle crosses the stop line while its signal is red.

Starts when the front of the vehicle crosses the stop line and ends when it leaves the
intersection or the picture (task conventions):

- the front is the bottom point of the vehicle's box furthest along its lane's flow, and it
  must cross from the lane's side of the line;
- red comes only from the traffic lights (docs/SIGNAL_DESIGN.md). Where they aren't read
  there's no call: guessing red from waiting traffic would flag the first car away on green;
- the phase must have been red for at least red_grace_sec when the front crosses, since the
  first moment of red is too close to amber to call;
- leaving the intersection means reaching an outgoing lane or leaving the picture, and at most
  max_sec after the crossing.

Only lanes under the arm's signal count.
"""
from __future__ import annotations

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import (
    RED,
    RuleContext,
    front_points,
    lane_direction,
    last_incoming_lane,
    past_stop_line,
)
from src.scene.scene_map import SceneMap

LABEL = "red_light"


def find(context: RuleContext) -> list[Segment]:
    p, scene, features = context.params, context.scene, context.features
    if not context.phases:
        return []
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        crossing = _crossing(rows, scene)
        if crossing is None:
            continue
        index, lane = crossing
        timeline = context.phases.get(scene.lanes[lane].arm)
        if timeline is None or not scene.lanes[lane].signal:
            continue
        crossed = float(rows["t"][index - 1] + rows["t"][index]) / 2
        before = rows[(rows["t"] >= crossed - p["red_grace_sec"]) & (rows["t"] <= rows["t"][index])]
        if not (timeline.at(before["frame"]) == RED).all():
            continue
        end = min(_leaves(rows, index, scene, features.dt), crossed + p["max_sec"])
        segments.append(Segment(crossed, end, LABEL, (int(info["track_id"]),)))
    return segments


def _crossing(rows: np.ndarray, scene: SceneMap) -> tuple[int, int] | None:
    """(row, lane) where the vehicle's front first crosses its incoming lane's stop line, coming
    from that lane; None if it never does in view."""
    lane = last_incoming_lane(rows, scene)
    if lane is None or scene.lanes[lane].arm not in scene.stop_lines:
        return None
    past = past_stop_line(front_points(rows, lane_direction(lane, scene)), lane, scene)
    crossings = np.flatnonzero(past[1:] & ~past[:-1] & (rows["lane"][:-1] == lane)) + 1
    return (int(crossings[0]), lane) if len(crossings) else None


def _leaves(rows: np.ndarray, index: int, scene: SceneMap, dt: float) -> float:
    """When the vehicle leaves the intersection after row `index`: it reaches an outgoing lane,
    or its track ends (it leaves the picture)."""
    outgoing = [k for k, lane in enumerate(scene.lanes) if lane.role == "out"]
    reached = np.flatnonzero(np.isin(rows["lane"][index:], outgoing))
    last = index + int(reached[0]) if len(reached) else len(rows) - 1
    return float(rows["t"][last] + dt / 2)

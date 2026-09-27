"""red_light: a vehicle crosses the stop line while its signal is red.

Starts when the front of the vehicle crosses the stop line and ends when it leaves the
intersection or the picture (task conventions):

- the front is the bottom point of the vehicle's box furthest along its lane's flow, and it
  must cross from the lane's side of the line;
- a vehicle first seen already past the line (it came along the kerb, or crossed while hidden),
  near it (max_line_heights of its box heights) and followed for min_arrival_sec, counts from
  its first sighting: the dev labels mark a moped that arrives there on red and waits for the
  green as red_light (and stop_line);
- red comes only from the traffic lights (docs/SIGNAL_DESIGN.md). Where they aren't read
  there's no call: guessing red from waiting traffic would flag the first car away on green;
- the phase must have been red for at least red_grace_sec when the front crosses (or the
  vehicle is first seen), since the first moment of red is too close to amber to call;
- leaving the intersection means reaching an outgoing lane or an exit zone, or leaving the
  picture, and at most max_sec after the vehicle moves on from past the line (it may wait
  there for the green first).

Only lanes under the arm's signal count.
"""
from __future__ import annotations

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import (
    RED,
    PhaseTimeline,
    RuleContext,
    front_points,
    lane_direction,
    last_incoming_lane,
    nearest_incoming_lane,
    past_stop_line,
    stop_line_crossing,
)
from src.scene.geometry import distance_to_polyline
from src.scene.scene_map import SceneMap

LABEL = "red_light"


def find(context: RuleContext) -> list[Segment]:
    p, scene, features = context.params, context.scene, context.features
    if not context.phases:
        return []
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        passed = stop_line_crossing(rows, scene) or _first_seen_past(rows, scene, p, features.dt)
        if passed is None:
            continue
        index, lane = passed
        timeline = context.phases.get(scene.lanes[lane].arm)
        if timeline is None or not scene.lanes[lane].signal:
            continue
        if index > 0:
            crossed = float(rows["t"][index - 1] + rows["t"][index]) / 2
        else:
            crossed = float(rows["t"][0])
        if not _red_for(timeline, crossed, p["red_grace_sec"], features.fps):
            continue
        end = _leaves(rows, index, scene, features.dt, p["max_sec"])
        segments.append(Segment(crossed, end, LABEL, (int(info["track_id"]),)))
    return segments


def _first_seen_past(
    rows: np.ndarray, scene: SceneMap, p: dict, dt: float
) -> tuple[int, int] | None:
    """(0, lane) for a vehicle first seen already past its arm's stop line, out of the junction,
    near the line and followed for at least min_arrival_sec; None otherwise."""
    if rows["t"][-1] - rows["t"][0] + dt < p["min_arrival_sec"]:
        return None
    first = rows[:1]
    ground = np.column_stack([first["x"], first["y"]])
    lane = last_incoming_lane(first, scene)
    if lane is None:
        lane = nearest_incoming_lane(ground, scene)
    if lane is None or scene.lanes[lane].arm not in scene.stop_lines or first["junction"][0]:
        return None
    gap = distance_to_polyline(ground, scene.stop_lines[scene.lanes[lane].arm])[0]
    if gap > p["max_line_heights"] * max(float(first["height"][0]), 1.0):
        return None
    past = past_stop_line(front_points(first, lane_direction(lane, scene)), lane, scene)
    return (0, lane) if past[0] else None


def _red_for(timeline: PhaseTimeline, t: float, grace_sec: float, fps: float) -> bool:
    """Has the signal been red on every frame from grace_sec before t to t? (A video that
    starts in red counts as red since its first frame.)"""
    frames = np.arange(max(0, round((t - grace_sec) * fps)), round(t * fps) + 1)
    return bool((timeline.at(frames) == RED).all())


def _leaves(rows: np.ndarray, index: int, scene: SceneMap, dt: float, max_sec: float) -> float:
    """When the vehicle leaves the intersection after row `index`: it reaches an outgoing lane
    or an exit zone, or its track ends (it leaves the picture); at most max_sec after it moves
    on from past the line, since it may wait there for the green first."""
    after = rows[index:]
    outgoing = [k for k, lane in enumerate(scene.lanes) if lane.role == "out"]
    reached = np.flatnonzero(np.isin(after["lane"], outgoing) | (after["exit"] >= 0))
    last = int(reached[0]) if len(reached) else len(after) - 1
    waited = np.flatnonzero(after["stationary"][: last + 1])
    moves_on = float(after["t"][waited[-1]]) if len(waited) else float(after["t"][0])
    return min(float(after["t"][last] + dt / 2), moves_on + max_sec)

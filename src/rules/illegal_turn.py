"""illegal_turn: a turn from the wrong lane, or in a prohibited direction.

Starts when the vehicle starts turning and ends when the turn is complete (task conventions):

- where it came from: the incoming lane whose stop line its front crosses;
- where it went: the road it leaves by, meaning the first exit zone (`exit:<arm>`, where a road
  leaves the picture) or outgoing lane it reaches after that;
- illegal: that road isn't in the lane's `exits`. Unknown exits mean no call, and U-turns (back
  to the lane's own arm) are left to illegal_u_turn;
- start: where the heading first leaves the lane's direction by more than start_deg. A vehicle
  going straight on from a turn-only lane never turns, so its event starts at the stop line;
- end: where the heading settles within settle_deg of its direction when the new road is
  reached. Going straight on, the event ends where that road is reached.
"""
from __future__ import annotations

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import RuleContext, lane_direction, span, stop_line_crossing
from src.scene.scene_map import SceneMap

LABEL = "illegal_turn"


def find(context: RuleContext) -> list[Segment]:
    p, scene, features = context.params, context.scene, context.features
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        crossing = stop_line_crossing(rows, scene)
        if crossing is None:
            continue
        first, lane = crossing
        exits = scene.lanes[lane].exits
        left_by = _road_left_by(rows, first, scene)
        if exits is None or left_by is None:
            continue
        arm, reached = left_by
        if arm == scene.lanes[lane].arm or arm in exits:
            continue
        start, end = _turn(rows, first, reached, lane_direction(lane, scene), p)
        segments.append(Segment(*span(rows["t"], start, end, features.dt), LABEL,
                                (int(info["track_id"]),)))
    return segments


def _road_left_by(rows: np.ndarray, first: int, scene: SceneMap) -> tuple[str, int] | None:
    """(arm, row) of the first exit zone or outgoing lane the vehicle reaches from row `first`."""
    outgoing = {k: lane.arm for k, lane in enumerate(scene.lanes) if lane.role == "out"}
    after = rows[first:]
    in_zone = after["exit"] >= 0
    in_outgoing = np.isin(after["lane"], list(outgoing))
    hits = np.flatnonzero(in_zone | in_outgoing)
    if len(hits) == 0:
        return None
    i = int(hits[0])
    arm = scene.exit_arms[int(after["exit"][i])] if in_zone[i] else outgoing[int(after["lane"][i])]
    return arm, first + i


def _turn(
    rows: np.ndarray, first: int, reached: int, direction: np.ndarray, p: dict
) -> tuple[int, int]:
    """First and last row of the turn between the stop line (row `first`) and the new road
    (row `reached`), judged on rows where the vehicle moves (elsewhere heading is noise)."""
    rows_between = np.arange(first, reached + 1)
    moving = rows_between[rows["moving"][rows_between]]
    if len(moving) == 0:
        return first, reached
    heading = np.degrees(np.arctan2(rows["vy"][moving], rows["vx"][moving]))
    away = np.abs(_wrap(heading - np.degrees(np.arctan2(direction[1], direction[0]))))
    turning = np.flatnonzero(away > p["start_deg"])
    if len(turning) == 0:  # straight on: from the stop line to the new road
        return first, reached
    settled = np.abs(_wrap(heading - heading[-1])) <= p["settle_deg"]
    unsettled = np.flatnonzero(~settled[turning[0] :]) + turning[0]
    end = unsettled[-1] + 1 if len(unsettled) else turning[0]
    return int(moving[turning[0]]), int(moving[min(end, len(moving) - 1)])


def _wrap(degrees: np.ndarray) -> np.ndarray:
    """Angles folded into -180..180."""
    return (degrees + 180.0) % 360.0 - 180.0

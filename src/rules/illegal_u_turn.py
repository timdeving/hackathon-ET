"""illegal_u_turn: a U-turn where markings or signs prohibit it.

Starts when the vehicle starts turning and ends when the turn is complete (task conventions):
from where its heading first leaves its direction before the turn by more than start_deg, to
where it has turned by turn_deg, which must happen within max_sec. Whether it's prohibited comes
from the scene map: the lane the vehicle turned from has `exits` that leave out its own arm, or
the turn passes through a `no_uturn` zone. Unknown exits and no zone mean no call (unknown means
don't judge).
"""
from __future__ import annotations

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES, ground_points
from src.rules.common import RuleContext, last_incoming_lane, span
from src.scene.scene_map import SceneMap

LABEL = "illegal_u_turn"


def find(context: RuleContext) -> list[Segment]:
    p, features = context.params, context.features
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        for first, last in _u_turns(rows, p["turn_deg"], p["start_deg"], p["max_sec"]):
            if _prohibited(rows, first, last, context.scene, p["max_sec"]):
                start, end = span(rows["t"], first, last, features.dt)
                segments.append(Segment(start, end, LABEL, (int(info["track_id"]),)))
    return segments


def _u_turns(
    rows: np.ndarray, turn_deg: float, start_deg: float, max_sec: float
) -> list[tuple[int, int]]:
    """(first, last) rows of each U-turn: the heading, followed while moving, reverses by
    turn_deg within max_sec. first is where it left its earlier direction by start_deg."""
    moving = np.flatnonzero(rows["moving"])
    if len(moving) < 2:
        return []
    t = rows["t"][moving]
    heading = np.unwrap(np.arctan2(rows["vy"][moving], rows["vx"][moving]))
    turn, leave = np.radians(turn_deg), np.radians(start_deg)
    found, earliest = [], 0
    for j in range(len(moving)):
        while t[j] - t[earliest] > max_sec:
            earliest += 1
        change = np.abs(heading[j] - heading[earliest:j])
        if len(change) == 0 or change.max() < turn:
            continue
        before = earliest + int(change.argmax())  # the heading the vehicle turned away from
        left = np.flatnonzero(np.abs(heading[before : j + 1] - heading[before]) > leave)
        found.append((int(moving[before + left[0]]), int(moving[j])))
        earliest = j + 1  # look for the next U-turn after this one
    return found


def _prohibited(rows: np.ndarray, first: int, last: int, scene: SceneMap, max_sec: float) -> bool:
    """Is this U-turn banned: from a lane whose exits leave out its own arm, or through a
    no-U-turn zone?"""
    recent = rows[: first + 1]
    recent = recent[recent["t"] >= rows["t"][first] - max_sec]  # the lane just before the turn
    lane = last_incoming_lane(recent, scene)
    if lane is not None:
        exits = scene.lanes[lane].exits
        if exits is not None and scene.lanes[lane].arm not in exits:
            return True
    return bool(scene.in_no_uturn_zone(ground_points(rows[first : last + 1])).any())

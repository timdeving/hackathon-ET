"""solid_line_crossing: a vehicle crosses a solid marking, in a lane change or other manoeuvre.

Starts when a wheel crosses the line and ends when the vehicle is fully in the new lane (task
conventions). The bottom corners of the vehicle's box stand in for its wheels: the event runs
from one corner crossing the line to the other corner crossing it to the same side, within
max_sec. A corner that crosses and comes back before the other follows was a touch, not a
crossing.
"""
from __future__ import annotations

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import RuleContext
from src.scene.geometry import crosses_polyline, side_of_polyline

LABEL = "solid_line_crossing"


def find(context: RuleContext) -> list[Segment]:
    p, features = context.params, context.features
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        if len(rows) < 2:
            continue
        corners = {
            "left": np.column_stack([rows["left_x"], rows["left_y"]]),
            "right": np.column_stack([rows["right_x"], rows["right_y"]]),
        }
        between = (rows["t"][:-1] + rows["t"][1:]) / 2  # when a move from row i to i+1 happens
        for line in context.scene.solid_lines.values():
            crossings = []  # (time, corner, side it ends up on)
            for corner, xy in corners.items():
                moves = np.flatnonzero(crosses_polyline(xy[:-1], xy[1:], line))
                sides = np.sign(side_of_polyline(xy[moves + 1], line))
                for time, side in zip(between[moves].tolist(), sides.tolist(), strict=True):
                    crossings.append((time, corner, side))
            for start, end in _pair_corners(sorted(crossings), p["max_sec"], features.dt):
                segments.append(Segment(start, end, LABEL, (int(info["track_id"]),)))
    return segments


def _pair_corners(
    crossings: list[tuple[float, str, float]], max_sec: float, dt: float
) -> list[tuple[float, float]]:
    """(start, end) of each time one corner crosses and the other follows it to the same side."""
    spans = []
    pending = None
    for crossing in crossings:
        time, corner, side = crossing
        if (
            pending is not None
            and corner != pending[1]
            and side == pending[2]
            and time - pending[0] <= max_sec
        ):
            spans.append((pending[0], max(time, pending[0] + dt)))  # never zero length
            pending = None
        else:
            pending = crossing
    return spans

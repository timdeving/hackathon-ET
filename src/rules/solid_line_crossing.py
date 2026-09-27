"""solid_line_crossing: a vehicle crosses a solid marking, in a lane change or other manoeuvre.

Starts when a wheel crosses the line and ends when the vehicle is fully in the new lane (task
conventions). On this camera's diagonal view the bottom corners of a vehicle's box aren't its
wheels: the box is much wider than the car, and its two corners straddle a lane line even while
the car drives along it. So the crossing is where the vehicle's ground point (its bottom centre)
crosses the line, and the event runs from before_sec before that (a wheel reaches the line
first) to after_sec after it (the vehicle settles in the new lane).

A vehicle whose ground point crosses the same line more than once is driving along it (a bus
astride two lanes, a queue creeping), not changing lanes, and gets no call.
"""
from __future__ import annotations

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import RuleContext
from src.scene.geometry import crosses_polyline

LABEL = "solid_line_crossing"


def find(context: RuleContext) -> list[Segment]:
    p, features = context.params, context.features
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        if len(rows) < 2:
            continue
        ground = np.column_stack([rows["x"], rows["y"]])
        for line in context.scene.solid_lines.values():
            moves = np.flatnonzero(crosses_polyline(ground[:-1], ground[1:], line))
            if len(moves) != 1:
                continue
            crossed = float(rows["t"][moves[0]] + rows["t"][moves[0] + 1]) / 2
            segments.append(
                Segment(crossed - p["before_sec"], crossed + p["after_sec"], LABEL,
                        (int(info["track_id"]),))
            )
    return segments

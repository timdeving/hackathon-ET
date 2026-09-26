"""wrong_way: a vehicle moving against its lane's direction, including in the oncoming lane.

Starts when the vehicle enters a lane going the other way and ends when it's back in a lane
going its way, or leaves the picture (task conventions). "Against" means the cosine between its
heading and the lane's flow (the feature `agreement`) is below `against`, held for min_sec.
Slow movement doesn't count (min_speed): reversing out of a space, or creeping in a queue, would
otherwise read as wrong-way driving.
"""
from __future__ import annotations

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import RuleContext, runs

LABEL = "wrong_way"


def find(context: RuleContext) -> list[Segment]:
    p, features = context.params, context.features
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        against = (rows["agreement"] < p["against"]) & (rows["speed"] >= p["min_speed"])
        for start, end in runs(rows["t"], against, features.dt, p["gap_sec"], p["min_sec"]):
            segments.append(Segment(start, end, LABEL, (int(info["track_id"]),)))
    return segments

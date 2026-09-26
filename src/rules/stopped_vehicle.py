"""stopped_vehicle: a vehicle stationary on the carriageway for 10 s or more, not queued at a
signal.

Starts when the vehicle stops (dated back to that moment, not 10 s later) and ends when it moves
again or its track ends (task conventions). Not counted, each judged over most of the stop:

- stops off the road, in a parking area or at a bus stop (scene map);
- stops in the junction, where vehicles wait for a gap to turn;
- queued vehicles: within queue_gap of their own box heights of their lane's stop line, or of a
  stationary vehicle ahead of them in the same lane.
"""
from __future__ import annotations

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import FrameIndex, RuleContext, mostly, span, spread, stops
from src.scene.geometry import distance_to_polyline
from src.scene.scene_map import SceneMap

LABEL = "stopped_vehicle"


def find(context: RuleContext) -> list[Segment]:
    p, scene, features = context.params, context.scene, context.features
    others = FrameIndex(features.rows_of(VEHICLES))
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        for first, last in stops(rows, features.dt, p["min_sec"], p["gap_sec"]):
            stop = rows[first : last + 1]
            elsewhere = stop["parking"] | stop["bus_stop"] | stop["junction"]
            if not mostly(stop["on_road"]) or mostly(elsewhere):
                continue
            if mostly(_queued(stop[spread(len(stop))], others, scene, p["queue_gap"])):
                continue
            start, end = span(rows["t"], first, last, features.dt)
            segments.append(Segment(start, end, LABEL, (int(info["track_id"]),)))
    return segments


def _queued(rows: np.ndarray, others: FrameIndex, scene: SceneMap, gap: float) -> np.ndarray:
    """For each row: is the vehicle queued there? Within `gap` of its own box heights of its
    lane's stop line, or of a stationary vehicle ahead of it in the same lane."""
    queued = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        lane = int(row["lane"])
        if lane < 0:
            continue
        here = np.array([[row["x"], row["y"]]])
        reach = gap * row["height"]
        line = scene.stop_lines.get(scene.lanes[lane].arm)
        if scene.lanes[lane].role == "in" and line is not None:
            if distance_to_polyline(here, line)[0] <= reach:
                queued[i] = True
                continue
        near = others.at(int(row["frame"]))
        still = near["stationary"] & (near["track_id"] != row["track_id"])
        ahead = near[still & (near["lane"] == lane)]
        if len(ahead):
            flow = scene.flow_direction(np.array([lane]), here)[0]
            along = (np.column_stack([ahead["x"], ahead["y"]]) - here) @ flow
            queued[i] = bool(((along > 0) & (along <= reach)).any())
    return queued

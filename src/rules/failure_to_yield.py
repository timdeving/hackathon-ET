"""failure_to_yield: a vehicle drives through a crossing while a pedestrian is on it, or stepping
onto it.

Starts when the vehicle enters the crossing and ends when it leaves (task conventions): one
event per pass of one vehicle. The vehicle counts as on the crossing while any part of its
footprint (the bottom footprint_share of its box) is on it, so the pass lasts from its front
arriving to its rear leaving. A pedestrian counts while their feet are on the crossing or
within approach_margin_px of it. A queued vehicle standing on the crossing isn't driving through
it: the vehicle must be moving (min_speed) at a moment the pedestrian is there.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.events import Segment
from src.features.tracks import PERSON, VEHICLES, ground_points
from src.rules.common import RuleContext, footprint_points, runs

LABEL = "failure_to_yield"


def find(context: RuleContext) -> list[Segment]:
    p, scene, features = context.params, context.scene, context.features
    busy = _frames_with_pedestrians(context)
    if not busy:
        return []
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        points = footprint_points(rows, p["footprint_share"])
        on = scene.crossing_index(points.reshape(-1, 2)).reshape(len(rows), -1)
        for crossing, frames in busy.items():
            on_this = (on == crossing).any(axis=1)
            for start, end in runs(rows["t"], on_this, features.dt):
                during = on_this & (rows["t"] > start) & (rows["t"] < end)
                together = during & np.isin(rows["frame"], frames)
                if (together & (rows["speed"] >= p["min_speed"])).any():
                    segments.append(Segment(start, end, LABEL, (int(info["track_id"]),)))
    return segments


def _frames_with_pedestrians(context: RuleContext) -> dict[int, np.ndarray]:
    """For each crossing, the frames in which a pedestrian is on it or about to step onto it."""
    frames: dict[int, list[np.ndarray]] = defaultdict(list)
    for _, rows in context.features.tracks_of([PERSON], pedestrians_only=True):
        at = context.scene.crossing_index(ground_points(rows), context.params["approach_margin_px"])
        for crossing in np.unique(at[at >= 0]).tolist():
            frames[crossing].append(rows["frame"][at == crossing])
    return {crossing: np.unique(np.concatenate(parts)) for crossing, parts in frames.items()}

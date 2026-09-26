"""jaywalking: a pedestrian on the carriageway outside a crossing.

Starts when the person steps onto the road and ends when they leave it (task conventions).
Their position is their feet, the bottom-centre of the box. "On the road" means at least
edge_margin_px inside the road outline, so people waiting at the kerb don't count, and medians
and refuge islands aren't road (scene map). Riders and people seen inside vehicles are on the
road without being pedestrians, so they are skipped.
"""
from __future__ import annotations

from src.events import Segment
from src.features.tracks import PERSON, ground_points
from src.rules.common import RuleContext, runs

LABEL = "jaywalking"


def find(context: RuleContext) -> list[Segment]:
    p, scene, features = context.params, context.scene, context.features
    segments = []
    for info, rows in features.tracks_of([PERSON], pedestrians_only=True):
        feet = ground_points(rows)
        off_crossing = scene.on_road(feet, p["edge_margin_px"]) & (scene.crossing_index(feet) < 0)
        for start, end in runs(rows["t"], off_crossing, features.dt, p["gap_sec"], p["min_sec"]):
            segments.append(Segment(start, end, LABEL, (int(info["track_id"]),)))
    return segments

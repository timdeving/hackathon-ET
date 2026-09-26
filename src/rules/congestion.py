"""congestion: standstill or crawling traffic across all lanes of one direction.

Starts when the queue stops moving and ends when it clears (task conventions). A direction is
the lanes of one arm with one role (incoming or outgoing) in the scene map. At each analysed
frame it's jammed when at least min_vehicles vehicles are in its lanes, they occupy at least
lane_share of them, and their median speed is below crawl. A jam must last min_sec: long enough
to outlast a red light, since an ordinary red-light queue that clears on the next green isn't
congestion (our labelling convention, pending organizer question Q1).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import FrameIndex, RuleContext, runs
from src.scene.scene_map import SceneMap

LABEL = "congestion"


def find(context: RuleContext) -> list[Segment]:
    p, features = context.params, context.features
    vehicles = features.rows_of(VEHICLES)
    if len(vehicles) == 0:
        return []
    grid = np.arange(0, int(vehicles["frame"].max()) + 1, features.stride)  # analysed frames
    segments = []
    for lanes in _directions(context.scene).values():
        mine = FrameIndex(vehicles[np.isin(vehicles["lane"], lanes)])
        jammed = np.zeros(len(grid), dtype=bool)
        for frame in np.unique(mine.rows["frame"]):
            here = mine.at(int(frame))
            jammed[int(frame) // features.stride] = (
                len(here) >= p["min_vehicles"]
                and len(np.unique(here["lane"])) >= p["lane_share"] * len(lanes)
                and float(np.median(here["speed"])) < p["crawl"]
            )
        spans = runs(grid / features.fps, jammed, features.dt, p["gap_sec"], p["min_sec"])
        segments += [Segment(start, end, LABEL) for start, end in spans]
    return segments


def _directions(scene: SceneMap) -> dict[tuple[str, str], list[int]]:
    """Lane indices grouped by (arm, role): the directions traffic flows in."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, lane in enumerate(scene.lanes):
        groups[(lane.arm, lane.role)].append(index)
    return dict(groups)

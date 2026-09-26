"""What every rule shares: its inputs, and turning per-row conditions into timed segments."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.features.tracks import TrackFeatures
from src.scene.scene_map import SceneMap


@dataclass(frozen=True)
class RuleContext:
    """Everything a rule may look at for one video."""

    features: TrackFeatures
    scene: SceneMap
    params: Mapping[str, Any]  # the rule's own section of params.yaml: rules.<label>


def runs(
    t: np.ndarray, flags: np.ndarray, dt: float, gap_sec: float = 0.0, min_sec: float = 0.0
) -> list[tuple[float, float]]:
    """The time spans in which `flags` holds, over one track's rows.

    Gaps where it fails for at most gap_sec are bridged, so a flicker doesn't split an event;
    then spans shorter than min_sec are dropped. Each span reaches half an update beyond its
    first and last rows: the true start lies between the row before and the first row, and
    halfway is the best guess (likewise for the end).
    """
    hits = np.flatnonzero(flags)
    if len(hits) == 0:
        return []
    spans = []
    first = last = hits[0]
    for i in hits[1:]:
        if t[i] - t[last] > gap_sec + dt * 1.5:  # more than gap_sec of failing rows between
            spans.append((first, last))
            first = i
        last = i
    spans.append((first, last))
    return [
        (t[a] - dt / 2, t[b] + dt / 2)
        for a, b in spans
        if t[b] - t[a] + dt >= min_sec
    ]


def footprint_points(rows: np.ndarray, share: float) -> np.ndarray:
    """(N, 6, 2) points covering the part of each box that touches the road.

    The bottom `share` of a vehicle's box is roughly its outline on the ground: its bottom
    corners and centre, and the same three points share x box height higher. Sampling both
    means both the front and the rear of a vehicle count, whichever way it's driving.
    """
    lift = share * rows["height"]
    centre_x, centre_y = rows["x"], rows["y"]
    bottom = [
        (rows["left_x"], rows["left_y"]),
        (centre_x, centre_y),
        (rows["right_x"], rows["right_y"]),
    ]
    points = [np.column_stack([x, y]) for x, y in bottom]
    points += [np.column_stack([x, y - lift]) for x, y in bottom]
    return np.stack(points, axis=1)

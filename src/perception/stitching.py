"""Track stitching for Part A: join track fragments that belong to one object.

The tracker forgets an object it hasn't matched for tracker.lost_seconds (1 s), so a car hidden
behind a bus for longer comes back under a new ID. In the samples about a quarter of all tracks
vanish mid-picture with a plausible continuation. Broken IDs reset stopped_vehicle's 10-second
clock and split wrong-way and pedestrian paths. Part A has the whole video, so it can look at both
ends of every gap: a track that vanishes away from the picture's edges is joined to one of the
same kind that appears a few seconds later where the first one's motion says it should be.

Joined fragments share one track_id (the first fragment's); the tracker's own ID stays in
`tracker_id`. The gap is left as it is: the per-object features interpolate across it. Part B
never uses this, since it looks at future frames.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.perception.pipeline import TRACK_DTYPE, PerceptionResult

log = logging.getLogger(__name__)

STITCHED_DTYPE = np.dtype(TRACK_DTYPE.descr + [("tracker_id", np.int32)])
NO_JOIN = 1e6  # assignment cost of a pair that may not be joined: above any accepted cost
GAP_TIE_BREAK = 0.01  # per second of gap: between equally close candidates, prefer the sooner


@dataclasses.dataclass(frozen=True)
class _Ends:
    """The first or the last appearance of every track, as parallel arrays."""

    track_id: np.ndarray  # (T,)
    group: np.ndarray  # (T,) class group, as the tracker groups classes
    t: np.ndarray  # (T,) seconds
    point: np.ndarray  # (T, 2) bottom-centre of the raw detection box, video pixels
    height: np.ndarray  # (T,) box height, pixels
    velocity: np.ndarray  # (T, 2) of the point, pixels per second
    at_edge: np.ndarray  # (T,) the box touches the picture's border margin


def stitch_tracks(result: PerceptionResult, params: Mapping[str, Any]) -> PerceptionResult:
    """The same result with fragments of one object joined under one track_id.

    params: the whole of configs/params.yaml; the `stitching` section and the tracker's class
    groups are used. The track table gains a `tracker_id` column holding the tracker's own ID.
    """
    table = np.sort(result.tracks, order=["track_id", "frame"])
    stitched = np.zeros(len(table), dtype=STITCHED_DTYPE)
    for name in TRACK_DTYPE.names:
        stitched[name] = table[name]
    stitched["tracker_id"] = table["track_id"]
    if len(table) == 0:
        return dataclasses.replace(result, tracks=stitched)

    settings = params["stitching"]
    group_of = {
        class_id: group
        for group, class_ids in enumerate(params["tracker"]["class_groups"].values())
        for class_id in class_ids
    }
    first, last = _track_ends(table, result, group_of, settings)
    links = _links(last, first, settings)

    # Follow each chain A -> B -> C from its start: every link's earlier track ends before its
    # later one does, so in order of that end time an earlier track's root is always final.
    end_time = dict(zip(last.track_id.tolist(), last.t.tolist(), strict=True))
    root = np.arange(int(table["track_id"].max()) + 1)
    for earlier, later in sorted(links, key=lambda link: end_time[link[0]]):
        root[later] = root[earlier]
    stitched["track_id"] = root[stitched["tracker_id"]]
    log.info("%s: %d track fragments joined, %d tracks -> %d", result.info.name, len(links),
             len(first.track_id), len(first.track_id) - len(links))
    return dataclasses.replace(result, tracks=np.sort(stitched, order=["track_id", "frame"]))


def _track_ends(
    table: np.ndarray,
    result: PerceptionResult,
    group_of: Mapping[int, int],
    settings: Mapping[str, Any],
) -> tuple[_Ends, _Ends]:
    """The first and the last appearance of every track in a table sorted by track and frame."""
    info = result.info
    ids, starts = np.unique(table["track_id"], return_index=True)
    bounds = [*starts.tolist(), len(table)]
    t = table["frame"] / info.fps
    x = (table["det_x0"] + table["det_x1"]) / 2
    y = table["det_y1"]
    height = table["det_y1"] - table["det_y0"]
    margin_x, margin_y = settings["edge_margin"] * info.width, settings["edge_margin"] * info.height
    at_edge = (
        (table["det_x0"] < margin_x)
        | (table["det_x1"] > info.width - margin_x)
        | (table["det_y0"] < margin_y)
        | (table["det_y1"] > info.height - margin_y)
    )
    window = max(2, round(settings["velocity_sec"] * info.fps / result.stride))  # rows

    def velocity(rows: slice) -> np.ndarray:
        """Least-squares velocity of the point over some rows; zero from fewer than two."""
        times = t[rows]
        if len(times) < 2:
            return np.zeros(2)
        return np.array([np.polyfit(times, x[rows], 1)[0], np.polyfit(times, y[rows], 1)[0]])

    ends: dict[str, list] = {"first": [], "last": []}
    for i, track_id in enumerate(ids.tolist()):
        begin, end = bounds[i], bounds[i + 1]
        classes, which = np.unique(table["class_id"][begin:end], return_inverse=True)
        votes = np.bincount(which, weights=table["score"][begin:end])
        voted = int(classes[votes.argmax()])  # the detector flips between car, truck and bus
        group = group_of.get(voted, -1 - voted)  # a class in no group is its own group
        for name, index, span in (
            ("first", begin, slice(begin, min(end, begin + window))),
            ("last", end - 1, slice(max(begin, end - window), end)),
        ):
            ends[name].append((track_id, group, t[index], (x[index], y[index]), height[index],
                               velocity(span), at_edge[index]))
    return _as_ends(ends["first"]), _as_ends(ends["last"])


def _as_ends(items: list[tuple]) -> _Ends:
    track_id, group, t, point, height, velocity, at_edge = zip(*items, strict=True)
    return _Ends(
        track_id=np.array(track_id),
        group=np.array(group),
        t=np.array(t),
        point=np.array(point, dtype=np.float64),
        height=np.maximum(np.array(height, dtype=np.float64), 1.0),
        velocity=np.array(velocity, dtype=np.float64),
        at_edge=np.array(at_edge, dtype=bool),
    )


def _links(last: _Ends, first: _Ends, settings: Mapping[str, Any]) -> list[tuple[int, int]]:
    """(earlier, later) track IDs to join: each ending track to at most one starting track.

    A pair may be joined if both are the same kind of road user, the later one starts after the
    earlier one ends but within max_gap_sec, neither end touches the picture's border (objects
    that leave the picture really are gone), the box heights agree within max_size_change, and
    the two ends, each carried by its own velocity to the middle of the gap, meet within
    max_distance box heights. The closest pairs are chosen by optimal assignment.
    """
    gap = first.t[None, :] - last.t[:, None]
    size_change = first.height[None, :] / last.height[:, None]
    meet_last = last.point[:, None, :] + last.velocity[:, None, :] * gap[..., None] / 2
    meet_first = first.point[None, :, :] - first.velocity[None, :, :] * gap[..., None] / 2
    scale = (last.height[:, None] + first.height[None, :]) / 2
    mismatch = np.linalg.norm(meet_last - meet_first, axis=2) / scale

    allowed = (
        (last.group[:, None] == first.group[None, :])
        & (gap > 0)
        & (gap <= settings["max_gap_sec"])
        & ~last.at_edge[:, None]
        & ~first.at_edge[None, :]
        & (size_change <= settings["max_size_change"])
        & (size_change >= 1 / settings["max_size_change"])
        & (mismatch <= settings["max_distance"])
    )
    if not allowed.any():
        return []
    cost = np.where(allowed, mismatch + GAP_TIE_BREAK * gap, NO_JOIN)
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(last.track_id[r]), int(first.track_id[c]))
        for r, c in zip(rows, cols, strict=True)
        if allowed[r, c]
    ]

"""What every rule shares: its inputs, and turning per-row conditions into timed segments."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from src.features.tracks import TrackFeatures
from src.scene.geometry import side_of_polyline
from src.scene.scene_map import SceneMap

# Signal phase codes. The GPU PC's src/scene/signals.py returns these (docs/SIGNAL_DESIGN.md §3).
UNKNOWN, RED, AMBER, GREEN = 0, 1, 2, 3


class PhaseTimeline(Protocol):
    """One arm's signal phase over a video, read from the traffic lights."""

    def at(self, frames: np.ndarray) -> np.ndarray:
        """The phase code at each of these frame numbers; UNKNOWN where it isn't known."""
        ...


@dataclass(frozen=True)
class RuleContext:
    """Everything a rule may look at for one video."""

    features: TrackFeatures
    scene: SceneMap
    params: Mapping[str, Any]  # the rule's own section of params.yaml: rules.<label>
    # Signal phase per arm, where the lights can be read; empty until they can.
    phases: Mapping[str, PhaseTimeline] = field(default_factory=dict)


def runs(
    t: np.ndarray, flags: np.ndarray, dt: float, gap_sec: float = 0.0, min_sec: float = 0.0
) -> list[tuple[float, float]]:
    """The time spans in which `flags` holds, over one track's rows (or any time grid).

    Gaps where it fails for at most gap_sec are bridged, so a flicker doesn't split an event;
    then spans shorter than min_sec are dropped. Each span reaches half an update beyond its
    first and last rows: the true start lies between the row before and the first row, and
    halfway is the best guess (likewise for the end).
    """
    return [
        span(t, a, b, dt)
        for a, b in index_runs(t, flags, dt, gap_sec)
        if t[b] - t[a] + dt >= min_sec
    ]


def index_runs(
    t: np.ndarray, flags: np.ndarray, dt: float, gap_sec: float = 0.0
) -> list[tuple[int, int]]:
    """First and last row (inclusive) of each run of `flags`, bridging gaps up to gap_sec."""
    hits = np.flatnonzero(flags)
    if len(hits) == 0:
        return []
    spans = []
    first = last = hits[0]
    for i in hits[1:]:
        if t[i] - t[last] > gap_sec + dt * 1.5:  # more than gap_sec of failing rows between
            spans.append((int(first), int(last)))
            first = i
        last = i
    spans.append((int(first), int(last)))
    return spans


def span(t: np.ndarray, first: int, last: int, dt: float) -> tuple[float, float]:
    """The times rows first..last cover, half an update beyond each end (see runs())."""
    return float(t[first] - dt / 2), float(t[last] + dt / 2)


def stops(rows: np.ndarray, dt: float, min_sec: float, gap_sec: float) -> list[tuple[int, int]]:
    """First and last row of each time one object stands still for at least min_sec.

    A stop runs from where the object stopped moving to where it moves again (speed below
    moving_speed), as the task's conventions date stops. The stationary flag alone would start
    late and end early: it waits for a lower speed (hysteresis).
    """
    extended: list[tuple[int, int]] = []
    for first, last in index_runs(rows["t"], rows["stationary"], dt, gap_sec):
        while first > 0 and not rows["moving"][first - 1]:
            first -= 1
        while last < len(rows) - 1 and not rows["moving"][last + 1]:
            last += 1
        if extended and first <= extended[-1][1]:  # never moved in between: one stop
            first = extended.pop()[0]
        extended.append((first, last))
    return [(a, b) for a, b in extended if rows["t"][b] - rows["t"][a] + dt >= min_sec]


def mostly(flags: np.ndarray) -> bool:
    """True for at least half of the rows: how a condition is judged over a whole stop."""
    return len(flags) > 0 and float(np.mean(flags)) >= 0.5


def spread(n: int, at_most: int = 50) -> np.ndarray:
    """Up to at_most row indices spread evenly over n rows, for judging a long stop cheaply."""
    return np.unique(np.linspace(0, n - 1, min(n, at_most)).astype(int)) if n else np.zeros(0, int)


def front_points(rows: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """(N, 2) the front of each vehicle: the bottom point of its box (a corner or the centre)
    furthest along the travel direction `flow`."""
    bottom = np.stack(
        [
            np.column_stack([rows["left_x"], rows["left_y"]]),
            np.column_stack([rows["x"], rows["y"]]),
            np.column_stack([rows["right_x"], rows["right_y"]]),
        ],
        axis=1,
    )  # (N, 3, 2)
    return bottom[np.arange(len(rows)), (bottom @ flow).argmax(axis=1)]


def past_stop_line(points: np.ndarray, lane_index: int, scene: SceneMap) -> np.ndarray:
    """For each point: is it beyond the stop line of the lane's arm, on the junction side?"""
    lane = scene.lanes[lane_index]
    line = scene.stop_lines[lane.arm]
    flow = lane_direction(lane_index, scene)
    beyond = line.mean(axis=0) + flow * 10.0  # a point just past the line, to learn which side
    downstream = np.sign(side_of_polyline(beyond[None], line))[0]
    return np.sign(side_of_polyline(points, line)) == downstream


def lane_direction(lane_index: int, scene: SceneMap) -> np.ndarray:
    """The unit direction of a lane where it ends: for an incoming lane, at its stop line."""
    flow = scene.lanes[lane_index].flow
    step = flow[-1] - flow[-2]
    return step / max(float(np.linalg.norm(step)), 1e-9)


def stop_line_crossing(rows: np.ndarray, scene: SceneMap) -> tuple[int, int] | None:
    """(row, lane) where the vehicle's front first crosses its incoming lane's stop line, coming
    from that lane; None if it never does in view."""
    lane = last_incoming_lane(rows, scene)
    if lane is None or scene.lanes[lane].arm not in scene.stop_lines:
        return None
    past = past_stop_line(front_points(rows, lane_direction(lane, scene)), lane, scene)
    crossings = np.flatnonzero(past[1:] & ~past[:-1] & (rows["lane"][:-1] == lane)) + 1
    return (int(crossings[0]), lane) if len(crossings) else None


def last_incoming_lane(rows: np.ndarray, scene: SceneMap) -> int | None:
    """The last incoming lane (index into scene.lanes) among these rows, or None."""
    incoming = np.array([lane.role == "in" for lane in scene.lanes] + [False])
    in_lane = incoming[rows["lane"]]  # lane -1 picks the final False
    hits = np.flatnonzero(in_lane)
    return int(rows["lane"][hits[-1]]) if len(hits) else None


class FrameIndex:
    """Rows of many tracks sorted by frame, so everything seen in one frame is one slice."""

    def __init__(self, rows: np.ndarray) -> None:
        self.rows = np.sort(rows, order=["frame", "track_id"])

    def at(self, frame: int) -> np.ndarray:
        frames = self.rows["frame"]
        first = np.searchsorted(frames, frame, side="left")
        return self.rows[first : np.searchsorted(frames, frame, side="right")]


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

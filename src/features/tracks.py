"""Per-object features for the rules: where each tracked object is, how it moves, and in which
zone of the scene map.

Input is perception's track table: video pixels, one row per track per analysed frame in which
the tracker matched it. Output is one row per track per analysed frame, from its first
appearance to its last, in reference-picture pixels:

- frames where the tracker briefly lost the object are filled by linear interpolation;
- boxes are smoothed over time, so detector jitter doesn't read as motion;
- points are mapped into the reference picture (camera alignment), then looked up in the scene
  map.

Speeds are in box heights per second (decided 2026-09-26). Perspective makes a pixel mean
different distances across the picture, but an object's own height shrinks with distance the
same way, so the ratio compares across the picture until the map has ground points in metres.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.perception.pipeline import PerceptionResult
from src.scene.scene_map import SceneMap

# COCO class ids, as the detector reports them.
PERSON = 0
VEHICLES = (2, 3, 5, 7)  # car, motorcycle, bus, truck: a "vehicle" in the rules
TWO_WHEELERS = (1, 3)  # bicycle, motorcycle: what a rider rides
ENCLOSED = (2, 5, 7)  # car, bus, truck: what a person can be seen inside


class PointMapper(Protocol):
    """Maps video pixels to reference-picture pixels: camera alignment (the GPU PC's
    src/scene/alignment.py), or NoAlignment until that exists (HANDOVER.md §5.2)."""

    def to_reference(self, points: np.ndarray, frames: np.ndarray | None = None) -> np.ndarray:
        ...


class NoAlignment:
    """The identity: treats a video as framed exactly like the reference picture."""

    def to_reference(self, points: np.ndarray, frames: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(points, dtype=np.float64)


FEATURE_DTYPE = np.dtype(
    [
        ("track_id", np.int32),
        ("frame", np.int32),
        ("t", np.float64),  # seconds: frame / fps, as the harness computes t_sec
        ("filled", np.bool_),  # the tracker had lost the object here: position interpolated
        ("x", np.float64),  # ground point: bottom-centre of the box, in reference pixels
        ("y", np.float64),
        ("height", np.float64),  # box height, in reference pixels
        ("left_x", np.float64),  # bottom-left corner of the box
        ("left_y", np.float64),
        ("right_x", np.float64),  # bottom-right corner of the box
        ("right_y", np.float64),
        ("vx", np.float64),  # velocity of the ground point, reference pixels per second
        ("vy", np.float64),
        ("speed", np.float64),  # box heights per second
        ("moving", np.bool_),  # speed >= moving_speed; below it the direction is noise
        ("stationary", np.bool_),  # stopped, with hysteresis (stop_enter / stop_exit)
        ("on_road", np.bool_),  # carriageway, islands excluded
        ("lane", np.int16),  # index into SceneMap.lanes, or -1
        ("crossing", np.int16),  # index into SceneMap.crossings, or -1
        ("junction", np.bool_),
        ("parking", np.bool_),
        ("bus_stop", np.bool_),
        # cosine between the heading and the lane's flow: +1 with it, -1 against it; NaN when
        # not moving or not in a lane
        ("agreement", np.float64),
    ]
)

TRACK_INFO_DTYPE = np.dtype(
    [
        ("track_id", np.int32),
        ("class_id", np.int16),  # score-weighted vote over the track's detections
        ("rider", np.bool_),  # a person on a bicycle or motorcycle
        ("occupant", np.bool_),  # a person seen inside a car, bus or truck
        ("start", np.int64),  # the track's rows are rows[start:stop]
        ("stop", np.int64),
    ]
)


@dataclass(frozen=True)
class TrackFeatures:
    """Features of every track of one video."""

    rows: np.ndarray  # FEATURE_DTYPE, sorted by track, then frame
    tracks: np.ndarray  # TRACK_INFO_DTYPE, one per track, in the same order
    fps: float
    stride: int

    @property
    def dt(self) -> float:
        """Seconds between two analysed frames."""
        return self.stride / self.fps

    def tracks_of(
        self, class_ids: Sequence[int], pedestrians_only: bool = False
    ) -> Iterator[tuple[np.void, np.ndarray]]:
        """(info, rows) of each track whose voted class is in class_ids.

        pedestrians_only skips riders and people seen inside vehicles, who are on the road
        without being pedestrians.
        """
        for index in np.flatnonzero(np.isin(self.tracks["class_id"], class_ids)):
            info = self.tracks[index]
            if pedestrians_only and (info["rider"] or info["occupant"]):
                continue
            yield info, self.rows[info["start"] : info["stop"]]

    def rows_of(self, class_ids: Sequence[int], pedestrians_only: bool = False) -> np.ndarray:
        """All rows of the tracks tracks_of() yields, in one array."""
        parts = [rows for _, rows in self.tracks_of(class_ids, pedestrians_only)]
        return np.concatenate(parts) if parts else self.rows[:0]


def ground_points(rows: np.ndarray) -> np.ndarray:
    """(N, 2) ground points of feature rows, for scene-map lookups."""
    return np.column_stack([rows["x"], rows["y"]])


def compute_features(
    result: PerceptionResult,
    mapper: PointMapper,
    scene: SceneMap | None,
    params: Mapping[str, Any],
    causal: bool = False,
) -> TrackFeatures:
    """Features of every track in a perception result.

    params: the `features` section of configs/params.yaml. scene=None leaves every zone empty
    (lane and crossing -1). causal=True uses past frames only, for Part B: trailing smoothing and
    backward differences instead of centred ones.
    """
    fps, stride = result.info.fps, result.stride
    table = np.sort(result.tracks, order=["track_id", "frame"])
    ids, firsts = np.unique(table["track_id"], return_index=True)
    bounds = [*firsts.tolist(), len(table)]

    pieces, infos, start = [], [], 0
    for i, track_id in enumerate(ids.tolist()):
        observed = table[bounds[i] : bounds[i + 1]]
        rows = _one_track(observed, fps, stride, mapper, params, causal)
        pieces.append(rows)
        infos.append((track_id, _voted_class(observed), False, False, start, start + len(rows)))
        start += len(rows)
    rows = np.concatenate(pieces) if pieces else np.zeros(0, dtype=FEATURE_DTYPE)
    tracks = np.array(infos, dtype=TRACK_INFO_DTYPE)

    _mark_riders_and_occupants(table, tracks, params)
    _add_zones(rows, scene)
    return TrackFeatures(rows, tracks, fps, stride)


def _one_track(
    observed: np.ndarray,
    fps: float,
    stride: int,
    mapper: PointMapper,
    params: Mapping[str, Any],
    causal: bool,
) -> np.ndarray:
    """Feature rows of one track: every analysed frame from its first to its last."""
    frames = np.arange(observed["frame"][0], observed["frame"][-1] + 1, stride)
    rows = np.zeros(len(frames), dtype=FEATURE_DTYPE)
    rows["track_id"] = observed["track_id"][0]
    rows["frame"] = frames
    rows["t"] = frames / fps
    rows["filled"] = ~np.isin(frames, observed["frame"])

    # The raw detections, not the tracker's Kalman boxes: centred smoothing of the detections has
    # no lag, while the Kalman box trails a braking or turning object, and event boundaries have
    # to be right to a fraction of a second.
    window = _window(params["smooth_sec"], fps / stride)
    box = {
        side: _smooth(np.interp(frames, observed["frame"], observed[f"det_{side}"]), window, causal)
        for side in ("x0", "y0", "x1", "y1")
    }
    centre = (box["x0"] + box["x1"]) / 2
    corners = {
        "ground": np.column_stack([centre, box["y1"]]),
        "top": np.column_stack([centre, box["y0"]]),
        "left": np.column_stack([box["x0"], box["y1"]]),
        "right": np.column_stack([box["x1"], box["y1"]]),
    }
    mapped = {name: mapper.to_reference(points, frames) for name, points in corners.items()}
    rows["x"], rows["y"] = mapped["ground"].T
    rows["left_x"], rows["left_y"] = mapped["left"].T
    rows["right_x"], rows["right_y"] = mapped["right"].T
    rows["height"] = np.maximum(np.linalg.norm(mapped["ground"] - mapped["top"], axis=1), 1.0)

    rows["vx"] = _derivative(rows["x"], rows["t"], causal)
    rows["vy"] = _derivative(rows["y"], rows["t"], causal)
    rows["speed"] = np.hypot(rows["vx"], rows["vy"]) / rows["height"]
    rows["moving"] = rows["speed"] >= params["moving_speed"]
    rows["stationary"] = _hysteresis(rows["speed"], params["stop_enter"], params["stop_exit"])
    return rows


def _window(seconds: float, updates_per_second: float) -> int:
    """An odd number of samples spanning about `seconds`, so a centred window stays centred."""
    samples = max(1, round(seconds * updates_per_second))
    return samples if samples % 2 else samples + 1


def _smooth(values: np.ndarray, window: int, causal: bool) -> np.ndarray:
    """Moving average over `window` samples: centred, or trailing (causal). Near the ends the
    window shrinks rather than padding, so the first and last positions stay where they were
    seen."""
    values = values.astype(np.float64)
    if window <= 1 or len(values) < 2:
        return values
    kernel, ones = np.ones(window), np.ones(len(values))
    if causal:
        sums = np.convolve(values, kernel)[: len(values)]
        counts = np.convolve(ones, kernel)[: len(values)]
    else:
        sums = np.convolve(values, kernel, mode="same")
        counts = np.convolve(ones, kernel, mode="same")
    return sums / counts


def _derivative(values: np.ndarray, t: np.ndarray, causal: bool) -> np.ndarray:
    """Rate of change per second: central differences, or backward ones (causal)."""
    if len(values) < 2:
        return np.zeros(len(values))
    if causal:
        rate = np.diff(values) / np.diff(t)
        return np.concatenate([[rate[0]], rate])
    return np.gradient(values, t)


def _hysteresis(speed: np.ndarray, enter: float, leave: float) -> np.ndarray:
    """Stationary from where speed drops below `enter` until it rises above `leave`. Two
    thresholds keep the state from flickering while speed hovers around one of them."""
    state = np.zeros(len(speed), dtype=bool)
    stopped = False
    for i, value in enumerate(speed):
        if stopped and value > leave:
            stopped = False
        elif not stopped and value < enter:
            stopped = True
        state[i] = stopped
    return state


def _voted_class(observed: np.ndarray) -> int:
    """The class the detector gave the track most, weighting each detection by its score: the
    detector flips a vehicle between car, truck and bus from frame to frame."""
    classes, which = np.unique(observed["class_id"], return_inverse=True)
    return int(classes[np.bincount(which, weights=observed["score"]).argmax()])


def _mark_riders_and_occupants(
    table: np.ndarray, tracks: np.ndarray, params: Mapping[str, Any]
) -> None:
    """Flag person tracks that ride a two-wheeler, or sit inside a car, bus or truck.

    A rider's box overlaps a bicycle or motorcycle box by more than rider_overlap of its own
    area in at least half of its frames; an occupant's box lies inside a vehicle box by more than
    occupant_overlap in at least occupant_frames of them. Someone walking past a parked car
    overlaps it only for a moment, so neither flag catches them.
    """
    people = table[table["class_id"] == PERSON]
    if len(people) == 0:
        return
    on_two_wheeler = _covered_share(people, table[np.isin(table["class_id"], TWO_WHEELERS)])
    inside_vehicle = _covered_share(people, table[np.isin(table["class_id"], ENCLOSED)])
    for index in np.flatnonzero(tracks["class_id"] == PERSON):
        mine = people["track_id"] == tracks["track_id"][index]
        riding = (on_two_wheeler[mine] > params["rider_overlap"]).mean()
        inside = (inside_vehicle[mine] > params["occupant_overlap"]).mean()
        tracks["rider"][index] = riding >= 0.5
        tracks["occupant"][index] = inside >= params["occupant_frames"]


def _covered_share(people: np.ndarray, others: np.ndarray) -> np.ndarray:
    """For each person row, the largest share of its box that one of the `others` boxes of the
    same frame covers (raw detections, video pixels)."""
    shares = np.zeros(len(people))
    if len(others) == 0:
        return shares
    # Both sorted by frame, so each frame's rows are one slice, found by binary search.
    order = np.argsort(people["frame"], kind="stable")
    people_frames = people["frame"][order]
    others = np.sort(others, order="frame")
    corners = ("det_x0", "det_y0", "det_x1", "det_y1")
    for frame in np.intersect1d(people_frames, others["frame"]):
        mine = order[_frame_slice(people_frames, frame)]
        theirs = others[_frame_slice(others["frame"], frame)]
        p = np.column_stack([people[k][mine] for k in corners])
        o = np.column_stack([theirs[k] for k in corners])
        width = np.minimum(p[:, None, 2], o[None, :, 2]) - np.maximum(p[:, None, 0], o[None, :, 0])
        height = np.minimum(p[:, None, 3], o[None, :, 3]) - np.maximum(p[:, None, 1], o[None, :, 1])
        overlap = np.clip(width, 0, None) * np.clip(height, 0, None)
        area = np.maximum((p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]), 1e-6)
        shares[mine] = (overlap / area[:, None]).max(axis=1)
    return shares


def _frame_slice(sorted_frames: np.ndarray, frame: int) -> slice:
    """The rows of `frame` in an array sorted by frame."""
    return slice(
        np.searchsorted(sorted_frames, frame, side="left"),
        np.searchsorted(sorted_frames, frame, side="right"),
    )


def _add_zones(rows: np.ndarray, scene: SceneMap | None) -> None:
    """Scene-map zones of every ground point, and how each moving object's heading agrees with
    its lane's direction."""
    rows["lane"] = -1
    rows["crossing"] = -1
    rows["agreement"] = np.nan
    if scene is None or len(rows) == 0:
        return
    ground = ground_points(rows)
    rows["on_road"] = scene.on_road(ground)
    rows["lane"] = scene.lane_index(ground)
    rows["crossing"] = scene.crossing_index(ground)
    rows["junction"] = scene.in_junction(ground)
    rows["parking"] = scene.in_parking(ground)
    rows["bus_stop"] = scene.at_bus_stop(ground)

    velocity = np.column_stack([rows["vx"], rows["vy"]])
    heading = velocity / np.maximum(np.linalg.norm(velocity, axis=1, keepdims=True), 1e-9)
    flow = scene.flow_direction(rows["lane"].astype(np.int64), ground)
    judged = rows["moving"] & (rows["lane"] >= 0)
    rows["agreement"][judged] = (heading[judged] * flow[judged]).sum(axis=1)
